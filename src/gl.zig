//! Central GL: An implementation of opengl 1.3

pub const Context = struct {
    gpa: std.mem.Allocator,
    bound_render_target: centralgpu.Image = undefined,

    clear_color: u32 = 0xff_00_00_00,
    viewport: struct { x: i32, y: i32, width: isize, height: isize } = undefined,
    scissor: struct { x: i32, y: i32, width: isize, height: isize } = undefined,

    should_clear_color_attachment: bool = false,

    modelview_matrix_stack: [8][16]f32 = [_][16]f32{.{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    }} ** 8,
    modelview_matrix_stack_index: usize = 0,

    projection_matrix_stack: [8][16]f32 = [_][16]f32{.{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    }} ** 8,
    projection_matrix_stack_index: usize = 0,

    matrix_mode: enum {
        modelview,
        projection,
    } = .modelview,

    triangle_count: usize = 0,
    triangle_positions: std.ArrayListUnmanaged([3]centralgpu.WarpVec4(f32)) = .empty,
    triangle_colors: std.ArrayListUnmanaged([3]u32) = .empty,
    triangle_tex_coords: std.ArrayListUnmanaged([3][2]f32) = .empty,

    textures: std.ArrayListUnmanaged(struct {
        texture_data: []centralgpu.Rgba32,
        descriptor: centralgpu.ImageDescriptor,
    }) = .empty,
    texture_binding: [4]u32 = @splat(0),
    ///Maps from GL_TEXTURE_0..80
    texture_units: [80]u32 = @splat(0),
    texture_unit_active: u32 = 0,

    //Vertex Registers
    vertex_colour: u32 = 0xff_ff_ff_ff,
    vertex_uv: [2]f32 = @splat(0),

    //Vertex scratch
    vertex_scratch: std.MultiArrayList(struct {
        position: [4]f32,
        colour: u32,
        uv: [2]f32,
    }) = .empty,

    draw_commands: std.ArrayListUnmanaged(DrawCommandState) = .empty,

    has_begun: bool = false,
    primitive_mode: enum {
        triangle_list,
        triangle_strip,
        triangle_fan,
        quad_list,
        polygon,
    } = .triangle_list,

    flush_callback: ?*const fn () void = null,
};

const DrawCommandState = struct {
    triangle_id_start: u32,
    triangle_count: u32,

    scratch_vertex_start: u32,
    scratch_vertex_end: u32,

    image_base: [*]const u8,
    image_descriptor: centralgpu.ImageDescriptor,
};

pub var current_context: ?*Context = null;

pub export fn glClearColor(r: f32, g: f32, b: f32, a: f32) void {
    const context = current_context.?;

    context.clear_color = @bitCast(centralgpu.Rgba32.fromNormalized(.{ r, g, b, a }));
}

pub export fn glClear(flags: u32) callconv(.c) void {
    const context = current_context.?;

    if (flags & GL_COLOR_BUFFER_BIT != 0) {
        context.should_clear_color_attachment = true;
    }
}

pub export fn glViewport(x: i32, y: i32, w: isize, h: isize) callconv(.c) void {
    const context = current_context.?;

    context.viewport = .{
        .x = x,
        .y = y,
        .width = w,
        .height = h,
    };
}

pub export fn glDepthRange(
    near_val: f64,
    far_val: f64,
) callconv(.c) void {
    log.info("glDepthRange: near: {}, far: {}", .{ near_val, far_val });
}

pub export fn glDepthFunc() callconv(.c) void {}

pub export fn glDepthMask() callconv(.c) void {}

pub export fn glGetIntegerv(
    pname: i32,
    params: [*c]i32,
) callconv(.c) void {
    switch (pname) {
        GL_MAX_TEXTURE_SIZE => {
            params.* = @intCast(std.math.maxInt(u16));
        },
        else => {
            @panic("");
        },
    }
}

pub export fn glFogi() callconv(.c) void {}
pub export fn glFogf() callconv(.c) void {}
pub export fn glFogfv() callconv(.c) void {}

pub export fn glScissor(x: i32, y: i32, w: isize, h: isize) callconv(.c) void {
    const context = current_context.?;

    context.scissor = .{
        .x = x,
        .y = y,
        .width = w,
        .height = h,
    };
}

pub export fn glTexSubImage2D() void {}

pub export fn glTexImage2D(
    target: i32,
    level: i32,
    components: i32,
    width: isize,
    height: isize,
    border: i32,
    format: i32,
    _type: i32,
    data: *const anyopaque,
) callconv(.c) void {
    _ = components; // autofix
    _ = format; // autofix
    _ = _type; // autofix
    const context = current_context.?;
    const texture = &context.textures.items[context.texture_binding[@intCast(target - GL_TEXTURE_2D)] - 1];

    // std.debug.assert(std.math.isPowerOfTwo(width));
    // std.debug.assert(std.math.isPowerOfTwo(height));

    const padded_width = centralgpu.computeTargetPaddedSize(@intCast(width));
    const padded_height = centralgpu.computeTargetPaddedSize(@intCast(height));

    const dest_data_size: usize = @intCast(padded_width * padded_height * @sizeOf(u32));

    const src_data_size: usize = @intCast(width * height * 3);

    const image_data = std.heap.page_allocator.alloc(centralgpu.Rgba32, dest_data_size) catch @panic("");

    const source_data_ptr: [*]const u8 = @ptrCast(data);
    const source_data: []const u8 = source_data_ptr[0..src_data_size];

    var y: usize = 0;

    while (y < height) : (y += 1) {
        var x: usize = 0;

        while (x < width) : (x += 1) {
            const src_index = x + y * @as(usize, @intCast(width));
            const index = centralgpu.mortonEncode(@splat(@intCast(x)), @splat(@intCast(y)))[0];

            image_data[index] = .{
                .r = source_data[src_index * 3 + 0],
                .g = source_data[src_index * 3 + 1],
                .b = source_data[src_index * 3 + 2],
                .a = 255,
            };
        }
    }

    texture.texture_data = image_data;
    texture.descriptor = .{
        .rel_ptr = 0,
        .width_log2 = @intCast(std.math.log2_int(usize, @intCast(width))),
        .height_log2 = @intCast(std.math.log2_int(usize, @intCast(height))),
        .sampler_filter = .nearest,
        .sampler_address_mode = .repeat,
        .border_colour_shift_amount = 4,
    };

    _ = level; // autofix
    _ = border; // autofix
}

pub export fn glCopyTexSubImage2D(
    target: i32,
    level: i32,
    x_offset: i32,
    y_offset: i32,
    x: i32,
    y: i32,
    width: isize,
    height: isize,
) void {
    _ = target; // autofix
    _ = level; // autofix
    _ = x_offset; // autofix
    _ = y_offset; // autofix
    _ = x; // autofix
    _ = y; // autofix
    _ = width; // autofix
    _ = height; // autofix

}

pub export fn glGenTextures(
    n: isize,
    textures: [*c]u32,
) callconv(.c) void {
    const context = current_context.?;

    const id_start: u32 = @intCast(context.textures.items.len + 1);

    context.textures.appendNTimes(context.gpa, undefined, @intCast(n)) catch @panic("oom");

    for (0..@as(usize, @intCast(n))) |k| {
        textures[k] = id_start + @as(u32, @intCast(k));
    }
}

pub export fn glDeleteTextures(
    n: isize,
    textures: [*c]u32,
) callconv(.c) void {
    _ = n; // autofix
    _ = textures; // autofix
    //TODO: free texture memory
}

pub export fn glActiveTexture(
    texture: i32,
) callconv(.c) void {
    const context = current_context.?;

    const active_index: u32 = @intCast(texture - GL_TEXTURE0);

    if (context.texture_unit_active != active_index) {
        context.texture_unit_active = active_index;
    }
}

pub export fn glBindTexture(
    target: i32,
    texture: u32,
) callconv(.c) void {
    const context = current_context.?;

    const binding_index: usize = @intCast(target - GL_TEXTURE_2D);

    if (texture != context.texture_units[context.texture_unit_active]) {
        context.texture_binding[binding_index] = texture;
        context.texture_units[context.texture_unit_active] = texture;
    }
}

pub export fn glTexParameteri(
    target: i32,
    pname: i32,
    params: i32,
) callconv(.c) void {
    _ = target; // autofix
    _ = pname; // autofix
    _ = params; // autofix
}

pub export fn glGetTexParameteriv(
    target: i32,
    pname: i32,
    params: [*c]i32,
) callconv(.c) void {
    _ = target; // autofix
    _ = pname; // autofix
    _ = params; // autofix
}

pub export fn glTexParameterf(
    target: i32,
    pname: i32,
    param: f32,
) callconv(.c) void {
    _ = param; // autofix
    _ = target; // autofix
    _ = pname; // autofix
}

pub export fn glGetTexParameterfv(
    target: i32,
    pname: i32,
    params: [*c]f32,
) callconv(.c) void {
    _ = target; // autofix
    _ = pname; // autofix
    _ = params; // autofix
}

pub export fn glGenerateMipmap(
    target: i32,
) callconv(.c) void {
    _ = target; // autofix
}

pub export fn glBegin(flags: u32) callconv(.c) void {
    const context = current_context.?;

    const previous_polygon_mode = context.primitive_mode;

    var new_polygon_mode: @TypeOf(context.primitive_mode) = previous_polygon_mode;

    switch (flags) {
        GL_TRIANGLES => {
            new_polygon_mode = .triangle_list;
        },
        GL_QUADS => {
            new_polygon_mode = .quad_list;
        },
        GL_TRIANGLE_STRIP => {
            new_polygon_mode = .triangle_strip;
        },
        GL_TRIANGLE_FAN => {
            new_polygon_mode = .triangle_fan;
        },
        GL_POLYGON => {
            new_polygon_mode = .polygon;
        },
        else => {
            std.log.info("flags: {}", .{flags});
            @panic("glBegin: flags not supported. centralgl only supports GL_TRIANGLES");
        },
    }

    std.debug.assert(!context.has_begun);

    context.has_begun = true;

    const need_new_command = context.draw_commands.items.len == 0 or new_polygon_mode != previous_polygon_mode;

    if (need_new_command) {
        startCommand(context);
    }

    context.primitive_mode = new_polygon_mode;
}

pub export fn glEnd() callconv(.c) void {
    const context = current_context.?;

    std.debug.assert(context.has_begun);

    endCommand(context);

    flushPrimitives(context);

    context.has_begun = false;
}

fn startCommand(context: *Context) void {
    if (!context.has_begun) {
        return;
    }

    const command = context.draw_commands.addOne(context.gpa) catch @panic("oom");

    const triangle_id_offset: usize = 0;

    command.triangle_id_start = @intCast(context.triangle_count + triangle_id_offset);
    command.triangle_count = 0;

    command.scratch_vertex_start = @intCast(context.vertex_scratch.len);
    command.scratch_vertex_end = command.scratch_vertex_start;

    if (context.texture_units[context.texture_unit_active] != 0) {
        const active_texture = &context.textures.items[context.texture_units[context.texture_unit_active] - 1];

        command.image_base = @ptrCast(active_texture.texture_data.ptr);
        command.image_descriptor = active_texture.descriptor;
    } else {
        //TODO: handle binding texture 0
    }
}

fn endCommand(context: *Context) void {
    if (!context.has_begun) {
        return;
    }

    if (context.draw_commands.items.len == 0) {
        return;
    }

    const last_command = &context.draw_commands.items[context.draw_commands.items.len - 1];

    last_command.scratch_vertex_end = @intCast(context.vertex_scratch.len);

    const command_scratch_vertex_count = last_command.scratch_vertex_end - last_command.scratch_vertex_start;

    var triangle_count: usize = 0;

    switch (context.primitive_mode) {
        .triangle_list => {
            triangle_count = command_scratch_vertex_count / 3;

            if (command_scratch_vertex_count % 3 != 0) {
                triangle_count = 0;
            }
        },
        .quad_list => {
            triangle_count = (command_scratch_vertex_count / 4) * 2;

            if (command_scratch_vertex_count % 4 != 0) {
                triangle_count = 0;
            }
        },
        .triangle_strip => {
            triangle_count = command_scratch_vertex_count - 2;
        },
        .triangle_fan => {
            triangle_count = command_scratch_vertex_count - 2;
        },
        .polygon => {
            triangle_count = command_scratch_vertex_count - 2;
        },
    }

    last_command.triangle_count = @intCast(triangle_count);
}

fn currentMatrix() *[16]f32 {
    const context = current_context.?;

    switch (context.matrix_mode) {
        .modelview => {
            return &context.modelview_matrix_stack[context.modelview_matrix_stack_index];
        },
        .projection => {
            return &context.projection_matrix_stack[context.projection_matrix_stack_index];
        },
    }
}

fn currentBelow() *[16]f32 {
    const context = current_context.?;

    switch (context.matrix_mode) {
        .modelview => {
            return &context.modelview_matrix_stack[context.modelview_matrix_stack_index -| 1];
        },
        .projection => {
            return &context.projection_matrix_stack[context.projection_matrix_stack_index -| 1];
        },
    }
}

pub export fn glTranslatef(x: f32, y: f32, z: f32) callconv(.c) void {
    const translation_matrix = matrix_math.fromTranslation(.{ x, y, z });

    glMultMatrixf(@ptrCast(&translation_matrix));
}

pub export fn glScalef(x: f32, y: f32, z: f32) callconv(.c) void {
    const scale_matrix = matrix_math.fromScale(.{ x, y, z });

    glMultMatrixf(@ptrCast(&scale_matrix));
}

pub export fn glCullFace(cull_face: i32) callconv(.c) void {
    _ = cull_face; // autofix

}

pub export fn glFrontFace(front_face: i32) callconv(.c) void {
    _ = front_face; // autofix
}

pub export fn glPolygonMode() callconv(.c) void {}

pub export fn glShadeModel() callconv(.c) void {}

pub export fn glAlphaFunc() callconv(.c) void {}

pub export fn glBlendFunc() callconv(.c) void {}

pub export fn glHint() callconv(.c) void {}

pub export fn glEnable() callconv(.c) void {}

pub export fn glDisable() callconv(.c) void {}

pub export fn glTexEnvf() callconv(.c) void {}

pub export fn glRotatef(angle_degrees: f32, _x: f32, _y: f32, _z: f32) callconv(.c) void {
    const angle = (angle_degrees * std.math.tau) / 360.0;

    const cos_angle = @cos(angle * 0.5);
    const sin_angle = @sin(angle * 0.5);

    if (false) {
        const mag = @sqrt(_x * _x + _y * _y + _z * _z);
        const x = _x / mag;
        const y = _y / mag;
        const z = _z / mag;

        const xx = x * x;
        const yy = x * y;
        const zz = z * z;

        const s = @cos(angle);
        const c = @sin(angle);

        const xy = x * y;
        const yx = xy;

        const xz = x * z;
        const yz = y * z;

        const xs = x * s;
        const ys = y * s;
        const zs = z * s;

        const M = struct {
            pub inline fn _M(row: usize, col: usize) usize {
                return col * 4 + row;
            }
        }._M;

        var rotation_matrix: [16]f32 = .{
            xx * (1 - c) + c,  xy * (1 - c) - zs, xz * (1 - c) + ys, 0,
            yx * (1 - c) + zs, yy * (1 - c) + c,  yz * (1 - c) + xs, 0,
            xz * (1 - c) - ys, yz * (1 - c) + xs, zz * (1 - c) + c,  0,
            0,                 0,                 0,                 1,
        };

        rotation_matrix = .{
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        };

        const one_c = 1 - c;
        const zx = z * x;

        rotation_matrix[M(0, 0)] = (one_c * xx) + c;
        rotation_matrix[M(0, 1)] = (one_c * xy) - zs;
        rotation_matrix[M(0, 2)] = (one_c * zx) + ys;

        rotation_matrix[M(1, 0)] = (one_c * xy) + zs;
        rotation_matrix[M(1, 1)] = (one_c * yy) + c;
        rotation_matrix[M(1, 2)] = (one_c * yz) - xs;

        rotation_matrix[M(2, 0)] = (one_c * zx) - ys;
        rotation_matrix[M(2, 1)] = (one_c * yz) + xs;
        rotation_matrix[M(2, 2)] = (one_c * zz) + c;
        // glMultTransposeMatrixf(&rotation_matrix);
        glMultMatrixf(&rotation_matrix);

        return;
    }

    var q_w = cos_angle;
    var q_x = _x * sin_angle;
    var q_y = _y * sin_angle;
    var q_z = _z * sin_angle;

    const q_x_sqr = q_x * q_x;
    const q_y_sqr = q_y * q_y;
    const q_z_sqr = q_z * q_z;
    const q_w_sqr = q_w * q_w;

    const mag_squared = q_x_sqr + q_y_sqr + q_z_sqr + q_w_sqr;
    const mag = @sqrt(mag_squared);
    q_w /= mag;
    q_x /= mag;
    q_y /= mag;
    q_z /= mag;

    const rotation_matrix: [16]f32 = .{
        2 * (q_w_sqr + q_x_sqr) - 1, 2 * (q_x * q_y - q_w * q_z), 2 * (q_x * q_z + q_w * q_y), 0,
        2 * (q_x * q_y + q_w * q_z), 2 * (q_w_sqr + q_y_sqr) - 1, 2 * (q_x * q_z - q_w * q_x), 0,
        2 * (q_x * q_z - q_w * q_y), 2 * (q_y * q_z + q_w * q_x), 2 * (q_w_sqr + q_z_sqr) - 1, 0,
        0,                           0,                           0,                           1,
    };

    glMultTransposeMatrixf(&rotation_matrix);
    // glMultMatrixf(&rotation_matrix);
}

pub export fn glMatrixMode(mode: u32) callconv(.c) void {
    const context = current_context.?;
    switch (mode) {
        GL_MODELVIEW => context.matrix_mode = .modelview,
        GL_PROJECTION => context.matrix_mode = .projection,
        else => @panic("Matrix mode is not supported"),
    }
}

pub export fn glPushMatrix() callconv(.c) void {
    const context = current_context.?;

    const previous_matrix = currentMatrix().*;

    switch (context.matrix_mode) {
        .modelview => {
            context.modelview_matrix_stack_index += 1;
        },
        .projection => {
            context.projection_matrix_stack_index += 1;
        },
    }

    currentMatrix().* = previous_matrix;
}

pub export fn glPopMatrix() callconv(.c) void {
    const context = current_context.?;

    currentMatrix().* = currentBelow().*;

    switch (context.matrix_mode) {
        .modelview => {
            context.modelview_matrix_stack_index -= 1;
        },
        .projection => {
            context.projection_matrix_stack_index -= 1;
        },
    }
}

pub export fn glLoadMatrix(matrix: *const [16]f32) callconv(.c) void {
    currentMatrix().* = matrix.*;
}

pub export fn glLoadIdentity() callconv(.c) void {
    currentMatrix().* = .{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
}

pub export fn glMultTransposeMatrixf(matrix: *const [16]f32) callconv(.c) void {
    const matrix_row_major: [16]f32 = .{
        matrix[0],  matrix[1],  matrix[2],  matrix[3],
        matrix[4],  matrix[5],  matrix[6],  matrix[7],
        matrix[8],  matrix[9],  matrix[10], matrix[11],
        matrix[12], matrix[13], matrix[14], matrix[15],
    };

    glMultMatrixf(&matrix_row_major);
}

pub export fn glMultMatrixf(matrix: *const [16]f32) callconv(.c) void {
    const current_matrix = currentMatrix();

    matmul4(current_matrix, current_matrix, matrix);
}

fn matmul4(product: *[16]f32, a: *const [16]f32, b: *const [16]f32) void {
    const M = struct {
        pub inline fn _M(row: usize, col: usize) usize {
            return col * 4 + row;
        }
    }._M;

    for (0..4) |i| {
        const ai0 = a[M(i, 0)];
        const ai1 = a[M(i, 1)];
        const ai2 = a[M(i, 2)];
        const ai3 = a[M(i, 3)];

        product[M(i, 0)] = ai0 * b[M(0, 0)] + ai1 * b[M(1, 0)] + ai2 * b[M(2, 0)] + ai3 * b[M(3, 0)];
        product[M(i, 1)] = ai0 * b[M(0, 1)] + ai1 * b[M(1, 1)] + ai2 * b[M(2, 1)] + ai3 * b[M(3, 1)];
        product[M(i, 2)] = ai0 * b[M(0, 2)] + ai1 * b[M(1, 2)] + ai2 * b[M(2, 2)] + ai3 * b[M(3, 2)];
        product[M(i, 3)] = ai0 * b[M(0, 3)] + ai1 * b[M(1, 3)] + ai2 * b[M(2, 3)] + ai3 * b[M(3, 3)];
    }
}

pub export fn glFrustum(
    _left: f64,
    _right: f64,
    _bottom: f64,
    _top: f64,
    _near_val: f64,
    _far_val: f64,
) callconv(.c) void {
    const left: f32 = @floatCast(_left);
    const right: f32 = @floatCast(_right);
    const bottom: f32 = @floatCast(_bottom);
    const top: f32 = @floatCast(_top);
    const near_val: f32 = @floatCast(_near_val);
    const far_val: f32 = @floatCast(_far_val);

    // const x = (2.0 * near_val) / (right - left);
    // const y = (2.0 * near_val) / (top - bottom);

    // const a: f32 = (right + left) / (right - left);
    // const b: f32 = (top + bottom) / (top - bottom);
    // const c: f32 = -(far_val + near_val) / (far_val - near_val);
    // const d: f32 = -(2 * far_val * near_val) / (far_val - near_val);

    const nearval = near_val;
    const farval = far_val;

    const x: f32 = (2.0 * nearval) / (right - left);
    const y: f32 = (2.0 * nearval) / (top - bottom);
    const a: f32 = (right + left) / (right - left);
    const b: f32 = (top + bottom) / (top - bottom);
    const c: f32 = -(farval + nearval) / (farval - nearval);
    const d: f32 = -(2.0 * farval * nearval) / (farval - nearval);

    var frustum_matrix: [16]f32 = undefined;

    const M = struct {
        pub inline fn _M(row: usize, col: usize) usize {
            return col * 4 + row;
        }
    }._M;

    // zig fmt: off

    frustum_matrix[M(0,0)] = x;    frustum_matrix[M(0,1)] = 0.0;  frustum_matrix[M(0,2)] = a;     frustum_matrix[M(0,3)] = 0.0;
    frustum_matrix[M(1,0)] = 0.0;  frustum_matrix[M(1,1)] = y;    frustum_matrix[M(1,2)] = b;     frustum_matrix[M(1,3)] = 0.0;
    frustum_matrix[M(2,0)] = 0.0;  frustum_matrix[M(2,1)] = 0.0;  frustum_matrix[M(2,2)] = c;     frustum_matrix[M(2,3)] = d;
    frustum_matrix[M(3,0)] = 0.0;  frustum_matrix[M(3,1)] = 0.0;  frustum_matrix[M(3,2)] = -1.0;  frustum_matrix[M(3,3)] = 0.0;

    // zig fmt: on

    // frustum_matrix = .{
    //     (2 * near_val) / (right - left), 0,                               a,  0,
    //     0,                               (2 * near_val) / (top - bottom), b,  0,
    //     0,                               0,                               c,  d,
    //     0,                               0,                               -1, 0,
    // };

    glMultMatrixf(&frustum_matrix);
    // glMultTransposeMatrixf(&frustum_matrix);
}

pub export fn glOrtho(
    _left: f64,
    _right: f64,
    _bottom: f64,
    _top: f64,
    _near_val: f64,
    _far_val: f64,
) callconv(.c) void {
    var ortho_matrix: [16]f32 = undefined;

    const left: f32 = @floatCast(_left);
    const right: f32 = @floatCast(_right);
    const bottom: f32 = @floatCast(_bottom);
    const top: f32 = @floatCast(_top);
    const near_val: f32 = @floatCast(_near_val);
    const far_val: f32 = @floatCast(_far_val);

    const t_x: f32 = -(right + left) / (right - left);
    const t_y: f32 = -(top + bottom) / (top - bottom);
    const t_z: f32 = -(far_val + near_val) / (far_val - near_val);

    ortho_matrix = .{
        2 / (right - left), 0,                  0,                         t_x,
        0,                  2 / (top - bottom), 0,                         t_y,
        0,                  0,                  -2 / (far_val - near_val), t_z,
        0,                  0,                  0,                         1,
    };

    glMultMatrixf(&ortho_matrix);
    // glMultTransposeMatrixf(&ortho_matrix);
}

pub export fn glVertex3f(x: f32, y: f32, z: f32) callconv(.c) void {
    const context = current_context.?;

    const current_modelview_matrix = context.modelview_matrix_stack[context.modelview_matrix_stack_index];
    const current_projection_matrix = context.projection_matrix_stack[context.projection_matrix_stack_index];

    const transformed = matrix_math.mulByVec4(@bitCast(current_modelview_matrix), .{ x, y, z, 1 });

    // matmul4(&matrix_product, &current_modelview_matrix, &current_projection_matrix);

    //TODO: move computation to primitive shader
    const projected_vertex = matrix_math.mulByVec4(@bitCast(current_projection_matrix), transformed);

    context.vertex_scratch.append(context.gpa, .{
        .position = projected_vertex,
        .colour = context.vertex_colour,
        .uv = context.vertex_uv,
    }) catch @panic("oom");
}

fn flushPrimitives(context: *Context) void {
    switch (context.primitive_mode) {
        .triangle_list => {
            const triangle_count = @divTrunc(context.vertex_scratch.len, 3);
            const group_count = @divTrunc(triangle_count, 8) + @intFromBool(@rem(triangle_count, 8) != 0);

            _ = context.triangle_positions.addManyAsSlice(context.gpa, group_count) catch @panic("oom");
            _ = context.triangle_colors.addManyAsSlice(context.gpa, triangle_count) catch @panic("oom");
            _ = context.triangle_tex_coords.addManyAsSlice(context.gpa, triangle_count) catch @panic("oom");

            const global_group_start = context.triangle_count / 8;
            const global_triangle_id_start = context.triangle_count;

            for (0..triangle_count) |triangle_index| {
                const group_index = @divTrunc(triangle_index, 8);

                const global_group_index = global_group_start + group_index;

                const triangle_group = &context.triangle_positions.items[global_group_index];
                const triangle_id = global_triangle_id_start + triangle_index;

                const triangle_local_index = @rem(triangle_id, 8);

                for (0..3) |tri_vertex_index| {
                    const scratch_index = triangle_index * 3 + tri_vertex_index;

                    triangle_group[tri_vertex_index].x[triangle_local_index] = context.vertex_scratch.items(.position)[scratch_index][0];
                    triangle_group[tri_vertex_index].y[triangle_local_index] = context.vertex_scratch.items(.position)[scratch_index][1];
                    triangle_group[tri_vertex_index].z[triangle_local_index] = context.vertex_scratch.items(.position)[scratch_index][2];
                    triangle_group[tri_vertex_index].w[triangle_local_index] = context.vertex_scratch.items(.position)[scratch_index][3];

                    context.triangle_colors.items[triangle_id][tri_vertex_index] = context.vertex_scratch.items(.colour)[scratch_index];
                    context.triangle_tex_coords.items[triangle_id][tri_vertex_index] = context.vertex_scratch.items(.uv)[scratch_index];

                    context.triangle_colors.items[triangle_id][tri_vertex_index] = 0xff_00_ff_00;
                }
            }

            context.triangle_count += triangle_count;
        },
        .quad_list => {
            const triangle_count = @divTrunc(context.vertex_scratch.len, 4) * 2;
            const group_count = @divTrunc(triangle_count, 8) + @intFromBool(@rem(triangle_count, 8) != 0);

            _ = context.triangle_positions.addManyAsSlice(context.gpa, group_count) catch @panic("oom");
            _ = context.triangle_colors.addManyAsSlice(context.gpa, triangle_count) catch @panic("oom");
            _ = context.triangle_tex_coords.addManyAsSlice(context.gpa, triangle_count) catch @panic("oom");

            const global_group_start = context.triangle_count / 8;
            const global_triangle_id_start = context.triangle_count;

            const quad_indices: [6]usize = .{
                0, 1, 2,
                // 2, 1, 3,
                3, 2, 0,
            };

            for (0..triangle_count) |triangle_index| {
                const group_index = @divTrunc(triangle_index, 8);

                const global_group_index = global_group_start + group_index;

                const triangle_group = &context.triangle_positions.items[global_group_index];
                const triangle_id = global_triangle_id_start + triangle_index;

                const triangle_local_index = @rem(triangle_id, 8);

                const input_quad_index = @divTrunc(triangle_index, 2);

                for (0..3) |tri_vertex_index| {
                    const stream_vertex_index = triangle_index * 3 + tri_vertex_index;

                    const scratch_index = input_quad_index * 4 + quad_indices[@rem(stream_vertex_index, 6)];

                    triangle_group[tri_vertex_index].x[triangle_local_index] = context.vertex_scratch.items(.position)[scratch_index][0];
                    triangle_group[tri_vertex_index].y[triangle_local_index] = context.vertex_scratch.items(.position)[scratch_index][1];
                    triangle_group[tri_vertex_index].z[triangle_local_index] = context.vertex_scratch.items(.position)[scratch_index][2];
                    triangle_group[tri_vertex_index].w[triangle_local_index] = context.vertex_scratch.items(.position)[scratch_index][3];

                    context.triangle_colors.items[triangle_id][tri_vertex_index] = context.vertex_scratch.items(.colour)[scratch_index];
                    context.triangle_tex_coords.items[triangle_id][tri_vertex_index] = context.vertex_scratch.items(.uv)[scratch_index];

                    context.triangle_colors.items[triangle_id][tri_vertex_index] = 0xff_00_ff_ff;
                }
            }

            context.triangle_count += triangle_count;
        },
        .triangle_strip => {
            const triangle_count = context.vertex_scratch.len - 2;
            const group_count = @divTrunc(triangle_count, 8) + @intFromBool(@rem(triangle_count, 8) != 0);

            _ = context.triangle_positions.addManyAsSlice(context.gpa, group_count) catch @panic("oom");
            _ = context.triangle_colors.addManyAsSlice(context.gpa, triangle_count) catch @panic("oom");
            _ = context.triangle_tex_coords.addManyAsSlice(context.gpa, triangle_count) catch @panic("oom");

            const global_group_start = context.triangle_count / 8;
            const global_triangle_id_start = context.triangle_count;

            for (0..triangle_count) |triangle_index| {
                const group_index = @divTrunc(triangle_index, 8);

                const global_group_index = global_group_start + group_index;

                const triangle_group = &context.triangle_positions.items[global_group_index];
                const triangle_id = global_triangle_id_start + triangle_index;

                const triangle_local_index = @rem(triangle_id, 8);

                const odd_indices: [3]usize = .{ 1, 2, 0 };
                const even_indices: [3]usize = .{ 2, 1, 0 };

                for (0..3) |tri_vertex_index| {
                    const is_even = triangle_index % 2 == 0;

                    const scratch_index = triangle_index + if (is_even) even_indices[tri_vertex_index] else odd_indices[tri_vertex_index];

                    triangle_group[tri_vertex_index].x[triangle_local_index] = context.vertex_scratch.items(.position)[scratch_index][0];
                    triangle_group[tri_vertex_index].y[triangle_local_index] = context.vertex_scratch.items(.position)[scratch_index][1];
                    triangle_group[tri_vertex_index].z[triangle_local_index] = context.vertex_scratch.items(.position)[scratch_index][2];
                    triangle_group[tri_vertex_index].w[triangle_local_index] = context.vertex_scratch.items(.position)[scratch_index][3];

                    context.triangle_colors.items[triangle_id][tri_vertex_index] = context.vertex_scratch.items(.colour)[scratch_index];
                    context.triangle_tex_coords.items[triangle_id][tri_vertex_index] = context.vertex_scratch.items(.uv)[scratch_index];

                    //debug visual
                    context.triangle_colors.items[triangle_id][tri_vertex_index] = 0xff_00_00_ff;
                }
            }

            context.triangle_count += triangle_count;
        },
        .triangle_fan, .polygon => {
            const triangle_count = context.vertex_scratch.len - 2;
            const group_count = @divTrunc(triangle_count, 8) + @intFromBool(@rem(triangle_count, 8) != 0);

            _ = context.triangle_positions.addManyAsSlice(context.gpa, group_count) catch @panic("oom");
            _ = context.triangle_colors.addManyAsSlice(context.gpa, triangle_count) catch @panic("oom");
            _ = context.triangle_tex_coords.addManyAsSlice(context.gpa, triangle_count) catch @panic("oom");

            const global_group_start = context.triangle_count / 8;
            const global_triangle_id_start = context.triangle_count;

            for (0..triangle_count) |triangle_index| {
                const group_index = @divTrunc(triangle_index, 8);

                const global_group_index = global_group_start + group_index;

                const triangle_group = &context.triangle_positions.items[global_group_index];
                const triangle_id = global_triangle_id_start + triangle_index;

                const triangle_local_index = @rem(triangle_id, 8);

                for (0..3) |tri_vertex_index| {
                    const scratch_index = switch (tri_vertex_index) {
                        0 => 0,
                        1 => if (triangle_index > 0) triangle_index + 1 else 1,
                        2 => if (triangle_index > 0) triangle_index + 2 else 2,
                        else => unreachable,
                    };

                    triangle_group[tri_vertex_index].x[triangle_local_index] = context.vertex_scratch.items(.position)[scratch_index][0];
                    triangle_group[tri_vertex_index].y[triangle_local_index] = context.vertex_scratch.items(.position)[scratch_index][1];
                    triangle_group[tri_vertex_index].z[triangle_local_index] = context.vertex_scratch.items(.position)[scratch_index][2];
                    triangle_group[tri_vertex_index].w[triangle_local_index] = context.vertex_scratch.items(.position)[scratch_index][3];

                    context.triangle_colors.items[triangle_id][tri_vertex_index] = context.vertex_scratch.items(.colour)[scratch_index];
                    context.triangle_tex_coords.items[triangle_id][tri_vertex_index] = context.vertex_scratch.items(.uv)[scratch_index];

                    context.triangle_colors.items[triangle_id][tri_vertex_index] = 0xff_ff_ff_00;
                }
            }

            context.triangle_count += triangle_count;
        },
    }

    context.vertex_scratch.clearRetainingCapacity();
}

pub export fn glColor4f(r: f32, g: f32, b: f32, a: f32) callconv(.c) void {
    const context = current_context.?;

    context.vertex_colour = @bitCast(centralgpu.Rgba32.fromNormalized(.{ r, g, b, a }));
}

pub export fn glTexCoord2f(u: f32, v: f32) callconv(.c) void {
    const context = current_context.?;

    context.vertex_uv = .{ u, v };
}

pub export fn glVertex2f(x: f32, y: f32) callconv(.c) void {
    glVertex3f(x, y, 0);
}

pub export fn glVertex2i(x: i32, y: i32) callconv(.c) void {
    glVertex3f(@floatFromInt(x), @floatFromInt(y), 0);
}

pub export fn glVertex3fv(vert: [*]const f32) callconv(.c) void {
    glVertex3f(vert[0], vert[1], vert[2]);
}

pub export fn glColor3f(r: f32, g: f32, b: f32) callconv(.c) void {
    glColor4f(r, g, b, 1);
}

pub export fn glColor3fv(col: [*]f32) callconv(.c) void {
    glColor4f(col[0], col[1], col[2], 1);
}

pub export fn glColor4fv(col: [*]f32) callconv(.c) void {
    glColor4f(col[0], col[1], col[2], col[3]);
}

pub export fn glColor4ubv(color: [*]u8) callconv(.c) void {
    glColor4f(
        @as(f32, @floatFromInt(color[0])) / 255.0,
        @as(f32, @floatFromInt(color[1])) / 255.0,
        @as(f32, @floatFromInt(color[2])) / 255.0,
        @as(f32, @floatFromInt(color[3])) / 255.0,
    );
}

pub export fn glFlush() callconv(.c) void {
    const context = current_context.?;

    defer {
        if (context.flush_callback != null) {
            context.flush_callback.?();
        }
    }

    if (context.should_clear_color_attachment) {
        const actual_width = centralgpu.computeTargetPaddedSize(context.bound_render_target.width);
        _ = actual_width; // autofix
        const actual_height = centralgpu.computeTargetPaddedSize(context.bound_render_target.height);
        _ = actual_height; // autofix

        @memset(
            context.bound_render_target.pixel_ptr[0 .. context.bound_render_target.width * context.bound_render_target.height],
            @bitCast(context.clear_color),
        );
    }

    std.log.info("draw_cmds: {}", .{context.draw_commands.items.len});

    const debug_view = true;

    context.scissor = .{ .x = context.viewport.x, .y = context.viewport.y, .width = context.viewport.width, .height = context.viewport.height };

    if (debug_view) {

        //debug all vertices in one pipeline

        const triangle_group_begin = 0;
        const triangle_group_count = context.triangle_count / 8 + @intFromBool(context.triangle_count % 8 != 0);

        for (triangle_group_begin..triangle_group_begin + triangle_group_count) |triangle_group_id| {
            const unfiorms: centralgpu.Uniforms = .{
                .vertex_positions = context.triangle_positions.items,
                .vertex_colours = context.triangle_colors.items,
                .vertex_texture_coords = context.triangle_tex_coords.items,
                .image_base = undefined,
                .image_descriptor = undefined,
            };

            var triangle_mask: centralgpu.WarpRegister(bool) = @splat(true);

            var triangle_id: centralgpu.WarpRegister(u32) = std.simd.iota(u32, 8);

            triangle_id += @splat(@intCast(triangle_group_id * 8));

            triangle_mask = centralgpu.vectorBoolAnd(triangle_mask, triangle_id >= @as(centralgpu.WarpRegister(u32), @splat(0)));
            triangle_mask = centralgpu.vectorBoolAnd(triangle_mask, triangle_id < @as(centralgpu.WarpRegister(u32), @splat(@intCast(context.triangle_count))));

            const viewport_x: f32 = @floatFromInt(context.viewport.x);
            const viewport_y: f32 = @floatFromInt(context.viewport.y);

            const viewport_width: f32 = @floatFromInt(context.viewport.width);
            const viewport_height: f32 = @floatFromInt(context.viewport.height);

            const projected_triangles = centralgpu.processGeometry(
                .{
                    .viewport_transform = .{
                        .translation_x = viewport_x + viewport_width * 0.5,
                        .translation_y = viewport_y + viewport_height * 0.5,
                        .scale_x = viewport_width * 0.5,
                        .scale_y = viewport_height * 0.5,
                    },
                },
                unfiorms,
                triangle_group_id * 8,
                triangle_mask,
            );

            if (std.simd.countTrues(projected_triangles.mask) == 0) {
                continue;
            }

            centralgpu.rasterize(
                .{
                    .scissor_min_x = context.scissor.x,
                    .scissor_min_y = context.scissor.y,
                    .scissor_max_x = @truncate(context.scissor.width),
                    .scissor_max_y = @truncate(context.scissor.height),
                    .render_target = context.bound_render_target,
                },
                unfiorms,
                triangle_group_id * 8,
                projected_triangles,
            );
        }
    } else {
        for (context.draw_commands.items) |draw_command| {
            if (draw_command.triangle_count == 0) {
                continue;
            }

            const triangle_group_begin = draw_command.triangle_id_start / 8;
            const triangle_group_count = draw_command.triangle_count / 8 + @intFromBool(draw_command.triangle_count % 8 != 0);

            std.log.info("draw_cmd: triangle_begin: {}", .{draw_command.triangle_id_start});
            std.log.info("draw_cmd: triangle_count: {}", .{draw_command.triangle_count});

            std.log.info("draw_cmd: group_begin: {}", .{triangle_group_begin});
            std.log.info("draw_cmd: group_count: {}", .{triangle_group_count});

            for (draw_command.triangle_id_start..draw_command.triangle_id_start + draw_command.triangle_count) |tri_id| {
                for (0..3) |tri_vert| {
                    const tri_group = context.triangle_positions.items[@divTrunc(tri_id, 8)][tri_vert];

                    std.log.info("triangle({}): pos_{}: {}, {}, {}, {}", .{
                        tri_id,
                        tri_vert,
                        tri_group.x[@rem(tri_id, 8)],
                        tri_group.y[@rem(tri_id, 8)],
                        tri_group.z[@rem(tri_id, 8)],
                        tri_group.w[@rem(tri_id, 8)],
                    });
                }
            }

            for (triangle_group_begin..triangle_group_begin + triangle_group_count) |triangle_group_id| {
                const unfiorms: centralgpu.Uniforms = .{
                    .vertex_positions = context.triangle_positions.items,
                    .vertex_colours = context.triangle_colors.items,
                    .vertex_texture_coords = context.triangle_tex_coords.items,
                    .image_base = @ptrCast(draw_command.image_base),
                    .image_descriptor = draw_command.image_descriptor,
                };

                var triangle_mask: centralgpu.WarpRegister(bool) = @splat(true);

                var triangle_id: centralgpu.WarpRegister(u32) = std.simd.iota(u32, 8);

                triangle_id += @splat(@intCast(triangle_group_id * 8));

                triangle_mask = centralgpu.vectorBoolAnd(triangle_mask, triangle_id >= @as(centralgpu.WarpRegister(u32), @splat(draw_command.triangle_id_start)));
                triangle_mask = centralgpu.vectorBoolAnd(triangle_mask, triangle_id < @as(centralgpu.WarpRegister(u32), @splat(draw_command.triangle_id_start + draw_command.triangle_count)));

                const viewport_x: f32 = @floatFromInt(context.viewport.x);
                const viewport_y: f32 = @floatFromInt(context.viewport.y);

                const viewport_width: f32 = @floatFromInt(context.viewport.width);
                const viewport_height: f32 = @floatFromInt(context.viewport.height);

                const projected_triangles = centralgpu.processGeometry(
                    .{
                        .viewport_transform = .{
                            .translation_x = viewport_x + viewport_width * 0.5,
                            .translation_y = viewport_y + viewport_height * 0.5,
                            .scale_x = viewport_width * 0.5,
                            .scale_y = viewport_height * 0.5,
                        },
                    },
                    unfiorms,
                    triangle_group_id * 8,
                    triangle_mask,
                );

                if (std.simd.countTrues(projected_triangles.mask) == 0) {
                    continue;
                }

                centralgpu.rasterize(
                    .{
                        .scissor_min_x = context.scissor.x,
                        .scissor_min_y = context.scissor.y,
                        .scissor_max_x = @truncate(context.scissor.width),
                        .scissor_max_y = @truncate(context.scissor.height),
                        .render_target = context.bound_render_target,
                    },
                    unfiorms,
                    triangle_group_id * 8,
                    projected_triangles,
                );
            }
        }
    }

    context.triangle_positions.clearRetainingCapacity();
    context.triangle_colors.clearRetainingCapacity();
    context.triangle_tex_coords.clearRetainingCapacity();
    context.draw_commands.clearRetainingCapacity();
    context.vertex_scratch.clearRetainingCapacity();
    context.triangle_count = 0;
    context.should_clear_color_attachment = false;
}

pub export fn glFinish() callconv(.c) void {
    glFlush();
}

pub const GL_TEXTURE_ENV: i32 = 0x2300;
pub const GL_TEXTURE_ENV_MODE: i32 = 0x2200;
pub const GL_TEXTURE_1D: i32 = 0x0DE0;
pub const GL_TEXTURE_2D: i32 = 0x0DE1;
pub const GL_TEXTURE_WRAP_S: i32 = 0x2802;
pub const GL_TEXTURE_WRAP_T: i32 = 0x2803;
pub const GL_TEXTURE_MAG_FILTER: i32 = 0x2800;
pub const GL_TEXTURE_MIN_FILTER: i32 = 0x2801;
pub const GL_TEXTURE_ENV_COLOR: i32 = 0x2201;
pub const GL_TEXTURE_GEN_S: i32 = 0x0C60;
pub const GL_TEXTURE_GEN_T: i32 = 0x0C61;
pub const GL_TEXTURE_GEN_R: i32 = 0x0C62;
pub const GL_TEXTURE_GEN_Q: i32 = 0x0C63;
pub const GL_TEXTURE_GEN_MODE: i32 = 0x2500;
pub const GL_TEXTURE_BORDER_COLOR: i32 = 0x1004;
pub const GL_TEXTURE_WIDTH: i32 = 0x1000;
pub const GL_TEXTURE_HEIGHT: i32 = 0x1001;
pub const GL_TEXTURE_BORDER: i32 = 0x1005;
pub const GL_TEXTURE_COMPONENTS: i32 = 0x1003;
pub const GL_TEXTURE_RED_SIZE: i32 = 0x805C;
pub const GL_TEXTURE_GREEN_SIZE: i32 = 0x805D;
pub const GL_TEXTURE_BLUE_SIZE: i32 = 0x805E;
pub const GL_TEXTURE_ALPHA_SIZE: i32 = 0x805F;
pub const GL_TEXTURE_LUMINANCE_SIZE: i32 = 0x8060;
pub const GL_TEXTURE_INTENSITY_SIZE: i32 = 0x8061;
pub const GL_NEAREST_MIPMAP_NEAREST: i32 = 0x2700;
pub const GL_NEAREST_MIPMAP_LINEAR: i32 = 0x2702;
pub const GL_LINEAR_MIPMAP_NEAREST: i32 = 0x2701;
pub const GL_LINEAR_MIPMAP_LINEAR: i32 = 0x2703;
pub const GL_OBJECT_LINEAR: i32 = 0x2401;
pub const GL_OBJECT_PLANE: i32 = 0x2501;
pub const GL_EYE_LINEAR: i32 = 0x2400;
pub const GL_EYE_PLANE: i32 = 0x2502;
pub const GL_SPHERE_MAP: i32 = 0x2402;
pub const GL_DECAL: i32 = 0x2101;
pub const GL_MODULATE: i32 = 0x2100;
pub const GL_NEAREST: i32 = 0x2600;
pub const GL_REPEAT: i32 = 0x2901;
pub const GL_CLAMP: i32 = 0x2900;
pub const GL_S: i32 = 0x2000;
pub const GL_T: i32 = 0x2001;
pub const GL_R: i32 = 0x2002;
pub const GL_Q: i32 = 0x2003;

pub const GL_FRONT_LEFT: i32 = 0x0400;
pub const GL_FRONT_RIGHT: i32 = 0x0401;
pub const GL_BACK_LEFT: i32 = 0x0402;
pub const GL_BACK_RIGHT: i32 = 0x0403;
pub const GL_AUX0: i32 = 0x0409;
pub const GL_AUX1: i32 = 0x040A;
pub const GL_AUX2: i32 = 0x040B;
pub const GL_AUX3: i32 = 0x040C;
pub const GL_COLOR_INDEX: i32 = 0x1900;
pub const GL_RED: i32 = 0x1903;
pub const GL_GREEN: i32 = 0x1904;
pub const GL_BLUE: i32 = 0x1905;
pub const GL_ALPHA: i32 = 0x1906;
pub const GL_LUMINANCE: i32 = 0x1909;
pub const GL_LUMINANCE_ALPHA: i32 = 0x190A;
pub const GL_ALPHA_BITS: i32 = 0x0D55;
pub const GL_RED_BITS: i32 = 0x0D52;
pub const GL_GREEN_BITS: i32 = 0x0D53;
pub const GL_BLUE_BITS: i32 = 0x0D54;
pub const GL_INDEX_BITS: i32 = 0x0D51;
pub const GL_SUBPIXEL_BITS: i32 = 0x0D50;
pub const GL_AUX_BUFFERS: i32 = 0x0C00;
pub const GL_READ_BUFFER: i32 = 0x0C02;
pub const GL_DRAW_BUFFER: i32 = 0x0C01;
pub const GL_DOUBLEBUFFER: i32 = 0x0C32;
pub const GL_STEREO: i32 = 0x0C33;
pub const GL_BITMAP: i32 = 0x1A00;
pub const GL_COLOR: i32 = 0x1800;
pub const GL_DEPTH: i32 = 0x1801;
pub const GL_STENCIL: i32 = 0x1802;
pub const GL_DITHER: i32 = 0x0BD0;
pub const GL_RGB: i32 = 0x1907;
pub const GL_RGBA: i32 = 0x1908;

pub const GL_RESCALE_NORMAL: i32 = 0x803A;
pub const GL_CLAMP_TO_EDGE: i32 = 0x812F;
pub const GL_MAX_ELEMENTS_VERTICES: i32 = 0x80E8;
pub const GL_MAX_ELEMENTS_INDICES: i32 = 0x80E9;
pub const GL_BGR: i32 = 0x80E0;
pub const GL_BGRA: i32 = 0x80E1;
pub const GL_UNSIGNED_BYTE_3_3_2: i32 = 0x8032;
pub const GL_UNSIGNED_BYTE_2_3_3_REV: i32 = 0x8362;
pub const GL_UNSIGNED_SHORT_5_6_5: i32 = 0x8363;
pub const GL_UNSIGNED_SHORT_5_6_5_REV: i32 = 0x8364;
pub const GL_UNSIGNED_SHORT_4_4_4_4: i32 = 0x8033;
pub const GL_UNSIGNED_SHORT_4_4_4_4_REV: i32 = 0x8365;
pub const GL_UNSIGNED_SHORT_5_5_5_1: i32 = 0x8034;
pub const GL_UNSIGNED_SHORT_1_5_5_5_REV: i32 = 0x8366;
pub const GL_UNSIGNED_INT_8_8_8_8: i32 = 0x8035;
pub const GL_UNSIGNED_INT_8_8_8_8_REV: i32 = 0x8367;
pub const GL_UNSIGNED_INT_10_10_10_2: i32 = 0x8036;
pub const GL_UNSIGNED_INT_2_10_10_10_REV: i32 = 0x8368;
pub const GL_LIGHT_MODEL_COLOR_CONTROL: i32 = 0x81F8;
pub const GL_SINGLE_COLOR: i32 = 0x81F9;
pub const GL_SEPARATE_SPECULAR_COLOR: i32 = 0x81FA;
pub const GL_TEXTURE_MIN_LOD: i32 = 0x813A;
pub const GL_TEXTURE_MAX_LOD: i32 = 0x813B;
pub const GL_TEXTURE_BASE_LEVEL: i32 = 0x813C;
pub const GL_TEXTURE_MAX_LEVEL: i32 = 0x813D;
pub const GL_SMOOTH_POINT_SIZE_RANGE: i32 = 0x0B12;
pub const GL_SMOOTH_POINT_SIZE_GRANULARITY: i32 = 0x0B13;
pub const GL_SMOOTH_LINE_WIDTH_RANGE: i32 = 0x0B22;
pub const GL_SMOOTH_LINE_WIDTH_GRANULARITY: i32 = 0x0B23;
pub const GL_ALIASED_POINT_SIZE_RANGE: i32 = 0x846D;
pub const GL_ALIASED_LINE_WIDTH_RANGE: i32 = 0x846E;
pub const GL_PACK_SKIP_IMAGES: i32 = 0x806B;
pub const GL_PACK_IMAGE_HEIGHT: i32 = 0x806C;
pub const GL_UNPACK_SKIP_IMAGES: i32 = 0x806D;
pub const GL_UNPACK_IMAGE_HEIGHT: i32 = 0x806E;
pub const GL_TEXTURE_3D: i32 = 0x806F;
pub const GL_PROXY_TEXTURE_3D: i32 = 0x8070;
pub const GL_TEXTURE_DEPTH: i32 = 0x8071;
pub const GL_TEXTURE_WRAP_R: i32 = 0x8072;
pub const GL_MAX_3D_TEXTURE_SIZE: i32 = 0x8073;
pub const GL_TEXTURE_BINDING_3D: i32 = 0x806A;

pub const GL_BYTE: i32 = 0x1400;
pub const GL_UNSIGNED_BYTE: i32 = 0x1401;
pub const GL_SHORT: i32 = 0x1402;
pub const GL_UNSIGNED_SHORT: i32 = 0x1403;
pub const GL_INT: i32 = 0x1404;
pub const GL_UNSIGNED_INT: i32 = 0x1405;
pub const GL_FLOAT: i32 = 0x1406;
pub const GL_2_BYTES: i32 = 0x1407;
pub const GL_3_BYTES: i32 = 0x1408;
pub const GL_4_BYTES: i32 = 0x1409;
pub const GL_DOUBLE: i32 = 0x140A;

pub const GL_TEXTURE0: i32 = 0x84C0;
pub const GL_TEXTURE1: i32 = 0x84C1;
pub const GL_TEXTURE2: i32 = 0x84C2;
pub const GL_TEXTURE3: i32 = 0x84C3;
pub const GL_TEXTURE4: i32 = 0x84C4;
pub const GL_TEXTURE5: i32 = 0x84C5;
pub const GL_TEXTURE6: i32 = 0x84C6;
pub const GL_TEXTURE7: i32 = 0x84C7;
pub const GL_TEXTURE8: i32 = 0x84C8;
pub const GL_TEXTURE9: i32 = 0x84C9;
pub const GL_TEXTURE10: i32 = 0x84CA;
pub const GL_TEXTURE11: i32 = 0x84CB;
pub const GL_TEXTURE12: i32 = 0x84CC;
pub const GL_TEXTURE13: i32 = 0x84CD;
pub const GL_TEXTURE14: i32 = 0x84CE;
pub const GL_TEXTURE15: i32 = 0x84CF;
pub const GL_TEXTURE16: i32 = 0x84D0;
pub const GL_TEXTURE17: i32 = 0x84D1;
pub const GL_TEXTURE18: i32 = 0x84D2;
pub const GL_TEXTURE19: i32 = 0x84D3;
pub const GL_TEXTURE20: i32 = 0x84D4;
pub const GL_TEXTURE21: i32 = 0x84D5;
pub const GL_TEXTURE22: i32 = 0x84D6;
pub const GL_TEXTURE23: i32 = 0x84D7;
pub const GL_TEXTURE24: i32 = 0x84D8;
pub const GL_TEXTURE25: i32 = 0x84D9;
pub const GL_TEXTURE26: i32 = 0x84DA;
pub const GL_TEXTURE27: i32 = 0x84DB;
pub const GL_TEXTURE28: i32 = 0x84DC;
pub const GL_TEXTURE29: i32 = 0x84DD;
pub const GL_TEXTURE30: i32 = 0x84DE;
pub const GL_TEXTURE31: i32 = 0x84DF;
pub const GL_ACTIVE_TEXTURE: i32 = 0x84E0;
pub const GL_CLIENT_ACTIVE_TEXTURE: i32 = 0x84E1;
pub const GL_MAX_TEXTURE_UNITS: i32 = 0x84E2;

pub const GL_TRIANGLES: u32 = 0x0004;
pub const GL_TRIANGLE_STRIP: u32 = 0x0005;
pub const GL_TRIANGLE_FAN: u32 = 0x0006;
pub const GL_QUADS: u32 = 0x0007;
pub const GL_POLYGON: u32 = 0x0009;

pub const GL_COLOR_BUFFER_BIT: u32 = 0x00004000;

pub const GL_MAX_LIST_NESTING: i32 = 0x0B31;
pub const GL_MAX_EVAL_ORDER: i32 = 0x0D30;
pub const GL_MAX_LIGHTS: i32 = 0x0D31;
pub const GL_MAX_CLIP_PLANES: i32 = 0x0D32;
pub const GL_MAX_TEXTURE_SIZE: i32 = 0x0D33;
pub const GL_MAX_PIXEL_MAP_TABLE: i32 = 0x0D34;
pub const GL_MAX_ATTRIB_STACK_DEPTH: i32 = 0x0D35;
pub const GL_MAX_MODELVIEW_STACK_DEPTH: i32 = 0x0D36;
pub const GL_MAX_NAME_STACK_DEPTH: i32 = 0x0D37;
pub const GL_MAX_PROJECTION_STACK_DEPTH: i32 = 0x0D38;
pub const GL_MAX_TEXTURE_STACK_DEPTH: i32 = 0x0D39;
pub const GL_MAX_VIEWPORT_DIMS: i32 = 0x0D3A;
pub const GL_MAX_CLIENT_ATTRIB_STACK_DEPTH: i32 = 0x0D3B;

//Matrix modes
pub const GL_MODELVIEW: u32 = 0x1700;
pub const GL_PROJECTION: u32 = 0x1701;
pub const GL_TEXTURE: u32 = 0x1702;

const log = std.log.scoped(.centralgpu_gl);
const std = @import("std");
const centralgpu = @import("centralgpu");
const matrix_math = @import("gl/matrix.zig");
