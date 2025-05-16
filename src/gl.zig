//! Central GL: An implementation of opengl 1.0

pub const GL_TRIANGLES: u32 = 0x0004;
pub const GL_TRIANGLE_STRIP: u32 = 0x0005;
pub const GL_TRIANGLE_FAN: u32 = 0x0006;
pub const GL_QUADS: u32 = 0x0007;

pub const Context = struct {
    gpa: std.mem.Allocator,
    bound_render_target: centralgpu.Image = undefined,

    clear_color: u32 = 0xff_00_00_00,
    viewport: struct { x: i32, y: i32, width: isize, height: isize } = undefined,
    scissor: struct { x: i32, y: i32, width: isize, height: isize } = undefined,

    //Base color texture
    texture_image: []u8 = &.{},
    texture_descriptor: centralgpu.ImageDescriptor = undefined,

    should_clear_color_attachment: bool = false,
    triangle_vertex_index: u32 = 0,

    modelview_matrix_stack: [2][16]f32 = [_][16]f32{@splat(0)} ** 2,
    modelview_matrix_stack_index: usize = 0,

    triangle_count: usize = 0,
    triangle_positions: std.ArrayListUnmanaged([3]centralgpu.WarpVec3(f32)) = .empty,
    triangle_colors: std.ArrayListUnmanaged([3]u32) = .empty,
    triangle_tex_coords: std.ArrayListUnmanaged([3][2]f32) = .empty,

    scale: [3]f32 = @splat(1),
    translate: [3]f32 = @splat(0),
};

pub var current_context: ?*Context = null;

pub const GL_COLOR_BUFFER_BIT: u32 = 0x00004000;

pub fn glClearColor(r: f32, g: f32, b: f32, a: f32) void {
    const context = current_context.?;

    context.clear_color = @bitCast(centralgpu.Rgba32.fromNormalized(.{ r, g, b, a }));
}

pub fn glClear(flags: u32) callconv(.c) void {
    const context = current_context.?;

    if (flags & GL_COLOR_BUFFER_BIT != 0) {
        context.should_clear_color_attachment = true;
    }
}

pub fn glViewport(x: i32, y: i32, w: isize, h: isize) callconv(.c) void {
    const context = current_context.?;

    context.viewport = .{
        .x = x,
        .y = y,
        .width = w,
        .height = h,
    };
}

pub fn glScissor(x: i32, y: i32, w: isize, h: isize) callconv(.c) void {
    const context = current_context.?;

    context.scissor = .{
        .x = x,
        .y = y,
        .width = w,
        .height = h,
    };
}

pub fn glTexImage2D(
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
    const context = current_context.?;

    const dest_data_size: usize = @intCast(width * height * @sizeOf(u32));
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

    context.texture_image = @ptrCast(image_data);
    context.texture_descriptor = .{
        .rel_ptr = 0,
        .width_log2 = @intCast(std.math.log2_int(usize, @intCast(width))),
        .height_log2 = @intCast(std.math.log2_int(usize, @intCast(height))),
        .sampler_filter = .nearest,
        .sampler_address_mode = .repeat,
    };

    _ = _type; // autofix
    _ = target; // autofix
    _ = level; // autofix
    _ = components; // autofix
    _ = border; // autofix
    _ = format; // autofix
}

pub fn glBegin(flags: u32) callconv(.c) void {
    if (flags != GL_TRIANGLES) {
        @panic("glBegin: flags not supported. centralgl only supports GL_TRIANGLES");
    }

    const context = current_context.?;
    _ = context; // autofix
}

pub fn glEnd() callconv(.c) void {}

pub fn glTranslate3f(x: f32, y: f32, z: f32) callconv(.c) void {
    const context = current_context.?;

    context.translate = .{ x, y, z };
}

pub fn glScale3f(x: f32, y: f32, z: f32) callconv(.c) void {
    const context = current_context.?;

    context.scale = .{ x, y, z };
}

pub fn glRotatef(angle: f32, x: f32, y: f32, z: f32) callconv(.c) void {
    _ = angle; // autofix
    _ = x; // autofix
    _ = y; // autofix
    _ = z; // autofix
    @panic("Unimplemented");
}

//Matrix modes
const GL_MODELVIEW: u32 = 0x1700;
const GL_PROJECTION: u32 = 0x1701;
const GL_TEXTURE: u32 = 0x1702;

pub fn glMatrixMode(mode: u32) callconv(.c) void {
    _ = mode; // autofix
}

pub fn glPushMatrix() callconv(.c) void {
    const context = current_context.?;
    context.modelview_matrix_stack_index += 1;
}

pub fn glPopMatrix() callconv(.c) void {
    const context = current_context.?;
    context.modelview_matrix_stack_index -= 1;
}

pub fn glLoadMatrix(matrix: *[16]f32) callconv(.c) void {
    const context = current_context.?;

    context.modelview_matrix_stack[context.modelview_matrix_stack_index] = matrix.*;
}

pub fn glLoadIdentity() callconv(.c) void {
    const context = current_context.?;

    context.modelview_matrix_stack[context.modelview_matrix_stack_index] = .{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
}

pub fn glMultMatrix(matrix: *[16]f32) callconv(.c) void {
    _ = matrix; // autofix
}

pub fn glVertex3f(x: f32, y: f32, z: f32) callconv(.c) void {
    const context = current_context.?;

    ensureVertexCapacity() catch @panic("oom");

    const triangle_group = &context.triangle_positions.items[(context.triangle_count - 1) / 8];

    const triangle_index: usize = @rem(context.triangle_count - 1, 8);

    triangle_group[context.triangle_vertex_index].x[triangle_index] = x * context.scale[0] + context.translate[0];
    triangle_group[context.triangle_vertex_index].y[triangle_index] = y * context.scale[1] + context.translate[1];
    triangle_group[context.triangle_vertex_index].z[triangle_index] = z * context.scale[2] + context.translate[2];

    context.triangle_vertex_index += 1;
}

pub fn glColor4f(r: f32, g: f32, b: f32, a: f32) callconv(.c) void {
    const context = current_context.?;

    ensureVertexCapacity() catch @panic("oom");

    const tri = &context.triangle_colors.items[context.triangle_count - 1];

    tri[context.triangle_vertex_index] = @bitCast(centralgpu.Rgba32.fromNormalized(.{ r, g, b, a }));
}

pub fn glTexCoord2f(u: f32, v: f32) callconv(.c) void {
    const context = current_context.?;

    ensureVertexCapacity() catch @panic("oom");

    const tri = &context.triangle_tex_coords.items[context.triangle_count - 1];

    tri[context.triangle_vertex_index] = .{ u, v };
}

pub fn glVertex2f(x: f32, y: f32) callconv(.c) void {
    glVertex3f(x, y, 0);
}

pub fn glVertex2i(x: i32, y: i32) callconv(.c) void {
    glVertex3f(@floatFromInt(x), @floatFromInt(y), 0);
}

pub fn glColor3f(r: f32, g: f32, b: f32) callconv(.c) void {
    glColor4f(r, g, b, 1);
}

pub fn glFlush() callconv(.c) void {
    const context = current_context.?;

    if (context.should_clear_color_attachment) {
        const actual_width = centralgpu.computeTargetPaddedSize(context.bound_render_target.width);
        const actual_height = centralgpu.computeTargetPaddedSize(context.bound_render_target.height);

        @memset(
            context.bound_render_target.pixel_ptr[0 .. actual_width * actual_height],
            @bitCast(context.clear_color),
        );
    }

    const triangle_group_count = context.triangle_count / 8 + @intFromBool(context.triangle_count % 8 != 0);

    for (0..triangle_group_count) |triangle_group_id| {
        const unfiorms: centralgpu.Uniforms = .{
            .vertex_positions = context.triangle_positions.items,
            .vertex_colours = context.triangle_colors.items,
            .vertex_texture_coords = context.triangle_tex_coords.items,
            .image_base = context.texture_image.ptr,
            .image_descriptor = context.texture_descriptor,
        };

        var triangle_mask: centralgpu.WarpRegister(bool) = @splat(true);

        if (triangle_group_id == triangle_group_count - 1) {
            const remainder = @rem(context.triangle_count, 8);

            triangle_mask = @splat(false);

            for (0..remainder) |mask_index| {
                triangle_mask[mask_index] = true;
            }
        }

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
                    .scale_y = -viewport_height * 0.5,
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

    context.triangle_positions.clearRetainingCapacity();
    context.triangle_colors.clearRetainingCapacity();
    context.triangle_tex_coords.clearRetainingCapacity();
    context.triangle_count = 0;
    context.triangle_vertex_index = 0;
    context.scale = @splat(1);
    context.translate = @splat(0);
    context.should_clear_color_attachment = false;
    context.modelview_matrix_stack_index = 0;
}

fn ensureVertexCapacity() !void {
    const context = current_context.?;

    if (context.triangle_count == 0) {
        context.triangle_count = 1;
        context.triangle_vertex_index = 0;
    }

    if (context.triangle_vertex_index > 2) {
        context.triangle_vertex_index = 0;
        context.triangle_count += 1;
    }

    if (context.triangle_vertex_index == 0) {
        _ = context.triangle_colors.addOne(context.gpa) catch @panic("");
        _ = context.triangle_tex_coords.addOne(context.gpa) catch @panic("");

        if (context.triangle_positions.items.len < context.triangle_count * 8) {
            _ = context.triangle_positions.addOne(context.gpa) catch @panic("");
        }
    }
}

const std = @import("std");
const centralgpu = @import("root.zig");
