//! Central GL: An implementation of opengl 1.5

pub const Context = struct {
    gpa: std.mem.Allocator,

    have_seen_viewport: bool = false,

    render_area_width: i32,
    render_area_height: i32,

    bound_render_target: centralgpu.Image = undefined,
    depth_image: []centralgpu.Depth24Stencil8 = &.{},

    viewport: struct { x: i32, y: i32, width: isize, height: isize } = undefined,
    scissor: struct { x: i32, y: i32, width: isize, height: isize } = undefined,

    clear_color: u32 = 0xff_00_00_00,
    clear_depth: f32 = 1,
    should_clear_color_attachment: bool = false,
    should_clear_depth_attachment: bool = false,
    should_clear_stencil_attachment: bool = false,

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
    triangle_positions: std.ArrayListUnmanaged([3]centralgpu.WarpVec3(f32)) = .empty,
    triangle_colors: std.ArrayListUnmanaged([3]u32) = .empty,
    triangle_tex_coords: std.ArrayListUnmanaged([3][4][2]f32) = .empty,

    indexed_geometry: struct {
        indices: std.ArrayListUnmanaged([3]centralgpu.WarpRegister(u32)) = .empty,
        vertex_positions: std.ArrayListUnmanaged(f32) = .empty,
    } = .{},

    default_texture_descriptor: centralgpu.ImageDescriptor = .{
        .width_log2 = 0,
        .height_log2 = 0,
        .max_mip_level = 0,
        .sampler_magnification_filter = .linear,
        .sampler_minification_filter = .linear,
        .sampler_mipmap_filter = .nearest,
        .sampler_address_mode_u = .repeat,
        .sampler_address_mode_v = .repeat,
        .rel_ptr = -1,
        .border_colour = .white,
    },
    textures: std.ArrayListUnmanaged(TextureState) = .empty,
    ///Maps from GL_TEXTURE_0..80
    texture_units: [80]u32 = @splat(0),
    texture_unit_enabled: [80]bool = @splat(false),
    texture_environments: [80]centralgpu.TextureEnvironment = [1]centralgpu.TextureEnvironment{.{}} ** 80,
    texture_unit_active: u32 = 0,

    //Vertex Registers
    vertex_colour: u32 = 0xff_ff_ff_ff,
    vertex_uv: [4][2]f32 = [1][2]f32{@splat(0)} ** 4,

    //Vertex scratch
    vertex_scratch: std.MultiArrayList(struct {
        position: [3]f32,
        colour: u32,
        //Indexed by texture unit
        uv: [4][2]f32,
    }) = .empty,

    vertex_array: struct {
        vertex_pointer: ?[*]const f32 = null,
        color_pointer: ?[*]const f32 = null,
        tex_coord_pointer: [4]?[*]const f32 = [1]?[*]const f32{null} ** 4,
        vertex_component_count: u32 = 0,
    } = .{},

    draw_commands: std.ArrayListUnmanaged(DrawCommandState) = .empty,

    //Draw state
    enable_alpha_test: bool = false,
    enable_scissor_test: bool = false,
    enable_depth_test: bool = false,
    enable_depth_write: bool = true,
    enable_stencil_test: bool = false,
    enable_face_cull: bool = false,
    enable_blend: bool = false,
    invert_depth_test: bool = false,

    stencil_mask: u8 = 0xff,
    stencil_ref: u8 = 1,

    alpha_ref: f32 = 0,

    blend_state: centralgpu.BlendState = .{
        .src_factor = .one,
        .dst_factor = .zero,
    },

    depth_min: f32 = 0,
    depth_max: f32 = 1,

    has_begun: bool = false,
    primitive_mode: enum {
        triangle_list,
        triangle_strip,
        triangle_fan,
        quad_list,
        polygon,
    } = .triangle_list,

    flush_callback: ?*const fn () void = null,

    profile_data: struct {
        flush_primitives_time: i128 = 0,
    } = .{},

    raster_tile_buffer: centralgpu.RasterTileBuffer,

    pixel_store: struct {
        unpack_row_length: usize = 0,
    } = .{},
};

pub const TextureState = struct {
    texture_data: []centralgpu.Rgba32 = &.{},
    descriptor: centralgpu.ImageDescriptor = .{
        .rel_ptr = 0,
        .max_mip_level = 0,
        .width_log2 = 0,
        .height_log2 = 0,
        .sampler_magnification_filter = .linear,
        .sampler_minification_filter = .linear,
        .sampler_mipmap_filter = .nearest,
        .sampler_address_mode_u = .repeat,
        .sampler_address_mode_v = .repeat,
        .border_colour = .black_transparent,
    },
    internal_format: i32 = 0,
    width: u32 = 0,
    height: u32 = 0,
    max_defined_level: u32 = 0,
    //This is for the scenario where an image has it's mipmapping parameter changed before glTexImage
    //We can't update the descriptor until we know the texture's width and/or height
    mipmapping_enabled: bool = false,
};

pub const DrawCommandState = struct {
    triangle_id_start: u32,
    triangle_count: u32,

    scratch_vertex_start: u32,
    scratch_vertex_end: u32,

    vertex_matrix: [4][4]f32,

    image_base: [4][*]const u8,
    image_descriptor: [4]centralgpu.ImageDescriptor,
    texture_environments: [4]centralgpu.TextureEnvironment,

    flags: Flags,
    blend_state: centralgpu.BlendState,
    stencil_mask: u8,
    stencil_ref: u8,

    alpha_ref: f32,

    scissor_x: i32,
    scissor_y: i32,
    scissor_width: i32,
    scissor_height: i32,

    viewport_x: i32,
    viewport_y: i32,
    viewport_width: i32,
    viewport_height: i32,
    depth_min: f32,
    depth_max: f32,

    pub const Flags = packed struct(u8) {
        enable_alpha_test: bool,
        enable_scissor_test: bool,
        enable_depth_test: bool,
        enable_depth_write: bool,
        enable_stencil_test: bool,
        enable_blend: bool,
        enable_backface_cull: bool,
        invert_depth_test: bool,
    };
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

    if (flags & GL_DEPTH_BUFFER_BIT != 0) {
        context.should_clear_depth_attachment = true;
    }

    if (flags & GL_STENCIL_BUFFER_BIT != 0) {
        context.should_clear_stencil_attachment = true;
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
    const context = current_context.?;

    context.depth_min = @floatCast(near_val);
    context.depth_max = @floatCast(far_val);
}

pub export fn glDepthFunc(
    func: i32,
) callconv(.c) void {
    const context = current_context.?;

    switch (func) {
        GL_LEQUAL => {
            context.invert_depth_test = false;
        },
        GL_GEQUAL => {
            context.invert_depth_test = true;
        },
        else => {
            std.log.info("glDepthFunc: func: 0x{x}\n", .{func});
            @panic("GL_DEPTH_FUNC");
        },
    }
}

pub export fn glDepthMask(
    flag: bool,
) callconv(.c) void {
    const context = current_context.?;

    context.enable_depth_write = flag;
}

pub export fn glStencilFunc(
    func: i32,
    ref: u32,
    mask: u32,
) void {
    const context = current_context.?;

    context.stencil_ref = @intCast(ref);
    context.stencil_mask = @truncate(mask);

    switch (func) {
        GL_EQUAL => {},
        else => {
            std.log.info("func: {}\n", .{func});
            @panic("glStencilFunc");
        },
    }
}

pub export fn glStencilOp(
    sfail: i32,
    dpfail: i32,
    dppass: i32,
) void {
    std.log.info("sfail = 0x{x}, dpfail = 0x{x}, dppass = 0x{x}\n", .{ sfail, dpfail, dppass });

    switch (sfail) {
        GL_KEEP => {},
        else => {
            std.log.info("op = 0x{x}\n", .{sfail});

            @panic("glStencilOp");
        },
    }

    if (dppass != GL_INCR) {
        @panic("glStencilOp: only GL_INCR supported");
    }
}

pub export fn glGetFloatv() callconv(.c) void {}

pub export fn glGetIntegerv(
    pname: i32,
    params: [*c]i32,
) callconv(.c) void {
    switch (pname) {
        GL_MAX_TEXTURE_SIZE => {
            params.* = @intCast(std.math.maxInt(u16));
        },
        GL_MAX_TEXTURE_UNITS => {
            params.* = 32;
        },
        else => {
            log.info("glGetIntegerV: 0x{x}", .{pname});
            @panic("Unsupported getIntegerV");
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

fn readSrcPixel(
    _type: i32,
    data: *const anyopaque,
    component_count: usize,
    index: usize,
) centralgpu.Rgba32 {
    switch (_type) {
        GL_UNSIGNED_BYTE => {
            const source_data_ptr: [*]const u8 = @ptrCast(data);

            if (component_count < 4) {
                // @panic("");
            }

            return .{
                .r = source_data_ptr[index * 4 + 0],
                .g = source_data_ptr[index * 4 + 1],
                .b = source_data_ptr[index * 4 + 2],
                .a = if (component_count < 4) 255 else source_data_ptr[index * 4 + 3],
            };
        },
        else => {
            log.info("data_type: 0x{x}", .{_type});
            @panic("Texture src data type out of range!");
        },
    }
}

fn computeMipCountAndTotalSize(width: usize, height: usize) struct { usize, usize } {
    var count: usize = 1;
    var total_size: usize = width * height;

    var current_width: usize = width;
    var current_height: usize = height;

    while (true) {
        current_width /= 2;
        current_height /= 2;

        if (current_width == 0 or current_height == 0) {
            break;
        }

        count += 1;
        total_size += current_width * current_height;
    }

    return .{ count, total_size };
}

pub export fn glTexSubImage2D(
    target: i32,
    level: i32,
    x_offset: i32,
    y_offset: i32,
    width: isize,
    height: isize,
    format: i32,
    _type: i32,
    data: *const anyopaque,
) callconv(.c) void {
    _ = format; // autofix

    const context = current_context.?;

    if (target != GL_TEXTURE_2D) {
        @panic("only GL_TEXTURE_2D is supported");
    }

    const texture = &context.textures.items[context.texture_units[context.texture_unit_active] - 1];

    texture.max_defined_level = @max(texture.max_defined_level, @as(u4, @intCast(level)));

    if (texture.mipmapping_enabled) {
        texture.descriptor.max_mip_level = @intCast(texture.max_defined_level);
    }

    const component_count: usize = internalFormatComponentCount(texture.internal_format);

    const start_x: usize = @intCast(x_offset);
    const start_y: usize = @intCast(y_offset);
    const end_x: usize = @intCast(x_offset + width);
    const end_y: usize = @intCast(y_offset + height);

    const row_width: usize = if (context.pixel_store.unpack_row_length == 0) @intCast(width) else context.pixel_store.unpack_row_length;

    const mip_base: u32 = centralgpu.imageMipBaseAddress(texture.descriptor, @splat(@intCast(level)))[0];

    const dest_data_ptr: [*]centralgpu.Rgba32 = texture.texture_data.ptr + mip_base;

    var y: usize = start_y;

    while (y < end_y) : (y += 1) {
        var x: usize = start_x;

        while (x < end_x) : (x += 1) {
            const src_x: usize = x - start_x;
            const src_y: usize = y - start_y;

            const src_index = @as(usize, @intCast(src_y)) * row_width + src_x;
            const index = centralgpu.mortonEncode(@splat(@intCast(x)), @splat(@intCast(y)))[0];

            dest_data_ptr[index] = readSrcPixel(_type, data, component_count, src_index);
        }
    }
}

pub export fn glDrawBuffer() callconv(.c) void {}

pub export fn glEnableClientState(
    cap: i32,
) callconv(.c) void {
    switch (cap) {
        GL_COLOR_ARRAY => {},
        else => {},
    }
}

pub export fn glDisableClientState(
    cap: i32,
) callconv(.c) void {
    const context = current_context.?;

    switch (cap) {
        GL_COLOR_ARRAY => {
            context.vertex_array.color_pointer = null;
        },
        GL_TEXTURE_COORD_ARRAY => {
            for (&context.vertex_array.tex_coord_pointer) |*ptr| {
                ptr.* = null;
            }
        },
        GL_VERTEX_ARRAY => {
            context.vertex_array.vertex_pointer = null;
        },
        else => {},
    }
}

pub export fn glVertexPointer(
    size: i32,
    @"type": i32,
    stride: isize,
    ptr: *const anyopaque,
) callconv(.c) void {
    _ = stride; // autofix
    _ = @"type";

    const context = current_context.?;

    context.vertex_array.vertex_pointer = @ptrCast(@alignCast(ptr));
    context.vertex_array.vertex_component_count = @intCast(size);
}

pub export fn glTexCoordPointer(
    size: i32,
    @"type": i32,
    stride: isize,
    ptr: *const anyopaque,
) callconv(.c) void {
    _ = size; // autofix
    _ = stride; // autofix
    _ = @"type";
    const context = current_context.?;
    context.vertex_array.tex_coord_pointer[context.texture_unit_active] = @ptrCast(@alignCast(ptr));
}

pub export fn glColorPointer(
    size: i32,
    @"type": i32,
    stride: isize,
    ptr: *const anyopaque,
) callconv(.c) void {
    _ = size; // autofix
    _ = stride; // autofix
    _ = @"type";
    const context = current_context.?;
    context.vertex_array.color_pointer = @ptrCast(@alignCast(ptr));
}

pub export fn glDrawArrays(
    mode: i32,
    first: i32,
    count: isize,
) callconv(.c) void {
    _ = first; // autofix

    switch (mode) {
        GL_TRIANGLES, GL_TRIANGLE_FAN, GL_TRIANGLE_STRIP, GL_POLYGON => {},
        else => return,
    }

    const context = current_context.?;

    const old_color = context.vertex_colour;
    defer {
        context.vertex_colour = old_color;
    }

    glBegin(@bitCast(mode));

    for (0..@intCast(count)) |vertex_index| {
        for (context.vertex_array.tex_coord_pointer, 0..) |tex_ptr, unit| {
            if (tex_ptr != null) {
                const u = tex_ptr.?[vertex_index * 2 + 0];
                const v = tex_ptr.?[vertex_index * 2 + 1];

                glMultiTexCoord2fARB(@as(i32, @intCast(unit)) + GL_TEXTURE0, u, v);
            }
        }

        if (context.vertex_array.color_pointer != null) {
            const r = context.vertex_array.color_pointer.?[vertex_index * 4 + 0];
            const g = context.vertex_array.color_pointer.?[vertex_index * 4 + 1];
            const b = context.vertex_array.color_pointer.?[vertex_index * 4 + 2];
            const a = context.vertex_array.color_pointer.?[vertex_index * 4 + 3];

            glColor4f(r, g, b, a);
        }

        switch (context.vertex_array.vertex_component_count) {
            2 => {
                const x = context.vertex_array.vertex_pointer.?[vertex_index * context.vertex_array.vertex_component_count + 0];
                const y = context.vertex_array.vertex_pointer.?[vertex_index * context.vertex_array.vertex_component_count + 1];

                glVertex2f(x, y);
            },
            3 => {
                const x = context.vertex_array.vertex_pointer.?[vertex_index * context.vertex_array.vertex_component_count + 0];
                const y = context.vertex_array.vertex_pointer.?[vertex_index * context.vertex_array.vertex_component_count + 1];
                const z = context.vertex_array.vertex_pointer.?[vertex_index * context.vertex_array.vertex_component_count + 2];

                glVertex3f(x, y, z);
            },
            else => @panic("vertex_component_count not supported"),
        }
    }

    glEnd();
}

pub export fn glDrawElements(
    mode: i32,
    count: isize,
    @"type": i32,
    indices: *const anyopaque,
) callconv(.c) void {
    _ = @"type";
    switch (mode) {
        GL_TRIANGLES, GL_TRIANGLE_FAN, GL_TRIANGLE_STRIP, GL_POLYGON => {},
        else => return,
    }

    const context = current_context.?;

    const old_color = context.vertex_colour;
    defer {
        context.vertex_colour = old_color;
    }

    const indices_u16: [*]const u16 = @ptrCast(@alignCast(indices));

    glBegin(@bitCast(mode));

    for (0..@intCast(count)) |element_index| {
        const vertex_index: usize = indices_u16[element_index];

        for (context.vertex_array.tex_coord_pointer, 0..) |tex_ptr, unit| {
            if (tex_ptr != null) {
                const u = tex_ptr.?[vertex_index * 2 + 0];
                const v = tex_ptr.?[vertex_index * 2 + 1];

                glMultiTexCoord2fARB(@as(i32, @intCast(unit)) + GL_TEXTURE0, u, v);
            }
        }

        if (context.vertex_array.color_pointer != null) {
            const r = context.vertex_array.color_pointer.?[vertex_index * 4 + 0];
            const g = context.vertex_array.color_pointer.?[vertex_index * 4 + 1];
            const b = context.vertex_array.color_pointer.?[vertex_index * 4 + 2];
            const a = context.vertex_array.color_pointer.?[vertex_index * 4 + 3];

            glColor4f(r, g, b, a);
        }

        switch (context.vertex_array.vertex_component_count) {
            2 => {
                const x = context.vertex_array.vertex_pointer.?[vertex_index * context.vertex_array.vertex_component_count + 0];
                const y = context.vertex_array.vertex_pointer.?[vertex_index * context.vertex_array.vertex_component_count + 1];

                glVertex2f(x, y);
            },
            3 => {
                const x = context.vertex_array.vertex_pointer.?[vertex_index * context.vertex_array.vertex_component_count + 0];
                const y = context.vertex_array.vertex_pointer.?[vertex_index * context.vertex_array.vertex_component_count + 1];
                const z = context.vertex_array.vertex_pointer.?[vertex_index * context.vertex_array.vertex_component_count + 2];

                glVertex3f(x, y, z);
            },
            else => @panic("vertex_component_count not supported"),
        }
    }

    glEnd();
}

fn internalFormatComponentCount(internal_format: i32) usize {
    if (internal_format <= 4) {
        return @intCast(internal_format);
    }

    switch (internal_format) {
        GL_RGB10_A2,
        GL_RGBA,
        => {
            return 4;
        },
        GL_RGB => {
            return 3;
        },
        else => {
            log.info("internalFormat: 0x{x}", .{internal_format});
            @panic("internalFormat out of range");
        },
    }
}

pub export fn glPixelStorei(
    pname: i32,
    param: i32,
) callconv(.c) void {
    std.log.info("glPixelStorei: 0x{x}, 0x{x}\n", .{ pname, param });
    const context = current_context.?;

    switch (pname) {
        GL_UNPACK_ROW_LENGTH => {
            context.pixel_store.unpack_row_length = @intCast(param);
        },
        else => {},
    }
}

pub export fn glTexImage2D(
    target: i32,
    level: i32,
    internal_format: i32,
    width: isize,
    height: isize,
    border: i32,
    format: i32,
    _type: i32,
    data: ?*const anyopaque,
) callconv(.c) void {
    const context = current_context.?;

    if (target != GL_TEXTURE_2D) {
        @panic("only GL_TEXTURE_2D is supported");
    }

    if (context.texture_units[context.texture_unit_active] == 0) {
        return;
    }

    const texture = &context.textures.items[context.texture_units[context.texture_unit_active] - 1];

    if (level != 0) {
        glTexSubImage2D(
            target,
            level,
            0,
            0,
            width,
            height,
            format,
            _type,
            data.?,
        );
        return;
    }

    texture.internal_format = internal_format;
    texture.width = @intCast(width);
    texture.height = @intCast(height);

    log.info("components = 0x{x}", .{internal_format});
    log.info("format = 0x{x}", .{format});
    log.info("format_type = 0x{x}", .{_type});

    if (format != GL_RGBA) {
        @panic("component type not supported");
    }

    std.debug.assert(std.math.isPowerOfTwo(width));
    std.debug.assert(std.math.isPowerOfTwo(height));

    const padded_width: usize = @intCast(width);
    const padded_height: usize = @intCast(height);

    const mip_count, const total_data_size = computeMipCountAndTotalSize(padded_width, padded_height);
    _ = mip_count; // autofix

    const dest_data_size: usize = @intCast(total_data_size * @sizeOf(u32));

    const image_data = std.heap.page_allocator.alloc(centralgpu.Rgba32, dest_data_size) catch @panic("");

    @memset(image_data, .{ .r = 0, .g = 0, .b = 0, .a = 0 });

    texture.texture_data = image_data;

    texture.descriptor.width_log2 = @intCast(std.math.log2_int(usize, @intCast(width)));
    texture.descriptor.height_log2 = @intCast(std.math.log2_int(usize, @intCast(height)));

    if (texture.mipmapping_enabled) {
        texture.descriptor.max_mip_level = @intCast(texture.max_defined_level);
    } else {
        texture.descriptor.max_mip_level = 0;
    }

    if (data != null) {
        glTexSubImage2D(
            target,
            level,
            0,
            0,
            width,
            height,
            format,
            _type,
            data.?,
        );
    } else {
        @memset(image_data, .{ .r = 0, .g = 0, .b = 0, .a = 0 });
    }

    _ = border; // autofix
}

pub export fn glCopyTexSubImage2D(
    target: i32,
    level: i32,
    x_offset: i32,
    y_offset: i32,
    _x: i32,
    _y: i32,
    _width: isize,
    _height: isize,
) void {
    if (target != GL_TEXTURE_2D) {
        @panic("Only GL_TEXTURE_2D is supported");
    }

    //Before reading from the framebuffer we must flush
    //We need to use flush without invoking the callback so we don't present the render result of the flush
    flushWithoutCallback();

    const context = current_context.?;
    const texture = &context.textures.items[context.texture_units[context.texture_unit_active] - 1];

    const width: u32 = @intCast(_width);
    const height: u32 = @intCast(_height);

    const x: u32 = @intCast(_x);

    var y: u32 = @intCast(_y);

    y += height;
    y = context.bound_render_target.height - y;

    for (0..height) |src_y_offset| {
        const src_y = y + src_y_offset;
        for (0..width) |src_x_offset| {
            const src_x = x + src_x_offset;
            const sample = centralgpu.renderTargetLoad(
                context.bound_render_target.pixel_ptr,
                context.bound_render_target.width,
                @intCast(src_x),
                @intCast(src_y),
            );

            const dst_x = @as(u32, @intCast(x_offset)) + src_x_offset;
            const dst_y = @as(u32, @intCast(y_offset)) + src_y_offset;

            centralgpu.imageStore(
                .{
                    true,  false, false, false,
                    false, false, false, false,
                },
                @ptrCast(texture.texture_data.ptr),
                texture.descriptor,
                @splat(@intCast(level)),
                .{
                    .x = @splat(@intCast(dst_x)),
                    .y = @splat(@intCast(texture.height - 1 - dst_y)),
                },
                @splat(@bitCast(sample)),
            );
        }
    }
}

pub export fn glGenTextures(
    n: isize,
    textures: [*c]u32,
) callconv(.c) void {
    const context = current_context.?;

    const id_start: u32 = @intCast(context.textures.items.len + 1);

    context.textures.appendNTimes(context.gpa, .{}, @intCast(n)) catch @panic("oom");

    for (0..@as(usize, @intCast(n))) |k| {
        textures[k] = id_start + @as(u32, @intCast(k));
    }
}

pub export fn glDeleteTextures(
    n: isize,
    textures: [*c]u32,
) callconv(.c) void {
    const context = current_context.?;

    for (0..@intCast(n)) |k| {
        const texture_handle = textures[k];
        const texture = &context.textures.items[texture_handle - 1];

        for (context.texture_units[0..]) |*bound_tex| {
            if (bound_tex.* == texture_handle) {
                bound_tex.* = 0;
            }
        }

        if (texture.texture_data.len != 0) {
            std.heap.page_allocator.free(texture.texture_data);
        }
    }
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
    if (target != GL_TEXTURE_2D) {
        @panic("Only GL_TEXTURE_2D is supported!");
    }

    const context = current_context.?;

    if (texture != 0 and texture - 1 >= context.textures.items.len) {
        const many_to_add = (texture) - @as(u32, @intCast(context.textures.items.len));

        context.textures.appendNTimes(context.gpa, .{}, many_to_add) catch @panic("oom");
    }

    std.debug.assert(texture -| 1 < context.textures.items.len);

    context.texture_units[context.texture_unit_active] = texture;
}

pub export fn glTexEnvi(
    target: i32,
    pname: i32,
    param: i32,
) callconv(.c) void {
    glTexEnvf(target, pname, @floatFromInt(param));
}

pub export fn glTexEnvf(
    target: i32,
    pname: i32,
    param: f32,
) callconv(.c) void {
    if (target != GL_TEXTURE_ENV) {
        @panic("Only GL_TEXTURE_ENV is supported!");
    }

    const context = current_context.?;

    const tex_env = &context.texture_environments[context.texture_unit_active];

    switch (pname) {
        GL_TEXTURE_ENV_MODE => {
            tex_env.function = switch (@as(i32, @intFromFloat(param))) {
                GL_MODULATE => .modulate,
                GL_REPLACE => .replace,
                GL_ADD => .add,
                GL_DECAL => .decal,
                else => tex_env.function,
            };
        },
        GL_COMBINE_RGB => {
            tex_env.function = switch (@as(i32, @intFromFloat(param))) {
                GL_MODULATE => .modulate,
                GL_REPLACE => .replace,
                GL_ADD => .add,
                GL_DECAL => .decal,
                else => {
                    std.log.info("texture env func: 0x{x}\n", .{@as(i32, @intFromFloat(param))});
                    @panic("texture env function not supported");
                },
            };
        },
        GL_SOURCE0_RGB => {},
        GL_SOURCE1_RGB => {},
        GL_SOURCE2_RGB => {},
        GL_RGB_SCALE => {
            tex_env.rgb_scale = param;
        },
        else => {
            std.debug.print("0x{x}\n", .{pname});
            @panic("Unsupported pname");
        },
    }
}

pub export fn glTexParameteri(
    target: i32,
    pname: i32,
    param: i32,
) callconv(.c) void {
    if (target != GL_TEXTURE_2D) {
        @panic("Only GL_TEXTURE_2D supported");
    }

    const context = current_context.?;

    var texture: ?*TextureState = null;
    var descriptor: *centralgpu.ImageDescriptor = undefined;

    if (context.texture_units[context.texture_unit_active] != 0) {
        texture = &context.textures.items[context.texture_units[context.texture_unit_active] - 1];
        descriptor = &texture.?.descriptor;
    } else {
        descriptor = &context.default_texture_descriptor;
    }

    switch (pname) {
        GL_TEXTURE_MAG_FILTER,
        => {
            switch (param) {
                GL_NEAREST,
                GL_NEAREST_MIPMAP_LINEAR,
                GL_NEAREST_MIPMAP_NEAREST,
                => {
                    descriptor.sampler_magnification_filter = .nearest;
                },
                GL_LINEAR,
                GL_LINEAR_MIPMAP_NEAREST,
                GL_LINEAR_MIPMAP_LINEAR,
                => {
                    descriptor.sampler_magnification_filter = .linear;
                },
                else => {},
            }
        },
        GL_TEXTURE_MIN_FILTER,
        => {
            switch (param) {
                GL_NEAREST,
                => {
                    descriptor.sampler_minification_filter = .nearest;
                    descriptor.max_mip_level = 0;

                    if (texture != null) texture.?.mipmapping_enabled = false;
                },
                GL_NEAREST_MIPMAP_LINEAR,
                GL_NEAREST_MIPMAP_NEAREST,
                => {
                    descriptor.sampler_minification_filter = .nearest;

                    if (texture != null) {
                        texture.?.mipmapping_enabled = true;
                        descriptor.max_mip_level = @intCast(texture.?.max_defined_level);
                    }
                },
                GL_LINEAR => {
                    descriptor.sampler_minification_filter = .linear;
                    descriptor.max_mip_level = 0;

                    if (texture != null) texture.?.mipmapping_enabled = false;
                },
                GL_LINEAR_MIPMAP_NEAREST,
                GL_LINEAR_MIPMAP_LINEAR,
                => {
                    descriptor.sampler_minification_filter = .linear;
                    descriptor.max_mip_level = descriptor.mipLevelCount();

                    if (texture != null) {
                        texture.?.mipmapping_enabled = true;
                        descriptor.max_mip_level = @intCast(texture.?.max_defined_level);
                    }
                },
                else => {},
            }

            if (param == GL_LINEAR_MIPMAP_LINEAR or param == GL_NEAREST_MIPMAP_LINEAR) {
                descriptor.sampler_mipmap_filter = .linear;
            }

            if (param == GL_LINEAR_MIPMAP_NEAREST or param == GL_NEAREST_MIPMAP_NEAREST) {
                descriptor.sampler_mipmap_filter = .nearest;
            }
        },
        GL_TEXTURE_WRAP_S => {
            descriptor.sampler_address_mode_u = switch (param) {
                GL_REPEAT => .repeat,
                GL_MIRRORED_REPEAT => .repeat_mirrored,
                GL_CLAMP_TO_EDGE => .clamp_to_edge,
                GL_CLAMP_TO_BORDER => .clamp_to_border,
                else => unreachable,
            };
        },
        GL_TEXTURE_WRAP_T => {
            descriptor.sampler_address_mode_v = switch (param) {
                GL_REPEAT => .repeat,
                GL_MIRRORED_REPEAT => .repeat_mirrored,
                GL_CLAMP_TO_EDGE => .clamp_to_edge,
                GL_CLAMP_TO_BORDER => .clamp_to_border,
                else => unreachable,
            };
        },
        else => {},
    }
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
    const param_int: i64 = @intFromFloat(param);

    glTexParameteri(target, pname, @truncate(param_int));
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
    if (target != GL_TEXTURE_2D) {
        @panic("Only GL_TEXTURE_2D is supported");
    }

    var descriptor: *centralgpu.ImageDescriptor = undefined;

    const context = current_context.?;

    const texture = &context.textures.items[context.texture_units[context.texture_unit_active] - 1];

    descriptor = &texture.descriptor;

    std.debug.assert(descriptor.max_mip_level == 0);

    const image_width = @as(usize, 1) << descriptor.width_log2;
    const image_height = @as(usize, 1) << descriptor.height_log2;

    var mip_width: usize = image_width;
    var mip_height: usize = image_height;

    var src_mip: u32 = 0;

    while (true) {
        if (mip_width > 1) mip_width /= 2;
        if (mip_height > 1) mip_height /= 2;

        centralgpu.imageBlit(
            descriptor.*,
            descriptor.*,
            src_mip,
            src_mip + 1,
            texture.texture_data.ptr,
            texture.texture_data.ptr,
            .linear,
        );

        texture.max_defined_level += 1;

        src_mip += 1;

        if (mip_width <= 1 and mip_height <= 1) {
            break;
        }
    }

    if (texture.mipmapping_enabled) {
        texture.descriptor.max_mip_level = @intCast(texture.max_defined_level);
    }
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

    startCommand(context);

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
    const command = context.draw_commands.addOne(context.gpa) catch @panic("oom");

    const triangle_id_offset: usize = 0;

    if (context.draw_commands.items.len == 1) {
        if (context.triangle_count != 0) {
            @panic("context.triangle_count != 0");
        }
    }

    if (context.triangle_count % 8 != 0) {
        const bump_forward = 8 - @rem(context.triangle_count, 8);

        context.triangle_count += bump_forward;

        _ = context.triangle_colors.addManyAsSlice(context.gpa, bump_forward) catch @panic("oom");
        _ = context.triangle_tex_coords.addManyAsSlice(context.gpa, bump_forward) catch @panic("oom");
    }

    command.triangle_id_start = @intCast(context.triangle_count + triangle_id_offset);
    command.triangle_count = 0;

    command.scratch_vertex_start = @intCast(context.vertex_scratch.len);
    command.scratch_vertex_end = 0;

    command.texture_environments = [1]centralgpu.TextureEnvironment{.{}} ** 4;

    for (0..4) |descriptor_index| {
        command.image_descriptor[descriptor_index].rel_ptr = -1;
    }

    {
        var descriptor_index: usize = 0;

        for (0..4) |texture_unit| {
            const texture_handle = context.texture_units[texture_unit];
            const texture_unit_enabled = context.texture_unit_enabled[texture_unit];
            const texture_env = context.texture_environments[texture_unit];

            if (texture_handle != 0 and texture_unit_enabled) {
                const active_texture = &context.textures.items[texture_handle - 1];

                command.image_base[descriptor_index] = @ptrCast(active_texture.texture_data.ptr);
                command.image_descriptor[descriptor_index] = active_texture.descriptor;
                command.texture_environments[descriptor_index] = texture_env;
                descriptor_index += 1;
            }
        }
    }

    command.flags.enable_alpha_test = context.enable_alpha_test;
    command.flags.enable_scissor_test = context.enable_scissor_test;
    command.flags.enable_depth_test = context.enable_depth_test;
    command.flags.enable_depth_write = context.enable_depth_write;
    command.flags.enable_stencil_test = context.enable_stencil_test;
    command.flags.enable_backface_cull = context.enable_face_cull;
    command.flags.enable_blend = context.enable_blend;
    command.flags.invert_depth_test = context.invert_depth_test;

    command.stencil_mask = context.stencil_mask;
    command.stencil_ref = context.stencil_ref;

    command.alpha_ref = context.alpha_ref;

    command.blend_state = context.blend_state;

    command.scissor_x = context.scissor.x;
    command.scissor_y = context.scissor.y;
    command.scissor_width = @truncate(context.scissor.width);
    command.scissor_height = @truncate(context.scissor.height);

    command.depth_min = context.depth_min;
    command.depth_max = context.depth_max;

    command.viewport_x = @intCast(context.viewport.x);
    command.viewport_y = @intCast(context.viewport.y);
    command.viewport_width = @intCast(context.viewport.width);
    command.viewport_height = @intCast(context.viewport.height);

    const current_modelview_matrix = context.modelview_matrix_stack[context.modelview_matrix_stack_index];
    const current_projection_matrix = context.projection_matrix_stack[context.projection_matrix_stack_index];

    var matrix_product: [16]f32 = undefined;

    matmul4(&matrix_product, &current_projection_matrix, &current_modelview_matrix);

    command.vertex_matrix = @bitCast(matrix_product);
}

fn endCommand(context: *Context) void {
    if (context.draw_commands.items.len == 0) {
        return;
    }

    const last_command = &context.draw_commands.items[context.draw_commands.items.len - 1];

    last_command.scratch_vertex_end = @intCast(context.vertex_scratch.len);

    const command_scratch_vertex_count = last_command.scratch_vertex_end - last_command.scratch_vertex_start;

    if (command_scratch_vertex_count == 0) {
        return;
    }

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

    if (command_scratch_vertex_count < 3) {
        triangle_count = 0;
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

pub export fn glAlphaFunc(func: i32, ref: f32) callconv(.c) void {
    _ = func; // autofix
    const context = current_context.?;

    context.alpha_ref = ref;
}

pub export fn glBlendFunc(
    source_factor: i32,
    dest_factor: i32,
) callconv(.c) void {
    const context = current_context.?;

    context.blend_state.src_factor = switch (source_factor) {
        GL_ONE => .one,
        GL_ZERO => .zero,
        GL_ONE_MINUS_SRC_ALPHA => .one_minus_src_alpha,
        GL_SRC_ALPHA => .src_alpha,
        else => @panic("BLEND FUNC VALUE NOT SUPPORTED"),
    };

    context.blend_state.dst_factor = switch (dest_factor) {
        GL_ONE => .one,
        GL_ZERO => .zero,
        GL_ONE_MINUS_SRC_ALPHA => .one_minus_src_alpha,
        GL_SRC_ALPHA => .src_alpha,
        else => @panic("BLEND FUNC VALUE NOT SUPPORTED"),
    };
}

pub export fn glHint() callconv(.c) void {}

pub export fn glEnable(cap: i32) callconv(.c) void {
    const context = current_context.?;

    switch (cap) {
        GL_TEXTURE_2D => {
            context.texture_unit_enabled[context.texture_unit_active] = true;
        },
        GL_ALPHA_TEST => {
            context.enable_alpha_test = true;
        },
        GL_SCISSOR_TEST => {
            context.enable_scissor_test = true;
        },
        GL_DEPTH_TEST => {
            context.enable_depth_test = true;
        },
        GL_CULL_FACE => {
            context.enable_face_cull = true;
        },
        GL_BLEND => {
            context.enable_blend = true;
        },
        GL_STENCIL_TEST => {
            context.enable_stencil_test = true;
        },
        else => {},
    }
}

pub export fn glDisable(cap: i32) callconv(.c) void {
    const context = current_context.?;

    switch (cap) {
        GL_TEXTURE_2D => {
            context.texture_unit_enabled[context.texture_unit_active] = false;
        },
        GL_ALPHA_TEST => {
            context.enable_alpha_test = false;
        },
        GL_SCISSOR_TEST => {
            context.enable_scissor_test = false;
        },
        GL_DEPTH_TEST => {
            context.enable_depth_test = false;
        },
        GL_CULL_FACE => {
            context.enable_face_cull = false;
        },
        GL_BLEND => {
            context.enable_blend = false;
        },
        GL_STENCIL_TEST => {
            context.enable_stencil_test = false;
        },
        else => {},
    }
}

pub export fn glRotatef(angle_degrees: f32, _x: f32, _y: f32, _z: f32) callconv(.c) void {
    const angle = (angle_degrees * std.math.tau) / 360.0;

    const cos_angle = @cos(angle * 0.5);
    const sin_angle = @sin(-angle * 0.5);

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

pub export fn glLoadMatrixf(matrix: *const [16]f32) callconv(.c) void {
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

    glMultMatrixf(&frustum_matrix);
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

    const M = struct {
        pub inline fn _M(row: usize, col: usize) usize {
            return col * 4 + row;
        }
    }._M;

    // zig fmt: off

    ortho_matrix[M(0,0)] = 2 / (right - left);    ortho_matrix[M(0,1)] = 0.0;  ortho_matrix[M(0,2)] = 0;     ortho_matrix[M(0,3)] = t_x;
    ortho_matrix[M(1,0)] = 0.0;  ortho_matrix[M(1,1)] = 2 / (top - bottom);    ortho_matrix[M(1,2)] = 0;     ortho_matrix[M(1,3)] = t_y;
    ortho_matrix[M(2,0)] = 0.0;  ortho_matrix[M(2,1)] = 0.0;  ortho_matrix[M(2,2)] = -2 / (far_val - near_val);     ortho_matrix[M(2,3)] = t_z;
    ortho_matrix[M(3,0)] = 0.0;  ortho_matrix[M(3,1)] = 0.0;  ortho_matrix[M(3,2)] = 0;  ortho_matrix[M(3,3)] = 1;

    // zig fmt: on

    glMultMatrixf(&ortho_matrix);
}

pub export fn glVertex3f(x: f32, y: f32, z: f32) callconv(.c) void {
    const context = current_context.?;

    context.vertex_scratch.append(context.gpa, .{
        .position = .{ x, y, z },
        .colour = context.vertex_colour,
        .uv = context.vertex_uv,
    }) catch @panic("oom");
}

fn flushPrimitives(context: *Context) void {
    const draw_command = context.draw_commands.getLast();

    if (draw_command.triangle_count == 0) {
        return;
    }

    const time_begin = std.time.nanoTimestamp();
    defer {
        const time_ns = std.time.nanoTimestamp() - time_begin;

        context.profile_data.flush_primitives_time += time_ns;
    }

    const triangle_count = draw_command.triangle_count;
    const global_triangle_id_start = draw_command.triangle_id_start;

    const vertex_start = context.indexed_geometry.vertex_positions.items.len / 3;
    const vertex_count = draw_command.scratch_vertex_end - draw_command.scratch_vertex_start;
    _ = vertex_count; // autofix

    const group_count = @divTrunc(triangle_count, 8) + @intFromBool(@rem(triangle_count, 8) != 0);

    const allocator = context.gpa;

    _ = context.indexed_geometry.indices.addManyAsSlice(allocator, group_count) catch @panic("oom");
    _ = context.indexed_geometry.vertex_positions.addManyAsSlice(allocator, context.vertex_scratch.len * 3) catch @panic("oom");

    //Update vertex data
    {
        for (context.vertex_scratch.items(.position), 0..) |vertex_position, vertex_index| {
            context.indexed_geometry.vertex_positions.items[(vertex_start + vertex_index) * 3 + 0] = vertex_position[0];
            context.indexed_geometry.vertex_positions.items[(vertex_start + vertex_index) * 3 + 1] = vertex_position[1];
            context.indexed_geometry.vertex_positions.items[(vertex_start + vertex_index) * 3 + 2] = vertex_position[2];
        }
    }

    _ = context.triangle_positions.addManyAsSlice(allocator, group_count) catch @panic("oom");
    _ = context.triangle_colors.addManyAsSlice(allocator, triangle_count) catch @panic("oom");
    _ = context.triangle_tex_coords.addManyAsSlice(allocator, triangle_count) catch @panic("oom");

    switch (context.primitive_mode) {
        .triangle_list => {
            for (0..triangle_count) |triangle_index| {
                const triangle_id = global_triangle_id_start + triangle_index;
                const group_index = @divTrunc(triangle_id, 8);
                const triangle_local_index = @rem(triangle_id, 8);

                const triangle_group_indices = &context.indexed_geometry.indices.items[group_index];

                inline for (0..3) |tri_vertex_index| {
                    const scratch_index = triangle_index * 3 + tri_vertex_index;

                    triangle_group_indices[tri_vertex_index][triangle_local_index] = @intCast(vertex_start + scratch_index);

                    context.triangle_colors.items[triangle_id][tri_vertex_index] = context.vertex_scratch.items(.colour)[scratch_index];
                    context.triangle_tex_coords.items[triangle_id][tri_vertex_index] = context.vertex_scratch.items(.uv)[scratch_index];
                }
            }
        },
        .quad_list => {
            const quad_indices: [6]usize = .{
                0, 1, 2,
                2, 3, 0,
            };

            for (0..triangle_count) |triangle_index| {
                const triangle_id = global_triangle_id_start + triangle_index;
                const group_index = @divTrunc(triangle_id, 8);
                const triangle_local_index = @rem(triangle_id, 8);

                const triangle_group_indices = &context.indexed_geometry.indices.items[group_index];

                const input_quad_index = @divTrunc(triangle_index, 2);

                inline for (0..3) |tri_vertex_index| {
                    const stream_vertex_index = triangle_index * 3 + tri_vertex_index;

                    const scratch_index = input_quad_index * 4 + quad_indices[@rem(stream_vertex_index, 6)];

                    triangle_group_indices[tri_vertex_index][triangle_local_index] = @intCast(vertex_start + scratch_index);

                    context.triangle_colors.items[triangle_id][tri_vertex_index] = context.vertex_scratch.items(.colour)[scratch_index];
                    context.triangle_tex_coords.items[triangle_id][tri_vertex_index] = context.vertex_scratch.items(.uv)[scratch_index];
                }
            }
        },
        .triangle_strip => {
            for (0..triangle_count) |triangle_index| {
                const triangle_id = global_triangle_id_start + triangle_index;
                const group_index = @divTrunc(triangle_id, 8);
                const triangle_local_index = @rem(triangle_id, 8);

                const triangle_group_indices = &context.indexed_geometry.indices.items[group_index];

                const even_indices: [3]usize = .{ 0, 1, 2 };
                const odd_indices: [3]usize = .{ 1, 0, 2 };

                inline for (0..3) |tri_vertex_index| {
                    const is_even = triangle_index % 2 == 0;

                    const scratch_index = triangle_index + if (is_even) even_indices[tri_vertex_index] else odd_indices[tri_vertex_index];

                    triangle_group_indices[tri_vertex_index][triangle_local_index] = @intCast(vertex_start + scratch_index);

                    context.triangle_colors.items[triangle_id][tri_vertex_index] = context.vertex_scratch.items(.colour)[scratch_index];
                    context.triangle_tex_coords.items[triangle_id][tri_vertex_index] = context.vertex_scratch.items(.uv)[scratch_index];
                }
            }
        },
        .triangle_fan, .polygon => {
            for (0..triangle_count) |triangle_index| {
                const triangle_id = global_triangle_id_start + triangle_index;
                const group_index = @divTrunc(triangle_id, 8);
                const triangle_local_index = @rem(triangle_id, 8);

                const triangle_group_indices = &context.indexed_geometry.indices.items[group_index];

                inline for (0..3) |tri_vertex_index| {
                    const scratch_index = switch (tri_vertex_index) {
                        0 => 0,
                        1 => triangle_index + 1,
                        2 => triangle_index + 2,
                        else => unreachable,
                    };

                    triangle_group_indices[tri_vertex_index][triangle_local_index] = @intCast(vertex_start + scratch_index);

                    context.triangle_colors.items[triangle_id][tri_vertex_index] = context.vertex_scratch.items(.colour)[scratch_index];
                    context.triangle_tex_coords.items[triangle_id][tri_vertex_index] = context.vertex_scratch.items(.uv)[scratch_index];
                }
            }
        },
    }

    context.vertex_scratch.clearRetainingCapacity();
    context.triangle_count += triangle_count;
}

pub export fn glColor4f(r: f32, g: f32, b: f32, a: f32) callconv(.c) void {
    const context = current_context.?;

    context.vertex_colour = @bitCast(centralgpu.Rgba32.fromNormalized(.{ r, g, b, a }));
}

pub export fn glTexCoord2f(u: f32, v: f32) callconv(.c) void {
    const context = current_context.?;

    context.vertex_uv[0] = .{ u, v };
}

pub export fn glMultiTexCoord2fARB(
    target: i32,
    u: f32,
    v: f32,
) callconv(.c) void {
    const context = current_context.?;

    const texture_unit: usize = @intCast(target - GL_TEXTURE0);

    context.vertex_uv[texture_unit] = .{ u, v };
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

    flushWithoutCallback();

    if (context.flush_callback != null) {
        context.flush_callback.?();
    }

    context.profile_data = .{};
}

pub fn flushWithoutCallback() callconv(.c) void {
    const context = current_context.?;

    if (context.should_clear_color_attachment or true) {
        const colour_image_words: []u32 = @ptrCast(@alignCast(context.bound_render_target.pixel_ptr[0 .. context.bound_render_target.width * context.bound_render_target.height]));

        const colour_image: []@Vector(8, u32) = @ptrCast(@alignCast(colour_image_words));

        @memset(colour_image, @splat(@bitCast(context.clear_color)));

        context.should_clear_color_attachment = false;
    }

    if (context.should_clear_depth_attachment or true) {
        //TODO: handle not clearing the stencil values
        const depth_clear = centralgpu.packDepthStencil(@splat(0), @splat(0xff));
        const depth_image: []@Vector(8, u32) = @ptrCast(@alignCast(context.depth_image));

        @memset(depth_image, depth_clear);

        context.should_clear_depth_attachment = false;
    }

    std.log.info("total draw_cmds: {}\n", .{context.draw_commands.items.len});
    std.log.info("total triangle: {}\n", .{context.triangle_count});
    // std.log.info("flush primitives time: {}ns\n", .{context.profile_data.flush_primitives_time});

    const time_begin = std.time.nanoTimestamp();
    defer {
        const time_ns = std.time.nanoTimestamp() - time_begin;
        std.log.info("glFlush: time_taken: {}ns\n", .{time_ns});
    }

    {
        @memset(context.raster_tile_buffer.tile_data, .{ .triangle_count = 0, .triangles = undefined });
    }

    const rasterizer_submit_time = std.time.nanoTimestamp();

    for (context.draw_commands.items, 0..) |draw_command, draw_index| {
        _ = draw_index; // autofix
        if (draw_command.triangle_count == 0) {
            continue;
        }

        const triangle_group_begin = draw_command.triangle_id_start / 8;
        const triangle_group_count = draw_command.triangle_count / 8 + @intFromBool(draw_command.triangle_count % 8 != 0);

        const actual_scissor_x: i32 = if (draw_command.flags.enable_scissor_test) draw_command.scissor_x else 0;
        const actual_scissor_y: i32 = if (draw_command.flags.enable_scissor_test) draw_command.scissor_y else 0;
        const actual_scissor_width: i32 = if (draw_command.flags.enable_scissor_test) draw_command.scissor_width else @intCast(context.render_area_width);
        const actual_scissor_height: i32 = if (draw_command.flags.enable_scissor_test) draw_command.scissor_height else @intCast(context.render_area_height);

        var triangle_id_start: u32 = @intCast(draw_command.triangle_id_start);

        const viewport_x: f32 = @floatFromInt(draw_command.viewport_x);
        const viewport_y: f32 = @floatFromInt(draw_command.viewport_y);

        const viewport_width: f32 = @floatFromInt(draw_command.viewport_width);
        const viewport_height: f32 = @floatFromInt(draw_command.viewport_height);

        var geometry_state: centralgpu.GeometryProcessState = .{
            .viewport_transform = .{
                .translation_x = viewport_x + viewport_width * 0.5,
                .translation_y = viewport_y + viewport_height * 0.5,
                .scale_x = viewport_width * 0.5,
                .scale_y = viewport_height * 0.5,
                .translation_z = (draw_command.depth_max + draw_command.depth_min) * 0.5,
                .scale_z = (draw_command.depth_max - draw_command.depth_min) * 0.5,
                .inverse_scale_x = undefined,
                .inverse_scale_y = undefined,
                .inverse_translation_x = undefined,
                .inverse_translation_y = undefined,
            },
            .backface_cull = draw_command.flags.enable_backface_cull,
        };
        geometry_state.viewport_transform.inverse_translation_x = -geometry_state.viewport_transform.translation_x;
        geometry_state.viewport_transform.inverse_translation_y = -geometry_state.viewport_transform.translation_y;

        geometry_state.viewport_transform.inverse_scale_x = 1 / geometry_state.viewport_transform.scale_x;
        geometry_state.viewport_transform.inverse_scale_y = 1 / geometry_state.viewport_transform.scale_y;

        const raster_state: centralgpu.RasterState = .{
            .scissor_min_x = actual_scissor_x,
            .scissor_min_y = actual_scissor_y,
            .scissor_max_x = actual_scissor_width,
            .scissor_max_y = actual_scissor_height,
            .render_target = context.bound_render_target,
            .depth_image = context.depth_image.ptr,
            .blend_state = draw_command.blend_state,
            .stencil_mask = draw_command.stencil_mask,
            .stencil_ref = draw_command.stencil_ref,
            .alpha_ref = draw_command.alpha_ref,
            .flags = .{
                .enable_depth_test = draw_command.flags.enable_depth_test,
                .enable_alpha_test = draw_command.flags.enable_alpha_test,
                .enable_depth_write = draw_command.flags.enable_depth_write and draw_command.flags.enable_depth_test,
                .enable_blend = draw_command.flags.enable_blend,
                .invert_depth_test = draw_command.flags.invert_depth_test,
                .enable_stencil_test = draw_command.flags.enable_stencil_test,
            },
        };

        const unfiorms: centralgpu.Uniforms = .{
            .vertex_matrix = draw_command.vertex_matrix,
            .vertex_positions = context.triangle_positions.items,
            .vertex_colours = context.triangle_colors.items,
            .vertex_texture_coords = context.triangle_tex_coords.items,
            .image_base = draw_command.image_base,
            .image_descriptor = draw_command.image_descriptor,
            .texture_environments = draw_command.texture_environments,
            .indexed_geometry = .{
                .indices = context.indexed_geometry.indices.items,
                .vertex_positions = context.indexed_geometry.vertex_positions.items,
                .vertex_colours = &.{},
                .vertex_texture_coords = undefined,
            },
        };

        const state_index: u32 = @intCast(context.raster_tile_buffer.stream_states.items.len);

        context.raster_tile_buffer.stream_states.append(context.gpa, .{
            .geometry_state = geometry_state,
            .raster_state = raster_state,
            .uniforms = unfiorms,
        }) catch @panic("");

        for (triangle_group_begin..triangle_group_begin + triangle_group_count) |triangle_group_id| {
            defer triangle_id_start += 8 - @rem(triangle_id_start, 8);

            var triangle_mask: centralgpu.WarpRegister(bool) = @splat(true);

            var triangle_id: centralgpu.WarpRegister(u32) = std.simd.iota(u32, 8);

            triangle_id += @as(centralgpu.WarpRegister(u32), @splat(@intCast(triangle_group_id * 8)));

            triangle_mask &= triangle_id < @as(centralgpu.WarpRegister(u32), @splat(draw_command.triangle_id_start + draw_command.triangle_count));

            const projected_triangles = centralgpu.processGeometry(
                geometry_state,
                unfiorms,
                triangle_group_id * 8,
                triangle_mask,
            );

            if (std.simd.countTrues(projected_triangles.mask) == 0) {
                continue;
            }

            centralgpu.rasterizeSubmit(
                &context.raster_tile_buffer,
                state_index,
                geometry_state,
                raster_state,
                unfiorms,
                triangle_group_id * 8,
                projected_triangles,
            );
        }
    }

    {
        std.log.info("rasterizer_submit_time: {}ns\n", .{std.time.nanoTimestamp() - rasterizer_submit_time});
    }

    const rasterizer_flush_time = std.time.nanoTimestamp();

    centralgpu.rasterizerFlushTiles(context.bound_render_target, &context.raster_tile_buffer);

    {
        std.log.info("rasterizer_flush_time: {}ns\n", .{std.time.nanoTimestamp() - rasterizer_flush_time});
    }

    context.raster_tile_buffer.stream_states.clearRetainingCapacity();
    context.raster_tile_buffer.stream_triangles.clearRetainingCapacity();

    context.indexed_geometry.indices.clearRetainingCapacity();
    context.indexed_geometry.vertex_positions.clearRetainingCapacity();
    context.triangle_positions.clearRetainingCapacity();
    context.triangle_colors.clearRetainingCapacity();
    context.triangle_tex_coords.clearRetainingCapacity();
    context.draw_commands.clearRetainingCapacity();
    context.vertex_scratch.clearRetainingCapacity();
    context.triangle_count = 0;
}

pub export fn glFinish() callconv(.c) void {
    glFlush();
}

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

// Texture mapping
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
pub const GL_MIRRORED_REPEAT: i32 = 0x8370;
pub const GL_CLAMP_TO_BORDER: i32 = 0x812D;

pub const GL_FOG: i32 = 0x0B60;
pub const GL_FOG_MODE: i32 = 0x0B65;
pub const GL_FOG_DENSITY: i32 = 0x0B62;
pub const GL_FOG_COLOR: i32 = 0x0B66;
pub const GL_FOG_INDEX: i32 = 0x0B61;
pub const GL_FOG_START: i32 = 0x0B63;
pub const GL_FOG_END: i32 = 0x0B64;
pub const GL_LINEAR: i32 = 0x2601;
pub const GL_EXP: i32 = 0x0800;
pub const GL_EXP2: i32 = 0x0801;

pub const GL_ALPHA_TEST: i32 = 0x0BC0;
pub const GL_ALPHA_TEST_REF: i32 = 0x0BC2;
pub const GL_ALPHA_TEST_FUNC: i32 = 0x0BC1;

pub const GL_SCISSOR_BOX: i32 = 0x0C10;
pub const GL_SCISSOR_TEST: i32 = 0x0C11;

pub const GL_STENCIL_BITS: i32 = 0x0D57;
pub const GL_STENCIL_TEST: i32 = 0x0B90;
pub const GL_STENCIL_CLEAR_VALUE: i32 = 0x0B91;
pub const GL_STENCIL_FUNC: i32 = 0x0B92;
pub const GL_STENCIL_VALUE_MASK: i32 = 0x0B93;
pub const GL_STENCIL_FAIL: i32 = 0x0B94;
pub const GL_STENCIL_PASS_DEPTH_FAIL: i32 = 0x0B95;
pub const GL_STENCIL_PASS_DEPTH_PASS: i32 = 0x0B96;
pub const GL_STENCIL_REF: i32 = 0x0B97;
pub const GL_STENCIL_WRITEMASK: i32 = 0x0B98;
pub const GL_STENCIL_INDEX: i32 = 0x1901;
pub const GL_KEEP: i32 = 0x1E00;
pub const GL_REPLACE: i32 = 0x1E01;
pub const GL_INCR: i32 = 0x1E02;
pub const GL_DECR: i32 = 0x1E03;

pub const GL_CURRENT_BIT: i32 = 0x00000001;
pub const GL_POINT_BIT: i32 = 0x00000002;
pub const GL_LINE_BIT: i32 = 0x00000004;
pub const GL_POLYGON_BIT: i32 = 0x00000008;
pub const GL_POLYGON_STIPPLE_BIT: i32 = 0x00000010;
pub const GL_PIXEL_MODE_BIT: i32 = 0x00000020;
pub const GL_LIGHTING_BIT: i32 = 0x00000040;
pub const GL_FOG_BIT: i32 = 0x00000080;
pub const GL_DEPTH_BUFFER_BIT: i32 = 0x00000100;
pub const GL_ACCUM_BUFFER_BIT: i32 = 0x00000200;
pub const GL_STENCIL_BUFFER_BIT: i32 = 0x00000400;
pub const GL_VIEWPORT_BIT: i32 = 0x00000800;
pub const GL_TRANSFORM_BIT: i32 = 0x00001000;
pub const GL_ENABLE_BIT: i32 = 0x00002000;
pub const GL_COLOR_BUFFER_BIT: i32 = 0x00004000;
pub const GL_HINT_BIT: i32 = 0x00008000;
pub const GL_EVAL_BIT: i32 = 0x00010000;
pub const GL_LIST_BIT: i32 = 0x00020000;
pub const GL_TEXTURE_BIT: i32 = 0x00040000;
pub const GL_SCISSOR_BIT: i32 = 0x00080000;
pub const GL_ALL_ATTRIB_BITS: i32 = 0x000FFFFF;

pub const GL_RG: i32 = 0x8227;
pub const GL_RG16: i32 = 0x822C;
pub const GL_RG16F: i32 = 0x822F;
pub const GL_RG16I: i32 = 0x8239;
pub const GL_RG16UI: i32 = 0x823A;
pub const GL_RG16_SNORM: i32 = 0x8F99;
pub const GL_RG32F: i32 = 0x8230;
pub const GL_RG32I: i32 = 0x823B;
pub const GL_RG32UI: i32 = 0x823C;
pub const GL_RG8: i32 = 0x822B;
pub const GL_RG8I: i32 = 0x8237;
pub const GL_RG8UI: i32 = 0x8238;
pub const GL_RG8_SNORM: i32 = 0x8F95;
pub const GL_RGB: i32 = 0x1907;
pub const GL_RGB10: i32 = 0x8052;
pub const GL_RGB10_A2: i32 = 0x8059;
pub const GL_RGB10_A2UI: i32 = 0x906F;
pub const GL_RGB12: i32 = 0x8053;
pub const GL_RGB16: i32 = 0x8054;
pub const GL_RGB16F: i32 = 0x881B;
pub const GL_RGB16I: i32 = 0x8D89;
pub const GL_RGB16UI: i32 = 0x8D77;
pub const GL_RGB16_SNORM: i32 = 0x8F9A;
pub const GL_RGB32F: i32 = 0x8815;
pub const GL_RGB32I: i32 = 0x8D83;
pub const GL_RGB32UI: i32 = 0x8D71;
pub const GL_RGB4: i32 = 0x804F;
pub const GL_RGB5: i32 = 0x8050;
pub const GL_RGB5_A1: i32 = 0x8057;
pub const GL_RGB8: i32 = 0x8051;
pub const GL_RGB8I: i32 = 0x8D8F;
pub const GL_RGB8UI: i32 = 0x8D7D;
pub const GL_RGB8_SNORM: i32 = 0x8F96;
pub const GL_RGB9_E5: i32 = 0x8C3D;
pub const GL_RGBA: i32 = 0x1908;
pub const GL_RGBA12: i32 = 0x805A;
pub const GL_RGBA16: i32 = 0x805B;
pub const GL_RGBA16F: i32 = 0x881A;
pub const GL_RGBA16I: i32 = 0x8D88;
pub const GL_RGBA16UI: i32 = 0x8D76;
pub const GL_RGBA16_SNORM: i32 = 0x8F9B;
pub const GL_RGBA2: i32 = 0x8055;
pub const GL_RGBA32F: i32 = 0x8814;
pub const GL_RGBA32I: i32 = 0x8D82;
pub const GL_RGBA32UI: i32 = 0x8D70;
pub const GL_RGBA4: i32 = 0x8056;
pub const GL_RGBA8: i32 = 0x8058;
pub const GL_RGBA8I: i32 = 0x8D8E;
pub const GL_RGBA8UI: i32 = 0x8D7C;
pub const GL_RGBA8_SNORM: i32 = 0x8F97;
pub const GL_RGBA_INTEGER: i32 = 0x8D99;
pub const GL_RGBA_MODE: i32 = 0x0C31;
pub const GL_RGB_INTEGER: i32 = 0x8D98;

pub const GL_POINT: i32 = 0x1B00;
pub const GL_LINE: i32 = 0x1B01;
pub const GL_FILL: i32 = 0x1B02;
pub const GL_CW: i32 = 0x0900;
pub const GL_CCW: i32 = 0x0901;
pub const GL_FRONT: i32 = 0x0404;
pub const GL_BACK: i32 = 0x0405;
pub const GL_POLYGON_MODE: i32 = 0x0B40;
pub const GL_POLYGON_SMOOTH: i32 = 0x0B41;
pub const GL_POLYGON_STIPPLE: i32 = 0x0B42;
pub const GL_EDGE_FLAG: i32 = 0x0B43;
pub const GL_CULL_FACE: i32 = 0x0B44;
pub const GL_CULL_FACE_MODE: i32 = 0x0B45;
pub const GL_FRONT_FACE: i32 = 0x0B46;
pub const GL_POLYGON_OFFSET_FACTOR: i32 = 0x8038;
pub const GL_POLYGON_OFFSET_UNITS: i32 = 0x2A00;
pub const GL_POLYGON_OFFSET_POINT: i32 = 0x2A01;
pub const GL_POLYGON_OFFSET_LINE: i32 = 0x2A02;
pub const GL_POLYGON_OFFSET_FILL: i32 = 0x8037;

pub const GL_BLEND: i32 = 0x0BE2;
pub const GL_BLEND_SRC: i32 = 0x0BE1;
pub const GL_BLEND_DST: i32 = 0x0BE0;
pub const GL_ZERO: i32 = 0;
pub const GL_ONE: i32 = 1;
pub const GL_SRC_COLOR: i32 = 0x0300;
pub const GL_ONE_MINUS_SRC_COLOR: i32 = 0x0301;
pub const GL_SRC_ALPHA: i32 = 0x0302;
pub const GL_ONE_MINUS_SRC_ALPHA: i32 = 0x0303;
pub const GL_DST_ALPHA: i32 = 0x0304;
pub const GL_ONE_MINUS_DST_ALPHA: i32 = 0x0305;
pub const GL_DST_COLOR: i32 = 0x0306;
pub const GL_ONE_MINUS_DST_COLOR: i32 = 0x0307;
pub const GL_SRC_ALPHA_SATURATE: i32 = 0x0308;

pub const GL_ACCUM: i32 = 0x0100;
pub const GL_ADD: i32 = 0x0104;
pub const GL_LOAD: i32 = 0x0101;
pub const GL_MULT: i32 = 0x0103;

pub const GL_COMBINE: i32 = 0x8570;
pub const GL_COMBINE_RGB: i32 = 0x8571;
pub const GL_COMBINE_ALPHA: i32 = 0x8572;
pub const GL_SOURCE0_RGB: i32 = 0x8580;
pub const GL_SOURCE1_RGB: i32 = 0x8581;
pub const GL_SOURCE2_RGB: i32 = 0x8582;
pub const GL_SOURCE0_ALPHA: i32 = 0x8588;
pub const GL_SOURCE1_ALPHA: i32 = 0x8589;
pub const GL_SOURCE2_ALPHA: i32 = 0x858A;
pub const GL_OPERAND0_RGB: i32 = 0x8590;
pub const GL_OPERAND1_RGB: i32 = 0x8591;
pub const GL_OPERAND2_RGB: i32 = 0x8592;
pub const GL_OPERAND0_ALPHA: i32 = 0x8598;
pub const GL_OPERAND1_ALPHA: i32 = 0x8599;
pub const GL_OPERAND2_ALPHA: i32 = 0x859A;
pub const GL_RGB_SCALE: i32 = 0x8573;
pub const GL_ADD_SIGNED: i32 = 0x8574;
pub const GL_INTERPOLATE: i32 = 0x8575;
pub const GL_SUBTRACT: i32 = 0x84E7;
pub const GL_CONSTANT: i32 = 0x8576;
pub const GL_PRIMARY_COLOR: i32 = 0x8577;
pub const GL_PREVIOUS: i32 = 0x8578;

pub const GL_NEVER: i32 = 0x0200;
pub const GL_LESS: i32 = 0x0201;
pub const GL_EQUAL: i32 = 0x0202;
pub const GL_LEQUAL: i32 = 0x0203;
pub const GL_GREATER: i32 = 0x0204;
pub const GL_NOTEQUAL: i32 = 0x0205;
pub const GL_GEQUAL: i32 = 0x0206;
pub const GL_ALWAYS: i32 = 0x0207;
pub const GL_DEPTH_TEST: i32 = 0x0B71;
pub const GL_DEPTH_BITS: i32 = 0x0D56;
pub const GL_DEPTH_CLEAR_VALUE: i32 = 0x0B73;
pub const GL_DEPTH_FUNC: i32 = 0x0B74;
pub const GL_DEPTH_RANGE: i32 = 0x0B70;
pub const GL_DEPTH_WRITEMASK: i32 = 0x0B72;
pub const GL_DEPTH_COMPONENT: i32 = 0x1902;

pub const GL_VERTEX_ARRAY: i32 = 0x8074;
pub const GL_NORMAL_ARRAY: i32 = 0x8075;
pub const GL_COLOR_ARRAY: i32 = 0x8076;
pub const GL_INDEX_ARRAY: i32 = 0x8077;
pub const GL_TEXTURE_COORD_ARRAY: i32 = 0x8078;
pub const GL_EDGE_FLAG_ARRAY: i32 = 0x8079;
pub const GL_VERTEX_ARRAY_SIZE: i32 = 0x807A;
pub const GL_VERTEX_ARRAY_TYPE: i32 = 0x807B;
pub const GL_VERTEX_ARRAY_STRIDE: i32 = 0x807C;
pub const GL_NORMAL_ARRAY_TYPE: i32 = 0x807E;
pub const GL_NORMAL_ARRAY_STRIDE: i32 = 0x807F;
pub const GL_COLOR_ARRAY_SIZE: i32 = 0x8081;
pub const GL_COLOR_ARRAY_TYPE: i32 = 0x8082;
pub const GL_COLOR_ARRAY_STRIDE: i32 = 0x8083;
pub const GL_INDEX_ARRAY_TYPE: i32 = 0x8085;
pub const GL_INDEX_ARRAY_STRIDE: i32 = 0x8086;
pub const GL_TEXTURE_COORD_ARRAY_SIZE: i32 = 0x8088;
pub const GL_TEXTURE_COORD_ARRAY_TYPE: i32 = 0x8089;
pub const GL_TEXTURE_COORD_ARRAY_STRIDE: i32 = 0x808A;
pub const GL_EDGE_FLAG_ARRAY_STRIDE: i32 = 0x808C;
pub const GL_VERTEX_ARRAY_POINTER: i32 = 0x808E;
pub const GL_NORMAL_ARRAY_POINTER: i32 = 0x808F;
pub const GL_COLOR_ARRAY_POINTER: i32 = 0x8090;
pub const GL_INDEX_ARRAY_POINTER: i32 = 0x8091;
pub const GL_TEXTURE_COORD_ARRAY_POINTER: i32 = 0x8092;
pub const GL_EDGE_FLAG_ARRAY_POINTER: i32 = 0x8093;
pub const GL_V2F: i32 = 0x2A20;
pub const GL_V3F: i32 = 0x2A21;
pub const GL_C4UB_V2F: i32 = 0x2A22;
pub const GL_C4UB_V3F: i32 = 0x2A23;
pub const GL_C3F_V3F: i32 = 0x2A24;
pub const GL_N3F_V3F: i32 = 0x2A25;
pub const GL_C4F_N3F_V3F: i32 = 0x2A26;
pub const GL_T2F_V3F: i32 = 0x2A27;
pub const GL_T4F_V4F: i32 = 0x2A28;
pub const GL_T2F_C4UB_V3F: i32 = 0x2A29;
pub const GL_T2F_C3F_V3F: i32 = 0x2A2A;
pub const GL_T2F_N3F_V3F: i32 = 0x2A2B;
pub const GL_T2F_C4F_N3F_V3F: i32 = 0x2A2C;
pub const GL_T4F_C4F_N3F_V4F: i32 = 0x2A2D;

pub const GL_MAP_COLOR: i32 = 0x0D10;
pub const GL_MAP_STENCIL: i32 = 0x0D11;
pub const GL_INDEX_SHIFT: i32 = 0x0D12;
pub const GL_INDEX_OFFSET: i32 = 0x0D13;
pub const GL_RED_SCALE: i32 = 0x0D14;
pub const GL_RED_BIAS: i32 = 0x0D15;
pub const GL_GREEN_SCALE: i32 = 0x0D18;
pub const GL_GREEN_BIAS: i32 = 0x0D19;
pub const GL_BLUE_SCALE: i32 = 0x0D1A;
pub const GL_BLUE_BIAS: i32 = 0x0D1B;
pub const GL_ALPHA_SCALE: i32 = 0x0D1C;
pub const GL_ALPHA_BIAS: i32 = 0x0D1D;
pub const GL_DEPTH_SCALE: i32 = 0x0D1E;
pub const GL_DEPTH_BIAS: i32 = 0x0D1F;
pub const GL_PIXEL_MAP_S_TO_S_SIZE: i32 = 0x0CB1;
pub const GL_PIXEL_MAP_I_TO_I_SIZE: i32 = 0x0CB0;
pub const GL_PIXEL_MAP_I_TO_R_SIZE: i32 = 0x0CB2;
pub const GL_PIXEL_MAP_I_TO_G_SIZE: i32 = 0x0CB3;
pub const GL_PIXEL_MAP_I_TO_B_SIZE: i32 = 0x0CB4;
pub const GL_PIXEL_MAP_I_TO_A_SIZE: i32 = 0x0CB5;
pub const GL_PIXEL_MAP_R_TO_R_SIZE: i32 = 0x0CB6;
pub const GL_PIXEL_MAP_G_TO_G_SIZE: i32 = 0x0CB7;
pub const GL_PIXEL_MAP_B_TO_B_SIZE: i32 = 0x0CB8;
pub const GL_PIXEL_MAP_A_TO_A_SIZE: i32 = 0x0CB9;
pub const GL_PIXEL_MAP_S_TO_S: i32 = 0x0C71;
pub const GL_PIXEL_MAP_I_TO_I: i32 = 0x0C70;
pub const GL_PIXEL_MAP_I_TO_R: i32 = 0x0C72;
pub const GL_PIXEL_MAP_I_TO_G: i32 = 0x0C73;
pub const GL_PIXEL_MAP_I_TO_B: i32 = 0x0C74;
pub const GL_PIXEL_MAP_I_TO_A: i32 = 0x0C75;
pub const GL_PIXEL_MAP_R_TO_R: i32 = 0x0C76;
pub const GL_PIXEL_MAP_G_TO_G: i32 = 0x0C77;
pub const GL_PIXEL_MAP_B_TO_B: i32 = 0x0C78;
pub const GL_PIXEL_MAP_A_TO_A: i32 = 0x0C79;
pub const GL_PACK_ALIGNMENT: i32 = 0x0D05;
pub const GL_PACK_LSB_FIRST: i32 = 0x0D01;
pub const GL_PACK_ROW_LENGTH: i32 = 0x0D02;
pub const GL_PACK_SKIP_PIXELS: i32 = 0x0D04;
pub const GL_PACK_SKIP_ROWS: i32 = 0x0D03;
pub const GL_PACK_SWAP_BYTES: i32 = 0x0D00;
pub const GL_UNPACK_ALIGNMENT: i32 = 0x0CF5;
pub const GL_UNPACK_LSB_FIRST: i32 = 0x0CF1;
pub const GL_UNPACK_ROW_LENGTH: i32 = 0x0CF2;
pub const GL_UNPACK_SKIP_PIXELS: i32 = 0x0CF4;
pub const GL_UNPACK_SKIP_ROWS: i32 = 0x0CF3;
pub const GL_UNPACK_SWAP_BYTES: i32 = 0x0CF0;
pub const GL_ZOOM_X: i32 = 0x0D16;
pub const GL_ZOOM_Y: i32 = 0x0D17;

const log = std.log.scoped(.centralgpu_gl);
const std = @import("std");
const centralgpu = @import("centralgpu");
const matrix_math = @import("gl/matrix.zig");
