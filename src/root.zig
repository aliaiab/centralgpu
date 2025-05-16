pub const Rgba32 = packed struct(u32) {
    r: u8,
    g: u8,
    b: u8,
    a: u8,

    pub fn fromNormalized(color: @Vector(4, f32)) Rgba32 {
        return .{
            .r = @intFromFloat(color[0] * 255),
            .g = @intFromFloat(color[1] * 255),
            .b = @intFromFloat(color[2] * 255),
            .a = @intFromFloat(color[3] * 255),
        };
    }
};

pub const XRgb888 = packed struct(u32) {
    b: u8,
    g: u8,
    r: u8,
    x: u8 = 0,
};

///Computes the actual, padded size in either x or y of a render target
pub fn computeTargetPaddedSize(value: usize) usize {
    return (@divTrunc(value, 64) + @intFromBool(@rem(value, 64) != 0)) * 64;
}

pub const Image = struct {
    pixel_ptr: [*]Rgba32,
    width: u32,
    height: u32,

    pub fn pixels(image: Image) []Rgba32 {
        return image.pixel_ptr[0 .. image.width * image.height];
    }
};

pub const ImageDescriptor = packed struct(u64) {
    rel_ptr: i32,
    width_log2: u4,
    height_log2: u4,
    sampler_filter: enum(u1) {
        nearest,
        bilinear,
    },
    sampler_address_mode: enum(u1) {
        repeat,
        clamp_to_edge,
    },
    _: u22 = 0,
};

pub fn WarpRegister(comptime T: type) type {
    const len = std.simd.suggestVectorLength(u32).?;

    return @Vector(len, T);
}

pub fn WarpVec2(comptime T: type) type {
    return struct {
        x: WarpRegister(T),
        y: WarpRegister(T),
    };
}

pub fn WarpVec3(comptime T: type) type {
    return struct {
        x: WarpRegister(T),
        y: WarpRegister(T),
        z: WarpRegister(T),

        pub fn splat(value: WarpRegister(T)) @This() {
            return .{
                .x = value,
                .y = value,
                .z = value,
                .w = value,
            };
        }

        pub fn add(lhs: @This(), rhs: @This()) @This() {
            return .{
                .x = lhs.x + rhs.x,
                .y = lhs.y + rhs.y,
                .z = lhs.z + rhs.z,
                .w = lhs.w + rhs.w,
            };
        }
    };
}

pub fn WarpVec4(comptime T: type) type {
    return struct {
        x: WarpRegister(T),
        y: WarpRegister(T),
        z: WarpRegister(T),
        w: WarpRegister(T),

        pub fn xy(self: @This()) WarpVec2(T) {
            return .{ .x = self.x, .y = self.y };
        }

        pub fn splat(value: WarpRegister(T)) @This() {
            return .{
                .x = value,
                .y = value,
                .z = value,
                .w = value,
            };
        }

        pub fn add(lhs: @This(), rhs: @This()) @This() {
            return .{
                .x = lhs.x + rhs.x,
                .y = lhs.y + rhs.y,
                .z = lhs.z + rhs.z,
                .w = lhs.w + rhs.w,
            };
        }

        pub fn mul(lhs: @This(), rhs: @This()) @This() {
            return .{
                .x = lhs.x * rhs.x,
                .y = lhs.y * rhs.y,
                .z = lhs.z * rhs.z,
                .w = lhs.w * rhs.w,
            };
        }
    };
}

pub const WarpProjectedTriangle = struct {
    mask: WarpRegister(bool),
    points: [3]WarpVec4(f32),
    unclipped_points: [3]WarpVec4(f32),
};

pub const Uniforms = struct {
    vertex_positions: []const [3]WarpVec3(f32),
    vertex_colours: []const [3]u32,
    vertex_texture_coords: []const [3][2]f32,

    image_base: [*]const u8,
    image_descriptor: ImageDescriptor,
};

pub const GeometryProcessState = struct {
    viewport_transform: struct {
        scale_x: f32,
        scale_y: f32,
        translation_x: f32,
        translation_y: f32,
    },
};

pub fn processGeometry(
    state: GeometryProcessState,
    uniforms: Uniforms,
    triangle_id_start: usize,
    input_mask: WarpRegister(bool),
) WarpProjectedTriangle {
    const in_triangle = uniforms.vertex_positions[triangle_id_start / 8];

    var out_triangle: [3]WarpVec4(f32) = undefined;
    var out_mask = input_mask;

    var cull_triangle: WarpRegister(bool) = @splat(true);

    for (0..3) |vertex_index| {
        out_triangle[vertex_index] = .{
            .x = in_triangle[vertex_index].x,
            .y = in_triangle[vertex_index].y,
            .z = in_triangle[vertex_index].z,
            .w = in_triangle[vertex_index].z,
        };

        const vector_minus_one: WarpRegister(f32) = @splat(-1);
        const vector_one: WarpRegister(f32) = @splat(1);

        cull_triangle = vectorBoolAnd(
            cull_triangle,
            vectorBoolOr(
                out_triangle[vertex_index].x < vector_minus_one,
                out_triangle[vertex_index].x > vector_one,
            ),
        );

        cull_triangle = vectorBoolAnd(
            cull_triangle,
            vectorBoolOr(
                out_triangle[vertex_index].y < vector_minus_one,
                out_triangle[vertex_index].y > vector_one,
            ),
        );

        const reciprocal_w = @as(WarpRegister(f32), @splat(1)) / out_triangle[vertex_index].w;

        out_triangle[vertex_index].x *= reciprocal_w;
        out_triangle[vertex_index].y *= reciprocal_w;
        out_triangle[vertex_index].z *= reciprocal_w;

        out_triangle[vertex_index].x = @mulAdd(
            WarpRegister(f32),
            out_triangle[vertex_index].x,
            @splat(state.viewport_transform.scale_x),
            @splat(state.viewport_transform.translation_x),
        );

        out_triangle[vertex_index].y = @mulAdd(
            WarpRegister(f32),
            out_triangle[vertex_index].y,
            @splat(state.viewport_transform.scale_y),
            @splat(state.viewport_transform.translation_y),
        );
    }

    out_mask = vectorBoolAnd(out_mask, vectorBoolNot(cull_triangle));

    return .{
        .mask = out_mask,
        .points = out_triangle,
        .unclipped_points = out_triangle,
    };
}

pub const RasterState = struct {
    scissor_min_x: i32,
    scissor_min_y: i32,
    scissor_max_x: i32,
    scissor_max_y: i32,

    render_target: Image,
};

///Rasterizer stage
pub fn rasterize(
    raster_state: RasterState,
    uniforms: Uniforms,
    triangle_id_start: usize,
    in_projected_triangle: WarpProjectedTriangle,
) void {
    @setRuntimeSafety(false);
    const projected_triangle = in_projected_triangle;

    const triangle_area = edgeFunctionSimd(
        projected_triangle.unclipped_points[0],
        projected_triangle.unclipped_points[1],
        projected_triangle.unclipped_points[2],
    );
    const triangle_area_reciprocal = @as(WarpRegister(f32), @splat(1)) / triangle_area;

    var bounds_min_x: WarpRegister(f32) = @splat(0);
    var bounds_min_y: WarpRegister(f32) = @splat(0);

    var bounds_max_x: WarpRegister(f32) = @splat(@floatFromInt(raster_state.render_target.width));
    var bounds_max_y: WarpRegister(f32) = @splat(@floatFromInt(raster_state.render_target.height));

    var triangle_min_x: WarpRegister(f32) = @splat(std.math.inf(f32));
    var triangle_min_y: WarpRegister(f32) = @splat(std.math.inf(f32));

    var triangle_max_x: WarpRegister(f32) = @splat(-std.math.inf(f32));
    var triangle_max_y: WarpRegister(f32) = @splat(-std.math.inf(f32));

    for (projected_triangle.points) |point| {
        triangle_min_x = @min(point.x, triangle_min_x);
        triangle_min_y = @min(point.y, triangle_min_y);

        triangle_max_x = @max(point.x, triangle_max_x);
        triangle_max_y = @max(point.y, triangle_max_y);
    }

    bounds_min_x = @max(bounds_min_x, triangle_min_x);
    bounds_min_y = @max(bounds_min_y, triangle_min_y);

    bounds_max_x = @min(bounds_max_x, triangle_max_x);
    bounds_max_y = @min(bounds_max_y, triangle_max_y);

    var start_x: WarpRegister(i32) = @intFromFloat(@floor(bounds_min_x));
    var start_y: WarpRegister(i32) = @intFromFloat(@floor(bounds_min_y));

    start_x = @max(start_x, @as(WarpRegister(i32), @splat(raster_state.scissor_min_x)));
    start_y = @max(start_y, @as(WarpRegister(i32), @splat(raster_state.scissor_min_y)));

    var end_x: WarpRegister(i32) = @intFromFloat(@ceil(bounds_max_x));
    var end_y: WarpRegister(i32) = @intFromFloat(@ceil(bounds_max_y));

    end_x = @min(end_x, @as(WarpRegister(i32), @splat(raster_state.scissor_max_x)));
    end_y = @min(end_y, @as(WarpRegister(i32), @splat(raster_state.scissor_max_y)));

    const mask_integer: u8 = @bitCast(projected_triangle.mask);

    const triangle_max_count = 8 - @clz(mask_integer);

    for (0..triangle_max_count) |triangle_index| {
        if (!projected_triangle.mask[triangle_index]) continue;

        //rasterisze triangle
        rasterizeTriangle(
            raster_state,
            uniforms,
            projected_triangle,
            triangle_id_start + triangle_index,
            triangle_index,
            triangle_area_reciprocal[triangle_index],
            start_x[triangle_index],
            start_y[triangle_index],
            end_x[triangle_index],
            end_y[triangle_index],
        );
    }
}

pub fn rasterizeTriangle(
    raster_state: RasterState,
    uniforms: Uniforms,
    projected_triangle: WarpProjectedTriangle,
    triangle_id: usize,
    triangle_index: usize,
    triangle_area_reciprocal: f32,
    start_x: i32,
    start_y: i32,
    end_x: i32,
    end_y: i32,
) void {
    @setRuntimeSafety(true);
    _ = triangle_index; // autofix
    const block_width = 4;
    const block_height = block_width;

    const block_start_x = @divTrunc(start_x, block_width);
    const block_start_y = @divTrunc(start_y, block_height);

    //Ceiling division
    const block_end_x = @divTrunc(end_x, block_width) + (@intFromBool(@rem(end_x, block_width) != 0));
    const block_end_y = @divTrunc(end_y, block_height) + (@intFromBool(@rem(end_y, block_height) != 0));

    const unclipped_vert_0: WarpVec2(f32) = .{
        .x = @splat(projected_triangle.unclipped_points[0].x[triangle_id]),
        .y = @splat(projected_triangle.unclipped_points[0].y[triangle_id]),
    };

    const unclipped_vert_1: WarpVec2(f32) = .{
        .x = @splat(projected_triangle.unclipped_points[1].x[triangle_id]),
        .y = @splat(projected_triangle.unclipped_points[1].y[triangle_id]),
    };

    const unclipped_vert_2: WarpVec2(f32) = .{
        .x = @splat(projected_triangle.unclipped_points[2].x[triangle_id]),
        .y = @splat(projected_triangle.unclipped_points[2].y[triangle_id]),
    };

    //per primitive processing
    const vertex_color_0_packed: WarpRegister(u32) = @splat(uniforms.vertex_colours[triangle_id][0]);
    const vertex_color_1_packed: WarpRegister(u32) = @splat(uniforms.vertex_colours[triangle_id][1]);
    const vertex_color_2_packed: WarpRegister(u32) = @splat(uniforms.vertex_colours[triangle_id][2]);

    const vertex_color_0 = unpackUnorm4x(vertex_color_0_packed);
    const vertex_color_1 = unpackUnorm4x(vertex_color_1_packed);
    const vertex_color_2 = unpackUnorm4x(vertex_color_2_packed);

    const vertex_texcoord_u_0: WarpRegister(f32) = @splat(uniforms.vertex_texture_coords[triangle_id][0][0]);
    const vertex_texcoord_u_1: WarpRegister(f32) = @splat(uniforms.vertex_texture_coords[triangle_id][1][0]);
    const vertex_texcoord_u_2: WarpRegister(f32) = @splat(uniforms.vertex_texture_coords[triangle_id][2][0]);

    const vertex_texcoord_v_0: WarpRegister(f32) = @splat(uniforms.vertex_texture_coords[triangle_id][0][1]);
    const vertex_texcoord_v_1: WarpRegister(f32) = @splat(uniforms.vertex_texture_coords[triangle_id][1][1]);
    const vertex_texcoord_v_2: WarpRegister(f32) = @splat(uniforms.vertex_texture_coords[triangle_id][2][1]);

    const block_count_x = raster_state.render_target.width / block_width + @intFromBool(raster_state.render_target.width % block_width != 0);

    var block_y: isize = block_start_y;

    while (block_y < block_end_y) : (block_y += 1) {
        var block_x: isize = block_start_x;

        while (block_x < block_end_x) : (block_x += 1) {
            const block_index: usize = @intCast(block_x + block_y * block_count_x);
            // const block_index = mortonEncodeScalar(@intCast(block_x), @intCast(block_y));
            const block_start_ptr = raster_state.render_target.pixel_ptr + block_index * block_width * block_height;

            for (0..2) |half_y_offset| {
                const y_offset: isize = @intCast(half_y_offset * 2);
                var execution_mask: WarpRegister(u32) = @splat(1);

                const target_start_ptr = block_start_ptr + @as(usize, @intCast(y_offset * block_width));

                const y: isize = block_y * block_height + y_offset;
                const x_base_offset = block_x * block_width;

                const swizzled_point_y_offset: WarpRegister(i32) = .{
                    0, 0, 0, 0,
                    1, 1, 1, 1,
                };

                const swizzled_point_x: WarpRegister(i32) = .{
                    0, 1, 2, 3,
                    0, 1, 2, 3,
                };

                //Pixel coordinate x
                var point_x_int: WarpRegister(i32) = @splat(@truncate(x_base_offset));

                point_x_int += swizzled_point_x;
                var point_y_int: WarpRegister(i32) = @splat(@truncate(y));

                //Pixel coordinate y
                point_y_int += swizzled_point_y_offset;

                var scissor_test: WarpRegister(bool) = @splat(true);

                scissor_test = vectorBoolAnd(scissor_test, point_x_int >= @as(WarpRegister(i32), @splat(raster_state.scissor_min_x)));
                scissor_test = vectorBoolAnd(scissor_test, point_y_int >= @as(WarpRegister(i32), @splat(raster_state.scissor_min_y)));

                scissor_test = vectorBoolAnd(scissor_test, point_x_int < @as(WarpRegister(i32), @splat(raster_state.scissor_max_x)));
                scissor_test = vectorBoolAnd(scissor_test, point_y_int < @as(WarpRegister(i32), @splat(raster_state.scissor_max_y)));

                execution_mask &= @intFromBool(scissor_test);

                var point_x: WarpRegister(f32) = @floatFromInt(point_x_int);
                var point_y: WarpRegister(f32) = @floatFromInt(point_y_int);

                point_x += @splat(0.5);
                point_y += @splat(0.5);

                var bary_0 = edgeFunctionSimdVec2(unclipped_vert_1, unclipped_vert_2, .{ .x = point_x, .y = point_y });
                var bary_1 = edgeFunctionSimdVec2(unclipped_vert_2, unclipped_vert_0, .{ .x = point_x, .y = point_y });
                var bary_2 = edgeFunctionSimdVec2(unclipped_vert_0, unclipped_vert_1, .{ .x = point_x, .y = point_y });

                bary_0 *= @splat(triangle_area_reciprocal);
                bary_1 *= @splat(triangle_area_reciprocal);
                bary_2 *= @splat(triangle_area_reciprocal);

                execution_mask &= @intFromBool(bary_0 >= @as(WarpRegister(f32), @splat(0)));
                execution_mask &= @intFromBool(bary_1 >= @as(WarpRegister(f32), @splat(0)));
                execution_mask &= @intFromBool(bary_2 >= @as(WarpRegister(f32), @splat(0)));

                execution_mask &= @intFromBool(bary_0 <= @as(WarpRegister(f32), @splat(1)));
                execution_mask &= @intFromBool(bary_1 <= @as(WarpRegister(f32), @splat(1)));
                execution_mask &= @intFromBool(bary_2 <= @as(WarpRegister(f32), @splat(1)));

                if (@reduce(.Or, execution_mask) == 0) {
                    continue;
                }

                const tex_u = bary_0 * vertex_texcoord_u_0 + bary_1 * vertex_texcoord_u_1 + bary_2 * vertex_texcoord_u_2;
                const tex_v = bary_0 * vertex_texcoord_v_0 + bary_1 * vertex_texcoord_v_1 + bary_2 * vertex_texcoord_v_2;

                const color_r = bary_0 * vertex_color_0.x + bary_1 * vertex_color_1.x + bary_2 * vertex_color_2.x;
                const color_g = bary_0 * vertex_color_0.y + bary_1 * vertex_color_1.y + bary_2 * vertex_color_2.y;
                const color_b = bary_0 * vertex_color_0.z + bary_1 * vertex_color_1.z + bary_2 * vertex_color_2.z;
                const color_a = bary_0 * vertex_color_0.w + bary_1 * vertex_color_1.w + bary_2 * vertex_color_2.w;

                const texture_sample = quadImageSample(
                    execution_mask != @as(WarpRegister(u1), @splat(0)),
                    @ptrCast(@alignCast(uniforms.image_base)),
                    uniforms.image_descriptor,
                    .{ .x = tex_u, .y = tex_v },
                );

                const color: WarpVec4(f32) = .{ .x = color_r, .y = color_g, .z = color_b, .w = color_a };
                _ = color; // autofix
                // const color = packUnorm4x(.{ .x = tex_u, .y = tex_v, .z = @splat(0), .w = @splat(1) });
                // const color = packUnorm4x(.{ .x = texture_sample.x, .y = texture_sample.y, .z = texture_sample.z, .w = @splat(1) });

                var color_result: WarpVec4(f32) = texture_sample;

                execution_mask &= ~(@intFromBool(texture_sample.x == @as(WarpRegister(f32), @splat(0))) &
                    @intFromBool(texture_sample.y == @as(WarpRegister(f32), @splat(0))) &
                    @intFromBool(texture_sample.z == @as(WarpRegister(f32), @splat(0))));

                color_result = color_result;

                // color_result = .mul(color_result, color);

                const packed_color = packUnorm4x(color_result);

                maskedStore(
                    u32,
                    execution_mask != @as(WarpRegister(u1), @splat(0)),
                    @ptrCast(@alignCast(target_start_ptr)),
                    packed_color,
                );
            }
        }
    }
}

inline fn edgeFunctionSimdVec2(
    a: WarpVec2(f32),
    b: WarpVec2(f32),
    c: WarpVec2(f32),
) WarpRegister(f32) {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

inline fn edgeFunctionSimd(
    a: WarpVec4(f32),
    b: WarpVec4(f32),
    c: WarpVec4(f32),
) WarpRegister(f32) {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

pub fn packUnorm4x(
    values: WarpVec4(f32),
) WarpRegister(u32) {
    const max_value: WarpRegister(f32) = @splat(std.math.maxInt(u8));

    @setRuntimeSafety(false);

    const x_in: WarpRegister(i32) = @intFromFloat(values.x * max_value);
    const y_in: WarpRegister(i32) = @intFromFloat(values.y * max_value);
    const z_in: WarpRegister(i32) = @intFromFloat(values.z * max_value);
    const w_in: WarpRegister(i32) = @intFromFloat(values.w * max_value);

    const zero: WarpRegister(i32) = @splat(0);
    const max_value_i32: WarpRegister(i32) = @splat(std.math.maxInt(u8));

    const x: WarpRegister(u32) = @abs(std.math.clamp(x_in, zero, max_value_i32));
    const y: WarpRegister(u32) = @abs(std.math.clamp(y_in, zero, max_value_i32));
    const z: WarpRegister(u32) = @abs(std.math.clamp(z_in, zero, max_value_i32));
    const w: WarpRegister(u32) = @abs(std.math.clamp(w_in, zero, max_value_i32));

    var result: WarpRegister(u32) = @splat(0);

    result |= w;
    result <<= @splat(8);
    result |= z;
    result <<= @splat(8);
    result |= y;
    result <<= @splat(8);
    result |= x;

    return result;
}

pub fn unpackUnorm4x(
    value: WarpRegister(u32),
) WarpVec4(f32) {
    var package: WarpRegister(u32) = value;

    package >>= @splat(8);
    const y: WarpRegister(u8) = @truncate(package);
    package >>= @splat(8);
    const z: WarpRegister(u8) = @truncate(package);
    package >>= @splat(8);
    const w: WarpRegister(u8) = @truncate(package);
    const x: WarpRegister(u8) = @truncate(value);

    const x_float: WarpRegister(f32) = @floatFromInt(x);
    const y_float: WarpRegister(f32) = @floatFromInt(y);
    const z_float: WarpRegister(f32) = @floatFromInt(z);
    const w_float: WarpRegister(f32) = @floatFromInt(w);

    const max_value: WarpRegister(f32) = @splat(std.math.maxInt(u8));

    return .{
        .x = x_float / max_value,
        .y = y_float / max_value,
        .z = z_float / max_value,
        .w = w_float / max_value,
    };
}

pub inline fn maskedLoad(
    comptime T: type,
    predicate: @Vector(8, bool),
    src: [*]const T,
) @Vector(8, T) {
    @setRuntimeSafety(false);

    switch (@import("builtin").cpu.arch) {
        .x86_64 => {
            const shift_amnt: @Vector(8, u32) = @splat(31);
            const mask: @Vector(8, u32) = @as(@Vector(8, u32), @intFromBool(predicate)) << shift_amnt;

            var values: @Vector(8, T) = undefined;

            asm volatile (
                \\vmaskmovps (%[src]), %[mask], %[values]
                : [values] "=v" (values),
                : [src] "r" (src),
                  [mask] "v" (mask),
            );

            return values;
        },
        else => {
            var result: @Vector(8, T) = undefined;

            //TODO: use avx2 vmaskmov
            inline for (0..8) |i| {
                if (predicate[i]) {
                    result[i] = src[i];
                }
            }

            return result;
        },
    }
}

pub inline fn maskedStore(
    comptime T: type,
    predicate: @Vector(8, bool),
    dest: *align(@alignOf(T)) @Vector(8, T),
    values: @Vector(8, T),
) void {
    @setRuntimeSafety(false);

    switch (@import("builtin").cpu.arch) {
        .x86_64 => {
            const shift_amnt: @Vector(8, u32) = @splat(31);
            const mask: @Vector(8, u32) = @as(@Vector(8, u32), @intFromBool(predicate)) << shift_amnt;

            asm volatile (
                \\vmaskmovps %[values], %[mask], (%[dest])
                :
                : [dest] "r" (dest),
                  [values] "v" (values),
                  [mask] "v" (mask),
            );
        },
        else => {
            inline for (0..8) |i| {
                if (predicate[i]) {
                    dest[i] = values[i];
                }
            }
        },
    }
}

pub inline fn maskedGather(
    comptime T: type,
    predicate: @Vector(8, bool),
    base: [*]const T,
    address: @Vector(8, u32),
) @Vector(8, T) {
    @setRuntimeSafety(false);

    switch (@import("builtin").cpu.arch) {
        .x86_64 => {
            const shift_amnt: @Vector(8, u32) = @splat(31);
            var mask: @Vector(8, u32) = @as(@Vector(8, u32), @intFromBool(predicate)) << shift_amnt;

            return asm (
                \\vgatherdps %[mask], (%[base], %[address], 4), %[ret]
                : [ret] "=&v" (-> @Vector(8, T)),
                  [mask] "+&v" (mask),
                : [address] "v" (address),
                  [base] "r" (base),
            );
        },
        else => {
            //TODO: use avx2 vpgather
            var result: @Vector(8, T) = undefined;

            inline for (0..8) |i| {
                if (predicate[i]) {
                    result[i] = base[address[i]];
                }
            }
            return result;
        },
    }
}

//Computes the derivative of value within the quad
pub fn quadComputeDerivative(value: WarpRegister(f32)) WarpVec2(f32) {
    const neighbour_x = quadReadNeigbhourX(value);
    const neighbour_y = quadReadNeigbhourY(value);

    const diff_x = value - neighbour_x;
    const diff_y = value - neighbour_y;

    return .{ .x = diff_x, .y = diff_y };
}

//Computes the coarse derivative of value within the quad
pub fn quadComputeDerivativeCoarse(value: WarpRegister(f32)) WarpVec2(f32) {
    const neighbour_x = quadReadNeigbhourX(value);
    const neighbour_y = quadReadNeigbhourY(value);

    const diff_x = value - neighbour_x;
    const diff_y = value - neighbour_y;

    return .{ .x = @splat(diff_x[0]), .y = @splat(diff_y[0]) };
}

///Reads the quad neighbour to the right for each warp thread
///Same as QuadReadAcrossX in Hlsl
pub fn quadReadNeigbhourX(warp_value: WarpRegister(f32)) WarpRegister(f32) {
    const shuffled = @shuffle(f32, warp_value, warp_value, WarpRegister(i32){
        1, 0, 3, 2,
        5, 4, 7, 6,
    });

    return shuffled;
}

///Reads the quad neighbour below each warp thread
///Same as QuadReadAcrossY in Hlsl
pub fn quadReadNeigbhourY(warp_value: WarpRegister(f32)) WarpRegister(f32) {
    const shuffled = @shuffle(f32, warp_value, warp_value, WarpRegister(i32){
        4, 5, 6, 7,
        0, 1, 2, 3,
    });

    return shuffled;
}

pub fn quadImageSample(
    execution_mask: WarpRegister(bool),
    base: [*]const u32,
    descriptor: ImageDescriptor,
    uv: WarpVec2(f32),
) WarpVec4(f32) {
    return imageSampleDerivative(
        execution_mask,
        base,
        descriptor,
        uv,
        quadComputeDerivativeCoarse(uv.x),
        quadComputeDerivativeCoarse(uv.y),
    );
}

pub fn imageSampleDerivative(
    execution_mask: WarpRegister(bool),
    base: [*]const u32,
    descriptor: ImageDescriptor,
    uv: WarpVec2(f32),
    u_derivative: WarpVec2(f32),
    v_derivative: WarpVec2(f32),
) WarpVec4(f32) {
    const texture_width: WarpRegister(f32) = @splat(@floatFromInt(@as(u32, 1) << @as(u5, descriptor.width_log2)));
    const texture_height: WarpRegister(f32) = @splat(@floatFromInt(@as(u32, 1) << @as(u5, descriptor.height_log2)));

    const scaled_dv_dx = v_derivative.x * texture_width;
    const scaled_dv_dy = v_derivative.y * texture_height;

    const scaled_du_dx = u_derivative.x * texture_width;
    const scaled_du_dy = u_derivative.y * texture_height;

    const length_x = @sqrt(scaled_du_dx * scaled_du_dx + scaled_dv_dx * scaled_dv_dx);
    const length_y = @sqrt(scaled_du_dy * scaled_du_dy + scaled_dv_dy * scaled_dv_dy);
    const rho_factor = @max(length_x, length_y);
    const mip_level = @log2(rho_factor);

    if (mip_level[0] - 3 >= 0) {
        return .{
            .x = rho_factor,
            .y = @splat(0),
            .z = @splat(0),
            .w = @splat(1),
        };
    }

    return imageLoad(execution_mask, base, descriptor, uv);
}

pub fn imageLoad(
    execution_mask: WarpRegister(bool),
    ///Base address from which the descriptor loads
    base: [*]const u32,
    descriptor: ImageDescriptor,
    ///Image coordinates
    uv: WarpVec2(f32),
) WarpVec4(f32) {
    @setRuntimeSafety(false);

    //idx = x + y * width
    const u_scale: WarpRegister(f32) = @splat(@floatFromInt((@as(u32, 1) << @as(u5, descriptor.width_log2)) - 1));
    const v_scale: WarpRegister(f32) = @splat(@floatFromInt((@as(u32, 1) << @as(u5, descriptor.height_log2)) - 1));

    //Handle wrapping
    var u: WarpRegister(f32) = uv.x;
    var v: WarpRegister(f32) = uv.y;

    switch (descriptor.sampler_address_mode) {
        .repeat => {
            u = u - @floor(u);
            v = v - @floor(v);
        },
        .clamp_to_edge => {
            u = @max(@min(u, @as(WarpRegister(f32), @splat(1))), @as(WarpRegister(f32), @splat(0)));
            v = @max(@min(v, @as(WarpRegister(f32), @splat(1))), @as(WarpRegister(f32), @splat(0)));
        },
    }

    const image_x_float: WarpRegister(f32) = u * u_scale;
    const image_y_float: WarpRegister(f32) = v * v_scale;

    const image_x: WarpRegister(i32) = @intFromFloat(@floor(image_x_float));
    const image_y: WarpRegister(i32) = @intFromFloat(@floor(image_y_float));

    const pixel_address = imageAddress(descriptor, image_x, image_y);

    const sample_packed = maskedGather(u32, execution_mask, base, @intCast(pixel_address));

    var sample = unpackUnorm4x(sample_packed);

    var pixel_address_float: WarpRegister(f32) = @floatFromInt(pixel_address);

    pixel_address_float /= (u_scale + @as(WarpRegister(f32), @splat(1))) * (v_scale + @as(WarpRegister(f32), @splat(1)));

    // sample.x = pixel_address_float;
    // sample.y = pixel_address_float;
    // sample.z = pixel_address_float;
    sample = sample;

    return sample;
}

///Computes the address of a pixel relative to the image level base from its x and y coordinates
pub inline fn imageAddress(
    descriptor: ImageDescriptor,
    x: WarpRegister(i32),
    y: WarpRegister(i32),
) WarpRegister(u32) {
    _ = descriptor; // autofix
    @setRuntimeSafety(false);

    //TODO: potentially allow for linear images and/or other tiling schemes

    const pixel_address = mortonEncode(@intCast(x), @intCast(y));

    return pixel_address;
}

pub inline fn mortonEncode(x: WarpRegister(u32), y: WarpRegister(u32)) WarpRegister(u32) {
    var index: WarpRegister(u32) = mortonPart1by1(y);

    index <<= @splat(1);

    index += mortonPart1by1(x);

    return index;
}

pub inline fn mortonPart1by1(x_in: WarpRegister(u32)) WarpRegister(u32) {
    var x: WarpRegister(u32) = x_in;

    x = (x ^ (x << @as(WarpRegister(u5), @splat(8)))) & @as(WarpRegister(u32), @splat(0x00ff00ff)); // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x << @as(WarpRegister(u5), @splat(4)))) & @as(WarpRegister(u32), @splat(0x0f0f0f0f)); // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x << @as(WarpRegister(u5), @splat(2)))) & @as(WarpRegister(u32), @splat(0x33333333)); // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x << @as(WarpRegister(u5), @splat(1)))) & @as(WarpRegister(u32), @splat(0x55555555)); // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    return x;
}

pub inline fn mortonEncodeScalar(x: usize, y: usize) usize {
    var index = mortonPart1by1Scalar(y);

    index <<= 1;

    index += mortonPart1by1Scalar(x);

    return index;
}

///Restricted to the range 0....2^16
pub inline fn mortonPart1by1Scalar(x_in: usize) usize {
    var x = x_in;

    //TODO: figure out which one is better (two shifts or one mask)
    // x &= 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210

    x = (x ^ (x << 8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x << 4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x << 2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x << 1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    return x;
}

pub fn vectorBoolAnd(a: WarpRegister(bool), b: WarpRegister(bool)) WarpRegister(bool) {
    const a_int = @intFromBool(a);
    const b_int = @intFromBool(b);
    const anded = a_int & b_int;

    return anded == @as(WarpRegister(u1), @splat(1));
}

pub fn vectorBoolOr(a: WarpRegister(bool), b: WarpRegister(bool)) WarpRegister(bool) {
    const a_int = @intFromBool(a);
    const b_int = @intFromBool(b);
    const anded = a_int | b_int;

    return anded == @as(WarpRegister(u1), @splat(1));
}

pub fn vectorBoolNot(a: WarpRegister(bool)) WarpRegister(bool) {
    const a_int = @intFromBool(a);

    return ~a_int == @as(WarpRegister(u1), @splat(1));
}

///An image copy that can scale, format cast and layout transition
pub fn blitRasterTargetToLinear(
    src_pixels: [*]const Rgba32,
    dst_pixels: [*]XRgb888,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) void {
    const block_width: usize = 4;
    const block_height: usize = block_width;

    const block_count_x: usize = @divTrunc(src_width, block_width) + @intFromBool(@rem(src_width, block_width) != 0);

    const scale_x: f32 = @as(f32, @floatFromInt(src_width)) / @as(f32, @floatFromInt(dst_width));
    const scale_y: f32 = @as(f32, @floatFromInt(src_height)) / @as(f32, @floatFromInt(dst_height));

    var dst_y: usize = 0;

    while (dst_y < dst_height) : (dst_y += 1) {
        var dst_x: usize = 0;

        const src_y_float: f32 = @as(f32, @floatFromInt(dst_y)) * scale_y;
        const src_y: usize = @intFromFloat(src_y_float);
        const src_block_y = @divTrunc(src_y, block_height);

        while (dst_x < dst_width) : (dst_x += 1) {
            const src_x_float: f32 = @as(f32, @floatFromInt(dst_x)) * scale_x;

            const src_x: usize = @intFromFloat(src_x_float);

            const src_block_x = @divTrunc(src_x, block_width);

            const block_index = src_block_x + src_block_y * block_count_x;
            // const block_index = mortonEncodeScalar(@intCast(src_block_x), @intCast(src_block_y));

            const block_pixels = src_pixels + block_index * block_width * block_height;

            const block_offset_x: usize = src_x & 0b11;
            const block_offset_y: usize = src_y & 0b11;

            const src_pixel: Rgba32 = block_pixels[block_offset_x + block_offset_y * 4];

            dst_pixels[dst_x + dst_y * dst_width] = .{
                .r = src_pixel.r,
                .g = src_pixel.g,
                .b = src_pixel.b,
            };
        }
    }
}

///An image copy that can scale, format cast and layout transition
pub fn blitRasterTargetToLinearSimd(
    src_pixels: [*]const Rgba32,
    dst_pixels: [*]XRgb888,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) void {
    const block_width_scalar: usize = 4;
    const block_height_scalar: usize = block_width_scalar;

    const block_width: WarpRegister(u32) = @splat(@intCast(block_width_scalar));
    const block_height: WarpRegister(u32) = @splat(@intCast(block_height_scalar));

    const src_width_vec: WarpRegister(u32) = @splat(@intCast(src_width));
    const src_height_vec: WarpRegister(u32) = @splat(@intCast(src_height));
    _ = src_height_vec; // autofix

    const zero_u32: WarpRegister(u32) = @splat(0);

    const block_count_x: WarpRegister(u32) = @divTrunc(src_width_vec, block_width) + @intFromBool(@rem(src_width_vec, block_width) != zero_u32);

    const scale_x: WarpRegister(f32) = @splat(@as(f32, @floatFromInt(src_width)) / @as(f32, @floatFromInt(dst_width)));
    const scale_y: f32 = @as(f32, @floatFromInt(src_height)) / @as(f32, @floatFromInt(dst_height));

    const warp_count_x: usize = dst_width / 8 + @intFromBool(@rem(dst_width, 8) != 0);

    var dst_y: u32 = 0;

    while (dst_y < dst_height) : (dst_y += 1) {
        var warp_x: u32 = 0;

        const src_y_float: f32 = @as(f32, @floatFromInt(dst_y)) * scale_y;
        const src_y: usize = @intFromFloat(src_y_float);
        const src_block_y = @divTrunc(src_y, block_height[0]);

        var dst_x: WarpRegister(u32) = std.simd.iota(u32, 8);

        while (warp_x < warp_count_x) : (warp_x += 1) {
            defer {
                dst_x += @splat(8);
            }

            var mask: WarpRegister(bool) = @splat(true);

            mask = dst_x < @as(WarpRegister(u32), @splat(@intCast(src_width)));

            const src_x_float = @as(WarpRegister(f32), @floatFromInt(dst_x)) * scale_x;

            const src_x: WarpRegister(u32) = @intFromFloat(src_x_float);

            const src_block_x = @divTrunc(src_x, block_width);

            const block_index = src_block_x + @as(WarpRegister(u32), @splat(@intCast(src_block_y))) * block_count_x;

            const block_pixels_offset = block_index * block_width * block_height;

            const block_mask: WarpRegister(u32) = @splat(0b11);

            const block_offset_x: WarpRegister(u32) = src_x & block_mask;
            const block_offset_y: WarpRegister(u32) = @as(WarpRegister(u32), @splat(@intCast(src_y))) & block_mask;

            const src_offset = block_offset_x + block_offset_y * @as(WarpRegister(u32), @splat(4));

            const src_pixel = maskedGather(u32, mask, @ptrCast(src_pixels), block_pixels_offset + src_offset);

            maskedStore(u32, mask, @ptrCast(@alignCast(dst_pixels + dst_y * dst_width)), src_pixel);
        }
    }
}

const std = @import("std");
