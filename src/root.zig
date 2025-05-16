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
    _: u23 = 0,
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
    };
}

pub const WarpProjectedTriangle = struct {
    mask: WarpRegister(bool),
    points: [3]WarpVec4(f32),
    unclipped_points: [3]WarpVec4(f32),
};

pub const Uniforms = struct {
    vertex_colours: []const [3]u32,
    vertex_positions: []const [3]WarpVec3(f32),
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
            .w = @splat(1),
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
};

///Rasterizer stage
pub fn rasterize(
    raster_state: RasterState,
    uniforms: Uniforms,
    render_target: Image,
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

    var bounds_max_x: WarpRegister(f32) = @splat(@floatFromInt(render_target.width));
    var bounds_max_y: WarpRegister(f32) = @splat(@floatFromInt(render_target.height));

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
            render_target,
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
    render_target: Image,
    projected_triangle: WarpProjectedTriangle,
    triangle_id: usize,
    triangle_index: usize,
    triangle_area_reciprocal: f32,
    start_x: i32,
    start_y: i32,
    end_x: i32,
    end_y: i32,
) void {
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

    const block_count_x = render_target.width / block_width + @intFromBool(render_target.width % block_width != 0);
    const block_count_y = render_target.height / block_height + @intFromBool(render_target.height % block_height != 0);
    _ = block_count_y; // autofix

    var block_y: isize = block_start_y;

    while (block_y < block_end_y) : (block_y += 1) {
        var block_x: isize = block_start_x;

        while (block_x < block_end_x) : (block_x += 1) {
            const block_index: usize = @intCast(block_x + block_y * block_count_x);
            const block_start_ptr = render_target.pixel_ptr + block_index * block_width * block_height;

            //quad rasterisation
            //quad warp layout:
            //0 1 | 2 3
            //4 5 | 6 7
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

                const color_r = bary_0 * vertex_color_0.x + bary_1 * vertex_color_1.x + bary_2 * vertex_color_2.x;
                const color_g = bary_0 * vertex_color_0.y + bary_1 * vertex_color_1.y + bary_2 * vertex_color_2.y;
                const color_b = bary_0 * vertex_color_0.z + bary_1 * vertex_color_1.z + bary_2 * vertex_color_2.z;
                const color_a = bary_0 * vertex_color_0.w + bary_1 * vertex_color_1.w + bary_2 * vertex_color_2.w;

                var color_r_deriv: WarpVec2(f32) = quadComputeDerivativeCoarse(color_r);
                var color_g_deriv: WarpVec2(f32) = quadComputeDerivativeCoarse(color_g);

                {
                    color_r_deriv.x *= @splat(200);
                    color_r_deriv.y *= @splat(200);

                    color_g_deriv.x *= @splat(200);
                    color_g_deriv.y *= @splat(200);
                }

                //computing dr/dx

                const color = packUnorm4x(.{ .x = color_r, .y = color_g, .z = color_b, .w = color_a });
                // const color = packUnorm4x(.{ .x = color_r_deriv.x, .y = color_g_deriv.y, .z = @splat(0), .w = @splat(1) });

                maskedStore(
                    u32,
                    execution_mask != @as(WarpRegister(u1), @splat(0)),
                    @ptrCast(@alignCast(target_start_ptr)),
                    color,
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
    _ = u_derivative; // autofix
    _ = v_derivative; // autofix
    _ = execution_mask; // autofix
    _ = base; // autofix
    _ = descriptor; // autofix
    _ = uv; // autofix
}

pub fn imageLoad(
    execution_mask: WarpRegister(bool),
    ///Base address from which the descriptor loads
    base: [*]const u32,
    descriptor: ImageDescriptor,
    ///Image coordinates
    xy: WarpVec2(u32),
) WarpVec4(f32) {
    _ = execution_mask; // autofix
    _ = xy; // autofix

    const image_base = base + descriptor.rel_ptr;
    _ = image_base; // autofix
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

const std = @import("std");
