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

pub const WarpColor = struct {
    r: WarpRegister(u8),
    g: WarpRegister(u8),
    b: WarpRegister(u8),
    a: WarpRegister(u8),
};

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

const Uniforms = struct {
    vertex_colours: []const [3]u32,
};

///Rasterizer stage
pub fn rasterize(
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

    const start_x: WarpRegister(u32) = @intFromFloat(@floor(bounds_min_x));
    const start_y: WarpRegister(u32) = @intFromFloat(@floor(bounds_min_y));

    const end_x: WarpRegister(u32) = @intFromFloat(@ceil(bounds_max_x));
    const end_y: WarpRegister(u32) = @intFromFloat(@ceil(bounds_max_y));

    for (0..8) |triangle_index| {
        if (!projected_triangle.mask[triangle_index]) continue;

        //rasterisze triangle
        rasterizeSimd(
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

pub fn rasterizeSimd(
    uniforms: Uniforms,
    render_target: Image,
    projected_triangle: WarpProjectedTriangle,
    triangle_id: usize,
    triangle_index: usize,
    triangle_area_reciprocal: f32,
    start_x: u32,
    start_y: u32,
    end_x: u32,
    end_y: u32,
) void {
    _ = triangle_index; // autofix
    const block_width = 4;
    const block_height = block_width;

    const block_start_x = start_x / block_width;
    const block_start_y = start_y / block_height;

    //Ceiling division
    const block_end_x = end_x / block_width + (@intFromBool(end_x % block_width != 0));
    const block_end_y = end_y / block_height + (@intFromBool(end_y % block_height != 0));

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

    std.debug.assert(render_target.width % block_width == 0);

    const block_count_x = render_target.width / block_width + @intFromBool(render_target.width % block_width != 0);
    const block_count_y = render_target.height / block_height + @intFromBool(render_target.height % block_height != 0);
    _ = block_count_y; // autofix

    for (block_start_y..block_end_y) |block_y| {
        for (block_start_x..block_end_x) |block_x| {
            const block_index = block_x + block_y * block_count_x;
            const block_start_ptr = render_target.pixel_ptr + block_index * block_width * block_height;

            //quad rasterisation
            //quad warp layout:
            //0 1 | 2 3
            //4 5 | 6 7
            for (0..2) |half_y_offset| {
                const y_offset = half_y_offset * 2;
                var execution_mask: WarpRegister(u32) = @splat(1);

                const target_start_ptr = block_start_ptr + y_offset * block_width;

                const y = block_y * block_height + y_offset;
                const x_base_offset = block_x * block_width;

                const swizzled_point_y_offset: WarpRegister(f32) = .{
                    0, 0, 0, 0,
                    1, 1, 1, 1,
                };

                const swizzled_point_x: WarpRegister(f32) = .{
                    0, 1, 2, 3,
                    0, 1, 2, 3,
                };

                var point_x: WarpRegister(f32) = @splat(@floatFromInt(x_base_offset));

                point_x += swizzled_point_x;
                var point_y: WarpRegister(f32) = @splat(@floatFromInt(y));

                point_y += swizzled_point_y_offset;

                point_x += @splat(0.5);
                point_y += @splat(0.5);

                var bary_0 = edgeFunctionSimdVec2(unclipped_vert_1, unclipped_vert_2, .{ .x = point_x, .y = point_y });
                var bary_1 = edgeFunctionSimdVec2(unclipped_vert_2, unclipped_vert_0, .{ .x = point_x, .y = point_y });
                var bary_2 = edgeFunctionSimdVec2(unclipped_vert_0, unclipped_vert_1, .{ .x = point_x, .y = point_y });

                const barycentric_sum = bary_0 + bary_1 + bary_2;
                _ = barycentric_sum; // autofix

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

                maskedStore(u32, execution_mask != @as(WarpRegister(u1), @splat(0)), @ptrCast(@alignCast(target_start_ptr)), color);
            }
        }
    }
}

pub fn rasterizeSingle(
    render_target: Image,
    triangle: [3]@Vector(2, f32),
    triangle_colors: [3]@Vector(4, f32),
) void {
    var bounds_min: @Vector(2, f32) = @splat(0);
    var bounds_max: @Vector(2, f32) = .{ @floatFromInt(render_target.width), @floatFromInt(render_target.height) };

    var triangle_min: @Vector(2, f32) = @splat(std.math.inf(f32));
    var triangle_max: @Vector(2, f32) = @splat(-std.math.inf(f32));

    for (triangle) |point| {
        triangle_min = @min(point, triangle_min);
        triangle_max = @max(point, triangle_max);
    }

    bounds_min = @max(bounds_min, triangle_min);
    bounds_max = @min(bounds_max, triangle_max);

    const start_x: usize = @intFromFloat(@floor(bounds_min[0]));
    const start_y: usize = @intFromFloat(@floor(bounds_min[1]));

    const end_x: usize = @intFromFloat(@ceil(bounds_max[0]));
    const end_y: usize = @intFromFloat(@ceil(bounds_max[1]));

    const block_start_x = start_x / 8;
    const block_start_y = start_y / 8;

    const block_end_x = end_x / 8 + (@intFromBool(end_x % 8 != 0));
    const block_end_y = end_y / 8 + (@intFromBool(end_y % 8 != 0));

    const triangle_area = edgeFunction(triangle[0], triangle[1], triangle[2]);
    const triangle_area_reciprocal = 1 / triangle_area;

    for (block_start_y..block_end_y) |block_y| {
        for (block_start_x..block_end_x) |block_x| {

            //rasterize block
            for (0..8) |y_offset| {
                const target_start_ptr = render_target.pixel_ptr + y_offset * 8 + 8 * 8 * block_x + 8 * 8 * block_y * (render_target.width / 8 + @intFromBool(render_target.width % 8 != 0));

                for (0..8) |x_offset| {
                    const x = block_x * 8 + x_offset;
                    const y = block_y * 8 + y_offset;

                    var point: @Vector(2, f32) = .{ @floatFromInt(x), @floatFromInt(y) };

                    point += @splat(0.5);

                    var w0 = edgeFunction(triangle[1], triangle[2], point);
                    var w1 = edgeFunction(triangle[2], triangle[0], point);
                    var w2 = edgeFunction(triangle[0], triangle[1], point);

                    w0 *= triangle_area_reciprocal;
                    w1 *= triangle_area_reciprocal;
                    w2 *= triangle_area_reciprocal;

                    if (w0 < 0 or w1 < 0 or w2 < 0) {
                        target_start_ptr[x_offset] = .{ .r = 255, .b = 255, .g = 255, .a = 255 };
                        continue;
                    }

                    const color_r = w0 * triangle_colors[0][0] + w1 * triangle_colors[1][0] + w2 * triangle_colors[2][0];
                    const color_g = w0 * triangle_colors[0][1] + w1 * triangle_colors[1][1] + w2 * triangle_colors[2][1];
                    const color_b = w0 * triangle_colors[0][2] + w1 * triangle_colors[1][2] + w2 * triangle_colors[2][2];
                    const color_a = w0 * triangle_colors[0][3] + w1 * triangle_colors[1][3] + w2 * triangle_colors[2][3];

                    const color: Rgba32 = .fromNormalized(.{ color_r, color_g, color_b, color_a });

                    target_start_ptr[x_offset] = color;
                }
            }
        }
    }

    if (true) return;

    for (start_y..end_y) |y| {
        for (start_x..end_x) |x| {
            var point: @Vector(2, f32) = .{ @floatFromInt(x), @floatFromInt(y) };

            point += @splat(0.5);

            var w0 = edgeFunction(triangle[1], triangle[2], point);
            var w1 = edgeFunction(triangle[2], triangle[0], point);
            var w2 = edgeFunction(triangle[0], triangle[1], point);

            w0 *= triangle_area_reciprocal;
            w1 *= triangle_area_reciprocal;
            w2 *= triangle_area_reciprocal;

            if (w0 < 0 or w1 < 0 or w2 < 0) {
                render_target.pixel_ptr[x + y * render_target.width] = .{ .r = 255, .b = 255, .g = 255, .a = 255 };
                continue;
            }

            const color_r = w0 * triangle_colors[0][0] + w1 * triangle_colors[1][0] + w2 * triangle_colors[2][0];
            const color_g = w0 * triangle_colors[0][1] + w1 * triangle_colors[1][1] + w2 * triangle_colors[2][1];
            const color_b = w0 * triangle_colors[0][2] + w1 * triangle_colors[1][2] + w2 * triangle_colors[2][2];
            const color_a = w0 * triangle_colors[0][3] + w1 * triangle_colors[1][3] + w2 * triangle_colors[2][3];

            const color: Rgba32 = .fromNormalized(.{ color_r, color_g, color_b, color_a });

            render_target.pixel_ptr[x + y * render_target.width] = color;
        }
    }
}

inline fn edgeFunction(
    a: @Vector(2, f32),
    b: @Vector(2, f32),
    c: @Vector(2, f32),
) f32 {
    return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0]);
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

pub fn unpackUnorm4x(value: WarpRegister(u32)) WarpVec4(f32) {
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

pub fn maskedLoad(comptime T: type, mask: @Vector(8, bool), ptr: [*]const T) @Vector(8, T) {
    @setRuntimeSafety(false);

    var result: @Vector(8, T) = undefined;

    //TODO: use avx2 vmaskmov
    inline for (0..8) |i| {
        if (mask[i]) {
            result[i] = ptr[i];
        }
    }

    return result;
}

pub fn maskedStore(comptime T: type, mask: @Vector(8, bool), ptr: [*]align(@alignOf(@Vector(8, T))) T, value: @Vector(8, T)) void {
    @setRuntimeSafety(false);

    //TODO: use avx2 vmaskmov
    inline for (0..8) |i| {
        if (mask[i]) {
            ptr[i] = value[i];
        }
    }
}

pub fn maskedGather(comptime T: type, mask: @Vector(8, bool), base: [*]const T, indices: @Vector(8, u32)) @Vector(8, T) {
    @setRuntimeSafety(false);

    //TODO: use avx2 vpgather
    var result: @Vector(8, T) = undefined;

    inline for (0..8) |i| {
        if (mask[i]) {
            result[i] = base[indices[i]];
        }
    }

    return result;
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

const std = @import("std");
