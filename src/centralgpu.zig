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

///A Combined Image/Sampler that points to multiple image mip lod levels
pub const ImageDescriptor = packed struct(u64) {
    rel_ptr: i32,
    width_log2: u4,
    height_log2: u4,
    min_mip_level: u4 = 0,
    max_mip_level: u4 = 0,
    sampler_filter: enum(u1) {
        nearest,
        bilinear,
    },
    sampler_address_mode: enum(u2) {
        repeat,
        clamp_to_edge,
        clamp_to_border,
    },
    ///Specifies how the border colour should be shifted
    ///border_colour = 0xff_ff_ff_ff << (border_colour_shift_amount << 3)
    border_colour_shift_amount: u3,
    _: u10 = 0,
};

const warp_register_len = std.simd.suggestVectorLength(u32).?;

pub fn WarpRegister(comptime T: type) type {
    return @Vector(warp_register_len, T);
}

pub fn WarpVec2(comptime T: type) type {
    return struct {
        x: WarpRegister(T),
        y: WarpRegister(T),

        pub inline fn neg(self: @This()) @This() {
            return .{
                .x = -self.x,
                .y = -self.y,
            };
        }

        pub inline fn add(left: @This(), right: @This()) @This() {
            return .{
                .x = left.x + right.x,
                .y = left.y + right.y,
            };
        }

        pub inline fn sub(left: @This(), right: @This()) @This() {
            return .{
                .x = left.x - right.x,
                .y = left.y - right.y,
            };
        }
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

        pub inline fn init(
            values: [4]f32,
        ) @This() {
            return .{
                .x = @splat(values[0]),
                .y = @splat(values[1]),
                .z = @splat(values[2]),
                .w = @splat(values[3]),
            };
        }

        pub inline fn splat(value: WarpRegister(T)) @This() {
            return .{
                .x = value,
                .y = value,
                .z = value,
                .w = value,
            };
        }

        pub inline fn neg(self: @This()) @This() {
            return .{
                .x = -self.x,
                .y = -self.y,
                .z = -self.z,
                .w = -self.w,
            };
        }

        pub inline fn add(left: @This(), right: @This()) @This() {
            return .{
                .x = left.x + right.x,
                .y = left.y + right.y,
                .z = left.z + right.z,
                .w = left.w + right.w,
            };
        }

        pub inline fn sub(left: @This(), right: @This()) @This() {
            return .{
                .x = left.x - right.x,
                .y = left.y - right.y,
                .z = left.z - right.z,
                .w = left.w - right.w,
            };
        }

        pub inline fn hadamardProduct(left: @This(), right: @This()) @This() {
            return .{
                .x = left.x * right.x,
                .y = left.y * right.y,
                .z = left.z * right.z,
                .w = left.w * right.w,
            };
        }

        ///Fused hadamard-add: (a * b + c)
        pub inline fn hadamardAdd(a: @This(), b: @This(), c: @This()) @This() {
            return .{
                .x = @mulAdd(WarpRegister(f32), a.x, b.x, c.x),
                .y = @mulAdd(WarpRegister(f32), a.y, b.y, c.y),
                .z = @mulAdd(WarpRegister(f32), a.z, b.z, c.z),
                .w = @mulAdd(WarpRegister(f32), a.w, b.w, c.w),
            };
        }

        pub inline fn scalarProduct(left: @This(), right: @This()) WarpRegister(T) {
            const xx = left.x * right.x;
            const yy = left.y * right.y;
            const zz = left.z * right.z;
            const ww = left.w * right.w;

            return xx + yy + zz + ww;
        }

        pub inline fn scale(left: @This(), right: WarpRegister(T)) @This() {
            return .{
                .x = left.x * right,
                .y = left.y * right,
                .z = left.z * right,
                .w = left.w * right,
            };
        }
    };
}

pub fn Mat4x4(comptime T: type) type {
    return struct {
        rows: [4]WarpVec4(T) = .{
            WarpVec4(T).init(.{ 1, 0, 0, 0 }),
            WarpVec4(T).init(.{ 0, 1, 0, 0 }),
            WarpVec4(T).init(.{ 0, 0, 1, 0 }),
            WarpVec4(T).init(.{ 0, 0, 0, 1 }),
        },

        pub inline fn init(values: [4][4]T) @This() {
            return .{
                .rows = .{
                    WarpVec4(T).init(values[0]),
                    WarpVec4(T).init(values[1]),
                    WarpVec4(T).init(values[2]),
                    WarpVec4(T).init(values[3]),
                },
            };
        }

        pub inline fn mulByVec4(left: @This(), right: WarpVec4(T)) WarpVec4(T) {
            const x_res = left.rows[0].scale(right.x);
            const y_res = left.rows[1].scale(right.y);
            const z_res = left.rows[2].scale(right.z);
            const w_res = left.rows[3].scale(right.w);

            var result: WarpVec4(T) = x_res;

            result = result.add(y_res);
            result = result.add(z_res);
            result = result.add(w_res);

            return result;
        }
    };
}

pub const WarpProjectedTriangle = struct {
    mask: WarpRegister(bool),
    ///After projection w actual stores reciprocal w
    points: [3]WarpVec4(f32),
};

pub const Uniforms = struct {
    vertex_positions: []const [3]WarpVec4(f32),
    vertex_colours: []const [3]u32,
    vertex_texture_coords: []const [3][4][2]f32,

    image_base: [4][*]const u8,
    image_descriptor: [4]ImageDescriptor,
    texture_environments: [4]TextureEnvironment,
};

pub const GeometryProcessState = struct {
    viewport_transform: struct {
        scale_x: f32,
        scale_y: f32,
        scale_z: f32,
        translation_x: f32,
        translation_y: f32,
        translation_z: f32,
    },
    backface_cull: bool,
};

pub fn processGeometry(
    state: GeometryProcessState,
    uniforms: Uniforms,
    triangle_id_start: usize,
    input_mask: WarpRegister(bool),
) WarpProjectedTriangle {
    @setRuntimeSafety(false);

    const in_triangle = uniforms.vertex_positions[triangle_id_start / 8];

    var out_triangle: [3]WarpVec4(f32) = undefined;
    var out_mask = input_mask;

    var cull_mask: WarpRegister(bool) = @splat(true);

    for (0..3) |vertex_index| {
        out_triangle[vertex_index] = .{
            .x = in_triangle[vertex_index].x,
            .y = in_triangle[vertex_index].y,
            .z = in_triangle[vertex_index].z,
            .w = in_triangle[vertex_index].w,
            // .w = @splat(1),
        };

        //mostly for debugging
        const cull_scale: WarpRegister(f32) = @splat(1);

        cull_mask = vectorBoolAnd(
            cull_mask,
            vectorBoolNot(vectorBoolAnd(
                -out_triangle[vertex_index].w * cull_scale <= out_triangle[vertex_index].x,
                out_triangle[vertex_index].x <= out_triangle[vertex_index].w * cull_scale,
            )),
        );

        cull_mask = vectorBoolAnd(
            cull_mask,
            vectorBoolNot(vectorBoolAnd(
                -out_triangle[vertex_index].w * cull_scale <= out_triangle[vertex_index].y,
                out_triangle[vertex_index].y <= out_triangle[vertex_index].w * cull_scale,
            )),
        );

        out_mask = vectorBoolAnd(
            out_mask,
            vectorBoolAnd(
                -out_triangle[vertex_index].w <= out_triangle[vertex_index].z,
                out_triangle[vertex_index].z <= out_triangle[vertex_index].w,
            ),
        );
    }
    out_mask = vectorBoolAnd(out_mask, vectorBoolNot(cull_mask));

    if (@reduce(.Or, out_mask) == false) {
        return .{ .mask = @splat(false), .points = undefined };
    }

    for (0..3) |vertex_index| {
        if (homogenous_raster == false) {
            const reciprocal_w = @as(WarpRegister(f32), @splat(1)) / out_triangle[vertex_index].w;

            out_triangle[vertex_index].x *= reciprocal_w;
            out_triangle[vertex_index].y *= reciprocal_w;
            out_triangle[vertex_index].z *= reciprocal_w;

            out_triangle[vertex_index].w = reciprocal_w;
        }
    }

    const eb_x = out_triangle[1].x - out_triangle[0].x;
    const eb_y = out_triangle[1].y - out_triangle[0].y;
    const ec_x = out_triangle[2].x - out_triangle[0].x;
    const ec_y = out_triangle[2].y - out_triangle[0].y;

    const backface_culled = eb_x * ec_y >= eb_y * ec_x;
    // const backface_culled: WarpRegister(bool) = @sp;

    for (0..3) |vertex_index| {
        if (homogenous_raster == false) {
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

        out_triangle[vertex_index].z = @mulAdd(
            WarpRegister(f32),
            out_triangle[vertex_index].z,
            @splat(state.viewport_transform.scale_z),
            @splat(state.viewport_transform.translation_z),
        );

        // out_triangle[vertex_index].z = @max(out_triangle[vertex_index].z, @as(WarpRegister(f32), @splat(-1)));
        // out_triangle[vertex_index].z = @min(out_triangle[vertex_index].z, @as(WarpRegister(f32), @splat(1)));
    }

    if (homogenous_raster == false) {
        if (state.backface_cull) {
            out_mask = vectorBoolAnd(out_mask, vectorBoolNot(backface_culled));
        }
    }

    return .{
        .mask = out_mask,
        .points = out_triangle,
    };
}

pub const BlendState = packed struct(u32) {
    src_factor: BlendConstant,
    dst_factor: BlendConstant,
    _: u28 = 0,

    pub const BlendConstant = enum(u2) {
        one,
        zero,
        one_minus_src_alpha,
        src_alpha,
    };
};

pub const RasterState = struct {
    scissor_min_x: i32,
    scissor_min_y: i32,
    scissor_max_x: i32,
    scissor_max_y: i32,

    flags: Flags,
    blend_state: BlendState,

    render_target: Image,
    depth_image: [*]f32,

    pub const Flags = packed struct(u32) {
        enable_alpha_test: bool,
        enable_depth_test: bool,
        enable_depth_write: bool,
        enable_blend: bool,
        _: u28 = 0,
    };
};

pub const TextureEnvironment = struct {
    rgb_scale: f32 = 1,
    function: Function = .modulate,

    pub const Function = enum {
        modulate,
        replace,
        add,
    };
};

const homogenous_raster = false;

///Rasterizer stage
pub fn rasterize(
    geometry_state: GeometryProcessState,
    raster_state: RasterState,
    uniforms: Uniforms,
    triangle_id_start: usize,
    in_projected_triangle: WarpProjectedTriangle,
) void {
    @setRuntimeSafety(false);
    var projected_triangle = in_projected_triangle;

    //clipless: clamp w away from zero
    if (homogenous_raster or true) {
        const max_inv_w: WarpRegister(f32) = @splat(@sqrt(std.math.floatMax(f32)));

        const w_lower_bound: WarpRegister(f32) = -max_inv_w;
        _ = w_lower_bound; // autofix
        const w_upper_bound: WarpRegister(f32) = max_inv_w;
        _ = w_upper_bound; // autofix

        inline for (0..3) |vertex| {
            _ = vertex; // autofix
            // projected_triangle.points[vertex].w = @max(@min(projected_triangle.points[vertex].w, w_upper_bound), w_lower_bound);
        }
    }

    var bounds_min: WarpVec2(f32) = undefined;
    var bounds_max: WarpVec2(f32) = undefined;

    {
        const p0: Homogenous2D = .{
            .x = projected_triangle.points[0].x,
            .y = projected_triangle.points[0].y,
            .w = projected_triangle.points[0].w,
        };
        const p1: Homogenous2D = .{
            .x = projected_triangle.points[1].x,
            .y = projected_triangle.points[1].y,
            .w = projected_triangle.points[1].w,
        };
        const p2: Homogenous2D = .{
            .x = projected_triangle.points[2].x,
            .y = projected_triangle.points[2].y,
            .w = projected_triangle.points[2].w,
        };

        bounds_min = homogenousProject(homogenousMin(p0, homogenousMin(p1, p2)));
        bounds_max = homogenousProject(homogenousMax(p0, homogenousMax(p1, p2)));

        const viewport_scale_x: WarpRegister(f32) = @splat(geometry_state.viewport_transform.scale_x);
        // _ = viewport_scale_x; // autofix
        const viewport_scale_y: WarpRegister(f32) = @splat(geometry_state.viewport_transform.scale_y);
        // _ = viewport_scale_y; // autofix

        const viewport_translation_x: WarpRegister(f32) = @splat(geometry_state.viewport_transform.translation_x);
        // _ = viewport_translation_x; // autofix
        const viewport_translation_y: WarpRegister(f32) = @splat(geometry_state.viewport_transform.translation_y);
        // _ = viewport_translation_y; // autofix

        bounds_min.x = bounds_min.x * viewport_scale_x + viewport_translation_x;
        bounds_min.y = bounds_min.y * viewport_scale_y + viewport_translation_y;

        bounds_max.x = bounds_max.x * viewport_scale_x + viewport_translation_x;
        bounds_max.y = bounds_max.y * viewport_scale_y + viewport_translation_y;
    }

    const triangle_area = edgeFunctionSimd(
        projected_triangle.points[0],
        projected_triangle.points[1],
        projected_triangle.points[2],
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

    if (homogenous_raster) {
        for (projected_triangle.points) |point| {
            _ = point; // autofix

            bounds_min_x = bounds_min.x;
            bounds_min_y = bounds_min.y;
            bounds_max_x = bounds_max.x;
            bounds_max_y = bounds_max.y;
        }
    } else {
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
    }

    var start_x: WarpRegister(i32) = @intFromFloat(@floor(bounds_min_x));
    var start_y: WarpRegister(i32) = @intFromFloat(@floor(bounds_min_y));

    start_x = @max(start_x, @as(WarpRegister(i32), @splat(raster_state.scissor_min_x)));
    start_y = @max(start_y, @as(WarpRegister(i32), @splat(raster_state.scissor_min_y)));

    var end_x: WarpRegister(i32) = @intFromFloat(@ceil(bounds_max_x));
    var end_y: WarpRegister(i32) = @intFromFloat(@ceil(bounds_max_y));

    end_x = @min(end_x, @as(WarpRegister(i32), @splat(raster_state.scissor_max_x)));
    end_y = @min(end_y, @as(WarpRegister(i32), @splat(raster_state.scissor_max_y)));

    const triangle_matrix: Mat3x3 = .{
        .{ projected_triangle.points[0].x, projected_triangle.points[1].x, projected_triangle.points[2].x },
        .{ projected_triangle.points[0].y, projected_triangle.points[1].y, projected_triangle.points[2].y },
        .{ projected_triangle.points[0].w, projected_triangle.points[1].w, projected_triangle.points[2].w },
    };

    const matrix_det = mat3x3Det(triangle_matrix);
    const matrix_det_recip = reciprocal(matrix_det);
    const matrix_inv = mat3x3InvDet(triangle_matrix, matrix_det_recip);
    const triangle_z_coords = mat3x3MulVec(triangle_matrix, .{
        .x = projected_triangle.points[0].z,
        .y = projected_triangle.points[1].z,
        .z = projected_triangle.points[2].z,
    });
    _ = triangle_z_coords; // autofix

    if (homogenous_raster) {
        //Backface cull
        projected_triangle.mask = vectorBoolAnd(projected_triangle.mask, matrix_det >= @as(WarpRegister(f32), @splat(0)));
    }

    const mask_integer: u8 = @bitCast(projected_triangle.mask);

    const triangle_max_count = 8 - @clz(mask_integer);

    for (0..triangle_max_count) |triangle_index| {
        if (!projected_triangle.mask[triangle_index]) continue;

        var per_triangle_matrix_inv: Mat3x3 = undefined;

        inline for (0..3) |row| {
            inline for (0..3) |column| {
                per_triangle_matrix_inv[row][column] = @splat(matrix_inv[row][column][triangle_index]);
            }
        }

        var per_triangle_matrix: Mat3x3 = undefined;

        inline for (0..3) |row| {
            inline for (0..3) |column| {
                per_triangle_matrix[row][column] = @splat(triangle_matrix[row][column][triangle_index]);
            }
        }

        //rasterisze triangle
        rasterizeTriangle(
            geometry_state,
            raster_state,
            uniforms,
            projected_triangle,
            per_triangle_matrix,
            per_triangle_matrix_inv,
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
    geometry_state: GeometryProcessState,
    raster_state: RasterState,
    uniforms: Uniforms,
    projected_triangle: WarpProjectedTriangle,
    triangle_matrix: Mat3x3,
    triangle_matrix_inv: Mat3x3,
    triangle_id: usize,
    triangle_index: usize,
    triangle_area_reciprocal: f32,
    start_x: i32,
    start_y: i32,
    end_x: i32,
    end_y: i32,
) void {
    _ = triangle_matrix; // autofix
    _ = geometry_state; // autofix
    @setRuntimeSafety(false);
    const block_width = 4;
    const block_height = block_width;

    const block_start_x = @divTrunc(start_x, block_width);
    const block_start_y = @divTrunc(start_y, block_height);

    //Ceiling division
    const block_end_x = @divTrunc(end_x, block_width) + (@intFromBool(@rem(end_x, block_width) != 0));
    const block_end_y = @divTrunc(end_y, block_height) + (@intFromBool(@rem(end_y, block_height) != 0));

    const vert_z_over_w_0: WarpRegister(f32) = @splat(projected_triangle.points[0].z[triangle_index]);
    const vert_z_over_w_1: WarpRegister(f32) = @splat(projected_triangle.points[1].z[triangle_index]);
    const vert_z_over_w_2: WarpRegister(f32) = @splat(projected_triangle.points[2].z[triangle_index]);

    //We store reciprocal w in the w of the projected triangle after processing
    const vertex_recip_w_0: WarpRegister(f32) = @splat(projected_triangle.points[0].w[triangle_index]);
    const vertex_recip_w_1: WarpRegister(f32) = @splat(projected_triangle.points[1].w[triangle_index]);
    const vertex_recip_w_2: WarpRegister(f32) = @splat(projected_triangle.points[2].w[triangle_index]);

    const recip_z_0 = reciprocal(vert_z_over_w_0) * vertex_recip_w_0;
    const recip_z_1 = reciprocal(vert_z_over_w_1) * vertex_recip_w_1;
    const recip_z_2 = reciprocal(vert_z_over_w_2) * vertex_recip_w_2;

    const vert_z_0 = reciprocal(recip_z_0);
    _ = vert_z_0; // autofix
    const vert_z_1 = reciprocal(recip_z_1);
    _ = vert_z_1; // autofix
    const vert_z_2 = reciprocal(recip_z_2);
    _ = vert_z_2; // autofix

    const unclipped_vert_0: WarpVec2(f32) = .{
        .x = @splat(projected_triangle.points[0].x[triangle_index]),
        .y = @splat(projected_triangle.points[0].y[triangle_index]),
    };

    const unclipped_vert_1: WarpVec2(f32) = .{
        .x = @splat(projected_triangle.points[1].x[triangle_index]),
        .y = @splat(projected_triangle.points[1].y[triangle_index]),
    };

    const unclipped_vert_2: WarpVec2(f32) = .{
        .x = @splat(projected_triangle.points[2].x[triangle_index]),
        .y = @splat(projected_triangle.points[2].y[triangle_index]),
    };

    //per primitive processing
    const vertex_color_0_packed: WarpRegister(u32) = @splat(uniforms.vertex_colours[triangle_id][0]);
    const vertex_color_1_packed: WarpRegister(u32) = @splat(uniforms.vertex_colours[triangle_id][1]);
    const vertex_color_2_packed: WarpRegister(u32) = @splat(uniforms.vertex_colours[triangle_id][2]);

    const vertex_color_0 = unpackUnorm4x(vertex_color_0_packed);
    const vertex_color_1 = unpackUnorm4x(vertex_color_1_packed);
    const vertex_color_2 = unpackUnorm4x(vertex_color_2_packed);

    const block_count_x = raster_state.render_target.width / block_width + @intFromBool(raster_state.render_target.width % block_width != 0);

    var block_y: isize = block_start_y;

    while (block_y < block_end_y) : (block_y += 1) {
        var block_x: isize = block_start_x;

        while (block_x < block_end_x) : (block_x += 1) {
            const block_index: usize = @intCast(block_x + block_y * block_count_x);
            // const block_index = mortonEncodeScalar(@intCast(block_x), @intCast(block_y));
            const block_offset = block_index * block_width * block_height;
            const block_start_ptr = raster_state.render_target.pixel_ptr + block_offset;

            for (0..2) |half_y_offset| {
                const y_offset: isize = @intCast(half_y_offset * 2);
                var execution_mask: WarpRegister(u32) = @splat(1);

                const target_start_ptr = block_start_ptr + @as(usize, @intCast(y_offset * block_width));
                const target_depth = raster_state.depth_image + block_offset + @as(usize, @intCast(y_offset * block_width));

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

                var bary_0: WarpRegister(f32) = undefined;
                var bary_1: WarpRegister(f32) = undefined;
                var bary_2: WarpRegister(f32) = undefined;

                if (homogenous_raster) {
                    const barycentrics = mat3x3MulVec(triangle_matrix_inv, .{
                        .x = point_x * reciprocal(@as(WarpRegister(f32), @splat(800))),
                        .y = point_y * reciprocal(@as(WarpRegister(f32), @splat(600))),
                        .z = @splat(1),
                    });

                    bary_0 = barycentrics.x;
                    bary_1 = barycentrics.y;
                    bary_2 = barycentrics.z;
                } else {
                    bary_0 = edgeFunctionSimdVec2(unclipped_vert_1, unclipped_vert_2, .{ .x = point_x, .y = point_y });
                    bary_1 = edgeFunctionSimdVec2(unclipped_vert_2, unclipped_vert_0, .{ .x = point_x, .y = point_y });
                    bary_2 = edgeFunctionSimdVec2(unclipped_vert_0, unclipped_vert_1, .{ .x = point_x, .y = point_y });

                    bary_0 *= @splat(triangle_area_reciprocal);
                    bary_1 *= @splat(triangle_area_reciprocal);
                    bary_2 *= @splat(triangle_area_reciprocal);
                }

                execution_mask &= @intFromBool(bary_0 >= @as(WarpRegister(f32), @splat(0)));
                execution_mask &= @intFromBool(bary_1 >= @as(WarpRegister(f32), @splat(0)));
                execution_mask &= @intFromBool(bary_2 >= @as(WarpRegister(f32), @splat(0)));

                if (@reduce(.Or, execution_mask) == 0) {
                    continue;
                }

                const fragment_recip_w = bary_0 * vertex_recip_w_0 + bary_1 * vertex_recip_w_1 + bary_2 * vertex_recip_w_2;

                const recip_z = bary_0 * recip_z_0 + bary_1 * recip_z_1 + bary_2 * recip_z_2;
                const pixel_z = reciprocal(recip_z);
                const fragment_w = reciprocal(fragment_recip_w);
                _ = fragment_w; // autofix

                bary_0 *= recip_z_0;
                bary_1 *= recip_z_1;
                bary_2 *= recip_z_2;
                bary_0 *= recip_z;
                bary_1 *= recip_z;
                bary_2 *= recip_z;

                const reciprocal_bary_sum = reciprocal(bary_0 + bary_1 + bary_2);

                bary_0 *= reciprocal_bary_sum;
                bary_1 *= reciprocal_bary_sum;
                bary_2 *= reciprocal_bary_sum;

                if (raster_state.flags.enable_depth_test) {
                    const previous_depth = maskedLoad(
                        f32,
                        execution_mask != @as(WarpRegister(u1), @splat(0)),
                        target_depth,
                    );

                    const depth_difference = pixel_z - previous_depth;

                    execution_mask &= @intFromBool(depth_difference <= @as(WarpRegister(f32), @splat(0)));
                }

                if (@reduce(.Or, execution_mask) == 0) {
                    continue;
                }

                const color_r = bary_0 * vertex_color_0.x + bary_1 * vertex_color_1.x + bary_2 * vertex_color_2.x;
                const color_g = bary_0 * vertex_color_0.y + bary_1 * vertex_color_1.y + bary_2 * vertex_color_2.y;
                const color_b = bary_0 * vertex_color_0.z + bary_1 * vertex_color_1.z + bary_2 * vertex_color_2.z;
                const color_a = bary_0 * vertex_color_0.w + bary_1 * vertex_color_1.w + bary_2 * vertex_color_2.w;

                const color: WarpVec4(f32) = .{ .x = color_r, .y = color_g, .z = color_b, .w = color_a };
                // const color = packUnorm4x(.{ .x = tex_u, .y = tex_v, .z = @splat(0), .w = @splat(1) });
                // const color = packUnorm4x(.{ .x = texture_sample.x, .y = texture_sample.y, .z = texture_sample.z, .w = @splat(1) });

                // var color_result: WarpVec4(f32) = texture_sample;
                var color_result: WarpVec4(f32) = color;

                color_result = color_result;

                var resultant_sample: WarpVec4(f32) = .init(.{ 1, 1, 1, 1 });

                for (uniforms.image_descriptor, uniforms.image_base, 0..) |image_descriptor, image_base, texture_unit| {
                    if (image_descriptor.rel_ptr != -1) {
                        const vertex_texcoord_u_0: WarpRegister(f32) = @splat(uniforms.vertex_texture_coords[triangle_id][0][texture_unit][0]);
                        const vertex_texcoord_u_1: WarpRegister(f32) = @splat(uniforms.vertex_texture_coords[triangle_id][1][texture_unit][0]);
                        const vertex_texcoord_u_2: WarpRegister(f32) = @splat(uniforms.vertex_texture_coords[triangle_id][2][texture_unit][0]);

                        const vertex_texcoord_v_0: WarpRegister(f32) = @splat(uniforms.vertex_texture_coords[triangle_id][0][texture_unit][1]);
                        const vertex_texcoord_v_1: WarpRegister(f32) = @splat(uniforms.vertex_texture_coords[triangle_id][1][texture_unit][1]);
                        const vertex_texcoord_v_2: WarpRegister(f32) = @splat(uniforms.vertex_texture_coords[triangle_id][2][texture_unit][1]);

                        const tex_u = bary_0 * vertex_texcoord_u_0 + bary_1 * vertex_texcoord_u_1 + bary_2 * vertex_texcoord_u_2;
                        const tex_v = bary_0 * vertex_texcoord_v_0 + bary_1 * vertex_texcoord_v_1 + bary_2 * vertex_texcoord_v_2;

                        const texture_environment = uniforms.texture_environments[texture_unit];

                        var texture_sample = quadImageSample(
                            execution_mask != @as(WarpRegister(u1), @splat(0)),
                            @ptrCast(@alignCast(image_base)),
                            image_descriptor,
                            .{ .x = tex_u, .y = tex_v },
                        );

                        texture_sample.x *= @splat(uniforms.texture_environments[texture_unit].rgb_scale);
                        texture_sample.y *= @splat(uniforms.texture_environments[texture_unit].rgb_scale);
                        texture_sample.z *= @splat(uniforms.texture_environments[texture_unit].rgb_scale);

                        switch (texture_environment.function) {
                            .modulate => {
                                resultant_sample.x *= texture_sample.x;
                                resultant_sample.y *= texture_sample.y;
                                resultant_sample.z *= texture_sample.z;
                                resultant_sample.w *= texture_sample.w;
                            },
                            .replace => {
                                resultant_sample = texture_sample;
                            },
                            .add => {
                                resultant_sample.x = resultant_sample.x + texture_sample.x;
                                resultant_sample.y = resultant_sample.y + texture_sample.y;
                                resultant_sample.z = resultant_sample.z + texture_sample.z;
                                resultant_sample.w *= texture_sample.w;
                            },
                        }
                    }
                }

                color_result = .hadamardProduct(resultant_sample, color_result);

                // color_result.x = @splat(@floatFromInt(triangle_id % 10));
                // color_result.x /= @splat(10);
                // color_result.y = @splat(@floatFromInt(triangle_id % 100));
                // color_result.y /= @splat(100);
                // color_result.z = @splat(1);
                // color_result.w = @splat(1);

                // color_result.x = tex_u;
                // color_result.y = tex_v;

                // color_result.x = pixel_z;
                // color_result.y = pixel_z;
                // color_result.z = pixel_z;

                // color_result.x *= @splat(0.001);
                // color_result.y *= @splat(0.001);
                // color_result.z *= @splat(0.001);

                // color_result.x = @splat(@floatFromInt(triangle_id % 200));
                // color_result.x /= @splat(200);
                // color_result.y = @splat(0);
                // color_result.z = @splat(0);
                // color_result.w = @splat(1);

                if (raster_state.flags.enable_alpha_test) {
                    //alpha test
                    execution_mask &= (@intFromBool(color_result.w > @as(WarpRegister(f32), @splat(0.01))));
                }

                if (raster_state.flags.enable_blend) {
                    const previous_colour_packed = maskedLoad(
                        u32,
                        execution_mask != @as(WarpRegister(u1), @splat(0)),
                        @ptrCast(@alignCast(target_start_ptr)),
                    );

                    const previous_colour = unpackUnorm4x(previous_colour_packed);

                    var output_colour: WarpVec4(f32) = color_result;

                    {
                        const destination_colour = previous_colour;
                        const source_colour = color_result;

                        const one: WarpVec4(f32) = .splat(@splat(1));
                        const zero: WarpVec4(f32) = .splat(@splat(0));
                        const one_minus_src_alpha: WarpVec4(f32) = .splat(@as(WarpRegister(f32), @splat(1)) - source_colour.w);
                        const source_alpha: WarpVec4(f32) = .splat(source_colour.w);

                        var source_factor: WarpVec4(f32) = one;
                        var dest_factor: WarpVec4(f32) = one_minus_src_alpha;

                        source_factor = switch (raster_state.blend_state.src_factor) {
                            .one => one,
                            .zero => zero,
                            .one_minus_src_alpha => one_minus_src_alpha,
                            .src_alpha => source_alpha,
                        };

                        dest_factor = switch (raster_state.blend_state.dst_factor) {
                            .one => one,
                            .zero => zero,
                            .one_minus_src_alpha => one_minus_src_alpha,
                            .src_alpha => source_alpha,
                        };

                        output_colour = source_colour.hadamardProduct(source_factor).add(destination_colour.hadamardProduct(dest_factor));
                    }

                    color_result = output_colour;
                }

                const packed_color = packUnorm4x(color_result);

                maskedStore(
                    u32,
                    execution_mask != @as(WarpRegister(u1), @splat(0)),
                    @ptrCast(@alignCast(target_start_ptr)),
                    packed_color,
                );

                if (raster_state.flags.enable_depth_write) {
                    maskedStore(
                        f32,
                        execution_mask != @as(WarpRegister(u1), @splat(0)),
                        @ptrCast(@alignCast(target_depth)),
                        pixel_z,
                    );
                }
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
    @setRuntimeSafety(false);

    const max_value: WarpRegister(f32) = @splat(std.math.maxInt(u8));

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
    @setRuntimeSafety(false);

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

pub inline fn quadImageSample(
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

pub inline fn imageSampleDerivative(
    execution_mask: WarpRegister(bool),
    base: [*]const u32,
    descriptor: ImageDescriptor,
    uv: WarpVec2(f32),
    u_derivative: WarpVec2(f32),
    v_derivative: WarpVec2(f32),
) WarpVec4(f32) {
    @setRuntimeSafety(false);

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
        // return .{
        // .x = mip_level / @as(WarpRegister(f32), @splat(10.0)),
        // .y = @splat(0),
        // .z = @splat(0),
        // .w = @splat(1),
        // };
    }

    const level: WarpRegister(u32) = @intFromFloat(mip_level);

    switch (descriptor.sampler_filter) {
        .nearest => return imageSampleNearest(execution_mask, base, descriptor, level, uv),
        .bilinear => return imageSampleBilinear(execution_mask, base, descriptor, level, uv),
    }
}

pub inline fn imageSampleBilinear(
    execution_mask: WarpRegister(bool),
    ///Base address from which the descriptor loads
    base: [*]const u32,
    descriptor: ImageDescriptor,
    level: WarpRegister(u32),
    uv: WarpVec2(f32),
) WarpVec4(f32) {
    @setRuntimeSafety(false);

    const u_scale: WarpRegister(f32) = @splat(@floatFromInt((@as(u32, 1) << @as(u5, descriptor.width_log2)) - 1));
    const v_scale: WarpRegister(f32) = @splat(@floatFromInt((@as(u32, 1) << @as(u5, descriptor.height_log2)) - 1));

    const inv_u_scale = @as(WarpRegister(f32), @splat(1)) / u_scale;
    const inv_v_scale = @as(WarpRegister(f32), @splat(1)) / v_scale;

    const half: WarpRegister(f32) = @splat(0.5);
    const one_and_half: WarpRegister(f32) = @splat(1.5);
    _ = one_and_half; // autofix

    const sample_loc_0_float = imageSamplerAddressFloat(
        descriptor,
        // uv.add(.{ .x = half * inv_u_scale, .y = half * inv_v_scale }),
        uv,
    );

    const sample_loc_0: WarpVec2(i32) = .{
        .x = @intFromFloat(@floor(sample_loc_0_float.x)),
        .y = @intFromFloat(@floor(sample_loc_0_float.y)),
    };
    const sample_loc_1 = imageSamplerAddress(
        descriptor,
        // uv.add(.{ .x = one_and_half * inv_u_scale, .y = half * inv_v_scale }),
        uv.add(.{ .x = half * inv_u_scale, .y = @splat(0) }),
    );
    const sample_loc_2 = imageSamplerAddress(
        descriptor,
        // uv.add(.{ .x = half * inv_u_scale, .y = one_and_half * inv_v_scale }),
        uv.add(.{ .x = @splat(0), .y = half * inv_v_scale }),
    );
    const sample_loc_3 = imageSamplerAddress(
        descriptor,
        // uv.add(.{ .x = one_and_half * inv_u_scale, .y = one_and_half * inv_v_scale }),
        uv.add(.{ .x = half * inv_u_scale, .y = half * inv_v_scale }),
    );

    const texel_point_x: WarpRegister(i32) = @intFromFloat(@floor(sample_loc_0_float.x));
    _ = texel_point_x; // autofix
    const texel_point_y: WarpRegister(i32) = @intFromFloat(@floor(sample_loc_0_float.y));
    _ = texel_point_y; // autofix
    const texel_point_offset_x = sample_loc_0_float.x - @floor(sample_loc_0_float.x);
    const texel_point_offset_y = sample_loc_0_float.y - @floor(sample_loc_0_float.y);

    const texel_0 = imageLoad(
        execution_mask,
        base,
        descriptor,
        level,
        sample_loc_0,
    );
    const texel_1 = imageLoad(
        execution_mask,
        base,
        descriptor,
        level,
        sample_loc_1,
    );
    const texel_2 = imageLoad(
        execution_mask,
        base,
        descriptor,
        level,
        sample_loc_2,
    );
    const texel_3 = imageLoad(
        execution_mask,
        base,
        descriptor,
        level,
        sample_loc_3,
    );

    const pixel_0 = unpackUnorm4x(texel_0);
    const pixel_1 = unpackUnorm4x(texel_1);
    const pixel_2 = unpackUnorm4x(texel_2);
    const pixel_3 = unpackUnorm4x(texel_3);

    const vector_one = WarpVec4(f32){ .x = @splat(1), .y = @splat(1), .z = @splat(1), .w = @splat(1) };

    const offset_x = WarpVec4(f32){
        .x = texel_point_offset_x,
        .y = texel_point_offset_x,
        .z = texel_point_offset_x,
        .w = texel_point_offset_x,
    };

    const offset_y = WarpVec4(f32){
        .x = texel_point_offset_y,
        .y = texel_point_offset_y,
        .z = texel_point_offset_y,
        .w = texel_point_offset_y,
    };

    const pixel_tx = pixel_1.hadamardAdd(offset_x, pixel_0.hadamardProduct(vector_one.sub(offset_x)));
    const pixel_bx = pixel_3.hadamardAdd(offset_x, pixel_2.hadamardProduct(vector_one.sub(offset_x)));
    const pixel_ty = pixel_bx.hadamardAdd(offset_y, pixel_tx.hadamardProduct(vector_one.sub(offset_y)));

    return pixel_ty;
}

pub inline fn imageSampleNearest(
    execution_mask: WarpRegister(bool),
    ///Base address from which the descriptor loads
    base: [*]const u32,
    descriptor: ImageDescriptor,
    level: WarpRegister(u32),
    uv: WarpVec2(f32),
) WarpVec4(f32) {
    @setRuntimeSafety(false);

    const texel_location = imageSamplerAddress(descriptor, uv);

    const sample_packed = imageLoad(
        execution_mask,
        base,
        descriptor,
        level,
        texel_location,
    );
    const sample = unpackUnorm4x(sample_packed);
    return sample;
}

pub inline fn imageLoad(
    execution_mask: WarpRegister(bool),
    ///Base address from which the descriptor loads
    base: [*]const u32,
    descriptor: ImageDescriptor,
    ///Specify the mip level to sample from
    level: WarpRegister(u32),
    ///Image coordinates
    physical_position: WarpVec2(i32),
) WarpRegister(u32) {
    @setRuntimeSafety(false);

    const pixel_address = imageAddress(descriptor, level, physical_position.x, physical_position.y);

    //0: opaque white
    //2: cyan
    //3: ...
    //4: transparent black

    const border_colour: WarpRegister(u32) = @splat(@truncate(@as(usize, 0xff_ff_ff_ff) << (@as(u6, descriptor.border_colour_shift_amount) << 3)));

    var actual_mask = execution_mask;

    const min_x: WarpRegister(i32) = @splat(0);
    const min_y: WarpRegister(i32) = @splat(0);

    const width: WarpRegister(i32) = @splat(@as(i32, 1) << @as(u5, descriptor.width_log2));
    const height: WarpRegister(i32) = @splat(@as(i32, 1) << @as(u5, descriptor.height_log2));

    actual_mask = vectorBoolAnd(actual_mask, physical_position.x >= min_x);
    actual_mask = vectorBoolAnd(actual_mask, physical_position.y >= min_y);
    actual_mask = vectorBoolAnd(actual_mask, physical_position.x < width);
    actual_mask = vectorBoolAnd(actual_mask, physical_position.y < height);

    const sample_packed = maskedGather(u32, actual_mask, base, @intCast(pixel_address));

    return @select(u32, actual_mask, sample_packed, border_colour);
}

///Returns the physical image coordinates for the normalized sampler coordinates
pub inline fn imageSamplerAddressFloat(
    descriptor: ImageDescriptor,
    ///Normalized image coordinates
    uv: WarpVec2(f32),
) WarpVec2(f32) {
    @setRuntimeSafety(false);

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
        .clamp_to_border => {},
    }

    const u_scale: WarpRegister(f32) = @splat(@floatFromInt((@as(u32, 1) << @as(u5, descriptor.width_log2)) - 1));
    const v_scale: WarpRegister(f32) = @splat(@floatFromInt((@as(u32, 1) << @as(u5, descriptor.height_log2)) - 1));

    const image_x_float: WarpRegister(f32) = u * u_scale;
    const image_y_float: WarpRegister(f32) = v * v_scale;

    return .{ .x = image_x_float, .y = image_y_float };
}

///Returns the physical image coordinates for the normalized sampler coordinates
pub inline fn imageSamplerAddress(
    descriptor: ImageDescriptor,
    ///Normalized image coordinates
    uv: WarpVec2(f32),
) WarpVec2(i32) {
    @setRuntimeSafety(false);

    const half: WarpRegister(f32) = @splat(0.5);

    const u_scale: WarpRegister(f32) = @splat(@floatFromInt((@as(u32, 1) << @as(u5, descriptor.width_log2)) - 1));
    const v_scale: WarpRegister(f32) = @splat(@floatFromInt((@as(u32, 1) << @as(u5, descriptor.height_log2)) - 1));

    const inv_u_scale = @as(WarpRegister(f32), @splat(1)) / u_scale;
    const inv_v_scale = @as(WarpRegister(f32), @splat(1)) / v_scale;

    const scaled_uv = imageSamplerAddressFloat(descriptor, .{
        .x = uv.x + half * inv_u_scale,
        .y = uv.y + half * inv_v_scale,
    });

    const image_x: WarpRegister(i32) = @intFromFloat(@floor(scaled_uv.x));
    const image_y: WarpRegister(i32) = @intFromFloat(@floor(scaled_uv.y));

    return .{ .x = image_x, .y = image_y };
}

///Computes the address of a pixel relative to the image level base from its x and y coordinates
pub inline fn imageAddress(
    descriptor: ImageDescriptor,
    level: WarpRegister(u32),
    x: WarpRegister(i32),
    y: WarpRegister(i32),
) WarpRegister(u32) {
    _ = level; // autofix
    _ = descriptor; // autofix
    @setRuntimeSafety(false);

    const level_base: WarpRegister(u32) = undefined;
    _ = level_base; // autofix

    var physical_x: WarpRegister(u32) = @intCast(x);
    var physical_y: WarpRegister(u32) = @intCast(y);

    physical_x = physical_x;
    physical_y = physical_y;

    //Map from logical addresses [0..width], [0..height] to physical mid addresses [0..mip_width, 0..mip_height]
    // physical_x >>= @intCast(level);
    // physical_y >>= @intCast(level);

    const pixel_address = mortonEncode(physical_x, physical_y);

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
    src_pitch_width: usize,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) void {
    const block_width: usize = 4;
    const block_height: usize = block_width;

    const block_count_x: usize = @divTrunc(src_pitch_width, block_width) + @intFromBool(@rem(src_pitch_width, block_width) != 0);

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

pub fn reciprocal(x: WarpRegister(f32)) WarpRegister(f32) {
    return @as(WarpRegister(f32), @splat(1)) / x;
}

const Mat3x3 = [3][3]WarpRegister(f32);

pub fn mat3x3Det(mat: Mat3x3) WarpRegister(f32) {
    // const row_1_x = mat[0][0];
    // const row_1_y = mat[0][1];
    // const row_1_z = mat[0][2];

    const column_1_x = mat[0][0];
    _ = column_1_x; // autofix
    const column_1_y = mat[1][0];
    _ = column_1_y; // autofix
    const column_1_z = mat[2][0];
    _ = column_1_z; // autofix

    const row_1_x = (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1]);
    _ = row_1_x; // autofix
    const row_1_y = (mat[0][2] * mat[2][1] - mat[0][1] * mat[2][2]);
    _ = row_1_y; // autofix
    const row_1_z = (mat[0][1] * mat[1][2] - mat[0][2] * mat[1][1]);
    _ = row_1_z; // autofix

    // const a = row_1_x;
    // const b = row_1_y;
    // const c = row_1_z;

    const m = &mat;

    const det = m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) -
        m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
        m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    // return a * (mat[1][1] * mat[2][2] - mat[2][1] * mat[1][2]) - b * (mat[1][0] * mat[2][2] - mat[2][0] * mat[1][2]) + c * (mat[1][0] * mat[2][1] - mat[2][0] * mat[1][1]);
    // return row_1_x * column_1_x + row_1_y * column_1_y + row_1_z * column_1_z;
    return det;
}

pub fn mat3x3InvDet(mat: Mat3x3, recip_det: WarpRegister(f32)) Mat3x3 {
    var result: Mat3x3 = undefined;

    result[0][0] = (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1]) * recip_det;
    result[0][1] = (mat[0][2] * mat[2][1] - mat[0][1] * mat[2][2]) * recip_det;
    result[0][2] = (mat[0][1] * mat[1][2] - mat[0][2] * mat[1][1]) * recip_det;
    result[1][0] = (mat[1][2] * mat[2][0] - mat[1][0] * mat[2][2]) * recip_det;
    result[1][1] = (mat[0][0] * mat[2][2] - mat[0][2] * mat[2][0]) * recip_det;
    result[1][2] = (mat[0][2] * mat[1][0] - mat[0][0] * mat[1][2]) * recip_det;
    result[2][0] = (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]) * recip_det;
    result[2][1] = (mat[0][1] * mat[2][0] - mat[0][0] * mat[2][1]) * recip_det;
    result[2][2] = (mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]) * recip_det;

    return result;
}

pub fn mat3x3MulVec(mat: Mat3x3, vec: WarpVec3(f32)) WarpVec3(f32) {
    var result: WarpVec3(f32) = undefined;

    result.x = mat[0][0] * vec.x + mat[0][1] * vec.y + mat[0][2] * vec.z;
    result.y = mat[1][0] * vec.x + mat[1][1] * vec.y + mat[1][2] * vec.z;
    result.z = mat[2][0] * vec.x + mat[2][1] * vec.y + mat[2][2] * vec.z;

    return result;
}

pub const Homogenous2D = struct {
    x: WarpRegister(f32),
    y: WarpRegister(f32),
    w: WarpRegister(f32),
};

pub fn homogenousMin(a: Homogenous2D, b: Homogenous2D) Homogenous2D {
    var result: Homogenous2D = undefined;

    result.x = @min(a.x * b.w, b.x * a.w);
    result.y = @min(a.y * b.w, b.y * a.w);
    result.w = a.w * b.w;

    return result;
}

pub fn homogenousMax(a: Homogenous2D, b: Homogenous2D) Homogenous2D {
    var result: Homogenous2D = undefined;

    result.x = @max(a.x * b.w, b.x * a.w);
    result.y = @max(a.y * b.w, b.y * a.w);
    result.w = a.w * b.w;

    return result;
}

pub fn homogenousProject(v: Homogenous2D) WarpVec2(f32) {
    const reciprocal_w = reciprocal(v.w);

    return .{ .x = v.x * reciprocal_w, .y = v.y * reciprocal_w };
}

const std = @import("std");
