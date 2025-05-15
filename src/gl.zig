//! Central GL: An implementation of opengl 1.0

pub const GL_TRIANGLES: u32 = 0x0004;
pub const GL_TRIANGLE_STRIP: u32 = 0x0005;
pub const GL_TRIANGLE_FAN: u32 = 0x0006;
pub const GL_QUADS: u32 = 0x0007;

pub const Context = struct {
    gpa: std.mem.Allocator,
    bound_render_target: centralgfx.Image = undefined,

    clear_color: u32 = 0xff_00_00_00,
    should_clear_color_attachment: bool = false,
    triangle_vertex_index: u32 = 0,

    modelview_matrix_stack: [2][16]f32 = [_][16]f32{@splat(0)} ** 2,
    modelview_matrix_stack_index: usize = 0,

    triangle_count: usize = 0,
    triangle_positions: std.ArrayListUnmanaged([3]centralgfx.WarpVec3(f32)) = .empty,
    triangle_colors: std.ArrayListUnmanaged([3]u32) = .empty,

    scale: [3]f32 = @splat(1),
    translate: [3]f32 = @splat(0),
};

pub var current_context: ?*Context = null;

pub const GL_COLOR_BUFFER_BIT: u32 = 0x00004000;

pub fn glClearColor(r: f32, g: f32, b: f32, a: f32) void {
    const context = current_context.?;

    context.clear_color = @bitCast(centralgfx.Rgba32.fromNormalized(.{ r, g, b, a }));
}

pub fn glClear(flags: u32) callconv(.c) void {
    const context = current_context.?;

    if (flags & GL_COLOR_BUFFER_BIT != 0) {
        context.should_clear_color_attachment = true;
    }
}

pub fn glViewport(x: i32, y: i32, w: isize, h: isize) void {
    _ = x; // autofix
    _ = y; // autofix
    _ = w; // autofix
    _ = h; // autofix
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

    tri[context.triangle_vertex_index] = @bitCast(centralgfx.Rgba32.fromNormalized(.{ r, g, b, a }));
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
        @memset(
            context.bound_render_target.pixel_ptr[0 .. context.bound_render_target.width * context.bound_render_target.height],
            @bitCast(context.clear_color),
        );
    }

    const triangle_group_count = context.triangle_count / 8 + @intFromBool(context.triangle_count % 8 != 0);

    for (0..triangle_group_count) |triangle_group_id| {
        var out_triangle: centralgfx.WarpProjectedTriangle = undefined;

        const in_triangle = context.triangle_positions.items[triangle_group_id];

        out_triangle.mask = @splat(true);

        if (triangle_group_id == triangle_group_count - 1) {
            const remainder = @rem(context.triangle_count, 8);

            out_triangle.mask = @splat(false);

            for (0..remainder) |mask_index| {
                out_triangle.mask[mask_index] = true;
            }
        }

        for (in_triangle, 0..) |tri, i| {
            out_triangle.points[i] = .{ .x = tri.x, .y = tri.y, .z = tri.z, .w = @splat(1) };

            const viewport_width: centralgfx.WarpRegister(f32) = @splat(@floatFromInt(context.bound_render_target.width));
            const viewport_height: centralgfx.WarpRegister(f32) = @splat(@floatFromInt(context.bound_render_target.height));

            out_triangle.points[i].x = out_triangle.points[i].x * viewport_width;
            out_triangle.points[i].y = out_triangle.points[i].y * -viewport_height;
        }

        out_triangle.unclipped_points = out_triangle.points;

        if (false) {
            for (out_triangle.points, 0..) |point, i| {
                _ = point; // autofix
                for (0..8) |tri_index| {
                    if (out_triangle.mask[tri_index]) {
                        std.log.info("[tri: {}][vtx: {}]out_triangle_pos: x: {}, y: {}, z: {}, col: x{x}", .{
                            tri_index,
                            i,
                            out_triangle.points[i].x[tri_index],
                            out_triangle.points[i].y[tri_index],
                            out_triangle.points[i].z[tri_index],
                            context.triangle_colors.items[triangle_group_id * 8 + tri_index][i],
                        });
                    }
                }
            }
        }

        centralgfx.rasterize(
            .{
                .vertex_colours = context.triangle_colors.items,
            },
            context.bound_render_target,
            triangle_group_id * 8,
            out_triangle,
        );
    }

    context.triangle_positions.clearRetainingCapacity();
    context.triangle_colors.clearRetainingCapacity();
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

        if (context.triangle_positions.items.len < context.triangle_count * 8) {
            _ = context.triangle_positions.addOne(context.gpa) catch @panic("");
        }
    }
}

const std = @import("std");
const centralgfx = @import("root.zig");
