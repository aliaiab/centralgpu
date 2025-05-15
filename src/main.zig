pub fn main() !void {
    const gpa = std.heap.smp_allocator;

    _ = sdl.SDL_SetAppMetadata("CentralGpu Example", "0.0.0", "CentralGpu Example");

    std.debug.assert(sdl.SDL_Init(0));
    defer sdl.SDL_Quit();

    var window: ?*sdl.SDL_Window = undefined;
    var renderer: ?*sdl.SDL_Renderer = undefined;

    const window_width = 640 / 2;
    const window_height = 480 / 2;

    _ = sdl.SDL_CreateWindowAndRenderer(
        "Centralgpu Example",
        1600,
        900,
        sdl.SDL_WINDOW_RESIZABLE,
        &window,
        &renderer,
    );
    defer sdl.SDL_DestroyRenderer(renderer);
    defer sdl.SDL_DestroyWindow(window);

    const target_texture = sdl.SDL_CreateTexture(
        renderer,
        sdl.SDL_PIXELFORMAT_RGBA32,
        sdl.SDL_TEXTUREACCESS_STREAMING,
        window_width,
        window_height,
    );
    defer sdl.SDL_DestroyTexture(target_texture);

    _ = sdl.SDL_SetTextureScaleMode(target_texture, sdl.SDL_SCALEMODE_NEAREST);

    const target_buffer = try gpa.alloc(centralgpu.Rgba32, window_width * window_height);
    defer gpa.free(target_buffer);

    const target_buffer_linear = try gpa.alloc(centralgpu.Rgba32, window_width * window_height);
    defer gpa.free(target_buffer_linear);

    var sdl_event: sdl.SDL_Event = undefined;

    const start_time = std.time.milliTimestamp();

    gl.current_context = try gpa.create(gl.Context);

    gl.current_context.?.* = .{
        .gpa = gpa,
        .bound_render_target = .{ .pixel_ptr = target_buffer.ptr, .width = window_width, .height = window_height },
    };

    while (true) {
        //event pump
        while (sdl.SDL_PollEvent(&sdl_event)) {
            switch (sdl_event.type) {
                sdl.SDL_EVENT_QUIT => return,
                else => {},
            }
        }

        const time_since_start: f32 = @floatFromInt(std.time.milliTimestamp() - start_time);

        const vertex_colors: [2][3]u32 = .{
            .{
                @bitCast(centralgpu.Rgba32.fromNormalized(.{ 1, 0, 0, 1 })),
                @bitCast(centralgpu.Rgba32.fromNormalized(.{ 0, 1, 0, 1 })),
                @bitCast(centralgpu.Rgba32.fromNormalized(.{ 0, 0, 1, 1 })),
            },
            .{
                @bitCast(centralgpu.Rgba32.fromNormalized(.{ 0.4, 0, 1, 1 })),
                @bitCast(centralgpu.Rgba32.fromNormalized(.{ 0, 1, 0, 1 })),
                @bitCast(centralgpu.Rgba32.fromNormalized(.{ 1, 0.4, 0, 1 })),
            },
        };

        if (false) {
            const time_begin = std.time.nanoTimestamp();

            // softraster.rasterizeSingle(
            //     .{
            //         .pixel_ptr = target_buffer.ptr,
            //         .width = window_width,
            //         .height = window_height,
            //     },
            //     .{
            //         .{ 200 + @sin(time_since_start * 0.001) * 300, 50 },
            //         .{ 450, 450 },
            //         .{ 50, 450 },
            //     },
            //     .{
            //         .{ 1, 0, 0, 1 },
            //         .{ 0, 1, 0, 1 },
            //         .{ 0, 0, 1, 1 },
            //     },
            // );

            //         .{ 200 + @sin(time_since_start * 0.001) * 300, 50 },
            //         .{ 450, 450 },
            //         .{ 50, 450 },

            var out_triangle: centralgpu.WarpProjectedTriangle = undefined;

            out_triangle.mask = @splat(false);
            out_triangle.mask[0] = true;
            out_triangle.mask[1] = true;

            out_triangle.points[0].x[0] = 200 + @sin(time_since_start * 0.001) * 300;
            out_triangle.points[0].y[0] = 50;

            out_triangle.points[1].x[0] = 450;
            out_triangle.points[1].y[0] = 450;

            out_triangle.points[2].x[0] = 50;
            out_triangle.points[2].y[0] = 450;

            out_triangle.points[0].x[1] = 50 + 200 + @cos(time_since_start * 0.001) * 300;
            out_triangle.points[0].y[1] = 50 + 50;

            out_triangle.points[1].x[1] = 50 + 450;
            out_triangle.points[1].y[1] = 50 + 450;

            out_triangle.points[2].x[1] = 50 + 50;
            out_triangle.points[2].y[1] = 50 + 450;

            out_triangle.unclipped_points = out_triangle.points;

            // // ^ 5733075ns

            // //   1159267ns
            centralgpu.rasterize(
                .{
                    .vertex_colours = &vertex_colors,
                },
                .{ .pixel_ptr = target_buffer.ptr, .width = window_width, .height = window_height },
                0,
                out_triangle,
            );

            std.debug.print("rasterize_time = {}ns \n", .{std.time.nanoTimestamp() - time_begin});
        }

        if (true) {
            gl.glViewport(0, 0, window_width, window_height);

            gl.glClearColor(0, 0.2, 0.3, 1);
            gl.glClear(gl.GL_COLOR_BUFFER_BIT);
            gl.glBegin(gl.GL_TRIANGLES);

            gl.glScale3f(0.25, 0.25, 1);
            gl.glTranslate3f(0.5, -0.5, 0);

            gl.glColor3f(1.0, 0.0, 0.0);
            gl.glVertex2i(0, 1);

            gl.glColor3f(0.0, 1.0, 0.0);
            gl.glVertex2i(-1, -1);

            gl.glColor3f(0.0, 0.0, 1.0);
            gl.glVertex2i(1, -1);

            gl.glColor3f(1.0, 0.0, 0.0);
            gl.glVertex2f(@sin(time_since_start * 0.001), 1);

            gl.glColor3f(0.0, 1.0, 0.0);
            gl.glVertex2f(-1 + @sin(time_since_start * 0.001), -1);

            gl.glColor3f(0.0, 0.0, 1.0);
            gl.glVertex2f(1 + @sin(time_since_start * 0.001), -1);

            gl.glEnd();

            const raster_time_gl = std.time.nanoTimestamp();
            gl.glFlush();

            std.debug.print("raster_time_gl (flush): {}ns\n", .{std.time.nanoTimestamp() - raster_time_gl});
        }

        const use_quad_raster = true;

        const image_transition_time_start = std.time.nanoTimestamp();

        if (use_quad_raster) {
            const block_width: usize = 4;
            const block_height: usize = 4;

            const block_count_x: usize = window_width / block_width + @intFromBool(window_width % block_width != 0);
            const block_count_y: usize = window_height / block_height + @intFromBool(window_height % block_height != 0);

            for (0..block_count_y) |block_y| {
                for (0..block_count_x) |block_x| {

                    //quad rasterisation
                    //quad warp layout:
                    //0 1 | 2 3
                    //4 5 | 6 7
                    for (0..2) |half_y_offset| {
                        const y_offset = half_y_offset * 2;
                        const block_index = block_x + block_y * block_count_x;

                        const target_start_ptr = target_buffer.ptr + block_index * block_width * block_height + y_offset * block_width;

                        const target_start_vec: [*]align(4) @Vector(4, u32) = @ptrCast(target_start_ptr);

                        const y = block_y * block_height + y_offset;
                        const x_base_offset = block_x * block_width;

                        for (0..2) |row| {
                            const out_index = x_base_offset + (y + row) * window_width;
                            const out_pixel_row: [*]align(4) @Vector(4, u32) = @ptrCast(&target_buffer_linear[out_index]);
                            const in_pixel_row: @Vector(4, u32) = target_start_vec[row];

                            out_pixel_row[0] = in_pixel_row;
                        }
                    }
                }
            }
        } else {
            //linearize the render target
            for (0..window_height) |y| {
                for (0..window_width) |x| {
                    const index = x + y * window_width;

                    const block_x = x / 8;
                    const block_y = y / 8;

                    const block_start_x = block_x * 8;
                    const block_start_y = block_y * 8;

                    const start_ptr = target_buffer.ptr + 8 * 8 * (block_x + block_y * (@as(usize, window_width) / 8 + @as(usize, @intFromBool(window_width % 8 != 0))));

                    const x_offset = x - block_start_x;
                    const y_offset = y - block_start_y;

                    target_buffer_linear[index] = start_ptr[x_offset + y_offset * 8];
                }
            }
        }

        std.debug.print("image layout transition: {}ns\n", .{std.time.nanoTimestamp() - image_transition_time_start});

        _ = sdl.SDL_UpdateTexture(target_texture, null, target_buffer_linear.ptr, window_width * @sizeOf(u32));
        _ = sdl.SDL_RenderTexture(renderer, target_texture, null, null);
        _ = sdl.SDL_RenderPresent(renderer);
    }
}

pub fn cosTurns(x: f32) f32 {
    const r = @abs(x) - @floor(@abs(x));

    return r;
}

fn cosCoff(comptime n: comptime_int) comptime_float {
    return (if (n % 2 == 0) 1 else -1) * repeatedMul(std.math.tau, 2 * n) / factorial(2 * n);
}

fn factorial(value: comptime_int) comptime_int {
    var result: comptime_int = 1;

    for (0..value) |k| {
        result *= (value - k);
    }

    return result;
}

fn repeatedMul(value: comptime_float, comptime n: comptime_int) comptime_float {
    var result: comptime_float = 1;

    for (0..n) |_| {
        result *= value;
    }

    return result;
}

const std = @import("std");
const sdl = @import("sdl");
const centralgpu = @import("root.zig");
const gl = @import("gl.zig");
