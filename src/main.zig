pub fn main() !void {
    const gpa = std.heap.smp_allocator;

    _ = sdl.SDL_SetAppMetadata("CentralGpu Example", "0.0.0", "CentralGpu Example");

    std.debug.assert(sdl.SDL_Init(0));
    defer sdl.SDL_Quit();

    var window: ?*sdl.SDL_Window = undefined;
    var renderer: ?*sdl.SDL_Renderer = undefined;

    const render_target_width: usize = 640;
    const render_target_height: usize = 480;

    _ = sdl.SDL_CreateWindowAndRenderer(
        "Centralgpu Example",
        640,
        480,
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
        render_target_width,
        render_target_height,
    );
    defer sdl.SDL_DestroyTexture(target_texture);

    _ = sdl.SDL_SetTextureScaleMode(target_texture, sdl.SDL_SCALEMODE_NEAREST);

    const padded_target_width = (render_target_width / 64 + @intFromBool(@rem(render_target_width, 64) != 0)) * 64;
    const padded_target_height = (render_target_height / 64 + @intFromBool(@rem(render_target_height, 64) != 0)) * 64;

    const target_buffer = try gpa.alloc(centralgpu.Rgba32, padded_target_width * padded_target_height);
    defer gpa.free(target_buffer);

    const target_buffer_linear = try gpa.alloc(centralgpu.Rgba32, render_target_width * render_target_height);
    defer gpa.free(target_buffer_linear);

    var sdl_event: sdl.SDL_Event = undefined;

    const start_time = std.time.milliTimestamp();

    gl.current_context = try gpa.create(gl.Context);

    gl.current_context.?.* = .{
        .gpa = gpa,
        .bound_render_target = .{ .pixel_ptr = target_buffer.ptr, .width = render_target_width, .height = render_target_height },
    };

    const test_texture_data = try std.fs.cwd().readFileAlloc(gpa, "zig-out/bin/shambler_base_color.data", std.math.maxInt(usize));
    defer gpa.free(test_texture_data);

    gl.glTexImage2D(
        0,
        0,
        0,
        1024,
        // 800,
        256,
        // 600,
        0,
        0,
        0,
        test_texture_data.ptr,
    );

    const use_gpu_for_scaling: bool = @import("builtin").mode == .Debug;

    while (true) {
        //event pump
        while (sdl.SDL_PollEvent(&sdl_event)) {
            switch (sdl_event.type) {
                sdl.SDL_EVENT_QUIT => return,
                else => {},
            }
        }

        const time_since_start: f32 = @floatFromInt(std.time.milliTimestamp() - start_time);

        if (true) {
            gl.glViewport(0, 0, render_target_width, render_target_height);
            gl.glScissor(0, 0, render_target_width, render_target_height);

            gl.glClearColor(0, 0.2, 0.3, 1);
            gl.glClear(gl.GL_COLOR_BUFFER_BIT);
            gl.glBegin(gl.GL_TRIANGLES);

            // gl.glScale3f(1, 0.5 * @sin(time_since_start * 0.0005), 1);
            // gl.glTranslate3f(-0.5, 0, 0);

            gl.glColor3f(1.0, 0.0, 0.0);
            gl.glTexCoord2f(0.5 * 2, 0);
            gl.glVertex3f(0, 1, @sin(time_since_start * 0.001) * 4);

            gl.glColor3f(0.0, 1.0, 0.0);
            gl.glTexCoord2f(0, 1 * 2);
            gl.glVertex3f(-1, -1, @sin(time_since_start * 0.001) * 4);

            gl.glColor3f(0.0, 0.0, 1.0);
            gl.glTexCoord2f(1 * 2, 1 * 2);
            gl.glVertex3f(1, -1, @sin(time_since_start * 0.001) * 4);

            gl.glColor3f(1.0, 0.0, 0.0);
            gl.glTexCoord2f(0, 0);
            gl.glVertex2f(@sin(time_since_start * 0.001) * 2, 1);

            gl.glColor3f(0.0, 1.0, 0.0);
            gl.glTexCoord2f(1, 0);
            gl.glVertex2f(-1 + @sin(time_since_start * 0.001) * 2, -1);

            gl.glColor3f(0.0, 0.0, 1.0);
            gl.glTexCoord2f(0, 1);
            gl.glVertex2f(1 + @sin(time_since_start * 0.001) * 2, -1);

            gl.glEnd();

            const raster_time_gl = std.time.nanoTimestamp();
            gl.glFlush();

            const raster_time_total_gl = std.time.nanoTimestamp() - raster_time_gl;
            const raster_time_total_ms = @as(f128, @floatFromInt(raster_time_total_gl)) / @as(f128, std.time.ns_per_ms);

            std.debug.print("raster_time_gl (flush): {d}ms, {}ns\n", .{ raster_time_total_ms, raster_time_total_gl });
        }

        const image_transition_time_start = std.time.nanoTimestamp();
        _ = image_transition_time_start; // autofix

        if (use_gpu_for_scaling) {
            const block_width: usize = 4;
            const block_height: usize = 4;

            const block_count_x: usize = render_target_width / block_width + @intFromBool(render_target_width % block_width != 0);
            const block_count_y: usize = render_target_height / block_height + @intFromBool(render_target_height % block_height != 0);

            for (0..block_count_y) |block_y| {
                for (0..block_count_x) |block_x| {
                    const block_index = block_y * block_count_x + block_x;
                    // const block_index = centralgpu.mortonEncodeScalar(@intCast(block_x), @intCast(block_y));

                    if (block_index * block_width * block_height >= target_buffer.len) {
                        continue;
                    }

                    const block_start_ptr: [*]centralgpu.Rgba32 = @ptrCast(&target_buffer[block_index * block_width * block_height]);

                    //quad rasterisation
                    //quad warp layout:
                    //0 1 | 2 3
                    //4 5 | 6 7
                    for (0..2) |half_y_offset| {
                        const y_offset = half_y_offset * 2;
                        // const block_index = block_x + block_y * block_count_x;

                        const target_start_vec: [*]align(4) @Vector(4, u32) = @ptrCast(block_start_ptr + y_offset * block_width);

                        const y = block_y * block_height + y_offset;
                        const x_base_offset = block_x * block_width;

                        for (0..2) |row| {
                            const out_index = x_base_offset + (y + row) * render_target_width;
                            const out_pixel_row: [*]align(4) @Vector(4, u32) = @ptrCast(&target_buffer_linear[out_index]);
                            const in_pixel_row: @Vector(4, u32) = target_start_vec[row];

                            out_pixel_row[0] = in_pixel_row;
                            // out_pixel_row[0] = @splat(0xff_ff_00_00);
                        }
                    }
                }
            }
        }

        const sdl_window_surface = sdl.SDL_GetWindowSurface(window);

        const time_surface_present = std.time.nanoTimestamp();

        if (!use_gpu_for_scaling and sdl.SDL_LockSurface(sdl_window_surface)) {
            switch (sdl_window_surface.*.format) {
                sdl.SDL_PIXELFORMAT_XRGB8888 => {
                    const surface_width: usize = @intCast(sdl_window_surface.*.w);
                    const surface_height: usize = @intCast(sdl_window_surface.*.h);

                    const pixel_ptr: [*]centralgpu.XRgb888 = @ptrCast(@alignCast(sdl_window_surface.*.pixels.?));

                    centralgpu.blitRasterTargetToLinear(
                        target_buffer.ptr,
                        pixel_ptr,
                        render_target_width,
                        render_target_height,
                        surface_width,
                        surface_height,
                    );
                },
                else => @panic("unsupported surface format"),
            }

            std.debug.print("time_surface_present: ({}ns)\n", .{std.time.nanoTimestamp() - time_surface_present});

            sdl.SDL_UnlockSurface(sdl_window_surface);
        }

        // std.debug.print("image layout transition: {}ns\n", .{std.time.nanoTimestamp() - image_transition_time_start});

        if (use_gpu_for_scaling) {
            _ = sdl.SDL_UpdateTexture(target_texture, null, target_buffer_linear.ptr, render_target_width * @sizeOf(u32));
            _ = sdl.SDL_RenderTexture(renderer, target_texture, null, null);
            _ = sdl.SDL_RenderPresent(renderer);
        } else {
            _ = sdl.SDL_UpdateWindowSurface(window);
        }
    }
}

const std = @import("std");
const sdl = @import("sdl");
const centralgpu = @import("root.zig");
const gl = @import("gl.zig");
