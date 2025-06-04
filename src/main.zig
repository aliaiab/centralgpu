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
        .render_area_width = 720,
        .render_area_height = 480,
        .bound_render_target = .{ .pixel_ptr = target_buffer.ptr, .width = render_target_width, .height = render_target_height },
        .depth_image = try gpa.alloc(centralgpu.Depth24Stencil8, render_target_width * render_target_height),
        .raster_tile_buffer = undefined,
    };
    const shambler_texture_data = try std.fs.cwd().readFileAlloc(gpa, "zig-out/bin/shambler_base_color.data", std.math.maxInt(usize));
    defer gpa.free(shambler_texture_data);
    // const test_texture_data = try std.fs.cwd().readFileAlloc(gpa, "zig-out/bin/texture_liquid.data", std.math.maxInt(usize));
    const test_texture_data = try std.fs.cwd().readFileAlloc(gpa, "zig-out/bin/wood_floor.data", std.math.maxInt(usize));
    defer gpa.free(test_texture_data);

    var shambler_texture: u32 = 0;

    gl.glGenTextures(1, &shambler_texture);

    std.debug.assert(shambler_texture != 0);

    gl.glBindTexture(gl.GL_TEXTURE_2D, shambler_texture);

    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR);
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR);

    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE);
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE);

    gl.glTexImage2D(
        gl.GL_TEXTURE_2D,
        0,
        gl.GL_RGBA,
        1024,
        256,
        // 800,
        // 600,
        // 242,
        // 84,
        0,
        gl.GL_RGBA,
        gl.GL_UNSIGNED_BYTE,
        shambler_texture_data.ptr,
    );

    var test_texture_handle: u32 = 0;

    gl.glGenTextures(1, &test_texture_handle);

    std.debug.assert(test_texture_handle != 0);

    gl.glBindTexture(gl.GL_TEXTURE_2D, test_texture_handle);

    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR);
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR);

    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE);
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE);

    gl.glTexImage2D(
        gl.GL_TEXTURE_2D,
        0,
        gl.GL_RGBA,
        // 1024,
        // 256,
        800,
        600,
        // 242,
        // 84,
        0,
        gl.GL_RGBA,
        gl.GL_UNSIGNED_BYTE,
        test_texture_data.ptr,
    );
    gl.glGenerateMipmap(gl.GL_TEXTURE_2D);
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0);

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

        const matrix_math = @import("gl/matrix.zig");

        var window_width_int: i32 = 0;
        var window_height_int: i32 = 0;

        _ = sdl.SDL_GetWindowSize(window, &window_width_int, &window_height_int);

        const aspect_ratio: f32 = @as(f32, @floatFromInt(window_width_int)) / @as(f32, @floatFromInt(window_height_int));

        if (getKeyPressed(sdl.SDL_SCANCODE_F)) {
            // gl.current_context.?.texture_descriptor.sampler_filter = switch (gl.current_context.?.texture_descriptor.sampler_filter) {
            //     .nearest => .bilinear,
            //     .bilinear => .nearest,
            // };
        }

        if (getKeyPressed(sdl.SDL_SCANCODE_T)) {
            // gl.current_context.?.texture_descriptor.sampler_address_mode = switch (gl.current_context.?.texture_descriptor.sampler_address_mode) {
            // .repeat => .clamp_to_edge,
            // .clamp_to_edge => .clamp_to_border,
            // .clamp_to_border => .repeat,
            // };
        }

        if (getKeyPressed(sdl.SDL_SCANCODE_B)) {
            // gl.current_context.?.texture_descriptor.border_colour_shift_amount +%= 1;

            // gl.current_context.?.texture_descriptor.border_colour_shift_amount %= 5;
        }

        if (true) {
            gl.glViewport(0, 0, render_target_width, render_target_height);
            gl.glScissor(0, 0, render_target_width, render_target_height);

            gl.glClearColor(0, 0.2, 0.3, 1);
            gl.glClear(gl.GL_COLOR_BUFFER_BIT);

            gl.glActiveTexture(gl.GL_TEXTURE0);
            gl.glBindTexture(gl.GL_TEXTURE_2D, test_texture_handle);
            gl.glMatrixMode(gl.GL_PROJECTION);
            gl.glLoadIdentity();
            gl.glMatrixMode(gl.GL_MODELVIEW);
            gl.glLoadIdentity();

            gl.glBegin(gl.GL_QUADS);

            gl.glTranslatef(0, 0, 0.1);

            gl.glTexCoord2f(0, 0);
            gl.glVertex2f(0, 0);

            gl.glTexCoord2f(1, 0);
            gl.glVertex2f(1, 0);

            gl.glTexCoord2f(0, 1);
            gl.glVertex2f(0, 1);

            gl.glTexCoord2f(1, 1);
            gl.glVertex2f(1, 1);

            gl.glTranslatef(-1, -0.5, 0.1);

            gl.glTexCoord2f(0, 0);
            gl.glVertex2f(0, 0);

            gl.glTexCoord2f(1, 0);
            gl.glVertex2f(1, 0);

            gl.glTexCoord2f(0, 1);
            gl.glVertex2f(0, 1);

            gl.glTexCoord2f(1, 1);
            gl.glVertex2f(1, 1);

            gl.glEnd();

            if (true) {
                gl.glBindTexture(gl.GL_TEXTURE_2D, shambler_texture);

                gl.glBegin(gl.GL_TRIANGLES);

                const projection_matrix = matrix_math.perspectiveNonReversedZ(
                    std.math.tau * 0.20,
                    aspect_ratio,
                    0.1,
                    500,
                );

                gl.glMatrixMode(gl.GL_PROJECTION);
                gl.glLoadIdentity();
                gl.glLoadMatrix(@ptrCast(&matrix_math.transpose(projection_matrix)));
                gl.glScalef(1, 1, -1);

                gl.glMatrixMode(gl.GL_MODELVIEW);
                gl.glLoadIdentity();
                // gl.glScale3f(1, 1, 1);
                // gl.glTranslate3f(@sin(time_since_start * 0.0005), 0, 0);

                // gl.glScale3f(1, 0.5 * @sin(time_since_start * 0.0005), 1);

                // gl.glTranslate3f(0, 0, @sin(time_since_start * 0.001) * 4);
                gl.glTranslatef(@sin(time_since_start * 0.001) * 0, 0, 5.5 + @sin(time_since_start * 0.001));
                gl.glRotatef(time_since_start * 0.0001 * 360, 0, 1, 0);

                const uv_scale: f32 = 2 + @cos(time_since_start * 0.001);
                // const uv_scale: f32 = 1;

                // gl.glColor3f(1.0, 0.0, 0.0);
                gl.glColor3f(1.0, 1.0, 1.0);
                gl.glTexCoord2f(0.5 * uv_scale, 0);
                gl.glVertex3f(0, 1, 0);

                // gl.glColor3f(0.0, 1.0, 0.0);
                gl.glColor3f(1.0, 1.0, 1.0);
                gl.glTexCoord2f(0, 1 * uv_scale);
                gl.glVertex3f(-1, -1, 0);

                // gl.glColor3f(0.0, 0.0, 1.0);
                gl.glColor3f(1.0, 1.0, 1.0);
                gl.glTexCoord2f(1 * uv_scale, 1 * uv_scale);
                gl.glVertex3f(1, -1, 0);

                gl.glLoadIdentity();
                gl.glTranslatef(0, 0, 1);
                gl.glBindTexture(gl.GL_TEXTURE_2D, test_texture_handle);

                // gl.glColor3f(1.0, 0.0, 0.0);
                // gl.glTexCoord2f(0, 0);
                // gl.glVertex2f(@sin(time_since_start * 0.001) * 2, 1);

                // gl.glColor3f(0.0, 1.0, 0.0);
                // gl.glTexCoord2f(1, 0);
                // gl.glVertex2f(-1 + @sin(time_since_start * 0.001) * 2, -1);

                // gl.glColor3f(0.0, 0.0, 1.0);
                // gl.glTexCoord2f(0, 1);
                // gl.glVertex2f(1 + @sin(time_since_start * 0.001) * 2, -1);

                gl.glEnd();
            }

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
            _ = sdl.SDL_RenderClear(renderer);
            _ = sdl.SDL_RenderTexture(renderer, target_texture, null, null);
            _ = sdl.SDL_RenderPresent(renderer);
        } else {
            _ = sdl.SDL_UpdateWindowSurface(window);
        }

        {
            var count: c_int = 0;
            if (previous_keyboard_state != null) gpa.free(previous_keyboard_state.?);

            previous_keyboard_state = sdl.SDL_GetKeyboardState(&count)[0..@intCast(count)];

            previous_keyboard_state = try gpa.dupe(bool, previous_keyboard_state.?);
        }
    }
}

///Returns true the first time the key is detected as down
fn getKeyPressed(key: sdl.SDL_Scancode) bool {
    var key_count: c_int = 0;

    const keys = sdl.SDL_GetKeyboardState(&key_count)[0..@intCast(key_count)];

    if (previous_keyboard_state == null) {
        return keys[key];
    }

    return keys[key] and !previous_keyboard_state.?[key];
}

var previous_keyboard_state: ?[]const bool = null;

const std = @import("std");
const sdl = @import("sdl");
const centralgpu = @import("centralgpu");
const gl = @import("gl.zig");
