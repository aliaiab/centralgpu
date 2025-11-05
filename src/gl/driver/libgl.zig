const gl_c = struct {
    pub const GL_VENDOR = 0x1F00;
    pub const GL_RENDERER = 0x1F01;
    pub const GL_VERSION = 0x1F02;
    pub const GL_EXTENSIONS = 0x1F03;
};

const WaylandContext = struct {
    shm: ?*wl.Shm,
    compositor: ?*wl.Compositor,
    wm_base: ?*xdg.WmBase,
};

const default_target_width = 640;
const default_target_height = 480;

var target_width: usize = default_target_width;
var target_height: usize = default_target_height;

var surface_width: usize = @intCast(default_target_width * 1);
var surface_height: usize = @intCast(default_target_height * 1);

export fn glGetString(name: i32) callconv(.c) [*:0]const u8 {
    const gpa = std.heap.smp_allocator;

    if (centralgpu_gl.current_context == null) {
        {
            const env_map = std.process.getEnvMap(gpa) catch @panic("oom");

            if (env_map.get("CENTRALGPU_TARGET_WIDTH")) |target_width_string| {
                target_width = std.fmt.parseInt(usize, target_width_string, 10) catch target_width;
            }

            if (env_map.get("CENTRALGPU_TARGET_HEIGHT")) |target_height_string| {
                target_height = std.fmt.parseInt(usize, target_height_string, 10) catch target_height;
            }

            surface_width = target_width;
            surface_height = target_height;

            if (env_map.get("CENTRALGPU_SURFACE_WIDTH")) |surface_width_string| {
                surface_width = std.fmt.parseInt(usize, surface_width_string, 10) catch surface_width;
            }

            if (env_map.get("CENTRALGPU_SURFACE_HEIGHT")) |surface_height_string| {
                surface_height = std.fmt.parseInt(usize, surface_height_string, 10) catch surface_height;
            }

            if (env_map.get("CENTRALGPU_SURFACE_SCALE")) |surface_scale_string| {
                const surface_scale = std.fmt.parseInt(usize, surface_scale_string, 10) catch 1;

                surface_width *= surface_scale;
                surface_height *= surface_scale;
            }
        }

        centralgpu_gl.current_context = gpa.create(centralgpu_gl.Context) catch @panic("oom");

        const target_buf = gpa.alloc(centralgpu.Rgba32, target_width * target_height) catch @panic("oom");

        const target_padded_width = centralgpu.computeTargetPaddedSize(target_width);
        const target_padded_height = centralgpu.computeTargetPaddedSize(target_height);

        const target_tile_width = target_padded_width / centralgpu.tile_width + @intFromBool(target_width % 16 != 0);
        const target_tile_height = target_padded_height / centralgpu.tile_height + @intFromBool(target_height % 16 != 0);

        const tile_buffer = gpa.alloc(centralgpu.RasterTileBuffer.Tile, target_tile_width * target_tile_height) catch @panic("oom");

        centralgpu_gl.current_context.?.* = .{
            .gpa = gpa,
            .render_area_width = @intCast(target_width),
            .render_area_height = @intCast(target_height),
            .bound_render_target = .{ .pixel_ptr = target_buf.ptr, .width = @intCast(target_width), .height = @intCast(target_height) },
            .depth_image = gpa.alloc(centralgpu.Depth24Stencil8, target_width * target_height) catch @panic("oom"),
            .flush_callback = &glFlushCallback,
            .viewport = .{
                .x = 0,
                .y = 0,
                .width = @intCast(target_width),
                .height = @intCast(target_height),
            },
            .scissor = .{
                .x = 0,
                .y = 0,
                .width = @intCast(target_width),
                .height = @intCast(target_height),
            },
            .raster_tile_buffer = .{
                .tile_data = tile_buffer,
                .gpa = gpa,
                .thread_pool = undefined,
            },
        };

        centralgpu_gl.current_context.?.raster_tile_buffer.thread_pool.init(.{
            .allocator = gpa,
        }) catch @panic("oom");

        initWayland() catch |e| {
            @panic(@errorName(e));
        };
    }

    switch (name) {
        gl_c.GL_VENDOR => {
            return "CENTRAL_GPU";
        },
        gl_c.GL_RENDERER => {
            return "CENTRAL_GPU_RASTER";
        },
        gl_c.GL_VERSION => {
            return "1.5.0";
        },
        gl_c.GL_EXTENSIONS => {
            return "GL_ARB_multitexture GL_ARB_texture_env_combine GL_ARB_texture_env_add GL_EXT_framebuffer_object";
        },
        else => {
            std.log.info("glGetString: name = {}", .{name});
            @panic("Unsupported gl string");
        },
    }
}

//Hack to get quakespasm to work
pub export fn SDL_GL_GetProcAddress(
    proc_name: [*:0]const u8,
) ?*const anyopaque {
    const proc_map = std.static_string_map.StaticStringMap(*const anyopaque).initComptime(.{
        .{ "glMultiTexCoord2fARB", &centralgpu_gl.glMultiTexCoord2fARB },
        .{ "glActiveTextureARB", &centralgpu_gl.glActiveTexture },
        .{ "glActiveTexture", &centralgpu_gl.glActiveTexture },
        .{ "glClientActiveTextureARB", &centralgpu_gl.glActiveTexture },
        .{ "glGenerateMipmapEXT", &centralgpu_gl.glGenerateMipmap },
        .{ "glGenerateMipmap", &centralgpu_gl.glGenerateMipmap },
    });

    std.debug.print("(centralgl) TRYING TO LOAD: {s}\n", .{proc_name});

    if (proc_map.get(std.mem.span(proc_name))) |proc| {
        return proc;
    } else {
        std.debug.print("(centralgl) FAILED TO LOAD: {s}", .{proc_name});
        return null;
    }
    return null;
}

var wayland_state: struct {
    display: *wl.Display,
    surface: *wl.Surface,
    buffer: *wl.Buffer,
    out_pixel_buffer: []align(1) centralgpu.XRgb888,
    running: bool = true,
} = undefined;

fn glFlushCallback() void {
    const pixel_ptr: [*]centralgpu.XRgb888 = @ptrCast(@alignCast(wayland_state.out_pixel_buffer.ptr));

    const time_start_ns = std.time.nanoTimestamp();

    centralgpu.blitRasterTargetToLinear(
        centralgpu_gl.current_context.?.bound_render_target.pixel_ptr,
        pixel_ptr,
        centralgpu_gl.current_context.?.bound_render_target.width,
        target_width,
        target_height,
        surface_width,
        surface_height,
    );

    {
        std.log.info("blitRaster_time: {}ns\n", .{std.time.nanoTimestamp() - time_start_ns});
    }

    if (wayland_state.display.roundtrip() != .SUCCESS) @panic("");

    wayland_state.surface.attach(wayland_state.buffer, 0, 0);
    wayland_state.surface.damage(0, 0, std.math.maxInt(i32), std.math.maxInt(i32));
    wayland_state.surface.commit();
}

fn initWayland() !void {
    const display = try wl.Display.connect(null);

    wayland_state.display = display;

    const registry = try display.getRegistry();

    var context = WaylandContext{
        .shm = null,
        .compositor = null,
        .wm_base = null,
    };

    wayland_state.running = true;

    registry.setListener(*WaylandContext, registryListener, &context);
    if (display.roundtrip() != .SUCCESS) return error.RoundtripFailed;

    const shm = context.shm orelse return error.NoWlShm;
    const compositor = context.compositor orelse return error.NoWlCompositor;
    const wm_base = context.wm_base orelse return error.NoXdgWmBase;

    const surface = try compositor.createSurface();
    const xdg_surface = try wm_base.getXdgSurface(surface);
    const xdg_toplevel = try xdg_surface.getToplevel();

    wayland_state.surface = surface;

    xdg_surface.setListener(*wl.Surface, xdgSurfaceListener, surface);
    xdg_toplevel.setListener(*bool, xdgToplevelListener, &wayland_state.running);

    const width = surface_width;
    const height = surface_height;
    const stride = width * 4;
    const size = stride * height;

    const fd = try posix.memfd_create("hello-zig-wayland", 0);
    try posix.ftruncate(fd, size);
    const data = try posix.mmap(
        null,
        size,
        posix.PROT.READ | posix.PROT.WRITE,
        .{ .TYPE = .SHARED },
        fd,
        0,
    );

    const Argb32 = centralgpu.XRgb888;

    const data_as_int_buffer: []align(1) u32 = @ptrCast(data);
    _ = data_as_int_buffer; // autofix
    const pixel_buffer: []align(1) Argb32 = @ptrCast(data);

    wayland_state.out_pixel_buffer = pixel_buffer;

    @memset(data, 0xff);

    const pool = try shm.createPool(fd, @intCast(size));

    const buffer = try pool.createBuffer(0, @intCast(width), @intCast(height), @intCast(stride), wl.Shm.Format.xrgb8888);

    wayland_state.buffer = buffer;

    surface.commit();
    if (display.roundtrip() != .SUCCESS) return error.RoundtripFailed;

    surface.attach(buffer, 0, 0);
    surface.commit();

    if (display.roundtrip() != .SUCCESS) return error.RoundtripFailed;
}

fn registryListener(registry: *wl.Registry, event: wl.Registry.Event, context: *WaylandContext) void {
    const mem = std.mem;

    switch (event) {
        .global => |global| {
            if (mem.orderZ(u8, global.interface, wl.Compositor.interface.name) == .eq) {
                context.compositor = registry.bind(global.name, wl.Compositor, 1) catch return;
            } else if (mem.orderZ(u8, global.interface, wl.Shm.interface.name) == .eq) {
                context.shm = registry.bind(global.name, wl.Shm, 1) catch return;
            } else if (mem.orderZ(u8, global.interface, xdg.WmBase.interface.name) == .eq) {
                context.wm_base = registry.bind(global.name, xdg.WmBase, 1) catch return;
            }
        },
        .global_remove => {},
    }
}

fn xdgSurfaceListener(xdg_surface: *xdg.Surface, event: xdg.Surface.Event, surface: *wl.Surface) void {
    switch (event) {
        .configure => |configure| {
            xdg_surface.ackConfigure(configure.serial);
            surface.commit();
        },
    }
}

fn xdgToplevelListener(_: *xdg.Toplevel, event: xdg.Toplevel.Event, running: *bool) void {
    switch (event) {
        .configure => {},
        .close => running.* = false,
    }
}

pub fn panic(msg: []const u8, stack_trace: ?*const std.builtin.StackTrace, ra: ?usize) noreturn {
    _ = stack_trace; // autofix
    _ = ra; // autofix

    std.log.err("panic: {s}", .{msg});

    std.posix.abort();
}

pub const std_options: std.Options = .{
    .log_level = .err,
};

comptime {
    _ = centralgpu_gl;
}

const centralgpu_gl = @import("centralgpu_gl");
const centralgpu = @import("centralgpu");
const sdl = @import("sdl");
const std = @import("std");

const posix = std.posix;
const wayland = @import("wayland");
const wl = wayland.client.wl;
const xdg = wayland.client.xdg;
