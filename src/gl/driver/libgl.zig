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

const target_width = 1024;
const target_height = 768;

export fn glGetString(name: i32) callconv(.c) [*:0]const u8 {
    const gpa = std.heap.smp_allocator;

    if (centralgpu_gl.current_context == null) {
        centralgpu_gl.current_context = gpa.create(centralgpu_gl.Context) catch @panic("oom");

        const target_buf = gpa.alloc(centralgpu.Rgba32, target_width * target_height) catch @panic("oom");

        centralgpu_gl.current_context.?.* = .{
            .gpa = gpa,
            .bound_render_target = .{ .pixel_ptr = target_buf.ptr, .width = target_width, .height = target_height },
            .depth_image = gpa.alloc(f32, target_width * target_height) catch @panic("oom"),
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
        };

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
            return "GL_ARB_multitexture GL_ARB_texture_env_combine GL_ARB_texture_env_add";
            // return "";
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
        .{ "glClientActiveTextureARB", &centralgpu_gl.glClientActiveTextureARB },
    });

    std.log.info("(centralgl) TRYING TO LOAD: {s}", .{proc_name});

    if (proc_map.get(std.mem.span(proc_name))) |proc| {
        return proc;
    } else {
        std.log.info("(centralgl) FAILED TO LOAD: {s}", .{proc_name});
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

// export fn __glDispatchInit() void {
// @panic("lol");
// }

export fn glXCreateWindow() void {
    @panic("Lol");
}

export fn glXCreateNewContext() void {
    @panic("Lol");
}

export fn glXCreateContextAttribsARB() void {
    @panic("Lol");
}

fn glFlushCallback() void {
    std.log.info("wayland_state: {}, {}, {}", .{ wayland_state.display, wayland_state.surface, wayland_state.buffer });

    const surface_width: usize = @intCast(target_width);
    const surface_height: usize = @intCast(target_height);

    const pixel_ptr: [*]centralgpu.XRgb888 = @ptrCast(@alignCast(wayland_state.out_pixel_buffer.ptr));

    centralgpu.blitRasterTargetToLinear(
        centralgpu_gl.current_context.?.bound_render_target.pixel_ptr,
        pixel_ptr,
        centralgpu_gl.current_context.?.bound_render_target.width,
        // centralgpu_gl.current_context.?.bound_render_target.height,
        640,
        480,
        surface_width,
        surface_height,
    );

    // @memset(wayland_state.out_pixel_buffer, .{ .r = 255, .g = 0, .b = 0, .x = 255 });

    // while (true) {
    if (wayland_state.display.roundtrip() != .SUCCESS) @panic("");

    wayland_state.surface.attach(wayland_state.buffer, 0, 0);
    wayland_state.surface.damage(0, 0, std.math.maxInt(i32), std.math.maxInt(i32));
    wayland_state.surface.commit();
    // }

    // // std.posix.exit(0);
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
    // defer surface.destroy();
    const xdg_surface = try wm_base.getXdgSurface(surface);
    // defer xdg_surface.destroy();
    const xdg_toplevel = try xdg_surface.getToplevel();
    // defer xdg_toplevel.destroy();

    wayland_state.surface = surface;

    xdg_surface.setListener(*wl.Surface, xdgSurfaceListener, surface);
    xdg_toplevel.setListener(*bool, xdgToplevelListener, &wayland_state.running);

    const width = target_width;
    const height = target_height;
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

    const pool = try shm.createPool(fd, size);
    // defer pool.destroy();

    const buffer = try pool.createBuffer(0, width, height, stride, wl.Shm.Format.xrgb8888);
    // defer buffer.destroy();

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

    std.posix.exit(0);
}

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
