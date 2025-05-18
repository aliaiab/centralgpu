export const __GLXGL_CORE_FUNCTIONS: [*:0]const u8 = "";

export fn __glXGLLoadGLXFunction(name: [*:0]const u8, ptr: *anyopaque, mutex: *anyopaque) *anyopaque {
    _ = name; // autofix
    _ = ptr; // autofix
    _ = mutex; // autofix
    @panic("");
}

pub fn panic(msg: []const u8, stack_trace: ?*const std.builtin.StackTrace, ra: ?usize) noreturn {
    _ = stack_trace; // autofix
    _ = ra; // autofix

    std.log.err("panic: {s}", .{msg});

    std.posix.exit(0);
}

const std = @import("std");
