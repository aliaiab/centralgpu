const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const sdl_dep = b.dependency("sdl", .{
        .target = target,
        .optimize = optimize,
        .preferred_link_mode = .static, // or .dynamic
    });
    const sdl_lib = sdl_dep.artifact("SDL3");

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const sdl_translate = b.addTranslateC(.{
        .optimize = optimize,
        .target = target,
        .root_source_file = sdl_dep.path("include/SDL3/SDL.h"),
    });

    sdl_translate.defineCMacro("SDL_DISABLE_OLD_NAMES", null);
    sdl_translate.addIncludePath(sdl_dep.path("include/"));

    exe_mod.linkLibrary(sdl_lib);

    exe_mod.addImport("sdl", sdl_translate.createModule());

    const exe = b.addExecutable(.{
        .name = "centralgpu_example",
        .root_module = exe_mod,
    });

    exe.use_llvm = true;

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const exe_unit_tests = b.addTest(.{
        .root_module = exe_mod,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_exe_unit_tests.step);
}
