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

    const centralgpu_module = b.createModule(.{
        .root_source_file = b.path("src/centralgpu.zig"),
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

    const sdl_module = sdl_translate.createModule();

    exe_mod.addImport("sdl", sdl_module);
    exe_mod.addImport("centralgpu", centralgpu_module);

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

    const centralgpu_gl_module = b.createModule(.{
        .root_source_file = b.path("src/gl.zig"),
        .target = target,
        .optimize = optimize,
    });
    centralgpu_gl_module.addImport("centralgpu", centralgpu_module);

    //Build the driver libraries
    {
        const libgl_module = b.createModule(.{
            .root_source_file = b.path("src/gl/driver/libgl.zig"),
            .target = target,
            .optimize = optimize,
        });

        libgl_module.addImport("sdl", sdl_module);
        libgl_module.addImport("centralgpu_gl", centralgpu_gl_module);
        libgl_module.addImport("centralgpu", centralgpu_module);
        libgl_module.linkLibrary(sdl_lib);

        {
            const Scanner = @import("wayland").Scanner;

            const scanner = Scanner.create(b, .{});

            const wayland_module = b.createModule(.{ .root_source_file = scanner.result });

            scanner.addSystemProtocol("stable/xdg-shell/xdg-shell.xml");
            scanner.addSystemProtocol("unstable/xdg-decoration/xdg-decoration-unstable-v1.xml");
            scanner.generate("wl_compositor", 1);
            scanner.generate("wl_shm", 1);
            scanner.generate("wl_seat", 1);
            scanner.generate("xdg_wm_base", 3);
            scanner.generate("zxdg_decoration_manager_v1", 1);

            libgl_module.addImport("wayland", wayland_module);

            libgl_module.linkSystemLibrary("wayland-client", .{
                .weak = true,
            });
        }

        const libgl_lib = b.addLibrary(.{
            .name = "GL",
            .linkage = .dynamic,
            .root_module = libgl_module,
            .version = .{
                .major = 1,
                .minor = 0,
                .patch = 0,
            },
        });

        b.installArtifact(libgl_lib);
    }

    if (false) {
        const libglx_module = b.createModule(.{
            .root_source_file = b.path("src/gl/driver/libglx.zig"),
            .target = target,
            .optimize = optimize,
        });

        const libgl_lib = b.addLibrary(.{
            .name = "GLX",
            .linkage = .dynamic,
            .root_module = libglx_module,
            .version = .{
                .major = 0,
                .minor = 0,
                .patch = 0,
            },
        });

        b.installArtifact(libgl_lib);
    }

    const exe_unit_tests = b.addTest(.{
        .root_module = exe_mod,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_exe_unit_tests.step);
}
