const std = @import("std");

pub fn build(b: *std.Build) !void {
    const test_step = b.step("test", "");
    const target = b.standardTargetOptions(.{});
    const opt = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "font-renderer",
        .root_source_file = b.path("src/main.zig"),
        .optimize = opt,
        .target = target,
    });

    b.installArtifact(exe);

    const test_exe = b.addTest(.{
        .name = "test",
        .root_source_file = b.path("src/main.zig"),
        .optimize = opt,
        .target = target,
    });

    const run_test = b.addRunArtifact(test_exe);
    test_step.dependOn(&run_test.step);
}
