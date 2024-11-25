const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const opt = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "font-renderer",
        .root_source_file = b.path("src/main.zig"),
        .optimize = opt,
        .target = target,
    });

    b.installArtifact(exe);
}
