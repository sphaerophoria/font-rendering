const std = @import("std");
const Allocator = std.mem.Allocator;

const Renderer = @This();
pub const Color = packed struct(u32) {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
};

canvas: []Color,
width: usize,

pub fn init(alloc: Allocator, width: usize, height: usize) !Renderer {
    const canvas = try alloc.alloc(Color, width * height);
    return .{
        .canvas = canvas,
        .width = @intCast(width),
    };
}

pub fn deinit(self: *Renderer, alloc: Allocator) void {
    alloc.free(self.canvas);
}

pub fn iWidth(self: Renderer) i64 {
    return @intCast(self.width);
}
pub fn calcHeight(self: Renderer) i64 {
    return @intCast(self.canvas.len / self.width);
}

pub fn drawCircle(self: *Renderer, cx: i64, cy: i64, r: i64, color: Color) void {
    const r2 = r * r;

    const min_y = self.clampedY(cy - r);
    const max_y = self.clampedY(cy + r);

    var y = min_y;
    while (y < max_y) {
        defer y += 1;
        // x^2 + y^2 < r^2
        // x^2 = r^2 - y^2
        // x = +-sqrt(r^2 - y^2)
        const y_offs = y - cy;
        // FIXME ew
        const x_offs: i64 = @intFromFloat(@sqrt(@as(f32, @floatFromInt(r2 - y_offs * y_offs))));

        const row = self.getRow(y);
        const write_start: usize = @intCast(self.clampedX(cx - x_offs));
        const write_end: usize = @intCast(self.clampedX(cx + x_offs));
        @memset(row[write_start..write_end], color);
    }
}

pub fn drawRect(self: *Renderer, min_x: i64, max_x: i64, min_y: i64, max_y: i64, color: Color) void {
    var y = self.clampedY(min_y);
    const end_y = self.clampedY(max_y);
    const start_x: usize = @intCast(self.clampedX(min_x));
    const end_x: usize = @intCast(self.clampedX(max_x));
    while (y < end_y) {
        defer y += 1;
        const row = self.getRow(y);
        @memset(row[start_x..end_x], color);
    }
}

pub fn drawLine(self: *Renderer, ax: i64, ay: i64, bx: i64, by: i64, width: i64, color: Color) void {
    const w_2 = @divFloor(width, 2);
    var y = self.clampedY(@min(ay, by) - w_2);
    const max_y = self.clampedY(@max(ay, by) + 1 + w_2);

    const ax_f: f32 = @floatFromInt(ax);
    const bx_f: f32 = @floatFromInt(bx);
    while (y < max_y) {
        defer y += 1;

        var t: f32 = @floatFromInt(y - ay);
        t /= @floatFromInt(by - ay);

        var t2: f32 = @floatFromInt(y + 1 - ay);
        t2 /= @floatFromInt(by - ay);
        if (by - ay == 0) {
            t = 0.0;
            t2 = 1.0;
        }
        const x1: i64 = @intFromFloat(std.math.lerp(ax_f, bx_f, std.math.clamp(t, 0.0, 1.0)));
        const x2: i64 = @intFromFloat(std.math.lerp(ax_f, bx_f, std.math.clamp(t2, 0.0, 1.0)));
        const row = self.getRow(y);
        const min_x: usize = @intCast(self.clampedX(@min(x1, x2) - w_2));
        const max_x: usize = @intCast(self.clampedX(@max(x1, x2) + w_2 + 1));
        @memset(row[min_x..max_x], color);
    }
}

pub fn writePpm(self: Renderer, writer: anytype) !void {
    try writer.print(
        \\P6
        \\{d} {d}
        \\255
        \\
    , .{ self.width, self.calcHeight() });

    for (self.canvas) |p| {
        try writer.writeAll(&.{ p.r, p.g, p.b });
    }
}

const Range = struct {
    start: i64,
    end: i64,
};

fn clampedY(self: Renderer, val: i64) i64 {
    return @intCast(std.math.clamp(val, 0, self.calcHeight()));
}

fn clampedX(self: Renderer, val: i64) i64 {
    return @intCast(std.math.clamp(val, 0, self.iWidth()));
}

fn getRow(self: Renderer, y: i64) []Color {
    const row_start: usize = @intCast(y * self.iWidth());
    const row_end: usize = row_start + self.width;
    return self.canvas[row_start..row_end];
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();
    var renderer = try Renderer.init(alloc, 800, 600);
    defer renderer.deinit(alloc);

    renderer.drawRect(0, 800, 0, 600, .{ .r = 0, .g = 255, .b = 0, .a = 255 });
    renderer.drawCircle(400, 300, 50, .{ .r = 255, .g = 0, .b = 0, .a = 255 });
    renderer.drawCircle(600, 300, 50, .{ .r = 0, .g = 0, .b = 255, .a = 255 });
    renderer.drawCircle(-25, -25, 50, .{ .r = 0, .g = 0, .b = 255, .a = 255 });
    renderer.drawCircle(810, 610, 50, .{ .r = 0, .g = 255, .b = 255, .a = 255 });
    renderer.drawLine(0, 0, 800, 600, 5, .{ .r = 255, .g = 255, .b = 0, .a = 255 });
    renderer.drawLine(400, 0, 400, 600, 5, .{ .r = 255, .g = 255, .b = 0, .a = 255 });
    renderer.drawLine(0, 300, 800, 300, 5, .{ .r = 255, .g = 255, .b = 0, .a = 255 });

    const f = try std.fs.cwd().createFile("test.ppm", .{});
    defer f.close();

    try renderer.writePpm(f.writer());
}
