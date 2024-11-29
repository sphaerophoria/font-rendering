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

const CanvasPoint = @Vector(2, i64);

pub fn findBezierTForY(ay: f32, by: f32, cy: f32, y: f32) [2]f32 {
    const denominator_i = ay - 2 * by + cy;

    if (denominator_i == 0) {
        // Linear function
        var ret = -1 * (-2 * by + cy + y);
        // FIXME: What if denom is 0
        ret /= 2 * (by - cy);
        return .{ ret, ret };
    }

    var numerator: f32 = -ay * cy + ay * y + by * by - 2 * by * y + cy * y;
    numerator = @sqrt(numerator);
    const offs_term: f32 = ay - by;
    const denominator: f32 = denominator_i;
    const out_1  = (numerator + offs_term) / denominator;
    const out_2  = -(numerator - offs_term) / denominator;
    return .{ out_1, out_2 };
}

const TangentLine = struct {
    a: @Vector(2, f32),
    b: @Vector(2, f32),
};

pub fn quadBezierTangentLine(a: @Vector(2, f32), b: @Vector(2, f32), c: @Vector(2, f32), t: f32) TangentLine {
    const t_splat: @Vector(2, f32) = @splat(t);
    const ab = std.math.lerp(a, b, t_splat);
    const bc = std.math.lerp(b, c, t_splat);
    return .{
        .a = ab,
        .b = bc,
    };
}

pub fn sampleQuadBezierCurve(a: @Vector(2, f32), b: @Vector(2, f32), c: @Vector(2, f32), t: f32) @Vector(2, f32) {
    const tangent_line = quadBezierTangentLine(a, b, c, t);
    return std.math.lerp(tangent_line.a, tangent_line.b, @splat(t));
}


test "bezier solving" {
    const curves = [_][3]@Vector(2, f32) {
        .{
            .{-20, 20},
            .{0, 0},
            .{20, 20},
        },
        .{
            .{-15, -30},
            .{5, 15},
            .{10, 20},
        },
        .{
            .{40, -30},
            .{80, -10},
            .{20, 10},
        },
    };

    const ts = [_]f32{
        0.0, 0.1, 0.4, 0.5, 0.8, 1.0
    };

    for (curves) |curve| {
        for (ts) |in_t| {
            const point1 = sampleQuadBezierCurve(
                curve[0],
                curve[1],
                curve[2],
                in_t,
            );

            var t1, var t2 = findBezierTForY(curve[0][1], curve[1][1], curve[2][1], point1[1]);
            if (@abs(t1 - in_t) > @abs(t2 - in_t)) {
                std.mem.swap(f32, &t1, &t2);
            }
            try std.testing.expectApproxEqAbs(t1, in_t, 0.001);

            if (t2 <= 1.0 and t2 >= 0.0) {

                const point2 = sampleQuadBezierCurve(
                    curve[0],
                    curve[1],
                    curve[2],
                    t2,
                );

                try std.testing.expectApproxEqAbs(point2[1], point1[1], 0.001);
            }
        }
    }
}

pub fn drawBezier(self: *Renderer, a: CanvasPoint, b: CanvasPoint, c: CanvasPoint, color: Color) void {
    const a_f: @Vector(2, f32) = @floatFromInt(a);
    const b_f: @Vector(2, f32) = @floatFromInt(b);
    const c_f: @Vector(2, f32) = @floatFromInt(c);

    const min_y = self.clampedY(std.mem.min(i64, &.{a[1], b[1], c[1]}));
    const max_y = self.clampedY(std.mem.max(i64, &.{a[1], b[1], c[1]}));

    var y = min_y;
    while (y < max_y) {
        defer y += 1;

        const t1s = findBezierTForY(a_f[1], b_f[1], c_f[1], @floatFromInt(y));
        const t2s = findBezierTForY(a_f[1], b_f[1], c_f[1], @floatFromInt(y + 1));
        for (t1s, t2s) |t1, t2| {
            if (!(t1 >= 0.0 and t1 <= 1.0 and t2 >= 0.0 and t2 <= 1.0)) {
                std.debug.print("Skipping due to T {d} {d}\n", .{t1, t2});
                continue;
            }
            const x1_f = sampleQuadBezierCurve(a_f, b_f, c_f, t1)[0];
            const x2_f = sampleQuadBezierCurve(a_f, b_f, c_f, t2)[0];
            const x1_px: i64 = @intFromFloat(@round(x1_f));
            const x2_px: i64 = @intFromFloat(@round(x2_f + 1));

            const x1_clamped: usize = @intCast(self.clampedX(@min(x1_px, x2_px)));
            const x2_clamped: usize = @intCast(self.clampedX(@max(x1_px, x2_px) + 1));

            if (x2_clamped - x1_clamped == 0) {
                std.debug.print("Nothing rendering sadge :(\n", .{});
            }

            const row = self.getRow(y);
            @memset(row[x1_clamped..x2_clamped], color);
        }
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

pub fn clampedY(self: Renderer, val: i64) i64 {
    return @intCast(std.math.clamp(val, 0, self.calcHeight()));
}

pub fn clampedX(self: Renderer, val: i64) i64 {
    return @intCast(std.math.clamp(val, 0, self.iWidth()));
}

pub fn getRow(self: Renderer, y: i64) []Color {
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
