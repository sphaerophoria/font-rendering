const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const Renderer = @import("Renderer.zig");

test {
    std.testing.refAllDeclsRecursive(@This());
}

const OffsetTable = packed struct {
    scaler: u32,
    num_tables: u16,
    search_range: u16,
    entry_selector: u16,
    range_shift: u16,
};

const TableDirectoryEntry = extern struct {
    tag: [4]u8,
    check_sum: u32,
    offset: u32,
    length: u32,

    fn getData(self: TableDirectoryEntry, data: []const u8) []const u8 {
        return data[self.offset .. self.offset + self.length];
    }
};

const GenericParser = struct {
    data: []const u8,
    idx: usize = 0,

    pub fn readVal(self: *GenericParser, comptime T: type) T {
        const size = @bitSizeOf(T) / 8;
        defer self.idx += size;
        return fixEndianness(std.mem.bytesToValue(T, self.data[self.idx .. self.idx + size]));
    }

    pub fn readArray(self: *GenericParser, comptime T: type, alloc: Allocator, len: usize) ![]T {
        const size = @bitSizeOf(T) / 8 * len;
        defer self.idx += size;
        return fixSliceEndianness(T, alloc, @alignCast(std.mem.bytesAsSlice(T, self.data[self.idx .. self.idx + size])));
    }
};

const Fixed = packed struct(u32) {
    frac: i16,
    integer: i16,
};

const HeadTable = packed struct {
    version: Fixed,
    font_revision: Fixed,
    check_sum_adjustment: u32,
    magic_number: u32,
    flags: u16,
    units_per_em: u16,
    created: i64,
    modified: i64,
    x_min: i16,
    y_min: i16,
    x_max: i16,
    y_max: i16,
    mac_style: u16,
    lowest_rec_ppem: u16,
    font_direction_hint: i16,
    index_to_loc_format: i16,
    glyph_data_format: i16,
};

const MaxpTable = packed struct {
    version: Fixed,
    num_glyphs: u16,
    max_points: u16,
    max_contours: u16,
    max_component_points: u16,
    max_component_contours: u16,
    max_zones: u16,
    max_twilight_points: u16,
    max_storage: u16,
    max_function_defs: u16,
    max_instruction_defs: u16,
    maxStackElements: u16,
    maxSizeOfInstructions: u16,
    maxComponentElements: u16,
    maxComponentDepth: u16,
};

pub const CmapTable = struct {
    cmap_bytes: []const u8,

    const Index = packed struct {
        version: u16,
        num_subtables: u16,
    };

    const SubtableLookup = packed struct {
        platform_id: u16,
        platform_specific_id: u16,
        offset: u32,

        fn isUnicodeBmp(self: SubtableLookup) bool {
            return (self.platform_id == 0 and self.platform_specific_id == 3) or // unicode + bmp
                (self.platform_id == 3 and self.platform_specific_id == 1) // windows + unicode ucs 2
            ;
        }
    };

    const Subtable = packed struct {
        platform_id: u16,
        platform_specific_id: u16,
        offset: u32,
    };

    pub const SubtableFormat4 = struct {
        format: u16,
        length: u16,
        language: u16,
        seg_count_x2: u16,
        search_range: u16,
        entry_selector: u16,
        range_shift: u16,
        end_code: []const u16,
        reserved_pad: u16,
        start_code: []const u16,
        id_delta: []const u16,
        id_range_offset: []const u16,
        glyph_indices: []const u16,

        fn getGlyphIndex(self: SubtableFormat4, c: u16) u16 {
            var i: usize = 0;
            while (i < self.end_code.len) {
                if (self.end_code[i] > c) {
                    break;
                }
                i += 1;
            }
            if (i >= self.end_code.len) unreachable;
            // [ id range ] [glyph indices ]
            //     |--------------|
            //     i   offs_bytes
            const byte_offset_from_id_offset = self.id_range_offset[i];
            if (byte_offset_from_id_offset == 0) {
                return self.id_delta[i] +% c;
            } else {
                const offs_from_loc = byte_offset_from_id_offset / 2 + (c - self.start_code[i]);
                const dist_to_end = self.id_range_offset.len - i;
                const glyph_index_index = offs_from_loc - dist_to_end;
                return self.glyph_indices[glyph_index_index] +% self.id_delta[i];
            }
        }
    };

    fn readIndex(self: CmapTable) Index {
        return fixEndianness(std.mem.bytesToValue(Index, self.cmap_bytes[0 .. @bitSizeOf(Index) / 8]));
    }

    fn readSubtableLookup(self: CmapTable, idx: usize) SubtableLookup {
        const subtable_size = @bitSizeOf(SubtableLookup) / 8;
        const start = @bitSizeOf(Index) / 8 + idx * subtable_size;
        const end = start + subtable_size;

        return fixEndianness(std.mem.bytesToValue(SubtableLookup, self.cmap_bytes[start..end]));
    }

    fn readSubtableFormat(self: CmapTable, offset: usize) u16 {
        return fixEndianness(std.mem.bytesToValue(u16, self.cmap_bytes[offset .. offset + 2]));
    }

    fn readSubtableFormat4(self: CmapTable, alloc: Allocator, offset: usize) !SubtableFormat4 {
        var generic_parser = GenericParser{ .data = self.cmap_bytes[offset..] };
        const format = generic_parser.readVal(u16);
        const length = generic_parser.readVal(u16);
        const language = generic_parser.readVal(u16);
        const seg_count_x2 = generic_parser.readVal(u16);
        const search_range = generic_parser.readVal(u16);
        const entry_selector = generic_parser.readVal(u16);
        const range_shift = generic_parser.readVal(u16);
        const end_code: []const u16 = try generic_parser.readArray(u16, alloc, seg_count_x2 / 2);
        // fixme errdefer
        const reserved_pad = generic_parser.readVal(u16);
        // fixme errdefer
        const start_code: []const u16 = try generic_parser.readArray(u16, alloc, seg_count_x2 / 2);
        // fixme errdefer
        const id_delta: []const u16 = try generic_parser.readArray(u16, alloc, seg_count_x2 / 2);
        // fixme errdefer
        const id_range_offset: []const u16 = try generic_parser.readArray(u16, alloc, seg_count_x2 / 2);
        const glyph_indices: []const u16 = try generic_parser.readArray(u16, alloc, (generic_parser.data.len - generic_parser.idx) / 2);

        return .{
            .format = format,
            .length = length,
            .language = language,
            .seg_count_x2 = seg_count_x2,
            .search_range = search_range,
            .entry_selector = entry_selector,
            .range_shift = range_shift,
            .end_code = end_code,
            .reserved_pad = reserved_pad,
            .start_code = start_code,
            .id_delta = id_delta,
            .id_range_offset = id_range_offset,
            .glyph_indices = glyph_indices,
        };
    }
};

fn fixEndianness(val: anytype) @TypeOf(val) {
    if (builtin.cpu.arch.endian() == .big) {
        return val;
    }
    switch (@typeInfo(@TypeOf(val))) {
        .Struct => {
            var ret = val;
            std.mem.byteSwapAllFields(@TypeOf(val), &ret);
            return ret;
        },
        .Int => {
            return std.mem.bigToNative(@TypeOf(val), val);
        },
        inline else => @compileError("Cannot fix endianness for " ++ @typeName(@TypeOf(val))),
    }
}

fn fixSliceEndianness(comptime T: type, alloc: Allocator, slice: []const T) ![]T {
    const duped = try alloc.alloc(T, slice.len);
    for (0..slice.len) |i| {
        duped[i] = fixEndianness(slice[i]);
    }
    return duped;
}

const GlyphTable = struct {
    data: []const u8,

    const GlyphCommon = packed struct {
        number_of_contours: i16,
        x_min: i16,
        y_min: i16,
        x_max: i16,
        y_max: i16,
    };

    const SimpleGlyphFlag = packed struct(u8) {
        on_curve_point: bool,
        x_short_vector: bool,
        y_short_vector: bool,
        repeat_flag: bool,
        x_is_same_or_positive_x_short_vector: bool,
        y_is_same_or_positive_y_short_vector: bool,
        overlap_simple: bool,
        reserved: bool,
    };

    const GlyphParseVariant = enum {
        short_pos,
        short_neg,
        long,
        repeat,

        fn fromBools(short: bool, is_same_or_positive_short: bool) GlyphParseVariant {
            if (short) {
                if (is_same_or_positive_short) {
                    return .short_pos;
                } else {
                    return .short_neg;
                }
            } else {
                if (is_same_or_positive_short) {
                    return .repeat;
                } else {
                    return .long;
                }
            }
        }
    };

    const SimpleGlyph = struct {
        common: GlyphCommon,
        end_pts_of_contours: []u16,
        instruction_length: u16,
        instructions: []u8,
        flags: []SimpleGlyphFlag,
        x_coordinates: []i16,
        y_coordinates: []i16,
    };

    fn getGlyphCommon(self: GlyphTable, start: usize) GlyphCommon {
        return fixEndianness(std.mem.bytesToValue(GlyphCommon, self.data[start .. start + @bitSizeOf(GlyphCommon) / 8]));
    }

    fn getGlyphSimple(self: GlyphTable, alloc: Allocator, start: usize, end: usize) !SimpleGlyph {
        var generic_parser = GenericParser{ .data = self.data[start..end] };
        const common = generic_parser.readVal(GlyphCommon);
        const end_pts_of_contours = try generic_parser.readArray(u16, alloc, @intCast(common.number_of_contours));
        errdefer alloc.free(end_pts_of_contours);
        const instruction_length = generic_parser.readVal(u16);
        const instructions = try generic_parser.readArray(u8, alloc, instruction_length);
        errdefer alloc.free(instructions);

        const num_contours = end_pts_of_contours[end_pts_of_contours.len - 1] + 1;

        const flags = try alloc.alloc(SimpleGlyphFlag, num_contours);
        errdefer alloc.free(flags);

        var i: usize = 0;
        while (i < num_contours) {
            defer i += 1;
            const flag_u8 = generic_parser.readVal(u8);
            const flag: SimpleGlyphFlag = @bitCast(flag_u8);
            std.debug.assert(flag.reserved == false);

            flags[i] = flag;

            if (flag.repeat_flag) {
                const num_repetitions = generic_parser.readVal(u8);
                @memset(flags[i + 1 .. i + 1 + num_repetitions], flag);
                i += num_repetitions;
            }
        }

        const x_coords = try alloc.alloc(i16, num_contours);
        errdefer alloc.free(x_coords);
        for (flags, 0..) |flag, idx| {
            const parse_variant = GlyphParseVariant.fromBools(flag.x_short_vector, flag.x_is_same_or_positive_x_short_vector);
            switch (parse_variant) {
                .short_pos => x_coords[idx] = generic_parser.readVal(u8),
                .short_neg => x_coords[idx] = -@as(i16, generic_parser.readVal(u8)),
                .long => x_coords[idx] = generic_parser.readVal(i16),
                .repeat => x_coords[idx] = 0,
            }
        }

        const y_coords = try alloc.alloc(i16, num_contours);
        errdefer alloc.free(y_coords);
        for (flags, 0..) |flag, idx| {
            const parse_variant = GlyphParseVariant.fromBools(flag.y_short_vector, flag.y_is_same_or_positive_y_short_vector);
            switch (parse_variant) {
                .short_pos => y_coords[idx] = generic_parser.readVal(u8),
                .short_neg => y_coords[idx] = -@as(i16, generic_parser.readVal(u8)),
                .long => y_coords[idx] = generic_parser.readVal(i16),
                .repeat => y_coords[idx] = 0,
            }
        }

        return .{
            .common = common,
            .end_pts_of_contours = end_pts_of_contours,
            .instruction_length = instruction_length,
            .instructions = instructions,
            .flags = flags,
            .x_coordinates = x_coords,
            .y_coordinates = y_coords,
        };
    }
};

pub const Ttf = struct {
    const HeaderTag = enum {
        cmap,
        head,
        maxp,
        loca,
        glyf,
    };

    head: HeadTable,
    maxp: MaxpTable,
    cmap: CmapTable,
    // FIXME: Assert type
    loca: []const u32,
    glyf: GlyphTable,

    pub fn init(alloc: Allocator, font_data: []const u8) !Ttf {
        const offset_table = fixEndianness(std.mem.bytesToValue(OffsetTable, font_data[0 .. @bitSizeOf(OffsetTable) / 8]));
        const table_directory_start = @bitSizeOf(OffsetTable) / 8;
        const table_directory_end = table_directory_start + @bitSizeOf(TableDirectoryEntry) * offset_table.num_tables / 8;
        const tables = std.mem.bytesAsSlice(TableDirectoryEntry, font_data[table_directory_start..table_directory_end]);
        var head: ?HeadTable = null;
        var maxp: ?MaxpTable = null;
        var cmap: ?CmapTable = null;
        var glyf: ?GlyphTable = null;
        var loca: ?[]const u32 = null;

        for (tables) |table_big| {
            const table = fixEndianness(table_big);
            const tag = std.meta.stringToEnum(HeaderTag, &table.tag) orelse continue;
            switch (tag) {
                .head => {
                    head = fixEndianness(std.mem.bytesToValue(HeadTable, table.getData(font_data)));
                },
                .loca => {
                    loca = try fixSliceEndianness(u32, alloc, @alignCast(std.mem.bytesAsSlice(u32, table.getData(font_data))));
                },
                .maxp => {
                    maxp = fixEndianness(std.mem.bytesToValue(MaxpTable, table.getData(font_data)));
                },
                .cmap => {
                    cmap = CmapTable{ .cmap_bytes = table.getData(font_data) };
                },
                .glyf => {
                    glyf = GlyphTable{ .data = table.getData(font_data) };
                },
            }
        }

        return .{
            .maxp = maxp orelse unreachable,
            .head = head orelse unreachable,
            .loca = loca orelse unreachable,
            .cmap = cmap orelse unreachable,
            .glyf = glyf orelse unreachable,
        };
    }
};

pub const CharGenerator = struct {
    idx: u8 = 0,

    const num_letters = 26;

    pub fn next(self: *CharGenerator) ?u8 {
        if (self.idx >= 2 * num_letters) return null;

        defer self.idx += 1;
        if (self.idx < num_letters) {
            return 'A' + self.idx;
        } else {
            return 'a' + self.idx - num_letters;
        }
    }
};

const IVec2 = @Vector(2, i16);

pub const GlyphSegmentIter = struct {
    glyph: GlyphTable.SimpleGlyph,
    x_acc: i16 = 0,
    y_acc: i16 = 0,

    idx: usize = 0,
    contour_idx: usize = 0,
    last_contour_last_point: IVec2 = .{ 0, 0 },

    pub const Output = union(enum) {
        line: struct {
            a: IVec2,
            b: IVec2,
            contour_id: usize,
        },
        bezier: struct {
            a: IVec2,
            b: IVec2,
            c: IVec2,
            contour_id: usize,
        },
    };

    pub fn init(glyph: GlyphTable.SimpleGlyph) GlyphSegmentIter {
        return GlyphSegmentIter{
            .glyph = glyph,
        };
    }

    pub fn next(self: *GlyphSegmentIter) ?Output {
        while (true) {
            if (self.idx >= self.glyph.x_coordinates.len) return null;
            defer self.idx += 1;

            const a = self.getPoint(self.idx);

            defer self.x_acc = a.pos[0];
            defer self.y_acc = a.pos[1];

            const b = self.getPoint(self.idx + 1);
            const c = self.getPoint(self.idx + 2);

            const ret = abcToCurve(a, b, c, self.contour_idx);
            if (self.glyph.end_pts_of_contours[self.contour_idx] == self.idx) {
                self.contour_idx += 1;
                self.last_contour_last_point = a.pos;
            }

            if (ret) |val| {
                return val;
            }
        }
    }

    const Point = struct {
        on_curve: bool,
        pos: IVec2,
    };

    fn abcToCurve(a: Point, b: Point, c: Point, contour_idx: usize) ?Output {
        if (a.on_curve and b.on_curve) {
            return .{ .line = .{
                .a = a.pos,
                .b = b.pos,
                .contour_id = contour_idx,
            } };
        } else if (b.on_curve) {
            return null;
        }

        std.debug.assert(!b.on_curve);

        const a_on = resolvePoint(a, b);
        const c_on = resolvePoint(c, b);

        return .{ .bezier = .{
            .a = a_on,
            .b = b.pos,
            .c = c_on,
            .contour_id = contour_idx,
        } };
    }

    // FIXME: Stateful APIs using contour idx are gross
    fn contourStart(self: GlyphSegmentIter) usize {
        if (self.contour_idx == 0) {
            return 0;
        } else {
            return self.glyph.end_pts_of_contours[self.contour_idx - 1] + 1;
        }
    }

    fn wrappedContourIdx(self: GlyphSegmentIter, idx: usize) usize {
        const contour_start = self.contourStart();
        const contour_len = self.glyph.end_pts_of_contours[self.contour_idx] + 1 - contour_start;

        std.debug.print("idx: {d}, contour_start: {d}, contour_len: {d}\n", .{ idx, contour_start, contour_len });

        return (idx - contour_start) % contour_len + contour_start;
    }

    fn getPoint(self: *GlyphSegmentIter, idx: usize) Point {
        var x_acc = self.x_acc;
        var y_acc = self.y_acc;

        for (self.idx..idx + 1) |i| {
            const wrapped_i = self.wrappedContourIdx(i);
            std.debug.print("wrapped i: {d}\n", .{wrapped_i});
            if (wrapped_i == self.contourStart()) {
                std.debug.print("idx is 0, last point {any}\n", .{self.last_contour_last_point});
                x_acc = self.last_contour_last_point[0];
                y_acc = self.last_contour_last_point[1];
            }
            x_acc += self.glyph.x_coordinates[wrapped_i];
            y_acc += self.glyph.y_coordinates[wrapped_i];
        }

        const pos = IVec2{
            x_acc,
            y_acc,
        };

        const on_curve = self.glyph.flags[self.wrappedContourIdx(idx)].on_curve_point;
        return .{
            .on_curve = on_curve,
            .pos = pos,
        };
    }

    fn resolvePoint(maybe_off: Point, off: Point) IVec2 {
        if (maybe_off.on_curve) return maybe_off.pos;
        std.debug.assert(off.on_curve == false);

        return (maybe_off.pos + off.pos) / IVec2{ 2, 2 };
    }
};
pub const GlyphPointIter = struct {
    glyph: GlyphTable.SimpleGlyph,
    x_acc: i16 = 0,
    y_acc: i16 = 0,
    idx: usize = 0,

    pub const Output = struct {
        point: IVec2,
        on_curve: bool,
    };

    pub fn init(glyph: GlyphTable.SimpleGlyph) GlyphPointIter {
        return .{
            .glyph = glyph,
        };
    }

    pub fn next(self: *GlyphPointIter) ?Output {
        if (self.idx >= self.glyph.x_coordinates.len) return null;
        defer self.idx += 1;

        self.x_acc += self.glyph.x_coordinates[self.idx];
        self.y_acc += self.glyph.y_coordinates[self.idx];

        const pos = IVec2{
            self.x_acc,
            self.y_acc,
        };

        const on_curve = self.glyph.flags[self.idx].on_curve_point;
        return .{
            .on_curve = on_curve,
            .point = pos,
        };
    }
};

pub fn readSubtable(alloc: Allocator, ttf_parser: Ttf) !CmapTable.SubtableFormat4 {
    // Otherwise locs are the wrong size
    std.debug.assert(ttf_parser.head.index_to_loc_format == 1);
    std.debug.assert(ttf_parser.head.magic_number == 0x5F0F3CF5);

    const index = ttf_parser.cmap.readIndex();
    std.log.debug("cmap index: {any}\n", .{index});
    const unicode_table_offs = blk: {
        for (0..index.num_subtables) |i| {
            const subtable = ttf_parser.cmap.readSubtableLookup(i);
            if (subtable.isUnicodeBmp()) {
                break :blk subtable.offset;
            }
        }
        return error.NoUnicodeBmpTables;
    };

    const format = ttf_parser.cmap.readSubtableFormat(unicode_table_offs);
    if (format != 4) {
        std.log.err("Can only handle unicode format 4", .{});
        return error.Unimplemented;
    }

    return try ttf_parser.cmap.readSubtableFormat4(alloc, unicode_table_offs);
}

pub fn glyphForChar(alloc: Allocator, ttf_parser: Ttf, subtable: CmapTable.SubtableFormat4, char: u16) !GlyphTable.SimpleGlyph {
    const glyph_index = subtable.getGlyphIndex(char);
    const glyf_start = ttf_parser.loca[glyph_index];
    const glyf_end = ttf_parser.loca[glyph_index + 1];

    const glyph_header = ttf_parser.glyf.getGlyphCommon(glyf_start);
    std.log.debug("{any}\n", .{glyph_header});

    std.debug.assert(glyph_header.number_of_contours >= 0);
    return try ttf_parser.glyf.getGlyphSimple(alloc, glyf_start, glyf_end);
}

pub const Vec2 = @Vector(2, f32);

pub fn sampleQuadBezierCurve(a: Vec2, b: Vec2, c: Vec2, t: f32) Vec2 {
    const t_splat: Vec2 = @splat(t);
    const ab = std.math.lerp(a, b, t_splat);
    const bc = std.math.lerp(b, c, t_splat);

    return std.math.lerp(ab, bc, t_splat);
}

pub fn flipY(in: IVec2, height: i16) IVec2 {
    return .{ in[0], height - in[1] };
}

const BBox = struct {
    const invalid = BBox{
        .min_x = std.math.maxInt(i64),
        .max_x = std.math.minInt(i64),
        .min_y = std.math.maxInt(i64),
        .max_y = std.math.minInt(i64),
    };

    min_x: i64,
    max_x: i64,
    min_y: i64,
    max_y: i64,
};

fn pointsBounds(points: []const IVec2) BBox {
    var ret = BBox.invalid;

    for (points) |point| {
        ret.min_x = @min(point[0], ret.min_x);
        ret.min_y = @min(point[1], ret.min_x);
        ret.max_x = @max(point[0], ret.max_x);
        ret.max_y = @max(point[1], ret.max_x);
    }

    return ret;
}

fn curveBounds(curve: GlyphSegmentIter.Output) BBox {
    switch (curve) {
        .line => |l| {
            return pointsBounds(&.{ l.a, l.b });
        },
        .bezier => |b| {
            return pointsBounds(&.{ b.a, b.b, b.c });
        },
    }
}

fn mergeBboxes(a: BBox, b: BBox) BBox {
    return .{
        .min_x = @min(a.min_x, b.min_x),
        .max_x = @max(a.max_x, b.max_x),
        .min_y = @min(a.min_y, b.min_y),
        .max_y = @max(a.max_y, b.max_y),
    };
}

const RowCurvePointInner = struct {
    x_pos: i64,
    entering: bool,
    contour_id: usize,

    pub fn format(value: RowCurvePointInner, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print("{d} ({}, {})", .{ value.x_pos, value.entering, value.contour_id });
    }
};

const RowCurvePoint = struct {
    x_pos: i64,
    entering: bool,

    pub fn format(value: RowCurvePoint, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print("{d} ({})", .{ value.x_pos, value.entering });
    }
};

fn sortRemoveDuplicateCurvePoints(alloc: Allocator, points: *std.ArrayList(RowCurvePointInner)) !void {
    var to_remove = std.ArrayList(usize).init(alloc);
    defer to_remove.deinit();
    for (0..points.items.len) |i| {
        const next_idx = (i + 1) % points.items.len;
        if (points.items[i].entering == points.items[next_idx].entering and points.items[i].contour_id == points.items[next_idx].contour_id) {
            if (points.items[i].entering) {
                if (points.items[i].x_pos > points.items[next_idx].x_pos) {
                    try to_remove.append(i);
                } else {
                    try to_remove.append(next_idx);
                }
            } else {
                if (points.items[i].x_pos > points.items[next_idx].x_pos) {
                    try to_remove.append(next_idx);
                } else {
                    try to_remove.append(i);
                }
            }
        }
    }

    while (to_remove.popOrNull()) |i| {
        if (points.items.len == 1) break;
        _ = points.swapRemove(i);
    }

    const lessThan = struct {
        fn f(_: void, lhs: RowCurvePointInner, rhs: RowCurvePointInner) bool {
            if (lhs.x_pos == rhs.x_pos) {
                return lhs.entering and !rhs.entering;
            }
            return lhs.x_pos < rhs.x_pos;
        }
    }.f;
    std.mem.sort(RowCurvePointInner, points.items, {}, lessThan);
}

fn findRowCurvePoints(alloc: Allocator, curves: []const GlyphSegmentIter.Output, y: i64) ![]RowCurvePoint {
    var ret = std.ArrayList(RowCurvePointInner).init(alloc);
    defer ret.deinit();

    for (curves) |curve| {
        switch (curve) {
            .line => |l| {
                const a_f: @Vector(2, f32) = @floatFromInt(l.a);
                const b_f: @Vector(2, f32) = @floatFromInt(l.b);
                const y_f: f32 = @floatFromInt(y);

                if (l.b[1] == l.a[1]) continue;
                const t = (y_f - a_f[1]) / (b_f[1] - a_f[1]);

                if (!(t >= 0.0 and t <= 1.0)) {
                    std.debug.print("Discarding point on line {any} -> {any}, t: {d}\n", .{ l.a, l.b, t });
                    continue;
                }

                const x = std.math.lerp(a_f[0], b_f[0], t);

                const x_pos_i: i64 = @intFromFloat(@round(x));
                const entering = l.a[1] < l.b[1];

                try ret.append(.{ .entering = entering, .x_pos = x_pos_i, .contour_id = l.contour_id });
            },
            .bezier => |b| {
                const a_f: @Vector(2, f32) = @floatFromInt(b.a);
                const b_f: @Vector(2, f32) = @floatFromInt(b.b);
                const c_f: @Vector(2, f32) = @floatFromInt(b.c);

                const ts = Renderer.findBezierTForY(a_f[1], b_f[1], c_f[1], @floatFromInt(y));

                for (ts, 0..) |t, i| {
                    if (!(t >= 0.0 and t <= 1.0)) {
                        continue;
                    }
                    const tangent_line = Renderer.quadBezierTangentLine(a_f, b_f, c_f, t);

                    const eps = 1e-7;
                    const at_apex = @abs(tangent_line.a[1] - tangent_line.b[1]) < eps;
                    const at_end = t < eps or @abs(t - 1.0) < eps;
                    const moving_up = tangent_line.a[1] < tangent_line.b[1] or b.a[1] < b.c[1];

                    // If we are at the apex, and at the very edge of a curve,
                    // we have to be careful. In this case we can only count
                    // one of the enter/exit events as we are only half of the
                    // parabola.
                    //
                    // U -> enter/exit
                    // \_ -> enter
                    // _/ -> exit
                    //  _
                    // / -> enter
                    // _
                    //  \-> exit

                    // The only special case is that we are at the apex, and at
                    // the end of the curve. In this case we only want to
                    // consider one of the two points. Otherwise we just ignore
                    // the apex as it's an immediate enter/exit. I.e. useless
                    //
                    // This boils down to the following condition
                    if (at_apex and (!at_end or i == 1)) continue;

                    const x_f = sampleQuadBezierCurve(a_f, b_f, c_f, t)[0];
                    const x_px: i64 = @intFromFloat(@round(x_f));
                    try ret.append(.{
                        .entering = moving_up,
                        .x_pos = x_px,
                        .contour_id = b.contour_id,
                    });
                }
            },
        }
    }

    try sortRemoveDuplicateCurvePoints(alloc, &ret);

    const real_ret = try alloc.alloc(RowCurvePoint, ret.items.len);
    for (0..ret.items.len) |i| {
        real_ret[i] = .{
            .entering = ret.items[i].entering,
            .x_pos = ret.items[i].x_pos,
        };
    }

    return real_ret;
}

test "find row points V" {
    // Double counted point at the apex of a V, should immediately go in and out
    const curves = [_]GlyphSegmentIter.Output{
        .{
            .line = .{
                .a = .{ -1.0, 1.0 },
                .b = .{ 0.0, 0.0 },
                .contour_id = 0,
            },
        },
        .{
            .line = .{
                .a = .{ 0.0, 0.0 },
                .b = .{ 1.0, 1.0 },
                .contour_id = 0,
            },
        },
    };

    const points = try findRowCurvePoints(std.testing.allocator, &curves, 0);
    defer std.testing.allocator.free(points);

    try std.testing.expectEqualSlices(
        RowCurvePoint,
        &.{
            .{
                .x_pos = 0,
                .entering = true,
            },
            .{
                .x_pos = 0,
                .entering = false,
            },
        },
        points,
    );
}

test "find row points X" {
    // Double entry and exit on the horizontal part where there's wraparound
    const curves = [_]GlyphSegmentIter.Output{
        .{
            .line = .{
                .a = .{ 5, 0 },
                .b = .{ 10, -10 },
                .contour_id = 0,
            },
        },
        .{
            .line = .{
                .a = .{ -10, -10 },
                .b = .{ -5, 0 },
                .contour_id = 0,
            },
        },
        .{
            .line = .{
                .a = .{ -5, 0 },
                .b = .{ -10, 10 },
                .contour_id = 0,
            },
        },
        .{
            .line = .{
                .a = .{ 10, 10 },
                .b = .{ 5, 0 },
                .contour_id = 0,
            },
        },
    };

    const points = try findRowCurvePoints(std.testing.allocator, &curves, 0);
    defer std.testing.allocator.free(points);

    try std.testing.expectEqualSlices(
        RowCurvePoint,
        &.{
            .{
                .x_pos = -5,
                .entering = true,
            },
            .{
                .x_pos = 5,
                .entering = false,
            },
        },
        points,
    );
}

test "find row points G" {
    // G has segment that goes
    //
    // |       ^
    // v____   |
    //      |  |
    //      v  |
    //
    // In this case we somehow have to avoid double counting the down arrow
    //

    const curves = [_]GlyphSegmentIter.Output{
        .{
            .line = .{
                .a = .{ 0, -5 },
                .b = .{ 0, 0 },
                .contour_id = 0,
            },
        },
        .{
            .line = .{
                .a = .{ 0, 0 },
                .b = .{ 5, 0 },
                .contour_id = 0,
            },
        },
        .{
            .line = .{
                .a = .{ 5, 0 },
                .b = .{ 5, 5 },
                .contour_id = 0,
            },
        },
        .{
            .line = .{
                .a = .{ 5, 5 },
                .b = .{ 10, 5 },
                .contour_id = 0,
            },
        },
        .{
            .line = .{
                .a = .{ 10, 5 },
                .b = .{ 10, -5 },
                .contour_id = 0,
            },
        },
    };

    const points = try findRowCurvePoints(std.testing.allocator, &curves, 0);
    defer std.testing.allocator.free(points);

    try std.testing.expectEqualSlices(
        RowCurvePoint,
        &.{
            .{
                .x_pos = 0,
                .entering = true,
            },
            .{
                .x_pos = 10,
                .entering = false,
            },
        },
        points,
    );
}

test "find row points horizontal line into bezier cw" {
    // Bottom inside of one of the holes in the letter B
    // shape like ___/. Here we want to ensure that after we exit the
    // quadratic, we have determined that we are outside the curve
    const curves = [_]GlyphSegmentIter.Output{
        .{
            .bezier = .{
                .a = .{ 855, 845 },
                .b = .{ 755, 713 },
                .c = .{ 608, 713 },
                .contour_id = 0,
            },
        },
        .{
            .line = .{
                .a = .{ 608, 713 },
                .b = .{ 369, 713 },
                .contour_id = 0,
            },
        },
        .{
            .line = .{
                .a = .{ 369, 713 },
                .b = .{ 369, 800 },
                .contour_id = 0,
            },
        },
    };

    const points = try findRowCurvePoints(std.testing.allocator, &curves, 713);
    defer std.testing.allocator.free(points);

    try std.testing.expectEqualSlices(
        RowCurvePoint,
        &.{
            .{
                .x_pos = 369,
                .entering = true,
            },
            .{
                .x_pos = 608,
                .entering = false,
            },
        },
        points,
    );
}

test "find row points bezier apex matching" {
    // Top of a C. There are two bezier curves that run into eachother at the
    // apex with a tangent of 0. This should result in an immediate in/out
    const curves = [_]GlyphSegmentIter.Output{
        .{
            .bezier = .{
                .a = .{ 350, 745 },
                .b = .{ 350, 135 },
                .c = .{ 743, 135 },
                .contour_id = 0,
            },
        },
        .{
            .bezier = .{
                .a = .{ 743, 135 },
                .b = .{ 829, 135 },
                .c = .{ 916, 167 },
                .contour_id = 0,
            },
        },
    };

    const points = try findRowCurvePoints(std.testing.allocator, &curves, 135);
    defer std.testing.allocator.free(points);

    try std.testing.expectEqualSlices(
        RowCurvePoint,
        &.{
            .{
                .x_pos = 743,
                .entering = true,
            },
            .{
                .x_pos = 743,
                .entering = false,
            },
        },
        points,
    );
}

test "find row points ascending line segments" {
    // Double counted point should be deduplicated as it's going in the same direction
    //
    // e.g. the o will be in two segments, but should count as one line cross
    //     /
    //    /
    //   o
    //  /
    // /

    const curves = [_]GlyphSegmentIter.Output{
        .{
            .line = .{
                .a = .{ 0, 0 },
                .b = .{ 1, 1 },
                .contour_id = 0,
            },
        },
        .{
            .line = .{
                .a = .{ 1, 1 },
                .b = .{ 2, 2 },
                .contour_id = 0,
            },
        },
    };

    const points = try findRowCurvePoints(std.testing.allocator, &curves, 1);
    defer std.testing.allocator.free(points);

    try std.testing.expectEqualSlices(
        RowCurvePoint,
        &.{
            .{
                .x_pos = 1,
                .entering = true,
            },
        },
        points,
    );
}

test "find row points bezier curve into line" {
    // In the following case
    //  |<-----
    //  |      \
    //  v
    //
    // If we end up on the horizontal line, we should see an exit followed by entry (counter clockwise)
    //
    const curves = [_]GlyphSegmentIter.Output{
        .{
            .bezier = .{
                .a = .{ 5, 0 },
                .b = .{ 5, 1 },
                .c = .{ 3, 1 },
                .contour_id = 0,
            },
        },
        .{
            .line = .{
                .a = .{ 3, 1 },
                .b = .{ 1, 1 },
                .contour_id = 0,
            },
        },
        .{
            .line = .{
                .a = .{ 1, 1 },
                .b = .{ 1, 0 },
                .contour_id = 0,
            },
        },
    };

    const points = try findRowCurvePoints(std.testing.allocator, &curves, 1);
    defer std.testing.allocator.free(points);

    try std.testing.expectEqualSlices(
        RowCurvePoint,
        &.{
            .{
                .x_pos = 1,
                .entering = false,
            },
            .{
                .x_pos = 3,
                .entering = true,
            },
        },
        points,
    );
}

fn renderChar(alloc: Allocator, c: u8, ttf_parser: Ttf, subtable: CmapTable.SubtableFormat4, renderer: *Renderer, out_dir: std.fs.Dir) !void {
    const out_w = renderer.iWidth();
    const out_h = renderer.calcHeight();
    renderer.drawRect(0, out_w, 0, out_h, .{ .r = 50, .g = 50, .b = 50, .a = 255 });
    const simple_glyph = try glyphForChar(alloc, ttf_parser, subtable, c);
    //const offs = IVec2{ 200, 200 };
    var iter = GlyphSegmentIter.init(simple_glyph);

    var curves = std.ArrayList(GlyphSegmentIter.Output).init(alloc);
    defer curves.deinit();

    var total_bbox = BBox.invalid;
    while (iter.next()) |item| {
        try curves.append(item);
        const item_bbox = curveBounds(item);
        total_bbox = mergeBboxes(total_bbox, item_bbox);
    }
    std.debug.print("object bbox: {any}\n", .{total_bbox});

    for (curves.items) |curve| {
        switch (curve) {
            .line => |l| {
                std.debug.print("line: {any} {any}\n", .{ l.a, l.b });
            },
            .bezier => |b| {
                std.debug.print("bezier: {any} {any} {any}\n", .{ b.a, b.b, b.c });
            },
        }
    }

    var y = total_bbox.min_y;
    while (y < total_bbox.max_y) {
        defer y += 1;
        const flipped_y = renderer.calcHeight() - y - 300;
        if (flipped_y >= renderer.calcHeight() or flipped_y < 0) continue;
        //if (y >= renderer.calcHeight() or y < 0) continue;
        const row = renderer.getRow(flipped_y);

        const row_curve_points = try findRowCurvePoints(alloc, curves.items, y);
        defer alloc.free(row_curve_points);

        std.debug.print("y: {d}, points: {any}\n", .{ y, row_curve_points });

        var winding_count: i64 = 0;
        var start: i64 = 0;
        for (row_curve_points) |point| {
            if (point.entering == false) {
                winding_count -= 1;
            } else {
                winding_count += 1;
                if (winding_count == 1) {
                    start = point.x_pos;
                }
            }
            // NOTE: Always see true first due to sorting
            if (winding_count == 0) {
                @memset(row[@intCast(renderer.clampedX(start))..@intCast(renderer.clampedX(point.x_pos))], .{ .r = 255, .g = 0, .b = 255, .a = 255 });
            }
        }
    }
    //std.log.debug("{any}", .{item});
    //const color = Renderer.Color{ .r = 0, .g = 0, .b = 0, .a = 255 };
    //const green = Renderer.Color{ .r = 0, .g = 255, .b = 0, .a = 255 };
    //const width = 5;
    ////const radius = 25;
    //switch (item) {
    //    .line => |l| {
    //        const a = flipY(l.a + offs, @intCast(out_h));
    //        const b = flipY(l.b + offs, @intCast(out_h));
    //        //renderer.drawCircle(a[0], a[1], radius, color);
    //        renderer.drawLine(a[0], a[1], b[0], b[1], width, color);
    //    },
    //    .bezier => |curve| {
    //        const a = flipY(curve.a + offs, @intCast(out_h));
    //        const b = flipY(curve.b + offs, @intCast(out_h));
    //        const point_c = flipY(curve.c + offs, @intCast(out_h));
    //        renderer.drawBezier(a, b, point_c, green);
    //        //renderer.drawCircle(a[0], a[1], radius, color);
    //        //renderer.drawLine(a[0], a[1], b[0], b[1], width, green);
    //        //renderer.drawLine(b[0], b[1], point_c[0], point_c[1], width, green);
    //        //renderer.drawLine(a[0], a[1], point_c[0], point_c[1], width, color);
    //    },
    //}
    //}
    //var point_iter = GlyphPointIter.init(simple_glyph);
    //const on_color = Renderer.Color{.r = 255, .g = 0, .b = 0, .a = 255};
    //const off_color = Renderer.Color{.r = 0, .g = 0, .b = 255, .a = 255};
    //while (point_iter.next()) |point| {
    //    const color = if (point.on_curve) on_color else off_color;
    //    const point_offs = flipY(point.point + offs, @intCast(out_h));
    //    renderer.drawCircle(point_offs[0], point_offs[1], 25.0, color);
    //}

    const fname = try std.fmt.allocPrint(alloc, "{c}.ppm", .{c});
    defer alloc.free(fname);
    const output = try out_dir.createFile(fname, .{});
    defer output.close();
    var output_buffered = std.io.bufferedWriter(output.writer());

    try renderer.writePpm(output_buffered.writer());
    try output_buffered.flush();
}

pub fn main() !void {
    const x: GlyphTable.SimpleGlyphFlag = @bitCast(@as(u8, 0x1));
    std.debug.print("{any}\n", .{x});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    //defer _ = gpa.deinit();

    const font_data = @embedFile("res/Hack-Regular.ttf");
    const alloc = gpa.allocator();

    const ttf_parser = try Ttf.init(alloc, font_data);
    const subtable = try readSubtable(alloc, ttf_parser);

    var chars = CharGenerator{};
    const out_w = 2048;
    const out_h = 2048;
    var renderer = try Renderer.init(alloc, out_w, out_h);
    defer renderer.deinit(alloc);

    std.fs.cwd().makeDir("out") catch {};
    var out_dir = try std.fs.cwd().openDir("out", .{});
    defer out_dir.close();

    //try renderChar(alloc, 'G', ttf_parser, subtable, &renderer, out_dir);
    while (chars.next()) |c| {
        try renderChar(alloc, c, ttf_parser, subtable, &renderer, out_dir);
    }

    //const num_countours = std.mem.bigToNative(i16, std.mem.bytesToValue(i16, ttf_parser.glyf.data[glyf_start..glyf_start + 2]));
    //std.log.debug("glyph index for {c} is {d} ({d}..{d}), {d} contours\n", .{char, glyph_index, ttf_parser.loca[glyph_index], ttf_parser.loca[glyph_index + 1], num_countours});

}
