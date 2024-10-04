const std = @import("std");
const fast = @import("fast.zig");
pub fn main() !void {
    std.debug.print("{any}\n", .{fast.noise2(34334, 50.0, 50.0)});
}
