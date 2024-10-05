const std = @import("std");
const fast = @import("fast.zig");
pub fn main() !void {
    const noise = try fast.Noise4.init(340340, std.heap.page_allocator);
    defer noise.deInit();
    std.debug.print("{any}\n", .{noise.improveXYZW(50.0, 50.0, 100.0, 100.0)});
}
