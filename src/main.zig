const std = @import("std");
const fast = @import("fast.zig");
pub fn main() !void {
    const noise = try fast.Noise2.init(340340, std.heap.page_allocator);
    defer noise.deInit();
    std.debug.print("{any}\n", .{noise.get(500000.0, 500000.0)});
}
