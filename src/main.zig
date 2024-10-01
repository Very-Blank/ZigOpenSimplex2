const std = @import("std");

const GRADIENTS2: [16]i64 = .{
    5,  2,  2,  5,
    -5, 2,  -2, 5,
    5,  -2, 2,  -5,
    -5, -2, -2, -5,
};

const STRETCH_CONSTANT2 = (1 / @sqrt(2 + 1) - 1) / 2;
const SQUISH_CONSTANT2 = (@sqrt(2 + 1) - 1) / 2;

pub fn main() !void {
    std.debug.print("All your {s} are belong to us.\n", .{"codebase"});
}
