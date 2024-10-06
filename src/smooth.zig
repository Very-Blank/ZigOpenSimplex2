const std = @import("std");

//K.jpg's OpenSimplex 2, Smooth variant
const PRIME_X: i64 = 0x5205402B9270C86F;
const PRIME_Y: i64 = 0x598CD327003817B5;
const PRIME_Z: i64 = 0x5BCC226E9FA0BACB;
const PRIME_W: i64 = 0x56CC5227E58F554B;
const HASH_MULTIPLIER: i64 = 0x53A3F72DEEC546F5;
const SEED_FLIP_3D: i64 = -0x52D547B2E96ED629;

const ROOT2OVER2: f64 = 0.7071067811865476;
const SKEW_2D: f64 = 0.366025403784439;
const UNSKEW_2D: f64 = -0.21132486540518713;

const ROOT3OVER3: f64 = 0.577350269189626;
const FALLBACK_ROTATE_3D: f64 = 2.0 / 3.0;
const ROTATE_3D_ORTHOGONALIZER: f64 = UNSKEW_2D;

const SKEW_4D: f32 = 0.309016994374947;
const UNSKEW_4D: f32 = -0.138196601125011;

const N_GRADS_2D_EXPONENT: i32 = 7;
const N_GRADS_3D_EXPONENT: i32 = 8;
const N_GRADS_4D_EXPONENT: i32 = 9;
const N_GRADS_2D: i32 = 1 << N_GRADS_2D_EXPONENT;
const N_GRADS_3D: i32 = 1 << N_GRADS_3D_EXPONENT;
const N_GRADS_4D: i32 = 1 << N_GRADS_4D_EXPONENT;

const NORMALIZER_2D: f64 = 0.05481866495625118;
const NORMALIZER_3D: f64 = 0.2781926117527186;
const NORMALIZER_4D: f64 = 0.11127401889945551;

const RSQUARED_2D: f32 = 2.0 / 3.0;
const RSQUARED_3D: f32 = 3.0 / 4.0;
const RSQUARED_4D: f32 = 4.0 / 5.0;

// Noise Evaluators
pub const Noise2 = struct {
    seed: i64,
    grads: *[N_GRADS_2D * 2]f32,
    allocator: std.mem.Allocator,

    pub fn init(seed: i64, allocator: std.mem.Allocator) !Noise2 {
        const noise = Noise2{
            .seed = seed,
            .grads = try allocator.create([N_GRADS_2D * 2]f32),
            .allocator = allocator,
        };

        for (0..N_GRADS_2D * 2) |i| {
            const grad_index = i % GRAD2_SRC.len;
            noise.grads.*[i] = @floatCast(GRAD2_SRC[grad_index] / NORMALIZER_2D);
        }

        return noise;
    }

    pub fn deInit(self: *const Noise2) void {
        self.allocator.destroy(self.grads);
    }

    // 2D Simplex noise, standard lattice orientation.
    pub fn standard(self: *const Noise2, x: f64, y: f64) f32 {
        // Get points for A2* lattice
        const s: f64 = SKEW_2D * (x + y);
        const xs: f64 = x + s;
        const ys: f64 = y + s;

        return self.unskewedBase(xs, ys);
    }

    // 2D Simplex noise, with Y pointing down the main diagonal.
    // Might be better for a 2D sandbox style game, where Y is vertical.
    // Probably slightly less optimal for heightmaps or continent maps,
    // unless your map is centered around an equator. It's a subtle
    // difference, but the option is here to make it an easy choice.
    pub fn improveX(self: *const Noise2, x: f64, y: f64) f32 {
        // Skew transform and rotation baked into one.
        const xx: f64 = x * ROOT2OVER2;
        const yy: f64 = y * (ROOT2OVER2 * (1.0 + 2.0 * SKEW_2D));

        return self.unskewedBase(yy + xx, yy - xx);
    }

    // 2D Simplex noise base.
    fn unskewedBase(self: *const Noise2, xs: f64, ys: f64) f32 {
        // Get base points and offsets.
        const xsb: i32 = fastFloor(xs);
        const ysb: i32 = fastFloor(ys);
        const xi: f32 = @floatCast(xs - @as(f64, @intFromFloat(xsb)));
        const yi: f32 = @floatCast(ys - @as(f64, @intFromFloat(ysb)));

        // Prime pre-multiplication for hash.
        const xsbp = @as(i64, @intCast(xsb)) *% PRIME_X;
        const ysbp = @as(i64, @intCast(ysb)) *% PRIME_Y;

        // Unskew.
        const t: f32 = (xi + yi) * @as(f32, @floatCast(UNSKEW_2D));
        const dx0: f32 = xi + t;
        const dy0: f32 = yi + t;

        // First vertex.
        const a0: f32 = RSQUARED_2D - dx0 * dx0 - dy0 * dy0;
        var value: f32 = (a0 * a0) * (a0 * a0) * self.grad2(xsbp, ysbp, dx0, dy0);

        // Second vertex.
        const a1: f32 = @as(f32, @floatCast((2.0 * (1.0 + 2.0 * UNSKEW_2D) * (1.0 / UNSKEW_2D + 2.0)))) * t + (@as(f32, @floatCast((-2.0 * (1.0 + 2.0 * UNSKEW_2D) * (1.0 + 2.0 * UNSKEW_2D)))) + a0);
        const dx1 = dx0 - (1.0 + 2.0 * UNSKEW_2D);
        const dy1 = dy0 - (1.0 + 2.0 * UNSKEW_2D);
        value += (a1 * a1) * (a1 * a1) * self.grad2(
            xsbp +% PRIME_X,
            ysbp +% PRIME_Y,
            dx1,
            dy1,
        );

        // Third and fourth vertices.
        // Nested conditionals were faster than compact bit logic/arithmetic.
        const xmyi: f32 = xi - yi;
        if (t < @as(f32, @floatCast(UNSKEW_2D))) {
            if (xi + xmyi > 1.0) {
                const dx2: f32 = dx0 - @as(f32, @floatCast((3.0 * UNSKEW_2D + 2.0)));
                const dy2: f32 = dy0 - @as(f32, @floatCast((3.0 * UNSKEW_2D + 2.0)));
                const a2: f32 = RSQUARED_2D - dx2 * dx2 - dy2 * dy2;
                if (a2 > 0.0) {
                    value += (a2 * a2) * (a2 * a2) * self.grad2(
                        xsbp +% PRIME_X << 1,
                        ysbp +% PRIME_Y,
                        dx2,
                        dy2,
                    );
                }
            } else {
                const dx2: f32 = dx0 - @as(f32, @floatCast(UNSKEW_2D));
                const dy2: f32 = dy0 - @as(f32, @floatCast(UNSKEW_2D));
                const a2: f32 = RSQUARED_2D - dx2 * dx2 - dy2 * dy2;
                if (a2 > 0.0) {
                    value +=
                        (a2 * a2) * (a2 * a2) * self.grad2(xsbp, ysbp +% PRIME_Y, dx2, dy2);
                }
            }

            if (yi - xmyi > 1.0) {
                const dx3: f32 = dx0 - @as(f32, @floatCast((3.0 * UNSKEW_2D + 1.0)));
                const dy3: f32 = dy0 - @as(f32, @floatCast((3.0 * UNSKEW_2D + 2.0)));
                const a3: f32 = RSQUARED_2D - dx3 * dx3 - dy3 * dy3;
                if (a3 > 0.0) {
                    value += (a3 * a3) * (a3 * a3) * self.grad2(
                        xsbp +% PRIME_X,
                        ysbp +% PRIME_Y << 1,
                        dx3,
                        dy3,
                    );
                }
            } else {
                const dx3 = dx0 - @as(f32, @floatCast(UNSKEW_2D + 1.0));
                const dy3 = dy0 - @as(f32, @floatCast(UNSKEW_2D));
                const a3 = RSQUARED_2D - dx3 * dx3 - dy3 * dy3;
                if (a3 > 0.0) {
                    value +=
                        (a3 * a3) * (a3 * a3) * self.grad2(xsbp +% PRIME_X, ysbp, dx3, dy3);
                }
            }
        } else {
            if (xi + xmyi < 0.0) {
                const dx2: f32 = dx0 + @as(f32, @floatCast(UNSKEW_2D + 1.0));
                const dy2: f32 = dy0 + @as(f32, @floatCast(UNSKEW_2D));
                const a2: f32 = RSQUARED_2D - dx2 * dx2 - dy2 * dy2;
                if (a2 > 0.0) {
                    value +=
                        (a2 * a2) * (a2 * a2) * self.grad2(xsbp -% PRIME_X, ysbp, dx2, dy2);
                }
            } else {
                const dx2: f32 = dx0 + @as(f32, @floatCast(UNSKEW_2D + 1.0));
                const dy2: f32 = dy0 + @as(f32, @floatCast(UNSKEW_2D));
                const a2: f32 = RSQUARED_2D - dx2 * dx2 - dy2 * dy2;
                if (a2 > 0.0) {
                    value +=
                        (a2 * a2) * (a2 * a2) * self.grad2(xsbp +% PRIME_X, ysbp, dx2, dy2);
                }
            }

            if (yi < xmyi) {
                const dx2: f32 = dx0 + @as(f32, @floatCast(UNSKEW_2D));
                const dy2: f32 = dy0 + @as(f32, @floatCast(UNSKEW_2D + 1.0));
                const a2: f32 = RSQUARED_2D - dx2 * dx2 - dy2 * dy2;
                if (a2 > 0.0) {
                    value +=
                        (a2 * a2) * (a2 * a2) * self.grad2(xsbp, ysbp -% PRIME_Y, dx2, dy2);
                }
            } else {
                const dx2: f32 = dx0 + @as(f32, @floatCast(UNSKEW_2D));
                const dy2: f32 = dy0 + @as(f32, @floatCast(UNSKEW_2D + 1.0));
                const a2: f32 = RSQUARED_2D - dx2 * dx2 - dy2 * dy2;
                if (a2 > 0.0) {
                    value +=
                        (a2 * a2) * (a2 * a2) * self.grad2(xsbp, ysbp +% PRIME_Y, dx2, dy2);
                }
            }
        }

        return value;
    }

    fn grad2(self: *const Noise2, xsvp: i64, ysvp: i64, dx: f32, dy: f32) f32 {
        var hash: i64 = self.seed ^ xsvp ^ ysvp;
        hash *%= HASH_MULTIPLIER;
        hash ^= hash >> (64 - N_GRADS_2D_EXPONENT + 1);
        const gi: usize = @intCast(hash & ((@as(i64, @intCast(N_GRADS_2D)) - 1) << 1));
        return @floatCast(self.grads.*[gi | 0] * dx + self.grads.*[gi | 1] * dy);
    }
};

// 3D OpenSimplex2 noise, with better visual isotropy in (X, Y).
// Recommended for 3D terrain and time-varied animations.
// The Z coordinate should always be the "different" coordinate in whatever your use case is.
// If Y is vertical in world coordinates, call noise3_ImproveXZ(x, z, Y) or use noise3_XZBeforeY.
// If Z is vertical in world coordinates, call noise3_ImproveXZ(x, y, Z).
// For a time varied animation, call noise3_ImproveXY(x, y, T).
pub const Noise3 = struct {
    seed: i64,
    grads: *[N_GRADS_3D * 4]f32,
    allocator: std.mem.Allocator,

    pub fn init(seed: i64, allocator: std.mem.Allocator) !Noise3 {
        const noise = Noise3{
            .seed = seed,
            .grads = try allocator.create([N_GRADS_3D * 4]f32),
            .allocator = allocator,
        };

        for (0..N_GRADS_3D * 4) |i| {
            const grad_index = i % GRAD3_SRC.len;
            noise.grads.*[i] = @floatCast(GRAD3_SRC[grad_index] / NORMALIZER_3D);
        }

        return noise;
    }

    pub fn deInit(self: *const Noise3) void {
        self.allocator.destroy(self.grads);
    }

    pub fn improveXY(self: *const Noise3, x: f64, y: f64, z: f64) f32 {
        // Re-orient the cubic lattices without skewing, so Z points up the main lattice diagonal,
        // and the planes formed by XY are moved far out of alignment with the cube faces.
        // Orthonormal rotation. Not a skew transform.
        const xy: f64 = x + y;
        const s2: f64 = xy * ROTATE_3D_ORTHOGONALIZER;
        const zz: f64 = z * ROOT3OVER3;
        const xr: f64 = x + s2 + zz;
        const yr: f64 = y + s2 + zz;
        const zr: f64 = xy * -ROOT3OVER3 + zz;

        // Evaluate both lattices to form a BCC lattice.
        return self.unrotatedBase(xr, yr, zr);
    }

    // 3D OpenSimplex2 noise, with better visual isotropy in (X, Z).
    // Recommended for 3D terrain and time-varied animations.
    // The Y coordinate should always be the "different" coordinate in whatever your use case is.
    // If Y is vertical in world coordinates, call noise3_ImproveXZ(x, Y, z).
    // If Z is vertical in world coordinates, call noise3_ImproveXZ(x, Z, y) or use noise3_ImproveXY.
    // For a time varied animation, call noise3_ImproveXZ(x, T, y) or use noise3_ImproveXY.
    pub fn improveXZ(self: *const Noise3, x: f64, y: f64, z: f64) f32 {
        // Re-orient the cubic lattices without skewing, so Y points up the main lattice diagonal,
        // and the planes formed by XZ are moved far out of alignment with the cube faces.
        // Orthonormal rotation. Not a skew transform.
        const xz: f64 = x + z;
        const s2: f64 = xz * ROTATE_3D_ORTHOGONALIZER;
        const yy: f64 = y * ROOT3OVER3;
        const xr: f64 = x + s2 + yy;
        const zr: f64 = z + s2 + yy;
        const yr: f64 = xz * -ROOT3OVER3 + yy;

        // Evaluate both lattices to form a BCC lattice.
        return self.unrotatedBase(xr, yr, zr);
    }

    // 3D OpenSimplex2 noise, fallback rotation option
    // Use noise3_ImproveXY or noise3_ImproveXZ instead, wherever appropriate.
    // They have less diagonal bias. This function's best use is as a fallback.
    pub fn fallback(self: *const Noise3, x: f64, y: f64, z: f64) f32 {
        // Re-orient the cubic lattices via rotation, to produce a familiar look.
        // Orthonormal rotation. Not a skew transform.
        const r: f64 = FALLBACK_ROTATE_3D * (x + y + z);
        const xr: f64 = r - x;
        const yr: f64 = r - y;
        const zr: f64 = r - z;

        // Evaluate both lattices to form a BCC lattice.
        return self.unrotatedBase(xr, yr, zr);
    }

    // Generate overlapping cubic lattices for 3D OpenSimplex2 noise.
    fn unrotatedBase(self: *const Noise3, xr: f64, yr: f64, zr: f64) f32 {
        var seed: i64 = self.seed;
        // Get base points and offsets.
        const xrb: i32 = fastRound(xr);
        const yrb: i32 = fastRound(yr);
        const zrb: i32 = fastRound(zr);
        const xi: f32 = @floatCast(xr - @as(f64, @floatFromInt(xrb)));
        const yi: f32 = @floatCast(yr - @as(f64, @floatFromInt(yrb)));
        const zi: f32 = @floatCast(zr - @as(f64, @floatFromInt(zrb)));

        // Prime pre-multiplication for hash. Also flip seed for second lattice copy.
        const xrbp: i64 = @as(i64, @intCast(xrb)) *% PRIME_X;
        const yrbp: i64 = @as(i64, @intCast(yrb)) *% PRIME_Y;
        const zrbp: i64 = @as(i64, @intCast(zrb)) *% PRIME_Z;
        const seed2: i64 = seed ^ SEED_FLIP_3D;

        // -1 if positive, 0 if negative.
        const xNMask: i32 = @as(i32, @intFromFloat(-0.5 - xi));
        const yNMask: i32 = @as(i32, @intFromFloat(-0.5 - yi));
        const zNMask: i32 = @as(i32, @intFromFloat(-0.5 - zi));

        // First vertex.
        const x0: f32 = xi + @as(f32, @floatFromInt(xNMask));
        const y0: f32 = yi + @as(f32, @floatFromInt(yNMask));
        const z0: f32 = zi + @as(f32, @floatFromInt(zNMask));
        const a0: f32 = RSQUARED_3D - x0 * x0 - y0 * y0 - z0 * z0;

        var value: f32 = (a0 * a0) * (a0 * a0) * self.grad3(
            self.seed,
            xrbp +% (@as(i64, @intCast(xNMask)) & PRIME_X),
            yrbp +% (@as(i64, @intCast(yNMask)) & PRIME_Y),
            zrbp +% (@as(i64, @intCast(zNMask)) & PRIME_Z),
            x0,
            y0,
            z0,
        );

        // Second vertex.
        const x1: f32 = xi - 0.5;
        const y1: f32 = yi - 0.5;
        const z1: f32 = zi - 0.5;
        const a1: f32 = RSQUARED_3D - x1 * x1 - y1 * y1 - z1 * z1;
        value += (a1 * a1) * (a1 * a1) * self.grad3(
            seed2,
            xrbp +% PRIME_X,
            yrbp +% PRIME_Y,
            zrbp +% PRIME_Z,
            x1,
            y1,
            z1,
        );

        // Shortcuts for building the remaining falloffs.
        // Derived by subtracting the polynomials with the offsets plugged in.
        const xAFlipMask0: f32 = @as(f32, @floatFromInt(((xNMask | 1) << 1))) * x1;
        const yAFlipMask0: f32 = @as(f32, @floatFromInt(((yNMask | 1) << 1))) * y1;
        const zAFlipMask0: f32 = @as(f32, @floatFromInt(((zNMask | 1) << 1))) * z1;
        const xAFlipMask1: f32 = @as(f32, @floatFromInt((-2 - (xNMask << 2)))) * x1 - 1.0;
        const yAFlipMask1: f32 = @as(f32, @floatFromInt((-2 - (yNMask << 2)))) * y1 - 1.0;
        const zAFlipMask1: f32 = @as(f32, @floatFromInt((-2 - (zNMask << 2)))) * z1 - 1.0;

        var skip5: bool = false;
        const a2: f32 = xAFlipMask0 + a0;
        if (a2 > 0.0) {
            const x2: f32 = x0 - @as(f32, @floatFromInt(xNMask | 1));
            const y2: f32 = y0;
            const z2: f32 = z0;
            value += (a2 * a2) * (a2 * a2) * self.grad3(
                seed,
                xrbp +% (@as(i64, @intCast(!xNMask)) & PRIME_X),
                yrbp +% (@as(i64, @intCast(yNMask)) & PRIME_Y),
                zrbp +% (@as(i64, @intCast(zNMask)) & PRIME_Z),
                x2,
                y2,
                z2,
            );
        } else {
            const a3: f32 = yAFlipMask0 + zAFlipMask0 + a0;
            if (a3 > 0.0) {
                const x3: f32 = x0;
                const y3: f32 = y0 - @as(f32, @floatFromInt(yNMask | 1));
                const z3: f32 = z0 - @as(f32, @floatFromInt(zNMask | 1));
                value += (a3 * a3) * (a3 * a3) * self.grad3(
                    seed,
                    xrbp +% (@as(i64, @intCast(xNMask)) & PRIME_X),
                    yrbp +% (@as(i64, @intCast(!yNMask)) & PRIME_Y),
                    zrbp +% (@as(i64, @intCast(!zNMask)) & PRIME_Z),
                    x3,
                    y3,
                    z3,
                );
            }

            const a4: f32 = xAFlipMask1 + a1;
            if (a4 > 0.0) {
                const x4: f32 = @as(f32, @intFromFloat(xNMask | 1)) + x1;
                const y4: f32 = y1;
                const z4: f32 = z1;
                value += (a4 * a4) * (a4 * a4) * self.grad3(
                    seed2,
                    xrbp +% ((@as(i64, @intCast(xNMask)) & PRIME_X << 1)),
                    yrbp +% PRIME_Y,
                    zrbp +% PRIME_Z,
                    x4,
                    y4,
                    z4,
                );
                skip5 = true;
            }
        }

        var skip9: bool = false;
        const a6 = yAFlipMask0 + a0;
        if (a6 > 0.0) {
            const x6: f32 = x0;
            const y6: f32 = y0 - @as(f32, @floatFromInt((yNMask | 1)));
            const z6: f32 = z0;
            value += (a6 * a6) * (a6 * a6) * self.grad3(
                seed,
                xrbp +% (@as(i64, @intCast(xNMask)) & PRIME_X),
                yrbp +% (@as(i64, @intCast(!yNMask)) & PRIME_Y),
                zrbp +% (@as(i64, @intCast(zNMask)) & PRIME_Z),
                x6,
                y6,
                z6,
            );
        } else {
            var a7: f32 = xAFlipMask0 + zAFlipMask0 + a0;
            if (a7 > 0.0) {
                const x7 = x0 - @as(f32, @floatFromInt((xNMask | 1)));
                const y7 = y0;
                const z7 = z0 - @as(f32, @floatFromInt((zNMask | 1)));
                value += (a7 * a7) * (a7 * a7) * grad3(
                    seed,
                    xrbp +% (@as(i64, @intCast(!xNMask)) & PRIME_X),
                    yrbp +% (@as(i64, @intCast(yNMask)) & PRIME_Y),
                    zrbp +% (@as(i64, @intCast(!zNMask)) & PRIME_Z),
                    x7,
                    y7,
                    z7,
                );
            }

            const a8: f32 = yAFlipMask1 + a1;
            if (a8 > 0.0) {
                const x8 = x1;
                const y8 = y1 + @as(f32, @floatFromInt((yNMask | 1)));
                const z8 = z1;
                value += (a8 * a8) * (a8 * a8) * self.grad3(
                    seed2,
                    xrbp +% PRIME_X,
                    yrbp +% (@as(i64, @intCast(yNMask)) & (PRIME_Y << 1)),
                    zrbp +% PRIME_Z,
                    x8,
                    y8,
                    z8,
                );

                skip9 = true;
            }
        }

        var skipD: bool = false;
        const aA: f32 = zAFlipMask0 + a0;
        if (aA > 0.0) {
            const xA: f32 = x0;
            const yA: f32 = y0;
            const zA: f32 = z0 - @as(f32, @floatFromInt((zNMask | 1)));
            value += (aA * aA) * (aA * aA) * self.grad3(
                seed,
                xrbp +% (@as(i64, @intCast(xNMask)) & PRIME_X),
                yrbp +% (@as(i64, @intCast(yNMask)) & PRIME_Y),
                zrbp +% (@as(i64, @intCast(!zNMask)) & PRIME_Z),
                xA,
                yA,
                zA,
            );
        } else {
            const aB = xAFlipMask0 + yAFlipMask0 + a0;
            if (aB > 0.0) {
                const xB: f32 = x0 - @as(f32, @floatFromInt((xNMask | 1)));
                const yB: f32 = y0 - @as(f32, @floatFromInt((yNMask | 1)));
                const zB: f32 = z0;
                value += (aB * aB) * (aB * aB) * self.grad3(
                    seed,
                    xrbp +% (@as(i64, @intCast(!xNMask)) & PRIME_X),
                    yrbp +% (@as(i64, @intCast(!yNMask)) & PRIME_Y),
                    zrbp +% (@as(i64, @intCast(zNMask)) & PRIME_Z),
                    xB,
                    yB,
                    zB,
                );
            }

            const aC: f32 = zAFlipMask1 + a1;
            if (aC > 0.0) {
                const xC: f32 = x1;
                const yC: f32 = y1;
                const zC: f32 = z1 + @as(f32, @floatFromInt((zNMask | 1)));
                value += (aC * aC) * (aC * aC) * self.grad3(
                    seed2,
                    xrbp +% PRIME_X,
                    yrbp +% PRIME_Y,
                    zrbp +% (@as(i64, @intCast(zNMask)) & (PRIME_Z << 1)),
                    xC,
                    yC,
                    zC,
                );
                skipD = true;
            }
        }

        if (!skip5) {
            const a5: f32 = yAFlipMask1 + zAFlipMask1 + a1;
            if (a5 > 0.0) {
                const x5: f32 = x1;
                const y5: f32 = y1 + @as(f32, @floatFromInt((yNMask | 1)));
                const z5: f32 = z1 + @as(f32, @floatFromInt((zNMask | 1)));
                value += (a5 * a5) * (a5 * a5) * self.grad3(
                    seed2,
                    xrbp +% PRIME_X,
                    yrbp +% (@as(i64, @intCast(yNMask)) & (PRIME_Y << 1)),
                    zrbp +% (@as(i64, @intCast(zNMask)) & (PRIME_Z << 1)),
                    x5,
                    y5,
                    z5,
                );
            }
        }

        if (!skip9) {
            const a9: f32 = xAFlipMask1 + zAFlipMask1 + a1;
            if (a9 > 0.0) {
                const x9: f32 = x1 + @as(f32, @floatFromInt((xNMask | 1)));
                const y9: f32 = y1;
                const z9: f32 = z1 + @as(f32, @floatFromInt((zNMask | 1)));
                value += (a9 * a9) * (a9 * a9) * self.grad3(
                    seed2,
                    xrbp +% (@as(i64, @intCast(xNMask)) & (PRIME_X << 1)),
                    yrbp +% PRIME_Y,
                    zrbp +% (@as(i64, @intCast(zNMask)) & (PRIME_Z << 1)),
                    x9,
                    y9,
                    z9,
                );
            }
        }

        if (!skipD) {
            const aD: f32 = xAFlipMask1 + yAFlipMask1 + a1;
            if (aD > 0.0) {
                const xD: f32 = x1 + @as(f32, @floatFromInt((xNMask | 1)));
                const yD: f32 = y1 + @as(f32, @floatFromInt((yNMask | 1)));
                const zD: f32 = z1;
                value += (aD * aD) * (aD * aD) * self.grad3(
                    seed2,
                    xrbp +% (@as(i64, @intCast(yNMask)) & (PRIME_Y << 1)),
                    yrbp +% (@as(i64, @intCast(yNMask)) & (PRIME_Y << 1)),
                    zrbp +% PRIME_Z,
                    xD,
                    yD,
                    zD,
                );
            }
        }

        return value;
    }

    fn grad3(
        self: *const Noise3,
        seed: i64,
        xrvp: i64,
        yrvp: i64,
        zrvp: i64,
        dx: f32,
        dy: f32,
        dz: f32,
    ) f32 {
        var hash: i64 = (seed ^ xrvp) ^ (yrvp ^ zrvp);
        hash *%= HASH_MULTIPLIER;
        hash ^= hash >> (64 - N_GRADS_3D_EXPONENT + 2);
        const gi: usize = @intCast(hash & ((@as(i64, @intCast(N_GRADS_3D)) - 1) << 2));
        return @floatCast(self.grads[gi | 0] * dx + self.grads[gi | 1] * dy + self.grads[gi | 2] * dz);
    }
};

// 4D OpenSimplex2 noise, with XYZ oriented like noise3_ImproveXY
// and W for an extra degree of freedom. W repeats eventually.
// Recommended for time-varied animations which texture a 3D object (W=time)
// in a space where Z is vertical
pub const Noise4 = struct {
    seed: i64,
    grads: *[N_GRADS_4D * 4]f32,
    allocator: std.mem.Allocator,

    pub fn init(seed: i64, allocator: std.mem.Allocator) !Noise4 {
        const noise = Noise4{
            .seed = seed,
            .grads = try allocator.create([N_GRADS_4D * 4]f32),
            .allocator = allocator,
        };

        for (0..N_GRADS_4D * 4) |i| {
            const grad_index = i % GRAD4_SRC.len;
            noise.grads.*[i] = @floatCast(GRAD4_SRC[grad_index] / NORMALIZER_4D);
        }

        return noise;
    }

    pub fn deInit(self: *const Noise4) void {
        self.allocator.destroy(self.grads);
    }

    pub fn improveXYZXY(self: *const Noise4, x: f64, y: f64, z: f64, w: f64) f32 {
        const xy = x + y;
        const s2 = xy * -0.21132486540518699998;
        const zz = z * 0.28867513459481294226;
        const ww = w * 0.2236067977499788;
        const xr = x + (zz + ww + s2);
        const yr = y + (zz + ww + s2);
        const zr = xy * -0.57735026918962599998 + (zz + ww);
        const wr = z * -0.866025403784439 + ww;

        return self.unskewedBase(xr, yr, zr, wr);
    }

    // 4D OpenSimplex2 noise, with XYZ oriented like noise3_ImproveXZ
    // and W for an extra degree of freedom. W repeats eventually.
    // Recommended for time-varied animations which texture a 3D object (W=time)
    // in a space where Y is vertical
    pub fn improveXYZXZ(self: *const Noise4, x: f64, y: f64, z: f64, w: f64) f32 {
        const xz = x + z;
        const s2 = xz * -0.21132486540518699998;
        const yy = y * 0.28867513459481294226;
        const ww = w * 0.2236067977499788;
        const xr = x + (yy + ww + s2);
        const zr = z + (yy + ww + s2);
        const yr = xz * -0.57735026918962599998 + (yy + ww);
        const wr = y * -0.866025403784439 + ww;

        return self.unskewedBase(xr, yr, zr, wr);
    }

    // 4D OpenSimplex2 noise, with XYZ oriented like noise3_Fallback
    // and W for an extra degree of freedom. W repeats eventually.
    // Recommended for time-varied animations which texture a 3D object (W=time)
    // where there isn't a clear distinction between horizontal and vertical
    pub fn improveXYZ(self: *const Noise4, x: f64, y: f64, z: f64, w: f64) f32 {
        const xyz = x + y + z;
        const ww = w * 0.2236067977499788;
        const s2 = xyz * -0.16666666666666666 + ww;
        const xs = x + s2;
        const ys = y + s2;
        const zs = z + s2;
        const ws = -0.5 * xyz + ww;

        return self.unskewedBase(xs, ys, zs, ws);
    }

    // 4D OpenSimplex2 noise, with XY and ZW forming orthogonal triangular-based planes.
    // Recommended for 3D terrain, where X and Y (or Z and W) are horizontal.
    // Recommended for noise(x, y, sin(time), cos(time)) trick.
    pub fn improveXYZW(self: *const Noise4, x: f64, y: f64, z: f64, w: f64) f32 {
        const s2 = (x + y) * -0.178275657951399372 + (z + w) * 0.215623393288842828;
        const t2 = (z + w) * -0.403949762580207112 + (x + y) * -0.375199083010075342;
        const xs = x + s2;
        const ys = y + s2;
        const zs = z + t2;
        const ws = w + t2;

        return self.unskewedBase(xs, ys, zs, ws);
    }

    // 4D OpenSimplex2 noise, fallback lattice orientation.

    pub fn fallback(self: *const Noise4, x: f64, y: f64, z: f64, w: f64) f32 {
        // Get points for A4 lattice
        const s = @as(f64, @floatCast(SKEW_4D)) * (x + y + z + w);
        const xs = x + s;
        const ys = y + s;
        const zs = z + s;
        const ws = w + s;

        return self.unskewedBase(xs, ys, zs, ws);
    }

    // 4D OpenSimplex2 noise base.
    fn unskewedBase(self: *const Noise4, xs: f64, ys: f64, zs: f64, ws: f64) f32 {
        var seed = self.seed;

        // Get base points and offsets
        const xsb: i32 = fastFloor(xs);
        const ysb: i32 = fastFloor(ys);
        const zsb: i32 = fastFloor(zs);
        const wsb: i32 = fastFloor(ws);
        var xsi: f32 = @floatCast(xs - @as(f64, @floatFromInt(xsb)));
        var ysi: f32 = @floatCast(xs - @as(f64, @floatFromInt(ysb)));
        var zsi: f32 = @floatCast(xs - @as(f64, @floatFromInt(zsb)));
        var wsi: f32 = @floatCast(xs - @as(f64, @floatFromInt(wsb)));

        // Determine which lattice we can be confident has a contributing point its corresponding cell's base simplex.
        // We only look at the spaces between the diagonal planes. This proved effective in all of my tests.
        const siSum: f32 = (xsi + ysi) + (zsi + wsi);
        const startingLattice: i32 = @intFromFloat(siSum * 1.25);

        // Offset for seed based on first lattice copy.
        seed += @as(i64, @intCast(startingLattice)) *% SEED_OFFSET_4D;

        // Offset for lattice point relative positions (skewed)
        const startingLatticeOffset: f32 = @as(f32, @floatFromInt(startingLattice)) * -LATTICE_STEP_4D;
        xsi += startingLatticeOffset;
        ysi += startingLatticeOffset;
        zsi += startingLatticeOffset;
        wsi += startingLatticeOffset;

        // Prep for vertex contributions.
        var ssi: f32 = (siSum + startingLatticeOffset * 4.0) * UNSKEW_4D;

        // Prime pre-multiplication for hash.
        var xsvp: i64 = @as(i64, @intCast(xsb)) *% PRIME_X;
        var ysvp: i64 = @as(i64, @intCast(ysb)) *% PRIME_Y;
        var zsvp: i64 = @as(i64, @intCast(zsb)) *% PRIME_Z;
        var wsvp: i64 = @as(i64, @intCast(wsb)) *% PRIME_W;

        // Five points to add, total, from five copies of the A4 lattice.
        var value: f32 = 0.0;
        for (0..5) |i| {
            // Next point is the closest vertex on the 4-simplex whose base vertex is the aforementioned vertex.
            const score0: f32 = 1.0 + ssi * (-1.0 / UNSKEW_4D); // Seems slightly faster than 1.0-xsi-ysi-zsi-wsi
            if (xsi >= ysi and xsi >= zsi and xsi >= wsi and xsi >= score0) {
                xsvp +%= PRIME_X;
                xsi -= 1.0;
                ssi -= UNSKEW_4D;
            } else if (ysi > xsi and ysi >= zsi and ysi >= wsi and ysi >= score0) {
                ysvp +%= PRIME_Y;
                ysi -= 1.0;
                ssi -= UNSKEW_4D;
            } else if (zsi > xsi and zsi > ysi and zsi >= wsi and zsi >= score0) {
                zsvp +%= PRIME_Z;
                zsi -= 1.0;
                ssi -= UNSKEW_4D;
            } else if (wsi > xsi and wsi > ysi and wsi > zsi and wsi >= score0) {
                wsvp +%= PRIME_W;
                wsi -= 1.0;
                ssi -= UNSKEW_4D;
            }

            // gradient contribution with falloff.
            const dx: f32 = xsi + ssi;
            const dy: f32 = ysi + ssi;
            const dz: f32 = zsi + ssi;
            const dw: f32 = wsi + ssi;
            var a: f32 = (dx * dx + dy * dy) + (dz * dz + dw * dw);
            if (a < RSQUARED_4D) {
                a -= RSQUARED_4D;
                a *= a;
                value += a * a * self.grad4(seed, xsvp, ysvp, zsvp, wsvp, dx, dy, dz, dw);
            }

            // Break from loop if we're done, skipping updates below.
            if (i == 4) {
                break;
            }

            // Update for next lattice copy shifted down by <-0.2, -0.2, -0.2, -0.2>.
            xsi += LATTICE_STEP_4D;
            ysi += LATTICE_STEP_4D;
            zsi += LATTICE_STEP_4D;
            wsi += LATTICE_STEP_4D;
            ssi += LATTICE_STEP_4D * 4.0 * UNSKEW_4D;
            seed -= SEED_OFFSET_4D;

            // Because we don't always start on the same lattice copy, there's a special reset case.
            if (i == startingLattice) {
                xsvp -= PRIME_X;
                ysvp -= PRIME_Y;
                zsvp -= PRIME_Z;
                wsvp -= PRIME_W;
                seed += SEED_OFFSET_4D * 5;
            }
        }

        return value;
    }

    fn grad4(
        self: *const Noise4,
        seed: i64,
        xsvp: i64,
        ysvp: i64,
        zsvp: i64,
        wsvp: i64,
        dx: f32,
        dy: f32,
        dz: f32,
        dw: f32,
    ) f32 {
        var hash: i64 = seed ^ (xsvp ^ ysvp) ^ (zsvp ^ wsvp);
        hash *%= HASH_MULTIPLIER;
        hash ^= hash >> (64 - N_GRADS_4D_EXPONENT + 2);
        const gi: usize = @intCast(hash & ((@as(i64, @intCast(N_GRADS_4D)) - 1) << 2));
        return @floatCast((self.grads[gi | 0] * dx + self.grads[gi | 1] * dy) + (self.grads[gi | 2] * dz + self.grads[gi | 3] * dw));
    }
};

fn fastFloor(x: f64) i32 {
    const xi: i32 = @intFromFloat(x);
    if (x < @as(f64, @floatFromInt(xi))) {
        return xi - 1;
    } else {
        return xi;
    }
}

fn fastRound(x: f64) i32 {
    if (x < 0.0) {
        return @intFromFloat(x - 0.5);
    } else {
        return @intFromFloat(x + 0.5);
    }
}

const GRAD2_SRC: [48]f64 = .{
    0.38268343236509,
    0.923879532511287,
    0.923879532511287,
    0.38268343236509,
    0.923879532511287,
    -0.38268343236509,
    0.38268343236509,
    -0.923879532511287,
    -0.38268343236509,
    -0.923879532511287,
    -0.923879532511287,
    -0.38268343236509,
    -0.923879532511287,
    0.38268343236509,
    -0.38268343236509,
    0.923879532511287,
    0.130526192220052,
    0.99144486137381,
    0.608761429008721,
    0.793353340291235,
    0.793353340291235,
    0.608761429008721,
    0.99144486137381,
    0.130526192220051,
    0.99144486137381,
    -0.130526192220051,
    0.793353340291235,
    -0.60876142900872,
    0.608761429008721,
    -0.793353340291235,
    0.130526192220052,
    -0.99144486137381,
    -0.130526192220052,
    -0.99144486137381,
    -0.608761429008721,
    -0.793353340291235,
    -0.793353340291235,
    -0.608761429008721,
    -0.99144486137381,
    -0.130526192220052,
    -0.99144486137381,
    0.130526192220051,
    -0.793353340291235,
    0.608761429008721,
    -0.608761429008721,
    0.793353340291235,
    -0.130526192220052,
    0.99144486137381,
};

const GRAD3_SRC: [192]f64 = .{
    2.22474487139,
    2.22474487139,
    -1.0,
    0.0,
    2.22474487139,
    2.22474487139,
    1.0,
    0.0,
    3.0862664687972017,
    1.1721513422464978,
    0.0,
    0.0,
    1.1721513422464978,
    3.0862664687972017,
    0.0,
    0.0,
    -2.22474487139,
    2.22474487139,
    -1.0,
    0.0,
    -2.22474487139,
    2.22474487139,
    1.0,
    0.0,
    -1.1721513422464978,
    3.0862664687972017,
    0.0,
    0.0,
    -3.0862664687972017,
    1.1721513422464978,
    0.0,
    0.0,
    -1.0,
    -2.22474487139,
    -2.22474487139,
    0.0,
    1.0,
    -2.22474487139,
    -2.22474487139,
    0.0,
    0.0,
    -3.0862664687972017,
    -1.1721513422464978,
    0.0,
    0.0,
    -1.1721513422464978,
    -3.0862664687972017,
    0.0,
    -1.0,
    -2.22474487139,
    2.22474487139,
    0.0,
    1.0,
    -2.22474487139,
    2.22474487139,
    0.0,
    0.0,
    -1.1721513422464978,
    3.0862664687972017,
    0.0,
    0.0,
    -3.0862664687972017,
    1.1721513422464978,
    0.0,
    -2.22474487139,
    -2.22474487139,
    -1.0,
    0.0,
    -2.22474487139,
    -2.22474487139,
    1.0,
    0.0,
    -3.0862664687972017,
    -1.1721513422464978,
    0.0,
    0.0,
    -1.1721513422464978,
    -3.0862664687972017,
    0.0,
    0.0,
    -2.22474487139,
    -1.0,
    -2.22474487139,
    0.0,
    -2.22474487139,
    1.0,
    -2.22474487139,
    0.0,
    -1.1721513422464978,
    0.0,
    -3.0862664687972017,
    0.0,
    -3.0862664687972017,
    0.0,
    -1.1721513422464978,
    0.0,
    -2.22474487139,
    -1.0,
    2.22474487139,
    0.0,
    -2.22474487139,
    1.0,
    2.22474487139,
    0.0,
    -3.0862664687972017,
    0.0,
    1.1721513422464978,
    0.0,
    -1.1721513422464978,
    0.0,
    3.0862664687972017,
    0.0,
    -1.0,
    2.22474487139,
    -2.22474487139,
    0.0,
    1.0,
    2.22474487139,
    -2.22474487139,
    0.0,
    0.0,
    1.1721513422464978,
    -3.0862664687972017,
    0.0,
    0.0,
    3.0862664687972017,
    -1.1721513422464978,
    0.0,
    -1.0,
    2.22474487139,
    2.22474487139,
    0.0,
    1.0,
    2.22474487139,
    2.22474487139,
    0.0,
    0.0,
    3.0862664687972017,
    1.1721513422464978,
    0.0,
    0.0,
    1.1721513422464978,
    3.0862664687972017,
    0.0,
    2.22474487139,
    -2.22474487139,
    -1.0,
    0.0,
    2.22474487139,
    -2.22474487139,
    1.0,
    0.0,
    1.1721513422464978,
    -3.0862664687972017,
    0.0,
    0.0,
    3.0862664687972017,
    -1.1721513422464978,
    0.0,
    0.0,
    2.22474487139,
    -1.0,
    -2.22474487139,
    0.0,
    2.22474487139,
    1.0,
    -2.22474487139,
    0.0,
    3.0862664687972017,
    0.0,
    -1.1721513422464978,
    0.0,
    1.1721513422464978,
    0.0,
    -3.0862664687972017,
    0.0,
    2.22474487139,
    -1.0,
    2.22474487139,
    0.0,
    2.22474487139,
    1.0,
    2.22474487139,
    0.0,
    1.1721513422464978,
    0.0,
    3.0862664687972017,
    0.0,
    3.0862664687972017,
    0.0,
    1.1721513422464978,
    0.0,
};

const GRAD4_SRC: [640]f64 = .{
    -0.6740059517812944,
    -0.3239847771997537,
    -0.3239847771997537,
    0.5794684678643381,
    -0.7504883828755602,
    -0.4004672082940195,
    0.15296486218853164,
    0.5029860367700724,
    -0.7504883828755602,
    0.15296486218853164,
    -0.4004672082940195,
    0.5029860367700724,
    -0.8828161875373585,
    0.08164729285680945,
    0.08164729285680945,
    0.4553054119602712,
    -0.4553054119602712,
    -0.08164729285680945,
    -0.08164729285680945,
    0.8828161875373585,
    -0.5029860367700724,
    -0.15296486218853164,
    0.4004672082940195,
    0.7504883828755602,
    -0.5029860367700724,
    0.4004672082940195,
    -0.15296486218853164,
    0.7504883828755602,
    -0.5794684678643381,
    0.3239847771997537,
    0.3239847771997537,
    0.6740059517812944,
    -0.6740059517812944,
    -0.3239847771997537,
    0.5794684678643381,
    -0.3239847771997537,
    -0.7504883828755602,
    -0.4004672082940195,
    0.5029860367700724,
    0.15296486218853164,
    -0.7504883828755602,
    0.15296486218853164,
    0.5029860367700724,
    -0.4004672082940195,
    -0.8828161875373585,
    0.08164729285680945,
    0.4553054119602712,
    0.08164729285680945,
    -0.4553054119602712,
    -0.08164729285680945,
    0.8828161875373585,
    -0.08164729285680945,
    -0.5029860367700724,
    -0.15296486218853164,
    0.7504883828755602,
    0.4004672082940195,
    -0.5029860367700724,
    0.4004672082940195,
    0.7504883828755602,
    -0.15296486218853164,
    -0.5794684678643381,
    0.3239847771997537,
    0.6740059517812944,
    0.3239847771997537,
    -0.6740059517812944,
    0.5794684678643381,
    -0.3239847771997537,
    -0.3239847771997537,
    -0.7504883828755602,
    0.5029860367700724,
    -0.4004672082940195,
    0.15296486218853164,
    -0.7504883828755602,
    0.5029860367700724,
    0.15296486218853164,
    -0.4004672082940195,
    -0.8828161875373585,
    0.4553054119602712,
    0.08164729285680945,
    0.08164729285680945,
    -0.4553054119602712,
    0.8828161875373585,
    -0.08164729285680945,
    -0.08164729285680945,
    -0.5029860367700724,
    0.7504883828755602,
    -0.15296486218853164,
    0.4004672082940195,
    -0.5029860367700724,
    0.7504883828755602,
    0.4004672082940195,
    -0.15296486218853164,
    -0.5794684678643381,
    0.6740059517812944,
    0.3239847771997537,
    0.3239847771997537,
    0.5794684678643381,
    -0.6740059517812944,
    -0.3239847771997537,
    -0.3239847771997537,
    0.5029860367700724,
    -0.7504883828755602,
    -0.4004672082940195,
    0.15296486218853164,
    0.5029860367700724,
    -0.7504883828755602,
    0.15296486218853164,
    -0.4004672082940195,
    0.4553054119602712,
    -0.8828161875373585,
    0.08164729285680945,
    0.08164729285680945,
    0.8828161875373585,
    -0.4553054119602712,
    -0.08164729285680945,
    -0.08164729285680945,
    0.7504883828755602,
    -0.5029860367700724,
    -0.15296486218853164,
    0.4004672082940195,
    0.7504883828755602,
    -0.5029860367700724,
    0.4004672082940195,
    -0.15296486218853164,
    0.6740059517812944,
    -0.5794684678643381,
    0.3239847771997537,
    0.3239847771997537,
    -0.753341017856078,
    -0.37968289875261624,
    -0.37968289875261624,
    -0.37968289875261624,
    -0.7821684431180708,
    -0.4321472685365301,
    -0.4321472685365301,
    0.12128480194602098,
    -0.7821684431180708,
    -0.4321472685365301,
    0.12128480194602098,
    -0.4321472685365301,
    -0.7821684431180708,
    0.12128480194602098,
    -0.4321472685365301,
    -0.4321472685365301,
    -0.8586508742123365,
    -0.508629699630796,
    0.044802370851755174,
    0.044802370851755174,
    -0.8586508742123365,
    0.044802370851755174,
    -0.508629699630796,
    0.044802370851755174,
    -0.8586508742123365,
    0.044802370851755174,
    0.044802370851755174,
    -0.508629699630796,
    -0.9982828964265062,
    -0.03381941603233842,
    -0.03381941603233842,
    -0.03381941603233842,
    -0.37968289875261624,
    -0.753341017856078,
    -0.37968289875261624,
    -0.37968289875261624,
    -0.4321472685365301,
    -0.7821684431180708,
    -0.4321472685365301,
    0.12128480194602098,
    -0.4321472685365301,
    -0.7821684431180708,
    0.12128480194602098,
    -0.4321472685365301,
    0.12128480194602098,
    -0.7821684431180708,
    -0.4321472685365301,
    -0.4321472685365301,
    -0.508629699630796,
    -0.8586508742123365,
    0.044802370851755174,
    0.044802370851755174,
    0.044802370851755174,
    -0.8586508742123365,
    -0.508629699630796,
    0.044802370851755174,
    0.044802370851755174,
    -0.8586508742123365,
    0.044802370851755174,
    -0.508629699630796,
    -0.03381941603233842,
    -0.9982828964265062,
    -0.03381941603233842,
    -0.03381941603233842,
    -0.37968289875261624,
    -0.37968289875261624,
    -0.753341017856078,
    -0.37968289875261624,
    -0.4321472685365301,
    -0.4321472685365301,
    -0.7821684431180708,
    0.12128480194602098,
    -0.4321472685365301,
    0.12128480194602098,
    -0.7821684431180708,
    -0.4321472685365301,
    0.12128480194602098,
    -0.4321472685365301,
    -0.7821684431180708,
    -0.4321472685365301,
    -0.508629699630796,
    0.044802370851755174,
    -0.8586508742123365,
    0.044802370851755174,
    0.044802370851755174,
    -0.508629699630796,
    -0.8586508742123365,
    0.044802370851755174,
    0.044802370851755174,
    0.044802370851755174,
    -0.8586508742123365,
    -0.508629699630796,
    -0.03381941603233842,
    -0.03381941603233842,
    -0.9982828964265062,
    -0.03381941603233842,
    -0.37968289875261624,
    -0.37968289875261624,
    -0.37968289875261624,
    -0.753341017856078,
    -0.4321472685365301,
    -0.4321472685365301,
    0.12128480194602098,
    -0.7821684431180708,
    -0.4321472685365301,
    0.12128480194602098,
    -0.4321472685365301,
    -0.7821684431180708,
    0.12128480194602098,
    -0.4321472685365301,
    -0.4321472685365301,
    -0.7821684431180708,
    -0.508629699630796,
    0.044802370851755174,
    0.044802370851755174,
    -0.8586508742123365,
    0.044802370851755174,
    -0.508629699630796,
    0.044802370851755174,
    -0.8586508742123365,
    0.044802370851755174,
    0.044802370851755174,
    -0.508629699630796,
    -0.8586508742123365,
    -0.03381941603233842,
    -0.03381941603233842,
    -0.03381941603233842,
    -0.9982828964265062,
    -0.3239847771997537,
    -0.6740059517812944,
    -0.3239847771997537,
    0.5794684678643381,
    -0.4004672082940195,
    -0.7504883828755602,
    0.15296486218853164,
    0.5029860367700724,
    0.15296486218853164,
    -0.7504883828755602,
    -0.4004672082940195,
    0.5029860367700724,
    0.08164729285680945,
    -0.8828161875373585,
    0.08164729285680945,
    0.4553054119602712,
    -0.08164729285680945,
    -0.4553054119602712,
    -0.08164729285680945,
    0.8828161875373585,
    -0.15296486218853164,
    -0.5029860367700724,
    0.4004672082940195,
    0.7504883828755602,
    0.4004672082940195,
    -0.5029860367700724,
    -0.15296486218853164,
    0.7504883828755602,
    0.3239847771997537,
    -0.5794684678643381,
    0.3239847771997537,
    0.6740059517812944,
    -0.3239847771997537,
    -0.3239847771997537,
    -0.6740059517812944,
    0.5794684678643381,
    -0.4004672082940195,
    0.15296486218853164,
    -0.7504883828755602,
    0.5029860367700724,
    0.15296486218853164,
    -0.4004672082940195,
    -0.7504883828755602,
    0.5029860367700724,
    0.08164729285680945,
    0.08164729285680945,
    -0.8828161875373585,
    0.4553054119602712,
    -0.08164729285680945,
    -0.08164729285680945,
    -0.4553054119602712,
    0.8828161875373585,
    -0.15296486218853164,
    0.4004672082940195,
    -0.5029860367700724,
    0.7504883828755602,
    0.4004672082940195,
    -0.15296486218853164,
    -0.5029860367700724,
    0.7504883828755602,
    0.3239847771997537,
    0.3239847771997537,
    -0.5794684678643381,
    0.6740059517812944,
    -0.3239847771997537,
    -0.6740059517812944,
    0.5794684678643381,
    -0.3239847771997537,
    -0.4004672082940195,
    -0.7504883828755602,
    0.5029860367700724,
    0.15296486218853164,
    0.15296486218853164,
    -0.7504883828755602,
    0.5029860367700724,
    -0.4004672082940195,
    0.08164729285680945,
    -0.8828161875373585,
    0.4553054119602712,
    0.08164729285680945,
    -0.08164729285680945,
    -0.4553054119602712,
    0.8828161875373585,
    -0.08164729285680945,
    -0.15296486218853164,
    -0.5029860367700724,
    0.7504883828755602,
    0.4004672082940195,
    0.4004672082940195,
    -0.5029860367700724,
    0.7504883828755602,
    -0.15296486218853164,
    0.3239847771997537,
    -0.5794684678643381,
    0.6740059517812944,
    0.3239847771997537,
    -0.3239847771997537,
    -0.3239847771997537,
    0.5794684678643381,
    -0.6740059517812944,
    -0.4004672082940195,
    0.15296486218853164,
    0.5029860367700724,
    -0.7504883828755602,
    0.15296486218853164,
    -0.4004672082940195,
    0.5029860367700724,
    -0.7504883828755602,
    0.08164729285680945,
    0.08164729285680945,
    0.4553054119602712,
    -0.8828161875373585,
    -0.08164729285680945,
    -0.08164729285680945,
    0.8828161875373585,
    -0.4553054119602712,
    -0.15296486218853164,
    0.4004672082940195,
    0.7504883828755602,
    -0.5029860367700724,
    0.4004672082940195,
    -0.15296486218853164,
    0.7504883828755602,
    -0.5029860367700724,
    0.3239847771997537,
    0.3239847771997537,
    0.6740059517812944,
    -0.5794684678643381,
    -0.3239847771997537,
    0.5794684678643381,
    -0.6740059517812944,
    -0.3239847771997537,
    -0.4004672082940195,
    0.5029860367700724,
    -0.7504883828755602,
    0.15296486218853164,
    0.15296486218853164,
    0.5029860367700724,
    -0.7504883828755602,
    -0.4004672082940195,
    0.08164729285680945,
    0.4553054119602712,
    -0.8828161875373585,
    0.08164729285680945,
    -0.08164729285680945,
    0.8828161875373585,
    -0.4553054119602712,
    -0.08164729285680945,
    -0.15296486218853164,
    0.7504883828755602,
    -0.5029860367700724,
    0.4004672082940195,
    0.4004672082940195,
    0.7504883828755602,
    -0.5029860367700724,
    -0.15296486218853164,
    0.3239847771997537,
    0.6740059517812944,
    -0.5794684678643381,
    0.3239847771997537,
    -0.3239847771997537,
    0.5794684678643381,
    -0.3239847771997537,
    -0.6740059517812944,
    -0.4004672082940195,
    0.5029860367700724,
    0.15296486218853164,
    -0.7504883828755602,
    0.15296486218853164,
    0.5029860367700724,
    -0.4004672082940195,
    -0.7504883828755602,
    0.08164729285680945,
    0.4553054119602712,
    0.08164729285680945,
    -0.8828161875373585,
    -0.08164729285680945,
    0.8828161875373585,
    -0.08164729285680945,
    -0.4553054119602712,
    -0.15296486218853164,
    0.7504883828755602,
    0.4004672082940195,
    -0.5029860367700724,
    0.4004672082940195,
    0.7504883828755602,
    -0.15296486218853164,
    -0.5029860367700724,
    0.3239847771997537,
    0.6740059517812944,
    0.3239847771997537,
    -0.5794684678643381,
    0.5794684678643381,
    -0.3239847771997537,
    -0.6740059517812944,
    -0.3239847771997537,
    0.5029860367700724,
    -0.4004672082940195,
    -0.7504883828755602,
    0.15296486218853164,
    0.5029860367700724,
    0.15296486218853164,
    -0.7504883828755602,
    -0.4004672082940195,
    0.4553054119602712,
    0.08164729285680945,
    -0.8828161875373585,
    0.08164729285680945,
    0.8828161875373585,
    -0.08164729285680945,
    -0.4553054119602712,
    -0.08164729285680945,
    0.7504883828755602,
    -0.15296486218853164,
    -0.5029860367700724,
    0.4004672082940195,
    0.7504883828755602,
    0.4004672082940195,
    -0.5029860367700724,
    -0.15296486218853164,
    0.6740059517812944,
    0.3239847771997537,
    -0.5794684678643381,
    0.3239847771997537,
    0.5794684678643381,
    -0.3239847771997537,
    -0.3239847771997537,
    -0.6740059517812944,
    0.5029860367700724,
    -0.4004672082940195,
    0.15296486218853164,
    -0.7504883828755602,
    0.5029860367700724,
    0.15296486218853164,
    -0.4004672082940195,
    -0.7504883828755602,
    0.4553054119602712,
    0.08164729285680945,
    0.08164729285680945,
    -0.8828161875373585,
    0.8828161875373585,
    -0.08164729285680945,
    -0.08164729285680945,
    -0.4553054119602712,
    0.7504883828755602,
    -0.15296486218853164,
    0.4004672082940195,
    -0.5029860367700724,
    0.7504883828755602,
    0.4004672082940195,
    -0.15296486218853164,
    -0.5029860367700724,
    0.6740059517812944,
    0.3239847771997537,
    0.3239847771997537,
    -0.5794684678643381,
    0.03381941603233842,
    0.03381941603233842,
    0.03381941603233842,
    0.9982828964265062,
    -0.044802370851755174,
    -0.044802370851755174,
    0.508629699630796,
    0.8586508742123365,
    -0.044802370851755174,
    0.508629699630796,
    -0.044802370851755174,
    0.8586508742123365,
    -0.12128480194602098,
    0.4321472685365301,
    0.4321472685365301,
    0.7821684431180708,
    0.508629699630796,
    -0.044802370851755174,
    -0.044802370851755174,
    0.8586508742123365,
    0.4321472685365301,
    -0.12128480194602098,
    0.4321472685365301,
    0.7821684431180708,
    0.4321472685365301,
    0.4321472685365301,
    -0.12128480194602098,
    0.7821684431180708,
    0.37968289875261624,
    0.37968289875261624,
    0.37968289875261624,
    0.753341017856078,
    0.03381941603233842,
    0.03381941603233842,
    0.9982828964265062,
    0.03381941603233842,
    -0.044802370851755174,
    0.044802370851755174,
    0.8586508742123365,
    0.508629699630796,
    -0.044802370851755174,
    0.508629699630796,
    0.8586508742123365,
    -0.044802370851755174,
    -0.12128480194602098,
    0.4321472685365301,
    0.7821684431180708,
    0.4321472685365301,
    0.508629699630796,
    -0.044802370851755174,
    0.8586508742123365,
    -0.044802370851755174,
    0.4321472685365301,
    -0.12128480194602098,
    0.7821684431180708,
    0.4321472685365301,
    0.4321472685365301,
    0.4321472685365301,
    0.7821684431180708,
    -0.12128480194602098,
    0.37968289875261624,
    0.37968289875261624,
    0.753341017856078,
    0.37968289875261624,
    0.03381941603233842,
    0.9982828964265062,
    0.03381941603233842,
    0.03381941603233842,
    -0.044802370851755174,
    0.8586508742123365,
    -0.044802370851755174,
    0.508629699630796,
    -0.044802370851755174,
    0.8586508742123365,
    0.508629699630796,
    -0.044802370851755174,
    -0.12128480194602098,
    0.7821684431180708,
    0.4321472685365301,
    0.4321472685365301,
    0.508629699630796,
    0.8586508742123365,
    -0.044802370851755174,
    -0.044802370851755174,
    0.4321472685365301,
    0.7821684431180708,
    -0.12128480194602098,
    0.4321472685365301,
    0.4321472685365301,
    0.7821684431180708,
    0.4321472685365301,
    -0.12128480194602098,
    0.37968289875261624,
    0.753341017856078,
    0.37968289875261624,
    0.37968289875261624,
    0.9982828964265062,
    0.03381941603233842,
    0.03381941603233842,
    0.03381941603233842,
    0.8586508742123365,
    -0.044802370851755174,
    -0.044802370851755174,
    0.508629699630796,
    0.8586508742123365,
    -0.044802370851755174,
    0.508629699630796,
    -0.044802370851755174,
    0.7821684431180708,
    -0.12128480194602098,
    0.4321472685365301,
    0.4321472685365301,
    0.8586508742123365,
    0.508629699630796,
    -0.044802370851755174,
    -0.044802370851755174,
    0.7821684431180708,
    0.4321472685365301,
    -0.12128480194602098,
    0.4321472685365301,
    0.7821684431180708,
    0.4321472685365301,
    0.4321472685365301,
    -0.12128480194602098,
    0.753341017856078,
    0.37968289875261624,
    0.37968289875261624,
    0.37968289875261624,
};
