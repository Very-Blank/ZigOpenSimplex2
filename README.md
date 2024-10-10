# OpenSimplex 2
A Zig translation of: <a href="https://github.com/KdotJPG/OpenSimplex2">OpenSimplex2</a>

Fast version should now be completely translation error free. As for the smooth version there are some mistakes still in the 3D and 4D noise base functions.

Successors to OpenSimplex Noise, plus updated OpenSimplex. Includes 2D, 3D, and 4D noise.

## Motivation

Legacy OpenSimplex used a different grid layout scheme to Simplex. It worked fine for common applications, but didn't produce as consistent contrast as the original Simplex point layout in 3D/4D. It also broke down completely when extended to 5D+, producing diagonal bands of visibly-varying value ranges.

Instead of vertex rearrangement, OpenSimplex2 focuses on congruent layouts constructed and traversed using disparate methods where need was concluded.

Gradient vector tables were also revisited to improve probability symmetry in both the old and new noise.

## Included

### OpenSimplex2(F)
 * Looks the most like Simplex.
 * Is about as fast as common Simplex implementations.
 * 3D is implemented by constructing a rotated body-centered-cubic grid as the offset union of two rotated cubic grids, and finding the closest two points on each.
 * 4D is implemented using 5 offset copies of the dual (reverse-skewed) grid to produce the target grid. This technique generalizes to any number of dimensions.
 * Bundled 2D function is just 2D Simplex with a nice gradient table. The 4D technique does extend to 2D, but this was found to be faster and I couldn't find evidence of limitations on its use. 3D and 4D are novel contributions to the field.

### OpenSimplex2S
 * Looks the most like 2014 OpenSimplex.
 * Uses large vertex contribution ranges like 2014 OpenSimplex, but has better uniformity in 3D and 4D.
 * 3D is implemented analogously to OpenSimplex2(F), but finds the closest four points on each to match the larger radius.
 * 4D is implemented using the ordinary skew and a pre-generated 4x4x4x4 lookup table. I have a work-in-progress more-procedural implementation in my project backlog, which I will swap in if I find it to be competitive in performance.
 * 2D is based on Simplex, but uses a different process to find which points are in range.
 * Recommended choice for ridged noise (if passing individual layers into `abs(x)`).
