//! Mask generation and application.
//!
//! A mask is a per-pixel scalar (0.0–1.0) that modulates premultiplied RGBA.
//! In premultiplied space, all four channels are multiplied by the mask value.
//!
//! Masks enable transparent output (rounded corners → PNG/WebP/AVIF/JXL)
//! without requiring compositing over a background.

/// Result of filling a mask row. Enables skipping work on uniform rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskFill {
    /// Every pixel is 1.0 — skip the multiply entirely.
    AllOpaque,
    /// Every pixel is 0.0 — zero the entire row.
    AllTransparent,
    /// Mixed values — must call [`mask_row`](crate::mask_row).
    Partial,
}

/// Row-level mask generator.
///
/// Implementations produce one `f32` per pixel (not per channel) for a given
/// output row. Values should be in `[0.0, 1.0]`.
///
/// The trait is object-safe and stateless per-row — `&self` not `&mut self` —
/// because mask geometry doesn't change between rows.
pub trait MaskSource {
    /// Fill `dst` with mask values for row `y`.
    ///
    /// `dst.len()` equals the image width in pixels.
    /// Returns a [`MaskFill`] hint so callers can skip no-op rows.
    fn fill_mask_row(&self, dst: &mut [f32], y: u32) -> MaskFill;
}

/// Antialiased rounded rectangle mask.
///
/// Generates a mask with 1.0 inside the rounded rectangle and 0.0 outside,
/// with a smooth antialiased transition at the corner arcs.
///
/// Supports per-corner radii. A circle is `radius = min(w, h) / 2` on all corners.
///
/// # Example
///
/// ```rust
/// use zenblend::mask::{RoundedRectMask, MaskSource, MaskFill};
///
/// let mask = RoundedRectMask::new(100, 100, [20.0; 4]);
/// let mut row = vec![0.0f32; 100];
/// match mask.fill_mask_row(&mut row, 5) {
///     MaskFill::Partial => { /* row has mixed values */ }
///     MaskFill::AllOpaque => { /* skip — all 1.0 */ }
///     MaskFill::AllTransparent => { /* zero the pixel row */ }
/// }
/// ```
pub struct RoundedRectMask {
    width: u32,
    height: u32,
    /// Per-corner radii: [top_left, top_right, bottom_right, bottom_left].
    radii: [f32; 4],
    /// Circle centers for each corner.
    centers: [(f32, f32); 4],
    /// Maximum radius (determines which rows need corner processing).
    max_radius: f32,
}

impl RoundedRectMask {
    /// Create a rounded rectangle mask.
    ///
    /// `radii` is `[top_left, top_right, bottom_right, bottom_left]` in pixels.
    /// Each radius is clamped to `min(width, height) / 2`.
    pub fn new(width: u32, height: u32, radii: [f32; 4]) -> Self {
        let max_r = (width.min(height) as f32) / 2.0;
        let radii = [
            radii[0].clamp(0.0, max_r),
            radii[1].clamp(0.0, max_r),
            radii[2].clamp(0.0, max_r),
            radii[3].clamp(0.0, max_r),
        ];
        let w = width as f32;
        let h = height as f32;
        let centers = [
            (radii[0], radii[0]),         // top-left
            (w - radii[1], radii[1]),     // top-right
            (w - radii[2], h - radii[2]), // bottom-right
            (radii[3], h - radii[3]),     // bottom-left
        ];
        let max_radius = radii.iter().copied().fold(0.0f32, f32::max);
        Self {
            width,
            height,
            radii,
            centers,
            max_radius,
        }
    }

    /// Create a uniform rounded rectangle (same radius on all corners).
    pub fn uniform(width: u32, height: u32, radius: f32) -> Self {
        Self::new(width, height, [radius; 4])
    }

    /// Create a circle mask (radius = min(width, height) / 2).
    pub fn circle(width: u32, height: u32) -> Self {
        let r = (width.min(height) as f32) / 2.0;
        Self::new(width, height, [r; 4])
    }
}

impl MaskSource for RoundedRectMask {
    fn fill_mask_row(&self, dst: &mut [f32], y: u32) -> MaskFill {
        debug_assert_eq!(dst.len(), self.width as usize);
        let h = self.height as f32;
        let yf = y as f32 + 0.5; // pixel center

        // Fast path: row is entirely inside all corner arcs
        let in_top = yf >= self.max_radius;
        let in_bottom = yf <= h - self.max_radius;
        if in_top && in_bottom {
            dst.fill(1.0);
            return MaskFill::AllOpaque;
        }

        // Row may intersect corners. Start with all 1.0.
        dst.fill(1.0);

        let mut any_partial = false;

        // Check each corner that this row intersects
        for (i, &(cx, cy)) in self.centers.iter().enumerate() {
            let r = self.radii[i];
            if r <= 0.0 {
                continue;
            }

            // Does this row fall within this corner's y-range?
            let in_corner_y = match i {
                0 => yf < r,     // top-left: y < radius
                1 => yf < r,     // top-right: y < radius
                2 => yf > h - r, // bottom-right: y > height - radius
                3 => yf > h - r, // bottom-left: y > height - radius
                _ => false,
            };
            if !in_corner_y {
                continue;
            }

            let dy = yf - cy;
            let dy2 = dy * dy;

            // Anti-aliasing zone: 1 pixel wide, centered on the circle edge.
            // r - 0.5 = fully inside, r + 0.5 = fully outside.
            let r_inner = r - 0.5;
            let r_outer = r + 0.5;
            let r_inner2 = r_inner * r_inner;
            let r_outer2 = r_outer * r_outer;

            // X range where the outer circle intersects this row
            let x_extent_outer = if r_outer2 > dy2 {
                libm::sqrtf(r_outer2 - dy2)
            } else {
                0.0
            };

            // X range affected by this corner
            let (x_start, x_end) = match i {
                0 | 3 => {
                    // Left corners: affect pixels 0..cx+x_extent
                    let start = 0usize;
                    let end = ((cx + x_extent_outer).ceil() as usize).min(self.width as usize);
                    (start, end)
                }
                1 | 2 => {
                    // Right corners: affect pixels cx-x_extent..width
                    let start = ((cx - x_extent_outer).floor().max(0.0)) as usize;
                    let end = self.width as usize;
                    (start, end)
                }
                _ => continue,
            };

            #[allow(clippy::needless_range_loop)]
            for x in x_start..x_end {
                let xf = x as f32 + 0.5;
                let dx = xf - cx;

                // Is this pixel inside the corner's bounding box?
                let in_corner_x = match i {
                    0 => xf < cx, // top-left: x < center_x
                    1 => xf > cx, // top-right: x > center_x
                    2 => xf > cx, // bottom-right: x > center_x
                    3 => xf < cx, // bottom-left: x < center_x
                    _ => false,
                };
                if !in_corner_x {
                    continue;
                }

                let dist2 = dx * dx + dy2;

                if dist2 >= r_outer2 {
                    // Outside circle — fully transparent
                    dst[x] = 0.0;
                    any_partial = true;
                } else if dist2 > r_inner2 {
                    // In the AA zone — smooth ramp
                    let dist = libm::sqrtf(dist2);
                    let coverage = r_outer - dist; // 1.0 at inner edge, 0.0 at outer edge
                    dst[x] = coverage.clamp(0.0, 1.0);
                    any_partial = true;
                }
                // else: inside circle — already 1.0
            }
        }

        if any_partial {
            MaskFill::Partial
        } else {
            // Row was entirely opaque (corners didn't clip anything)
            MaskFill::AllOpaque
        }
    }
}

/// Linear gradient mask.
///
/// Produces a gradient from 0.0 to 1.0 along the vector from `start` to `end`
/// (in pixel coordinates). Pixels before `start` are 0.0, after `end` are 1.0.
///
/// # Example
///
/// ```rust
/// use zenblend::mask::{LinearGradientMask, MaskSource};
///
/// // Top-to-bottom gradient on a 100×200 image
/// let mask = LinearGradientMask::new(100, 200, (50.0, 0.0), (50.0, 200.0));
/// let mut row = vec![0.0f32; 100];
/// mask.fill_mask_row(&mut row, 0);   // near 0.0
/// mask.fill_mask_row(&mut row, 199); // near 1.0
/// ```
pub struct LinearGradientMask {
    width: u32,
    // Direction vector (normalized)
    dx: f32,
    dy: f32,
    // 1.0 / gradient_length
    inv_len: f32,
    // Start point for origin reference
    sx: f32,
    sy: f32,
}

impl LinearGradientMask {
    /// Create a linear gradient mask.
    ///
    /// `start` and `end` are in pixel coordinates. The gradient ramps from 0.0
    /// at `start` to 1.0 at `end` along the vector between them.
    pub fn new(width: u32, _height: u32, start: (f32, f32), end: (f32, f32)) -> Self {
        let dx = end.0 - start.0;
        let dy = end.1 - start.1;
        let len = libm::sqrtf(dx * dx + dy * dy);
        let (dx, dy, inv_len) = if len > 0.0 {
            (dx / len, dy / len, 1.0 / len)
        } else {
            (0.0, 0.0, 0.0)
        };
        Self {
            width,
            dx,
            dy,
            inv_len,
            sx: start.0,
            sy: start.1,
        }
    }
}

impl MaskSource for LinearGradientMask {
    fn fill_mask_row(&self, dst: &mut [f32], y: u32) -> MaskFill {
        debug_assert_eq!(dst.len(), self.width as usize);
        let yf = y as f32 + 0.5;

        // If gradient has zero length, everything is opaque
        if self.inv_len <= 0.0 {
            dst.fill(1.0);
            return MaskFill::AllOpaque;
        }

        // Project first and last pixel centers onto the gradient vector
        let first_proj = ((0.5 - self.sx) * self.dx + (yf - self.sy) * self.dy) * self.inv_len;
        let last_proj = ((self.width as f32 - 0.5 - self.sx) * self.dx + (yf - self.sy) * self.dy)
            * self.inv_len;

        let (min_t, max_t) = if first_proj < last_proj {
            (first_proj, last_proj)
        } else {
            (last_proj, first_proj)
        };

        if min_t >= 1.0 {
            dst.fill(1.0);
            return MaskFill::AllOpaque;
        }
        if max_t <= 0.0 {
            dst.fill(0.0);
            return MaskFill::AllTransparent;
        }

        for (x, d) in dst.iter_mut().enumerate() {
            let xf = x as f32 + 0.5;
            let proj = ((xf - self.sx) * self.dx + (yf - self.sy) * self.dy) * self.inv_len;
            *d = proj.clamp(0.0, 1.0);
        }

        if min_t >= 1.0 - f32::EPSILON && max_t <= 1.0 + f32::EPSILON {
            MaskFill::AllOpaque
        } else if max_t <= f32::EPSILON {
            MaskFill::AllTransparent
        } else {
            MaskFill::Partial
        }
    }
}

/// Radial gradient mask.
///
/// 1.0 inside `inner_radius`, 0.0 outside `outer_radius`,
/// linear ramp between. The gradient is centered at `center`.
///
/// # Example
///
/// ```rust
/// use zenblend::mask::{RadialGradientMask, MaskSource};
///
/// // Spotlight centered at (50, 50)
/// let mask = RadialGradientMask::new(100, 100, (50.0, 50.0), 10.0, 40.0);
/// let mut row = vec![0.0f32; 100];
/// mask.fill_mask_row(&mut row, 50); // center row: inner=1.0, ramp, outer=0.0
/// ```
pub struct RadialGradientMask {
    width: u32,
    cx: f32,
    cy: f32,
    inner_r: f32,
    outer_r: f32,
    inv_ramp: f32, // 1.0 / (outer_r - inner_r)
}

impl RadialGradientMask {
    /// Create a radial gradient mask.
    ///
    /// - `center`: center point in pixel coordinates
    /// - `inner_radius`: fully opaque (1.0) inside this radius
    /// - `outer_radius`: fully transparent (0.0) outside this radius
    pub fn new(
        width: u32,
        _height: u32,
        center: (f32, f32),
        inner_radius: f32,
        outer_radius: f32,
    ) -> Self {
        let inner_r = inner_radius.max(0.0);
        let outer_r = outer_radius.max(inner_r);
        let ramp = outer_r - inner_r;
        let inv_ramp = if ramp > 0.0 { 1.0 / ramp } else { 0.0 };
        Self {
            width,
            cx: center.0,
            cy: center.1,
            inner_r,
            outer_r,
            inv_ramp,
        }
    }
}

impl MaskSource for RadialGradientMask {
    fn fill_mask_row(&self, dst: &mut [f32], y: u32) -> MaskFill {
        debug_assert_eq!(dst.len(), self.width as usize);
        let yf = y as f32 + 0.5;
        let dy = yf - self.cy;
        let dy2 = dy * dy;

        // Fast path: if the closest possible pixel is beyond outer_r, all transparent
        // The closest x to center is cx (clamped to row).
        let min_dx = if self.cx < 0.5 {
            0.5 - self.cx
        } else if self.cx > self.width as f32 - 0.5 {
            self.cx - (self.width as f32 - 0.5)
        } else {
            0.0
        };
        let min_dist2 = min_dx * min_dx + dy2;
        if min_dist2 >= self.outer_r * self.outer_r {
            dst.fill(0.0);
            return MaskFill::AllTransparent;
        }

        // Fast path: if the farthest possible pixel is within inner_r, all opaque
        let far_x = if self.cx < self.width as f32 * 0.5 {
            self.width as f32 - 0.5
        } else {
            0.5
        };
        let max_dx = (far_x - self.cx).abs();
        let max_dist2 = max_dx * max_dx + dy2;
        if max_dist2 <= self.inner_r * self.inner_r {
            dst.fill(1.0);
            return MaskFill::AllOpaque;
        }

        let mut all_opaque = true;
        let mut all_transparent = true;

        for (x, d) in dst.iter_mut().enumerate() {
            let xf = x as f32 + 0.5;
            let dx = xf - self.cx;
            let dist = libm::sqrtf(dx * dx + dy2);

            let v = if dist <= self.inner_r {
                1.0
            } else if dist >= self.outer_r {
                0.0
            } else {
                // Linear ramp: 1.0 at inner_r, 0.0 at outer_r
                1.0 - (dist - self.inner_r) * self.inv_ramp
            };
            *d = v;

            if v < 1.0 {
                all_opaque = false;
            }
            if v > 0.0 {
                all_transparent = false;
            }
        }

        if all_opaque {
            MaskFill::AllOpaque
        } else if all_transparent {
            MaskFill::AllTransparent
        } else {
            MaskFill::Partial
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn center_row_all_opaque() {
        let mask = RoundedRectMask::uniform(100, 100, 20.0);
        let mut row = vec![0.0f32; 100];
        // Row 50 is well inside all corners
        let fill = mask.fill_mask_row(&mut row, 50);
        assert_eq!(fill, MaskFill::AllOpaque);
        assert!(row.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn corner_row_is_partial() {
        let mask = RoundedRectMask::uniform(100, 100, 20.0);
        let mut row = vec![0.0f32; 100];
        // Row 0 intersects the top corners
        let fill = mask.fill_mask_row(&mut row, 0);
        assert_eq!(fill, MaskFill::Partial);
        // First pixel should be 0 (outside top-left corner)
        assert_eq!(row[0], 0.0);
        // Middle pixel should be 1.0 (between corners)
        assert_eq!(row[50], 1.0);
        // Last pixel should be 0 (outside top-right corner)
        assert_eq!(row[99], 0.0);
    }

    #[test]
    fn zero_radius_all_opaque() {
        let mask = RoundedRectMask::uniform(100, 100, 0.0);
        let mut row = vec![0.0f32; 100];
        for y in 0..100 {
            let fill = mask.fill_mask_row(&mut row, y);
            assert_eq!(fill, MaskFill::AllOpaque);
        }
    }

    #[test]
    fn circle_mask_symmetry() {
        let mask = RoundedRectMask::circle(100, 100);
        let mut row_top = vec![0.0f32; 100];
        let mut row_bot = vec![0.0f32; 100];
        mask.fill_mask_row(&mut row_top, 10);
        mask.fill_mask_row(&mut row_bot, 89);
        // Should be symmetric
        for x in 0..100 {
            assert!(
                (row_top[x] - row_bot[99 - x]).abs() < 1e-5,
                "asymmetry at x={x}: top={}, bot={}",
                row_top[x],
                row_bot[99 - x],
            );
        }
    }

    #[test]
    fn circle_mask_center_mostly_opaque() {
        let mask = RoundedRectMask::circle(100, 100);
        let mut row = vec![0.0f32; 100];
        // Center row of a 100px circle: nearly all pixels are 1.0,
        // but the very edge pixels may be in the AA zone (> 0.99).
        mask.fill_mask_row(&mut row, 50);
        let near_opaque = row.iter().filter(|&&v| v > 0.99).count();
        assert!(
            near_opaque >= 98,
            "center row should be nearly all opaque, got {near_opaque}/100"
        );
    }

    #[test]
    fn circle_mask_edge_mostly_transparent() {
        let mask = RoundedRectMask::circle(100, 100);
        let mut row = vec![0.0f32; 100];
        // Row 0 is at the very top of the circle — mostly outside
        let fill = mask.fill_mask_row(&mut row, 0);
        assert_eq!(fill, MaskFill::Partial);
        let transparent: usize = row.iter().filter(|&&v| v == 0.0).count();
        // Most pixels should be fully transparent at the top edge
        assert!(
            transparent >= 80,
            "expected mostly transparent at top edge, got {transparent}/100 transparent"
        );
    }

    #[test]
    fn per_corner_radii() {
        // Only top-left has radius, others are square
        let mask = RoundedRectMask::new(100, 100, [30.0, 0.0, 0.0, 0.0]);
        let mut row = vec![0.0f32; 100];
        let fill = mask.fill_mask_row(&mut row, 0);
        assert_eq!(fill, MaskFill::Partial);
        // Top-left corner pixel is outside
        assert_eq!(row[0], 0.0);
        // Top-right corner pixel is inside (no radius)
        assert_eq!(row[99], 1.0);
    }

    #[test]
    fn aa_values_in_range() {
        let mask = RoundedRectMask::uniform(200, 200, 50.0);
        let mut row = vec![0.0f32; 200];
        for y in 0..200 {
            mask.fill_mask_row(&mut row, y);
            for (x, &v) in row.iter().enumerate() {
                assert!((0.0..=1.0).contains(&v), "out of range at ({x},{y}): {v}");
            }
        }
    }

    #[test]
    fn aa_zone_has_fractional_values() {
        let mask = RoundedRectMask::uniform(200, 200, 50.0);
        let mut row = vec![0.0f32; 200];
        let mut found_fractional = false;
        for y in 0..200 {
            mask.fill_mask_row(&mut row, y);
            for &v in &row {
                if v > 0.0 && v < 1.0 {
                    found_fractional = true;
                    break;
                }
            }
            if found_fractional {
                break;
            }
        }
        assert!(
            found_fractional,
            "expected fractional AA values in corner arcs"
        );
    }

    // === LinearGradientMask tests ===

    #[test]
    fn linear_gradient_top_to_bottom() {
        let mask = LinearGradientMask::new(10, 100, (5.0, 0.0), (5.0, 100.0));
        let mut row = vec![0.0f32; 10];

        // Row 0: near start → near 0
        let fill = mask.fill_mask_row(&mut row, 0);
        assert_eq!(fill, MaskFill::Partial);
        assert!(row[5] < 0.02, "top should be near 0, got {}", row[5]);

        // Row 99: near end → near 1
        mask.fill_mask_row(&mut row, 99);
        assert!(row[5] > 0.98, "bottom should be near 1, got {}", row[5]);
    }

    #[test]
    fn linear_gradient_monotonic() {
        let mask = LinearGradientMask::new(1, 200, (0.5, 0.0), (0.5, 200.0));
        let mut row = vec![0.0f32; 1];
        let mut prev = -1.0f32;
        for y in 0..200 {
            mask.fill_mask_row(&mut row, y);
            assert!(
                row[0] >= prev - 1e-6,
                "not monotonic at y={y}: prev={prev}, cur={}",
                row[0]
            );
            prev = row[0];
        }
    }

    #[test]
    fn linear_gradient_endpoints() {
        let mask = LinearGradientMask::new(100, 1, (0.0, 0.5), (100.0, 0.5));
        let mut row = vec![0.0f32; 100];
        mask.fill_mask_row(&mut row, 0);
        // First pixel center (0.5) → 0.5/100 ≈ 0.005
        assert!(row[0] < 0.02);
        // Last pixel center (99.5) → 99.5/100 ≈ 0.995
        assert!(row[99] > 0.98);
    }

    #[test]
    fn linear_gradient_all_opaque() {
        // Gradient ends before the image starts (in y)
        let mask = LinearGradientMask::new(10, 100, (5.0, -200.0), (5.0, -100.0));
        let mut row = vec![0.0f32; 10];
        let fill = mask.fill_mask_row(&mut row, 0);
        assert_eq!(fill, MaskFill::AllOpaque);
    }

    #[test]
    fn linear_gradient_all_transparent() {
        // Gradient starts after the image ends
        let mask = LinearGradientMask::new(10, 100, (5.0, 200.0), (5.0, 300.0));
        let mut row = vec![0.0f32; 10];
        let fill = mask.fill_mask_row(&mut row, 0);
        assert_eq!(fill, MaskFill::AllTransparent);
    }

    // === RadialGradientMask tests ===

    #[test]
    fn radial_gradient_center_opaque() {
        let mask = RadialGradientMask::new(100, 100, (50.0, 50.0), 10.0, 40.0);
        let mut row = vec![0.0f32; 100];
        mask.fill_mask_row(&mut row, 50);
        // Center pixel (50.5, 50.5): distance ≈ 0.7 < inner_r=10 → 1.0
        assert_eq!(row[50], 1.0);
    }

    #[test]
    fn radial_gradient_outside_zero() {
        let mask = RadialGradientMask::new(100, 100, (50.0, 50.0), 10.0, 40.0);
        let mut row = vec![0.0f32; 100];
        mask.fill_mask_row(&mut row, 50);
        // Pixel at x=0 (0.5, 50.5): distance ≈ 49.5 > outer_r=40 → 0.0
        assert_eq!(row[0], 0.0);
    }

    #[test]
    fn radial_gradient_symmetry() {
        let mask = RadialGradientMask::new(100, 100, (50.0, 50.0), 10.0, 40.0);
        let mut row = vec![0.0f32; 100];
        mask.fill_mask_row(&mut row, 50);
        // Symmetric around center
        for x in 0..50 {
            assert!(
                (row[x] - row[99 - x]).abs() < 1e-5,
                "asymmetry at x={x}: left={}, right={}",
                row[x],
                row[99 - x]
            );
        }
    }

    #[test]
    fn radial_gradient_ramp() {
        let mask = RadialGradientMask::new(200, 200, (100.0, 100.0), 20.0, 80.0);
        let mut row = vec![0.0f32; 200];
        mask.fill_mask_row(&mut row, 100);
        // Check that values between inner and outer are fractional
        let mut found_fractional = false;
        for &v in &row {
            if v > 0.0 && v < 1.0 {
                found_fractional = true;
                break;
            }
        }
        assert!(found_fractional, "expected fractional values in ramp zone");
    }

    #[test]
    fn radial_gradient_far_row_transparent() {
        let mask = RadialGradientMask::new(100, 100, (50.0, 50.0), 5.0, 10.0);
        let mut row = vec![0.0f32; 100];
        // Row 0: dy=49.5, far beyond outer_r=10
        let fill = mask.fill_mask_row(&mut row, 0);
        assert_eq!(fill, MaskFill::AllTransparent);
    }
}
