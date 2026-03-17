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
            (radii[0], radii[0]),                 // top-left
            (w - radii[1], radii[1]),              // top-right
            (w - radii[2], h - radii[2]),          // bottom-right
            (radii[3], h - radii[3]),              // bottom-left
        ];
        let max_radius = radii.iter().copied().fold(0.0f32, f32::max);
        Self { width, height, radii, centers, max_radius }
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
                0 => yf < r,           // top-left: y < radius
                1 => yf < r,           // top-right: y < radius
                2 => yf > h - r,       // bottom-right: y > height - radius
                3 => yf > h - r,       // bottom-left: y > height - radius
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
                    0 => xf < cx,       // top-left: x < center_x
                    1 => xf > cx,       // top-right: x > center_x
                    2 => xf > cx,       // bottom-right: x > center_x
                    3 => xf < cx,       // bottom-left: x < center_x
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
        assert!(near_opaque >= 98, "center row should be nearly all opaque, got {near_opaque}/100");
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
        assert!(transparent >= 80, "expected mostly transparent at top edge, got {transparent}/100 transparent");
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
                assert!(
                    (0.0..=1.0).contains(&v),
                    "out of range at ({x},{y}): {v}"
                );
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
        assert!(found_fractional, "expected fractional AA values in corner arcs");
    }
}
