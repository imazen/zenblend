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

/// What kind of pixels a span contains.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpanKind {
    /// All pixels are 1.0 — skip entirely.
    Opaque,
    /// All pixels are 0.0 — zero the pixel data.
    Transparent,
    /// Mixed values — fill mask buffer and multiply this range only.
    Partial,
}

/// A contiguous horizontal span with uniform mask behavior.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaskSpan {
    /// First pixel (inclusive).
    pub start: u32,
    /// One past last pixel (exclusive).
    pub end: u32,
    /// What to do with this span.
    pub kind: SpanKind,
}

/// Inline collection of mask spans for one row.
///
/// Most mask shapes produce ≤ 5 spans per row (e.g., rounded rect:
/// transparent, partial, opaque, partial, transparent). The inline
/// capacity of 8 covers all practical cases without allocation.
#[derive(Clone, Debug)]
pub struct MaskSpans {
    spans: [MaskSpan; 8],
    len: u8,
}

impl MaskSpans {
    /// Create an empty span list.
    pub fn new() -> Self {
        Self {
            spans: [MaskSpan {
                start: 0,
                end: 0,
                kind: SpanKind::Opaque,
            }; 8],
            len: 0,
        }
    }

    /// Create a span list with a single span covering the full width.
    pub fn uniform(width: u32, kind: SpanKind) -> Self {
        let mut s = Self::new();
        s.push(MaskSpan {
            start: 0,
            end: width,
            kind,
        });
        s
    }

    /// Add a span. Panics if capacity (8) is exceeded.
    pub fn push(&mut self, span: MaskSpan) {
        assert!(
            (self.len as usize) < self.spans.len(),
            "MaskSpans overflow (max 8)"
        );
        self.spans[self.len as usize] = span;
        self.len += 1;
    }

    /// Number of spans.
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Whether there are no spans.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Iterate over spans.
    pub fn iter(&self) -> impl Iterator<Item = &MaskSpan> {
        self.spans[..self.len as usize].iter()
    }

    /// Convert to the legacy `MaskFill` hint (for callers that don't use spans).
    pub fn to_mask_fill(&self) -> MaskFill {
        if self.len == 1 {
            match self.spans[0].kind {
                SpanKind::Opaque => return MaskFill::AllOpaque,
                SpanKind::Transparent => return MaskFill::AllTransparent,
                SpanKind::Partial => return MaskFill::Partial,
            }
        }
        // Multiple spans → at least one non-uniform region
        MaskFill::Partial
    }

    /// Snap span boundaries to pixel multiples of `align`, expanding Partial
    /// spans and shrinking Opaque/Transparent neighbors.
    ///
    /// `align` is in **pixels** (e.g., 2 for AVX2 4ch RGBA, 4 for AVX-512 4ch).
    /// After alignment, every Partial span's `start` and `end` are multiples of
    /// `align` (clamped to `[0, width]`), so the SIMD kernel processes clean
    /// blocks with no scalar tail.
    ///
    /// This is always safe because:
    /// - Expanding Partial into Opaque territory: mask=1.0, multiply is identity.
    /// - Expanding Partial into Transparent territory: mask=0.0, multiply zeros.
    ///
    /// Spans that shrink to zero width are removed. Adjacent spans with the same
    /// kind after adjustment are merged.
    pub fn align_to(&mut self, align: u32) {
        if align <= 1 || self.len <= 1 {
            return;
        }

        let n = self.len as usize;
        let width = if n > 0 { self.spans[n - 1].end } else { return };

        // Pass 1: expand each Partial span's start down and end up to align boundary.
        for i in 0..n {
            if self.spans[i].kind == SpanKind::Partial {
                self.spans[i].start = round_down(self.spans[i].start, align);
                self.spans[i].end = round_up(self.spans[i].end, align).min(width);
            }
        }

        // Pass 2: resolve overlaps — Partial always wins over neighbors.
        // Walk left to right, clamping each span's start to the previous span's end.
        for i in 1..n {
            if self.spans[i].start < self.spans[i - 1].end {
                // Overlap: the earlier span was shrunk by the later's expansion,
                // or the later span was expanded into the earlier.
                // Whichever is Partial wins.
                if self.spans[i].kind == SpanKind::Partial {
                    // Partial expanded left — shrink the predecessor
                    self.spans[i - 1].end = self.spans[i].start;
                } else if self.spans[i - 1].kind == SpanKind::Partial {
                    // Predecessor expanded right — shrink this span
                    self.spans[i].start = self.spans[i - 1].end;
                } else {
                    // Neither is Partial (shouldn't happen in practice)
                    self.spans[i].start = self.spans[i - 1].end;
                }
            }
        }

        // Pass 3: remove zero-width spans and merge adjacent same-kind spans.
        let mut write = 0usize;
        for read in 0..n {
            if self.spans[read].start >= self.spans[read].end {
                continue; // zero-width, skip
            }
            if write > 0 && self.spans[write - 1].kind == self.spans[read].kind {
                // Merge with previous
                self.spans[write - 1].end = self.spans[read].end;
            } else {
                self.spans[write] = self.spans[read];
                write += 1;
            }
        }
        self.len = write as u8;
    }
}

/// Round down to nearest multiple of `align`.
#[inline]
fn round_down(val: u32, align: u32) -> u32 {
    val / align * align
}

/// Round up to nearest multiple of `align`.
#[inline]
fn round_up(val: u32, align: u32) -> u32 {
    val.div_ceil(align) * align
}

impl Default for MaskSpans {
    fn default() -> Self {
        Self::new()
    }
}

/// Preferred pixel alignment for mask span boundaries on the current platform.
///
/// This is the number of RGBA pixels processed per SIMD iteration in `mask_row`:
/// - x86_64 (AVX2): 2 pixels (8 floats per 256-bit register)
/// - AArch64 (NEON) / WASM32 (SIMD128): 1 pixel (4 floats per 128-bit register)
/// - Scalar fallback: 1 pixel
///
/// Pass this to [`MaskSpans::align_to`] to ensure Partial spans start and end
/// on SIMD block boundaries, eliminating scalar tails in the mask multiply kernel.
pub const fn mask_pixel_align() -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        2
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        1
    }
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

    /// Return spans describing which pixel ranges are opaque, transparent, or partial.
    ///
    /// This enables callers to skip or zero entire SIMD-aligned blocks without
    /// per-pixel mask multiplication. For rounded corners, the vast majority of
    /// pixels are in the opaque span — only the corner edges need actual multiply.
    ///
    /// The default implementation calls [`fill_mask_row`](Self::fill_mask_row) and
    /// returns a single span. Override for masks that can compute spans
    /// analytically (e.g., rounded rectangles know their corner extents from
    /// geometry and only fill partial sub-ranges).
    ///
    /// Spans must be non-overlapping, ordered by `start`, and cover `[0, width)`.
    fn mask_spans(&self, dst: &mut [f32], y: u32) -> MaskSpans {
        let width = dst.len() as u32;
        let fill = self.fill_mask_row(dst, y);
        match fill {
            MaskFill::AllOpaque => MaskSpans::uniform(width, SpanKind::Opaque),
            MaskFill::AllTransparent => MaskSpans::uniform(width, SpanKind::Transparent),
            MaskFill::Partial => MaskSpans::uniform(width, SpanKind::Partial),
        }
    }
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

    /// Fill a sub-range `[x_start..x_end)` of the mask buffer for row `y`.
    ///
    /// Same per-pixel logic as `fill_mask_row` but restricted to the given columns.
    /// Pixels outside corner arcs are set to 1.0; corner-affected pixels get AA coverage.
    fn fill_mask_region(&self, dst: &mut [f32], y: u32, x_start: u32, x_end: u32) {
        let h = self.height as f32;
        let yf = y as f32 + 0.5;
        let xs = x_start as usize;
        let xe = x_end as usize;

        // Start with opaque
        dst[xs..xe].fill(1.0);

        for (i, &(cx, cy)) in self.centers.iter().enumerate() {
            let r = self.radii[i];
            if r <= 0.0 {
                continue;
            }

            let in_corner_y = match i {
                0 => yf < r,
                1 => yf < r,
                2 => yf > h - r,
                3 => yf > h - r,
                _ => false,
            };
            if !in_corner_y {
                continue;
            }

            let dy = yf - cy;
            let dy2 = dy * dy;
            let r_inner = r - 0.5;
            let r_outer = r + 0.5;
            let r_inner2 = r_inner * r_inner;
            let r_outer2 = r_outer * r_outer;

            #[allow(clippy::needless_range_loop)]
            for x in xs..xe {
                let xf = x as f32 + 0.5;
                let dx = xf - cx;

                let in_corner_x = match i {
                    0 => xf < cx,
                    1 => xf > cx,
                    2 => xf > cx,
                    3 => xf < cx,
                    _ => false,
                };
                if !in_corner_x {
                    continue;
                }

                let dist2 = dx * dx + dy2;

                if dist2 >= r_outer2 {
                    dst[x] = 0.0;
                } else if dist2 > r_inner2 {
                    let dist = libm::sqrtf(dist2);
                    let coverage = r_outer - dist;
                    dst[x] = coverage.clamp(0.0, 1.0);
                }
            }
        }
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

    fn mask_spans(&self, dst: &mut [f32], y: u32) -> MaskSpans {
        let h = self.height as f32;
        let w = self.width;
        let yf = y as f32 + 0.5;

        // Fast path: row entirely inside all corner arcs
        let in_top = yf >= self.max_radius;
        let in_bottom = yf <= h - self.max_radius;
        if in_top && in_bottom {
            return MaskSpans::uniform(w, SpanKind::Opaque);
        }

        // Compute the x-extents affected by each active corner.
        // Left side: affected by top-left (i=0) or bottom-left (i=3)
        // Right side: affected by top-right (i=1) or bottom-right (i=2)
        let mut left_end = 0u32;
        let mut right_start = w;

        for (i, &(cx, _cy)) in self.centers.iter().enumerate() {
            let r = self.radii[i];
            if r <= 0.0 {
                continue;
            }

            let in_corner_y = match i {
                0 => yf < r,
                1 => yf < r,
                2 => yf > h - r,
                3 => yf > h - r,
                _ => false,
            };
            if !in_corner_y {
                continue;
            }

            let dy = yf - self.centers[i].1;
            let r_outer = r + 0.5;
            let r_outer2 = r_outer * r_outer;
            let dy2 = dy * dy;
            let x_extent = if r_outer2 > dy2 {
                libm::sqrtf(r_outer2 - dy2)
            } else {
                0.0
            };

            match i {
                0 | 3 => {
                    // Left corners: fill_mask_row checks in_corner_x as xf < cx,
                    // and scans 0..ceil(cx + x_extent). But only pixels with
                    // xf < cx are modified. The rightmost such pixel has
                    // x = ceil(cx - 0.5) - 1, so exclusive end = ceil(cx - 0.5).
                    // Use x_extent to also skip rows where the arc doesn't reach.
                    if x_extent > 0.0 {
                        let end = ((cx - 0.5).ceil().max(0.0) as u32).min(w);
                        left_end = left_end.max(end);
                    }
                }
                1 | 2 => {
                    // Right corners: in_corner_x checks xf > cx. The leftmost
                    // such pixel has x + 0.5 > cx, so x >= ceil(cx - 0.5).
                    // Pixel index: floor(cx + 0.5) = the first pixel with center > cx.
                    if x_extent > 0.0 {
                        let start = ((cx + 0.5).floor().max(0.0)) as u32;
                        right_start = right_start.min(start);
                    }
                }
                _ => {}
            }
        }

        // If no corners affected this row, it's all opaque
        if left_end == 0 && right_start == w {
            return MaskSpans::uniform(w, SpanKind::Opaque);
        }

        // Clamp so left_end <= right_start
        if left_end > right_start {
            // Corners overlap — fill entire row, return single partial span
            self.fill_mask_row(dst, y);
            let mut spans = MaskSpans::new();
            spans.push(MaskSpan {
                start: 0,
                end: w,
                kind: SpanKind::Partial,
            });
            return spans;
        }

        // Fill only the partial regions of the mask buffer, not the entire row.
        // The opaque center (left_end..right_start) is never read by apply_mask_spans.
        if left_end > 0 {
            self.fill_mask_region(dst, y, 0, left_end);
        }
        if right_start < w {
            self.fill_mask_region(dst, y, right_start, w);
        }

        // Build spans: [left partial] [opaque center] [right partial]
        let mut spans = MaskSpans::new();

        if left_end > 0 {
            spans.push(MaskSpan {
                start: 0,
                end: left_end,
                kind: SpanKind::Partial,
            });
        }

        if left_end < right_start {
            spans.push(MaskSpan {
                start: left_end,
                end: right_start,
                kind: SpanKind::Opaque,
            });
        }

        if right_start < w {
            spans.push(MaskSpan {
                start: right_start,
                end: w,
                kind: SpanKind::Partial,
            });
        }

        spans
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

    // === MaskSpans tests ===

    #[test]
    fn spans_all_opaque_center_row() {
        let mask = RoundedRectMask::uniform(100, 100, 20.0);
        let mut row = vec![0.0f32; 100];
        let spans = mask.mask_spans(&mut row, 50);
        assert_eq!(spans.len(), 1);
        let s = spans.iter().next().unwrap();
        assert_eq!(s.kind, SpanKind::Opaque);
        assert_eq!(s.start, 0);
        assert_eq!(s.end, 100);
    }

    #[test]
    fn spans_corner_row_has_partial_and_opaque() {
        let mask = RoundedRectMask::uniform(200, 200, 40.0);
        let mut row = vec![0.0f32; 200];
        // Row 5: intersects top-left and top-right corners
        let spans = mask.mask_spans(&mut row, 5);
        // Should have: [partial left corner] [opaque center] [partial right corner]
        assert!(
            spans.len() >= 2,
            "expected at least 2 spans, got {}",
            spans.len()
        );

        let kinds: Vec<SpanKind> = spans.iter().map(|s| s.kind).collect();
        assert!(
            kinds.contains(&SpanKind::Opaque),
            "expected an opaque span in {:?}",
            kinds
        );
        assert!(
            kinds.contains(&SpanKind::Partial),
            "expected a partial span in {:?}",
            kinds
        );

        // Spans should cover entire width
        let total: u32 = spans.iter().map(|s| s.end - s.start).sum();
        assert_eq!(total, 200);

        // Spans should be ordered and non-overlapping
        let mut prev_end = 0u32;
        for s in spans.iter() {
            assert_eq!(s.start, prev_end, "gap or overlap in spans");
            prev_end = s.end;
        }
    }

    #[test]
    fn spans_opaque_center_is_large() {
        // For a 1000px wide image with 20px corners, the opaque center should
        // be at least 900px — only ~20px on each side affected by corners.
        let mask = RoundedRectMask::uniform(1000, 1000, 20.0);
        let mut row = vec![0.0f32; 1000];
        let spans = mask.mask_spans(&mut row, 5);
        let opaque_pixels: u32 = spans
            .iter()
            .filter(|s| s.kind == SpanKind::Opaque)
            .map(|s| s.end - s.start)
            .sum();
        assert!(
            opaque_pixels >= 900,
            "expected ≥900 opaque pixels, got {opaque_pixels}"
        );
    }

    #[test]
    fn spans_default_impl_single_partial() {
        // LinearGradientMask uses the default mask_spans (no analytical override),
        // which returns a single Partial span for the whole row.
        let mask = LinearGradientMask::new(100, 1, (0.0, 0.5), (100.0, 0.5));
        let mut row = vec![0.0f32; 100];
        let spans = mask.mask_spans(&mut row, 0);
        assert_eq!(spans.len(), 1);
        let s = spans.iter().next().unwrap();
        assert_eq!(s.kind, SpanKind::Partial);
        assert_eq!(s.start, 0);
        assert_eq!(s.end, 100);
    }

    #[test]
    fn spans_apply_matches_fill() {
        // Verify that apply_mask_spans produces the same result as fill_mask_row + mask_row
        let mask = RoundedRectMask::uniform(100, 100, 20.0);

        for y in [0, 5, 10, 50, 90, 95, 99] {
            // Method 1: fill + mask_row
            let mut pixels1 = vec![0.5f32; 400]; // 100 pixels × 4ch
            let mut mask_buf1 = vec![0.0f32; 100];
            let fill = mask.fill_mask_row(&mut mask_buf1, y);
            match fill {
                MaskFill::AllOpaque => {}
                MaskFill::AllTransparent => pixels1.fill(0.0),
                MaskFill::Partial => crate::mask_row(&mut pixels1, &mask_buf1),
            }

            // Method 2: apply_mask_spans
            let mut pixels2 = vec![0.5f32; 400];
            let mut mask_buf2 = vec![0.0f32; 100];
            crate::apply_mask_spans(&mut pixels2, &mut mask_buf2, &mask, y);

            // Results must match
            for (i, (a, b)) in pixels1.iter().zip(pixels2.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "mismatch at y={y} index={i}: fill+mask={a}, spans={b}"
                );
            }
        }
    }

    // === align_to tests ===

    #[test]
    fn align_noop_for_single_span() {
        let mut spans = MaskSpans::uniform(100, SpanKind::Opaque);
        spans.align_to(4);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans.iter().next().unwrap().kind, SpanKind::Opaque);
    }

    #[test]
    fn align_expands_partial_shrinks_neighbors() {
        // [Transparent 0..3] [Partial 3..7] [Opaque 7..20]
        let mut spans = MaskSpans::new();
        spans.push(MaskSpan {
            start: 0,
            end: 3,
            kind: SpanKind::Transparent,
        });
        spans.push(MaskSpan {
            start: 3,
            end: 7,
            kind: SpanKind::Partial,
        });
        spans.push(MaskSpan {
            start: 7,
            end: 20,
            kind: SpanKind::Opaque,
        });

        spans.align_to(4);

        // Partial should expand: start 3→0, end 7→8
        // Transparent shrinks to zero (removed), Opaque shrinks from 7→8
        // Result: [Partial 0..8] [Opaque 8..20]
        assert_eq!(spans.len(), 2, "spans: {:?}", spans);
        let v: Vec<(u32, u32, SpanKind)> = spans.iter().map(|s| (s.start, s.end, s.kind)).collect();
        assert_eq!(v[0], (0, 8, SpanKind::Partial));
        assert_eq!(v[1], (8, 20, SpanKind::Opaque));
    }

    #[test]
    fn align_preserves_coverage() {
        // Alignment must not change total pixel coverage
        let mut spans = MaskSpans::new();
        spans.push(MaskSpan {
            start: 0,
            end: 15,
            kind: SpanKind::Transparent,
        });
        spans.push(MaskSpan {
            start: 15,
            end: 185,
            kind: SpanKind::Opaque,
        });
        spans.push(MaskSpan {
            start: 185,
            end: 200,
            kind: SpanKind::Transparent,
        });

        let total_before: u32 = spans.iter().map(|s| s.end - s.start).sum();
        spans.align_to(4);
        let total_after: u32 = spans.iter().map(|s| s.end - s.start).sum();
        assert_eq!(
            total_before, total_after,
            "coverage changed after alignment"
        );
    }

    #[test]
    fn align_partial_boundaries_are_multiples() {
        let mask = RoundedRectMask::uniform(200, 200, 40.0);
        let mut row = vec![0.0f32; 200];
        let mut spans = mask.mask_spans(&mut row, 5);
        spans.align_to(4);
        for s in spans.iter() {
            if s.kind == SpanKind::Partial {
                assert_eq!(
                    s.start % 4,
                    0,
                    "Partial span start {} not aligned to 4",
                    s.start
                );
                // end can be width (200) which is already aligned
                assert!(
                    s.end % 4 == 0 || s.end == 200,
                    "Partial span end {} not aligned to 4",
                    s.end
                );
            }
        }
    }

    #[test]
    fn align_correctness_with_real_mask() {
        // apply_mask_spans uses align_to internally — verify it still matches
        // the naive fill+mask_row approach
        let mask = RoundedRectMask::uniform(200, 200, 40.0);
        for y in [0, 3, 10, 39, 100, 160, 195, 199] {
            let mut pixels1 = vec![0.7f32; 800];
            let mut mask_buf1 = vec![0.0f32; 200];
            let fill = mask.fill_mask_row(&mut mask_buf1, y);
            match fill {
                MaskFill::AllOpaque => {}
                MaskFill::AllTransparent => pixels1.fill(0.0),
                MaskFill::Partial => crate::mask_row(&mut pixels1, &mask_buf1),
            }

            let mut pixels2 = vec![0.7f32; 800];
            let mut mask_buf2 = vec![0.0f32; 200];
            crate::apply_mask_spans(&mut pixels2, &mut mask_buf2, &mask, y);

            for (i, (a, b)) in pixels1.iter().zip(pixels2.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "mismatch at y={y} i={i}: naive={a}, spans={b}"
                );
            }
        }
    }
}
