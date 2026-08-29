//! Blend modes for premultiplied linear f32 RGBA pixel rows.
//!
//! `zenblend` provides Porter-Duff and artistic blend operations on
//! premultiplied linear f32 data. All functions operate on `&mut [f32]`
//! slices (4ch RGBA, interleaved). SIMD-accelerated where available.
//!
//! # Usage
//!
//! ```rust
//! use zenblend::{BlendMode, blend_row, blend_row_solid, blend_row_solid_opaque};
//!
//! let mut fg = vec![0.5, 0.0, 0.0, 0.5,  0.0, 0.3, 0.0, 1.0];
//! let bg =     vec![0.0, 0.3, 0.0, 1.0,  0.0, 0.0, 0.5, 0.5];
//! blend_row(&mut fg, &bg, BlendMode::SrcOver);
//! ```
//!
//! All pixel data must be **premultiplied linear f32**, 4 channels (RGBA).
//! Lengths must be equal and divisible by 4.

#![forbid(unsafe_code)]

mod blend;
pub mod mask;
mod simd;

/// Dev-only: the ORIGINAL unpremultiplying Overlay, kept reachable so
/// `benches/tier_isolation.rs` can measure the new premultiplied form against
/// what actually shipped before — the tier arm compares against the premul
/// form at 1 lane, which is a different question.
#[cfg(feature = "_dev")]
#[doc(hidden)]
pub fn __bench_overlay_unpremul(fg: &mut [f32], bg: &[f32]) {
    crate::blend::__overlay_unpremul_reference(fg, bg)
}

/// Dev-only originals for the other two reduced modes.
#[cfg(feature = "_dev")]
#[doc(hidden)]
pub fn __bench_linear_light_unpremul(fg: &mut [f32], bg: &[f32]) {
    crate::blend::__linear_light_unpremul_reference(fg, bg)
}
#[cfg(feature = "_dev")]
#[doc(hidden)]
pub fn __bench_pin_light_unpremul(fg: &mut [f32], bg: &[f32]) {
    crate::blend::__pin_light_unpremul_reference(fg, bg)
}
#[cfg(feature = "_dev")]
#[doc(hidden)]
pub fn __bench_color_dodge_unpremul(fg: &mut [f32], bg: &[f32]) {
    crate::blend::__color_dodge_unpremul_reference(fg, bg)
}
#[cfg(feature = "_dev")]
#[doc(hidden)]
pub fn __bench_divide_unpremul(fg: &mut [f32], bg: &[f32]) {
    crate::blend::__divide_unpremul_reference(fg, bg)
}
#[cfg(feature = "_dev")]
#[doc(hidden)]
pub fn __bench_color_burn_unpremul(fg: &mut [f32], bg: &[f32]) {
    crate::blend::__color_burn_unpremul_reference(fg, bg)
}
#[cfg(feature = "_dev")]
#[doc(hidden)]
pub fn __bench_vivid_light_unpremul(fg: &mut [f32], bg: &[f32]) {
    crate::blend::__vivid_light_unpremul_reference(fg, bg)
}
#[cfg(feature = "_dev")]
#[doc(hidden)]
pub fn __bench_hard_mix_unpremul(fg: &mut [f32], bg: &[f32]) {
    crate::blend::__hard_mix_unpremul_reference(fg, bg)
}
#[cfg(feature = "_dev")]
#[doc(hidden)]
pub fn __bench_soft_light_unpremul(fg: &mut [f32], bg: &[f32]) {
    crate::blend::__soft_light_unpremul_reference(fg, bg)
}

/// Porter-Duff and artistic blend modes.
///
/// Blend mode selection happens once per row (match on the enum), not per pixel.
/// The inner loop for each mode is a tight SIMD kernel.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BlendMode {
    // === Porter-Duff ===
    /// Output = 0. Clears both color and alpha.
    Clear,
    /// Output = Src. Background is completely replaced.
    Src,
    /// Output = Dst. Foreground is completely ignored.
    Dst,
    /// Output = Src + Dst * (1 - Src.a). Standard alpha compositing.
    #[default]
    SrcOver,
    /// Output = Dst + Src * (1 - Dst.a).
    DstOver,
    /// Output = Src * Dst.a.
    SrcIn,
    /// Output = Dst * Src.a.
    DstIn,
    /// Output = Src * (1 - Dst.a).
    SrcOut,
    /// Output = Dst * (1 - Src.a).
    DstOut,
    /// Output = Src * Dst.a + Dst * (1 - Src.a).
    SrcAtop,
    /// Output = Dst * Src.a + Src * (1 - Dst.a).
    DstAtop,
    /// Output = Src * (1 - Dst.a) + Dst * (1 - Src.a).
    Xor,

    // === Artistic (separable) ===
    /// Multiply: Src * Dst. Darkens.
    Multiply,
    /// Screen: Src + Dst - Src * Dst. Lightens.
    Screen,
    /// Overlay: Multiply if Dst < 0.5, Screen otherwise.
    Overlay,
    /// Darken: min(Src, Dst).
    Darken,
    /// Lighten: max(Src, Dst).
    Lighten,
    /// Hard light: Multiply if Src < 0.5, Screen otherwise.
    HardLight,
    /// Soft light (W3C formula).
    SoftLight,
    /// Color dodge: Dst / (1 - Src).
    ColorDodge,
    /// Color burn: 1 - (1 - Dst) / Src.
    ColorBurn,
    /// Difference: |Src - Dst|.
    Difference,
    /// Exclusion: Src + Dst - 2 * Src * Dst.
    Exclusion,

    // === Additional separable modes ===
    /// Linear burn: max(0, Src + Dst - 1). Additive darken.
    LinearBurn,
    /// Linear dodge: min(1, Src + Dst). Additive lighten.
    LinearDodge,
    /// Vivid light: ColorBurn(2·Src) if Src < 0.5, ColorDodge(2·Src - 1) otherwise.
    VividLight,
    /// Linear light: LinearBurn(2·Src) if Src < 0.5, LinearDodge(2·Src - 1) otherwise.
    LinearLight,
    /// Pin light: Darken(2·Src) if Src < 0.5, Lighten(2·Src - 1) otherwise.
    PinLight,
    /// Hard mix: 0 or 1 per channel (threshold via VividLight).
    HardMix,
    /// Divide: min(1, Dst / Src). Flat-field correction.
    Divide,
    /// Subtract: max(0, Dst - Src).
    Subtract,
    /// Plus (SVG/CSS): clamp(S + D, 0, 1) on premultiplied values directly.
    /// Unlike artistic modes, this operates on premultiplied data without unpremultiply.
    Plus,
}

/// Blend foreground over background in-place.
///
/// Both `fg` (destination, modified) and `bg` (source of background pixels)
/// must be premultiplied linear f32, 4ch RGBA. Lengths must be equal and
/// divisible by 4.
///
/// After this call, `fg` contains the blended result.
///
/// # Panics
///
/// Panics if `fg.len() != bg.len()` or lengths are not divisible by 4.
#[inline]
pub fn blend_row(fg: &mut [f32], bg: &[f32], mode: BlendMode) {
    assert_eq!(fg.len(), bg.len(), "fg and bg must have equal length");
    assert_eq!(fg.len() % 4, 0, "length must be divisible by 4");
    blend::dispatch_blend_row(fg, bg, mode);
}

/// Blend foreground over a solid background pixel in-place.
///
/// `fg` is premultiplied linear f32, 4ch RGBA (modified in-place).
/// `pixel` is a single premultiplied linear f32 RGBA pixel.
///
/// More efficient than `blend_row` for solid backgrounds: the pixel
/// stays in registers, no row buffer needed.
///
/// # Panics
///
/// Panics if `fg.len()` is not divisible by 4.
#[inline]
pub fn blend_row_solid(fg: &mut [f32], pixel: &[f32; 4], mode: BlendMode) {
    assert_eq!(fg.len() % 4, 0, "length must be divisible by 4");
    blend::dispatch_blend_row_solid(fg, pixel, mode);
}

/// Blend foreground over a solid opaque background pixel in-place.
///
/// Like [`blend_row_solid`], but the background pixel is known to be opaque
/// (alpha = 1.0). For `SrcOver`, the output alpha is forced to 1.0,
/// avoiding a multiply.
///
/// # Panics
///
/// Panics if `fg.len()` is not divisible by 4.
#[inline]
pub fn blend_row_solid_opaque(fg: &mut [f32], pixel: &[f32; 4], mode: BlendMode) {
    assert_eq!(fg.len() % 4, 0, "length must be divisible by 4");
    blend::dispatch_blend_row_solid_opaque(fg, pixel, mode);
}

/// Multiply premultiplied RGBA row by a per-pixel mask.
///
/// `mask` has one `f32` per pixel (`mask.len() == fg.len() / 4`).
/// Each mask value is broadcast to all 4 channels of the corresponding pixel.
/// In premultiplied space, this correctly modulates both color and alpha.
///
/// Use [`mask::MaskFill`] hints from [`mask::MaskSource::fill_mask_row`] to skip
/// no-op rows (all-opaque) or zero entire rows (all-transparent).
///
/// # Panics
///
/// Panics if `fg.len() != mask.len() * 4` or `fg.len()` is not divisible by 4.
#[inline]
pub fn mask_row(fg: &mut [f32], mask: &[f32]) {
    assert_eq!(
        fg.len(),
        mask.len() * 4,
        "fg must have 4× as many elements as mask"
    );
    assert_eq!(fg.len() % 4, 0, "fg length must be divisible by 4");
    simd::mask_row_apply(fg, mask);
}

/// Multiply premultiplied RGBA row by a constant mask value.
///
/// Equivalent to `mask_row` with a uniform mask, but avoids the mask buffer.
///
/// # Panics
///
/// Panics if `fg.len()` is not divisible by 4.
#[inline]
pub fn mask_row_constant(fg: &mut [f32], alpha: f32) {
    assert_eq!(fg.len() % 4, 0, "fg length must be divisible by 4");
    for v in fg.iter_mut() {
        *v *= alpha;
    }
}

/// Apply a mask to a premultiplied RGBA row using span hints.
///
/// Uses [`mask::MaskSource::mask_spans`] to identify opaque, transparent, and partial
/// regions. Opaque spans are skipped entirely. Transparent spans are zeroed.
/// Only partial spans invoke per-pixel mask multiplication — typically a small
/// fraction of the row (e.g., corner arcs for rounded rectangles).
///
/// `mask_buf` is a scratch buffer with one `f32` per pixel (`fg.len() / 4`).
/// Only the partial-span portions are filled by the mask source.
///
/// # Robustness against buggy `MaskSource` implementations
///
/// Spans returned by the trait impl are validated before use: they must be
/// non-overlapping, ordered by `start`, fit inside `[0, width]`, and cover
/// every pixel in the row. If validation fails (whether from a bug or a
/// hostile impl), this function falls back to filling the entire row via
/// [`mask::MaskSource::fill_mask_row`] and applying it as a single Partial
/// span — no panic, no slice OOB, no skipped pixels.
///
/// # Panics
///
/// Panics if `fg.len() != mask_buf.len() * 4` or `fg.len()` is not divisible by 4.
#[inline]
pub fn apply_mask_spans(fg: &mut [f32], mask_buf: &mut [f32], mask: &dyn mask::MaskSource, y: u32) {
    assert_eq!(
        fg.len(),
        mask_buf.len() * 4,
        "fg must have 4× as many elements as mask_buf"
    );
    assert_eq!(fg.len() % 4, 0, "fg length must be divisible by 4");

    let width = mask_buf.len() as u32;
    let mut spans = mask.mask_spans(mask_buf, y);

    // Validate spans before alignment — alignment assumes a well-formed input.
    // A buggy/hostile MaskSource impl could produce spans that overflow the
    // row, overlap, or leave gaps; align_to + the per-span slice indexing
    // below would panic on slice OOB. Fall back to a safe full-row Partial
    // path instead of crashing the caller.
    if !spans_cover_row(&spans, width) {
        let fill = mask.fill_mask_row(mask_buf, y);
        match fill {
            mask::MaskFill::AllOpaque => return,
            mask::MaskFill::AllTransparent => {
                fg.fill(0.0);
                return;
            }
            mask::MaskFill::Partial => {
                simd::mask_row_apply(fg, mask_buf);
                return;
            }
        }
    }

    // `align_to` widens Partial spans out to vector boundaries, absorbing a
    // few pixels from the neighbouring Opaque/Transparent spans. Its safety
    // argument is "expanding into Opaque territory is an identity multiply,
    // into Transparent territory it zeroes" — but that is only true if
    // `mask_buf` actually *holds* 1.0 / 0.0 at those pixels, and a MaskSource
    // is entitled to leave them untouched: before alignment they sat in a
    // uniform span that is never read. `RoundedRectMask` does exactly that —
    // it fills only the two corner regions and leaves the opaque centre alone
    // (see `mask_spans`), so with an odd corner extent the widened Partial
    // span used to reach over stale scratch bytes and multiply the pixel data
    // by them. Snapshot the pre-alignment spans and write the constants that
    // argument assumes, so it holds for every MaskSource impl.
    let pre = spans.clone();
    spans.align_to(mask::mask_pixel_align());
    materialize_aligned_margins(&pre, &spans, mask_buf);

    for span in spans.iter() {
        let px_start = span.start as usize;
        let px_end = span.end as usize;
        // Defensive clamp: align_to should preserve coverage, but guard
        // against any future change that could produce out-of-range indices.
        if px_end > mask_buf.len() || px_start > px_end {
            continue;
        }
        let ch_start = px_start * 4;
        let ch_end = px_end * 4;

        match span.kind {
            mask::SpanKind::Opaque => {} // skip
            mask::SpanKind::Transparent => {
                fg[ch_start..ch_end].fill(0.0);
            }
            mask::SpanKind::Partial => {
                simd::mask_row_apply(&mut fg[ch_start..ch_end], &mask_buf[px_start..px_end]);
            }
        }
    }
}

/// Write the uniform mask values that [`mask::MaskSpans::align_to`]'s
/// expansion assumes are already present.
///
/// A pixel that alignment moved out of an Opaque span and into a Partial one
/// must read as `1.0`, and one moved out of a Transparent span must read as
/// `0.0`. The mask source had no reason to write either — before alignment
/// those pixels were in a span the applier never reads — so the values are
/// whatever the caller's scratch buffer happened to contain (commonly zeros,
/// or the previous row's mask, both of which corrupt the output).
///
/// Only the absorbed margins are touched: a post-alignment Partial span
/// intersects a *pre*-alignment Partial span exactly where the source already
/// filled the buffer, and those pre-spans are skipped here. Everything else in
/// the intersection is at most `align - 1` pixels per boundary.
fn materialize_aligned_margins(
    pre: &mask::MaskSpans,
    post: &mask::MaskSpans,
    mask_buf: &mut [f32],
) {
    for p in post.iter() {
        if p.kind != mask::SpanKind::Partial {
            continue;
        }
        for q in pre.iter() {
            let value = match q.kind {
                mask::SpanKind::Opaque => 1.0,
                mask::SpanKind::Transparent => 0.0,
                // Already filled by the mask source — leave it alone.
                mask::SpanKind::Partial => continue,
            };
            let lo = (p.start.max(q.start) as usize).min(mask_buf.len());
            let hi = (p.end.min(q.end) as usize).min(mask_buf.len());
            if lo < hi {
                mask_buf[lo..hi].fill(value);
            }
        }
    }
}

/// Validate that spans are well-formed for a row of the given width.
///
/// Requires: spans non-empty, ordered by `start`, non-overlapping, all within
/// `[0, width]`, every pixel `0..width` covered exactly once. Empty rows
/// (width == 0) are valid only with no spans.
fn spans_cover_row(spans: &mask::MaskSpans, width: u32) -> bool {
    if width == 0 {
        return spans.is_empty();
    }
    if spans.is_empty() {
        return false;
    }
    let mut prev_end = 0u32;
    let mut first = true;
    for span in spans.iter() {
        if span.start > span.end || span.end > width {
            return false;
        }
        if first {
            if span.start != 0 {
                return false;
            }
            first = false;
        } else if span.start != prev_end {
            return false;
        }
        prev_end = span.end;
    }
    prev_end == width
}

/// Multiply R, G, B by per-pixel mask; leave alpha untouched.
///
/// `mask` has one `f32` per pixel (`mask.len() == fg.len() / 4`).
/// Each mask value is multiplied against the R, G, B channels of the
/// corresponding pixel while the alpha channel is preserved.
///
/// Use case: gain map application, vignette without opacity change,
/// color grading masks.
///
/// # Panics
///
/// Panics if `fg.len() != mask.len() * 4` or `fg.len()` is not divisible by 4.
#[inline]
pub fn mask_row_rgb(fg: &mut [f32], mask: &[f32]) {
    assert_eq!(
        fg.len(),
        mask.len() * 4,
        "fg must have 4× as many elements as mask"
    );
    assert_eq!(fg.len() % 4, 0, "fg length must be divisible by 4");
    simd::mask_row_rgb_apply(fg, mask);
}

/// Linearly interpolate between two RGBA rows using a per-pixel factor.
///
/// `out[px*4+c] = a[px*4+c] + (b[px*4+c] - a[px*4+c]) * t[px]`
///
/// `t` has one `f32` per pixel. `a`, `b`, `out` have 4ch RGBA.
/// When `t=0` → `a`, when `t=1` → `b`, when `t=0.5` → midpoint.
///
/// Use case: mask-gated adjustments — interpolate between original and adjusted image.
///
/// # Panics
///
/// Panics if slices have mismatched lengths or aren't divisible by 4.
#[inline]
pub fn lerp_row(a: &[f32], b: &[f32], t: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len(), "a and b must have equal length");
    assert_eq!(a.len(), out.len(), "a and out must have equal length");
    assert_eq!(a.len(), t.len() * 4, "a must have 4× as many elements as t");
    assert_eq!(a.len() % 4, 0, "length must be divisible by 4");
    simd::lerp_row_apply(a, b, t, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn src_over_opaque_fg_ignores_bg() {
        let mut fg = [1.0, 0.0, 0.0, 1.0];
        let bg = [0.0, 1.0, 0.0, 1.0];
        blend_row(&mut fg, &bg, BlendMode::SrcOver);
        assert_eq!(fg, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn src_over_transparent_fg_passes_bg() {
        let mut fg = [0.0, 0.0, 0.0, 0.0];
        let bg = [0.0, 0.5, 0.0, 1.0];
        blend_row(&mut fg, &bg, BlendMode::SrcOver);
        assert_eq!(fg, [0.0, 0.5, 0.0, 1.0]);
    }

    #[test]
    fn src_over_semi_transparent() {
        // 50% red over opaque green
        let mut fg = [0.5, 0.0, 0.0, 0.5];
        let bg = [0.0, 1.0, 0.0, 1.0];
        blend_row(&mut fg, &bg, BlendMode::SrcOver);
        assert!((fg[0] - 0.5).abs() < 1e-6);
        assert!((fg[1] - 0.5).abs() < 1e-6);
        assert!((fg[2] - 0.0).abs() < 1e-6);
        assert!((fg[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn src_over_solid() {
        let mut fg = [0.0, 0.0, 0.0, 0.0];
        let pixel = [0.0, 0.25, 0.0, 0.5];
        blend_row_solid(&mut fg, &pixel, BlendMode::SrcOver);
        assert_eq!(fg, [0.0, 0.25, 0.0, 0.5]);
    }

    #[test]
    fn src_over_solid_opaque() {
        let mut fg = [0.3, 0.0, 0.0, 0.3];
        let pixel = [1.0, 1.0, 1.0, 1.0];
        blend_row_solid_opaque(&mut fg, &pixel, BlendMode::SrcOver);
        assert!((fg[0] - 1.0).abs() < 1e-6);
        assert!((fg[1] - 0.7).abs() < 1e-6);
        assert!((fg[2] - 0.7).abs() < 1e-6);
        assert_eq!(fg[3], 1.0);
    }

    #[test]
    fn src_over_multi_pixel() {
        let mut fg = [
            1.0, 0.0, 0.0, 1.0, // opaque red
            0.0, 0.25, 0.0, 0.5, // 50% green
            0.0, 0.0, 0.0, 0.0, // transparent
        ];
        let bg = [
            0.0, 0.0, 1.0, 1.0, // opaque blue
            0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
        ];
        blend_row(&mut fg, &bg, BlendMode::SrcOver);

        // Pixel 0: opaque red → stays
        assert_eq!(&fg[0..4], &[1.0, 0.0, 0.0, 1.0]);
        // Pixel 1: 50% green over blue
        assert!((fg[4] - 0.0).abs() < 1e-6);
        assert!((fg[5] - 0.25).abs() < 1e-6);
        assert!((fg[6] - 0.5).abs() < 1e-6);
        assert!((fg[7] - 1.0).abs() < 1e-6);
        // Pixel 2: transparent → blue
        assert_eq!(&fg[8..12], &[0.0, 0.0, 1.0, 1.0]);
    }

    // === Porter-Duff operator tests ===

    #[test]
    fn clear_mode() {
        let mut fg = [0.5, 0.3, 0.1, 0.7];
        let bg = [0.1, 0.2, 0.3, 1.0];
        blend_row(&mut fg, &bg, BlendMode::Clear);
        assert_eq!(fg, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn src_mode() {
        let mut fg = [0.5, 0.3, 0.1, 0.7];
        let bg = [0.1, 0.2, 0.3, 1.0];
        blend_row(&mut fg, &bg, BlendMode::Src);
        assert_eq!(fg, [0.5, 0.3, 0.1, 0.7]);
    }

    #[test]
    fn dst_mode() {
        let mut fg = [0.5, 0.3, 0.1, 0.7];
        let bg = [0.1, 0.2, 0.3, 1.0];
        blend_row(&mut fg, &bg, BlendMode::Dst);
        assert_eq!(fg, [0.1, 0.2, 0.3, 1.0]);
    }

    #[test]
    fn dst_over_mode() {
        // dst_over = dst + src * (1 - dst.a)
        let mut fg = [0.5, 0.0, 0.0, 0.5];
        let bg = [0.0, 0.3, 0.0, 0.6];
        blend_row(&mut fg, &bg, BlendMode::DstOver);
        // out = bg + fg * (1 - bg.a) = [0.0, 0.3, 0.0, 0.6] + [0.5, 0.0, 0.0, 0.5] * 0.4
        //     = [0.2, 0.3, 0.0, 0.8]
        assert!((fg[0] - 0.2).abs() < 1e-6);
        assert!((fg[1] - 0.3).abs() < 1e-6);
        assert!((fg[2] - 0.0).abs() < 1e-6);
        assert!((fg[3] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn src_in_mode() {
        // src_in = src * dst.a
        let mut fg = [0.5, 0.3, 0.1, 0.7];
        let bg = [0.1, 0.2, 0.3, 0.5];
        blend_row(&mut fg, &bg, BlendMode::SrcIn);
        assert!((fg[0] - 0.25).abs() < 1e-6);
        assert!((fg[1] - 0.15).abs() < 1e-6);
        assert!((fg[2] - 0.05).abs() < 1e-6);
        assert!((fg[3] - 0.35).abs() < 1e-6);
    }

    #[test]
    fn dst_in_mode() {
        // dst_in = dst * src.a
        let mut fg = [0.5, 0.3, 0.1, 0.7];
        let bg = [0.4, 0.2, 0.6, 1.0];
        blend_row(&mut fg, &bg, BlendMode::DstIn);
        assert!((fg[0] - 0.28).abs() < 1e-6);
        assert!((fg[1] - 0.14).abs() < 1e-6);
        assert!((fg[2] - 0.42).abs() < 1e-6);
        assert!((fg[3] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn src_out_mode() {
        // src_out = src * (1 - dst.a)
        let mut fg = [0.5, 0.3, 0.1, 0.7];
        let bg = [0.1, 0.2, 0.3, 0.4];
        blend_row(&mut fg, &bg, BlendMode::SrcOut);
        assert!((fg[0] - 0.3).abs() < 1e-6);
        assert!((fg[1] - 0.18).abs() < 1e-6);
        assert!((fg[2] - 0.06).abs() < 1e-6);
        assert!((fg[3] - 0.42).abs() < 1e-6);
    }

    #[test]
    fn dst_out_mode() {
        // dst_out = dst * (1 - src.a)
        let mut fg = [0.5, 0.3, 0.1, 0.5];
        let bg = [0.4, 0.2, 0.6, 1.0];
        blend_row(&mut fg, &bg, BlendMode::DstOut);
        assert!((fg[0] - 0.2).abs() < 1e-6);
        assert!((fg[1] - 0.1).abs() < 1e-6);
        assert!((fg[2] - 0.3).abs() < 1e-6);
        assert!((fg[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn src_atop_mode() {
        // src_atop = src * dst.a + dst * (1 - src.a)
        let mut fg = [0.5, 0.0, 0.0, 0.5];
        let bg = [0.0, 0.4, 0.0, 0.8];
        blend_row(&mut fg, &bg, BlendMode::SrcAtop);
        // out = [0.5*0.8, 0.0, 0.0, 0.5*0.8] + [0.0, 0.4*0.5, 0.0, 0.8*0.5]
        //     = [0.4, 0.2, 0.0, 0.8]
        assert!((fg[0] - 0.4).abs() < 1e-6);
        assert!((fg[1] - 0.2).abs() < 1e-6);
        assert!((fg[2] - 0.0).abs() < 1e-6);
        assert!((fg[3] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn dst_atop_mode() {
        // dst_atop = dst * src.a + src * (1 - dst.a)
        let mut fg = [0.5, 0.0, 0.0, 0.5];
        let bg = [0.0, 0.4, 0.0, 0.8];
        blend_row(&mut fg, &bg, BlendMode::DstAtop);
        // out = [0.0*0.5, 0.4*0.5, 0.0, 0.8*0.5] + [0.5*0.2, 0.0, 0.0, 0.5*0.2]
        //     = [0.1, 0.2, 0.0, 0.5]
        assert!((fg[0] - 0.1).abs() < 1e-6);
        assert!((fg[1] - 0.2).abs() < 1e-6);
        assert!((fg[2] - 0.0).abs() < 1e-6);
        assert!((fg[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn xor_mode() {
        // xor = src * (1 - dst.a) + dst * (1 - src.a)
        let mut fg = [0.5, 0.0, 0.0, 0.5];
        let bg = [0.0, 0.4, 0.0, 0.8];
        blend_row(&mut fg, &bg, BlendMode::Xor);
        // out = [0.5*0.2, 0.0, 0.0, 0.5*0.2] + [0.0, 0.4*0.5, 0.0, 0.8*0.5]
        //     = [0.1, 0.2, 0.0, 0.5]
        assert!((fg[0] - 0.1).abs() < 1e-6);
        assert!((fg[1] - 0.2).abs() < 1e-6);
        assert!((fg[2] - 0.0).abs() < 1e-6);
        assert!((fg[3] - 0.5).abs() < 1e-6);
    }

    // === Artistic blend mode tests ===

    #[test]
    fn multiply_mode() {
        // Both opaque → multiply is just component-wise multiply
        let mut fg = [0.5, 0.3, 0.1, 1.0];
        let bg = [0.4, 0.6, 0.8, 1.0];
        blend_row(&mut fg, &bg, BlendMode::Multiply);
        assert!((fg[0] - 0.2).abs() < 1e-6);
        assert!((fg[1] - 0.18).abs() < 1e-6);
        assert!((fg[2] - 0.08).abs() < 1e-6);
        assert!((fg[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn screen_mode() {
        // Both opaque → screen = s + d - s*d
        let mut fg = [0.5, 0.3, 0.1, 1.0];
        let bg = [0.4, 0.6, 0.8, 1.0];
        blend_row(&mut fg, &bg, BlendMode::Screen);
        assert!((fg[0] - 0.7).abs() < 1e-6);
        assert!((fg[1] - 0.72).abs() < 1e-6);
        assert!((fg[2] - 0.82).abs() < 1e-6);
        assert!((fg[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn soft_light_mode() {
        // W3C SoftLight formula, both opaque.
        // s=0.3, d=0.8: s <= 0.5 → d - (1-2s)*d*(1-d) = 0.8 - 0.4*0.8*0.2 = 0.736
        let mut fg = [0.3, 0.0, 0.0, 1.0];
        let bg = [0.8, 0.0, 0.0, 1.0];
        blend_row(&mut fg, &bg, BlendMode::SoftLight);
        assert!((fg[0] - 0.736).abs() < 1e-5, "s<=0.5 case: {}", fg[0]);

        // s=0.8, d=0.5: s > 0.5, d > 0.25 → g = sqrt(d), result = d + (2s-1)(g-d)
        // g = sqrt(0.5) ≈ 0.7071, result = 0.5 + 0.6*(0.7071-0.5) = 0.6243
        let mut fg2 = [0.8, 0.0, 0.0, 1.0];
        let bg2 = [0.5, 0.0, 0.0, 1.0];
        blend_row(&mut fg2, &bg2, BlendMode::SoftLight);
        let expected = 0.5 + 0.6 * (0.5f32.sqrt() - 0.5);
        assert!(
            (fg2[0] - expected).abs() < 1e-5,
            "s>0.5,d>0.25 case: {} vs {}",
            fg2[0],
            expected
        );

        // s=0.8, d=0.1: s > 0.5, d <= 0.25 → g = ((16d-12)*d+4)*d
        // g = ((1.6-12)*0.1+4)*0.1 = (-10.4*0.1+4)*0.1 = 2.96*0.1 = 0.296
        // result = 0.1 + 0.6*(0.296-0.1) = 0.2176
        let mut fg3 = [0.8, 0.0, 0.0, 1.0];
        let bg3 = [0.1, 0.0, 0.0, 1.0];
        blend_row(&mut fg3, &bg3, BlendMode::SoftLight);
        let g = ((16.0 * 0.1 - 12.0) * 0.1 + 4.0) * 0.1;
        let expected3 = 0.1 + 0.6 * (g - 0.1);
        assert!(
            (fg3[0] - expected3).abs() < 1e-5,
            "s>0.5,d<=0.25 case: {} vs {}",
            fg3[0],
            expected3
        );
    }

    #[test]
    fn difference_mode() {
        // Both opaque → difference = |s - d|
        let mut fg = [0.5, 0.3, 0.8, 1.0];
        let bg = [0.3, 0.6, 0.2, 1.0];
        blend_row(&mut fg, &bg, BlendMode::Difference);
        assert!((fg[0] - 0.2).abs() < 1e-6);
        assert!((fg[1] - 0.3).abs() < 1e-6);
        assert!((fg[2] - 0.6).abs() < 1e-6);
        assert!((fg[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn exclusion_mode() {
        // Both opaque → exclusion = s + d - 2*s*d
        let mut fg = [0.5, 0.3, 0.1, 1.0];
        let bg = [0.4, 0.6, 0.8, 1.0];
        blend_row(&mut fg, &bg, BlendMode::Exclusion);
        assert!((fg[0] - 0.5).abs() < 1e-6); // 0.5 + 0.4 - 2*0.2 = 0.5
        assert!((fg[1] - 0.54).abs() < 1e-6); // 0.3 + 0.6 - 2*0.18 = 0.54
        assert!((fg[2] - 0.74).abs() < 1e-6); // 0.1 + 0.8 - 2*0.08 = 0.74
        assert!((fg[3] - 1.0).abs() < 1e-6);
    }

    // === Additional blend mode tests ===

    #[test]
    fn linear_burn_mode() {
        // Both opaque → max(0, s + d - 1)
        let mut fg = [0.8, 0.3, 0.1, 1.0];
        let bg = [0.4, 0.6, 0.2, 1.0];
        blend_row(&mut fg, &bg, BlendMode::LinearBurn);
        assert!((fg[0] - 0.2).abs() < 1e-6); // 0.8+0.4-1 = 0.2
        assert!((fg[1] - 0.0).abs() < 1e-6); // 0.3+0.6-1 = -0.1 → 0
        assert!((fg[2] - 0.0).abs() < 1e-6); // 0.1+0.2-1 = -0.7 → 0
        assert!((fg[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn linear_dodge_mode() {
        // Both opaque → min(1, s + d)
        let mut fg = [0.3, 0.7, 0.9, 1.0];
        let bg = [0.4, 0.6, 0.2, 1.0];
        blend_row(&mut fg, &bg, BlendMode::LinearDodge);
        assert!((fg[0] - 0.7).abs() < 1e-6); // 0.3+0.4 = 0.7
        assert!((fg[1] - 1.0).abs() < 1e-6); // 0.7+0.6 = 1.3 → 1
        assert!((fg[2] - 1.0).abs() < 1e-6); // 0.9+0.2 = 1.1 → 1
        assert!((fg[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn subtract_mode() {
        // Both opaque → max(0, d - s)
        let mut fg = [0.3, 0.7, 0.1, 1.0];
        let bg = [0.5, 0.2, 0.8, 1.0];
        blend_row(&mut fg, &bg, BlendMode::Subtract);
        assert!((fg[0] - 0.2).abs() < 1e-6); // 0.5-0.3 = 0.2
        assert!((fg[1] - 0.0).abs() < 1e-6); // 0.2-0.7 → 0
        assert!((fg[2] - 0.7).abs() < 1e-6); // 0.8-0.1 = 0.7
        assert!((fg[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn divide_mode() {
        // Both opaque → min(1, d / s)
        let mut fg = [0.5, 0.8, 0.1, 1.0];
        let bg = [0.25, 0.4, 0.5, 1.0];
        blend_row(&mut fg, &bg, BlendMode::Divide);
        assert!((fg[0] - 0.5).abs() < 1e-6); // 0.25/0.5 = 0.5
        assert!((fg[1] - 0.5).abs() < 1e-6); // 0.4/0.8 = 0.5
        assert!((fg[2] - 1.0).abs() < 1e-6); // 0.5/0.1 = 5.0 → 1
        assert!((fg[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn plus_mode_premul() {
        // Plus operates on premultiplied values directly
        let mut fg = [0.3, 0.2, 0.0, 0.5];
        let bg = [0.0, 0.4, 0.3, 0.6];
        blend_row(&mut fg, &bg, BlendMode::Plus);
        assert!((fg[0] - 0.3).abs() < 1e-6);
        assert!((fg[1] - 0.6).abs() < 1e-6);
        assert!((fg[2] - 0.3).abs() < 1e-6);
        assert!((fg[3] - 1.0).abs() < 1e-6); // 0.5+0.6 = 1.1 → 1.0
    }

    #[test]
    fn plus_commutative() {
        let a = [0.3, 0.2, 0.1, 0.5];
        let b = [0.1, 0.4, 0.3, 0.6];
        let mut fg1 = a;
        blend_row(&mut fg1, &b, BlendMode::Plus);
        let mut fg2 = b;
        blend_row(&mut fg2, &a, BlendMode::Plus);
        for i in 0..4 {
            assert!(
                (fg1[i] - fg2[i]).abs() < 1e-6,
                "Plus not commutative at {i}"
            );
        }
    }

    #[test]
    fn multiply_white_identity() {
        // Multiply with white = identity
        let original = [0.3, 0.5, 0.8, 1.0];
        let mut fg = original;
        let bg = [1.0, 1.0, 1.0, 1.0];
        blend_row(&mut fg, &bg, BlendMode::Multiply);
        for i in 0..4 {
            assert!(
                (fg[i] - original[i]).abs() < 1e-6,
                "Multiply(white) not identity at {i}"
            );
        }
    }

    #[test]
    fn subtract_self_is_zero() {
        let color = [0.5, 0.3, 0.8, 1.0];
        let mut fg = color;
        let bg = color;
        blend_row(&mut fg, &bg, BlendMode::Subtract);
        assert!((fg[0] - 0.0).abs() < 1e-6);
        assert!((fg[1] - 0.0).abs() < 1e-6);
        assert!((fg[2] - 0.0).abs() < 1e-6);
        assert!((fg[3] - 1.0).abs() < 1e-6); // alpha: SrcOver formula
    }

    #[test]
    fn vivid_light_mode() {
        // Both opaque, s=0.25 (<0.5) → ColorBurn(0.5, d)
        let mut fg = [0.25, 0.75, 0.5, 1.0];
        let bg = [0.8, 0.4, 0.6, 1.0];
        blend_row(&mut fg, &bg, BlendMode::VividLight);
        // s=0.25: ColorBurn(2*0.25=0.5, 0.8) = 1-(1-0.8)/0.5 = 1-0.4 = 0.6
        assert!((fg[0] - 0.6).abs() < 1e-5);
        assert!((fg[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn hard_mix_mode() {
        // Both opaque: threshold via VividLight → 0 or 1
        let mut fg = [0.9, 0.1, 0.5, 1.0];
        let bg = [0.8, 0.2, 0.5, 1.0];
        blend_row(&mut fg, &bg, BlendMode::HardMix);
        assert!(
            fg[0] == 0.0 || fg[0] == 1.0,
            "HardMix must be 0 or 1, got {}",
            fg[0]
        );
        assert!(
            fg[1] == 0.0 || fg[1] == 1.0,
            "HardMix must be 0 or 1, got {}",
            fg[1]
        );
        assert!((fg[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn pin_light_mode() {
        // Both opaque, s=0.2 → Darken(2*0.2, d) = min(0.4, d)
        let mut fg = [0.2, 0.8, 0.5, 1.0];
        let bg = [0.6, 0.3, 0.5, 1.0];
        blend_row(&mut fg, &bg, BlendMode::PinLight);
        assert!((fg[0] - 0.4).abs() < 1e-6); // min(0.4, 0.6) = 0.4
        // s=0.8 → Lighten(2*0.8-1, d) = max(0.6, 0.3) = 0.6
        assert!((fg[1] - 0.6).abs() < 1e-6);
        assert!((fg[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn linear_light_mode() {
        // Both opaque
        // s=0.25 → LinearBurn(0.5, d) = max(0, 0.5 + d - 1)
        let mut fg = [0.25, 0.75, 0.5, 1.0];
        let bg = [0.8, 0.4, 0.6, 1.0];
        blend_row(&mut fg, &bg, BlendMode::LinearLight);
        assert!((fg[0] - 0.3).abs() < 1e-5); // max(0, 0.5+0.8-1) = 0.3
        // s=0.75 → LinearDodge(0.5, d) = min(1, 0.5+d)
        assert!((fg[1] - 0.9).abs() < 1e-5); // min(1, 0.5+0.4) = 0.9
        assert!((fg[3] - 1.0).abs() < 1e-6);
    }

    // === Property tests ===

    #[test]
    fn src_over_identity_with_transparent_bg() {
        // SrcOver with transparent bg = identity
        let original = [0.3, 0.5, 0.1, 0.7];
        let mut fg = original;
        let bg = [0.0, 0.0, 0.0, 0.0];
        blend_row(&mut fg, &bg, BlendMode::SrcOver);
        for i in 0..4 {
            assert!((fg[i] - original[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn dst_over_identity_with_transparent_fg() {
        // DstOver with transparent fg = bg unchanged
        let mut fg = [0.0, 0.0, 0.0, 0.0];
        let bg = [0.3, 0.5, 0.1, 0.7];
        blend_row(&mut fg, &bg, BlendMode::DstOver);
        for i in 0..4 {
            assert!((fg[i] - bg[i]).abs() < 1e-6);
        }
    }

    // === mask_row_rgb tests ===

    #[test]
    fn mask_row_rgb_identity() {
        let mut fg = [0.3, 0.5, 0.8, 0.9];
        let mask = [1.0];
        mask_row_rgb(&mut fg, &mask);
        assert_eq!(fg, [0.3, 0.5, 0.8, 0.9]); // identity, alpha preserved
    }

    #[test]
    fn mask_row_rgb_zero_preserves_alpha() {
        let mut fg = [0.3, 0.5, 0.8, 0.9];
        let mask = [0.0];
        mask_row_rgb(&mut fg, &mask);
        assert_eq!(fg[0], 0.0);
        assert_eq!(fg[1], 0.0);
        assert_eq!(fg[2], 0.0);
        assert_eq!(fg[3], 0.9); // alpha untouched
    }

    #[test]
    fn mask_row_rgb_half() {
        let mut fg = [0.4, 0.6, 0.8, 1.0];
        let mask = [0.5];
        mask_row_rgb(&mut fg, &mask);
        assert!((fg[0] - 0.2).abs() < 1e-6);
        assert!((fg[1] - 0.3).abs() < 1e-6);
        assert!((fg[2] - 0.4).abs() < 1e-6);
        assert_eq!(fg[3], 1.0); // alpha untouched
    }

    #[test]
    fn mask_row_rgb_multi_pixel() {
        let mut fg = [0.4, 0.6, 0.8, 1.0, 0.2, 0.4, 0.6, 0.5, 0.1, 0.3, 0.5, 0.8];
        let mask = [0.5, 1.0, 0.0];
        mask_row_rgb(&mut fg, &mask);
        // Pixel 0: RGB halved, alpha preserved
        assert!((fg[0] - 0.2).abs() < 1e-6);
        assert_eq!(fg[3], 1.0);
        // Pixel 1: identity
        assert!((fg[4] - 0.2).abs() < 1e-6);
        assert_eq!(fg[7], 0.5);
        // Pixel 2: RGB zeroed, alpha preserved
        assert_eq!(fg[8], 0.0);
        assert_eq!(fg[11], 0.8);
    }

    // === lerp_row tests ===

    #[test]
    fn lerp_row_t_zero_is_a() {
        let a = [0.1, 0.2, 0.3, 0.4];
        let b = [0.9, 0.8, 0.7, 0.6];
        let t = [0.0];
        let mut out = [0.0; 4];
        lerp_row(&a, &b, &t, &mut out);
        for i in 0..4 {
            assert!((out[i] - a[i]).abs() < 1e-6, "t=0 should give a[{i}]");
        }
    }

    #[test]
    fn lerp_row_t_one_is_b() {
        let a = [0.1, 0.2, 0.3, 0.4];
        let b = [0.9, 0.8, 0.7, 0.6];
        let t = [1.0];
        let mut out = [0.0; 4];
        lerp_row(&a, &b, &t, &mut out);
        for i in 0..4 {
            assert!((out[i] - b[i]).abs() < 1e-6, "t=1 should give b[{i}]");
        }
    }

    #[test]
    fn lerp_row_t_half_is_midpoint() {
        let a = [0.0, 0.2, 0.4, 0.6];
        let b = [1.0, 0.8, 0.6, 0.4];
        let t = [0.5];
        let mut out = [0.0; 4];
        lerp_row(&a, &b, &t, &mut out);
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-6);
        assert!((out[2] - 0.5).abs() < 1e-6);
        assert!((out[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn lerp_row_multi_pixel() {
        let a = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let b = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let t = [0.25, 0.75];
        let mut out = [0.0; 8];
        lerp_row(&a, &b, &t, &mut out);
        // Pixel 0: 0 + (1-0)*0.25 = 0.25
        for val in &out[..4] {
            assert!((val - 0.25).abs() < 1e-6);
        }
        // Pixel 1: 1 + (0-1)*0.75 = 0.25
        for val in &out[4..8] {
            assert!((val - 0.25).abs() < 1e-6);
        }
    }

    // =========================================================================
    // Hardening tests (M1, M2 — security-audit-2026-05-06)
    // =========================================================================

    /// Buggy `MaskSource` impls that return malformed spans must not panic via
    /// slice OOB. `apply_mask_spans` validates spans and falls back to the
    /// per-pixel `fill_mask_row` path when validation fails.
    mod hostile_mask {
        use crate::mask::{MaskFill, MaskSource, MaskSpan, MaskSpans, SpanKind};

        /// Returns a span past the end of the row.
        pub struct OutOfBoundsSpans;
        impl MaskSource for OutOfBoundsSpans {
            fn fill_mask_row(&self, dst: &mut [f32], _y: u32) -> MaskFill {
                dst.fill(0.5);
                MaskFill::Partial
            }
            fn mask_spans(&self, dst: &mut [f32], _y: u32) -> MaskSpans {
                let w = dst.len() as u32;
                let mut s = MaskSpans::new();
                // start past width — would OOB on slice indexing
                s.push(MaskSpan {
                    start: 0,
                    end: w + 100,
                    kind: SpanKind::Partial,
                });
                s
            }
        }

        /// Returns spans with a gap in coverage.
        pub struct GappySpans;
        impl MaskSource for GappySpans {
            fn fill_mask_row(&self, dst: &mut [f32], _y: u32) -> MaskFill {
                dst.fill(1.0);
                MaskFill::AllOpaque
            }
            fn mask_spans(&self, dst: &mut [f32], _y: u32) -> MaskSpans {
                let w = dst.len() as u32;
                let mut s = MaskSpans::new();
                // Coverage stops at half — pixels [w/2..w) silently never processed
                s.push(MaskSpan {
                    start: 0,
                    end: w / 2,
                    kind: SpanKind::Opaque,
                });
                s
            }
        }

        /// Returns overlapping spans.
        pub struct OverlappingSpans;
        impl MaskSource for OverlappingSpans {
            fn fill_mask_row(&self, dst: &mut [f32], _y: u32) -> MaskFill {
                dst.fill(0.5);
                MaskFill::Partial
            }
            fn mask_spans(&self, dst: &mut [f32], _y: u32) -> MaskSpans {
                let w = dst.len() as u32;
                let mut s = MaskSpans::new();
                s.push(MaskSpan {
                    start: 0,
                    end: w,
                    kind: SpanKind::Opaque,
                });
                s.push(MaskSpan {
                    start: 0, // overlap
                    end: w,
                    kind: SpanKind::Opaque,
                });
                s
            }
        }

        /// Returns no spans at all.
        pub struct EmptySpans;
        impl MaskSource for EmptySpans {
            fn fill_mask_row(&self, dst: &mut [f32], _y: u32) -> MaskFill {
                dst.fill(1.0);
                MaskFill::AllOpaque
            }
            fn mask_spans(&self, _dst: &mut [f32], _y: u32) -> MaskSpans {
                MaskSpans::new()
            }
        }

        /// Returns reversed range (start > end).
        pub struct ReversedSpan;
        impl MaskSource for ReversedSpan {
            fn fill_mask_row(&self, dst: &mut [f32], _y: u32) -> MaskFill {
                dst.fill(0.0);
                MaskFill::AllTransparent
            }
            fn mask_spans(&self, dst: &mut [f32], _y: u32) -> MaskSpans {
                let w = dst.len() as u32;
                let mut s = MaskSpans::new();
                s.push(MaskSpan {
                    start: w,
                    end: 0,
                    kind: SpanKind::Transparent,
                });
                s
            }
        }
    }

    #[test]
    fn apply_mask_spans_oob_does_not_panic() {
        let mut fg = vec![0.5f32; 400];
        let mut buf = vec![0.0f32; 100];
        // Must not panic — fallback path uses fill_mask_row instead.
        apply_mask_spans(&mut fg, &mut buf, &hostile_mask::OutOfBoundsSpans, 0);
        // Fallback applied a partial mask of 0.5 → fg becomes 0.25
        assert!(fg.iter().all(|v| (v - 0.25).abs() < 1e-6));
    }

    #[test]
    fn apply_mask_spans_gappy_falls_back() {
        let mut fg = vec![0.7f32; 400];
        let mut buf = vec![0.0f32; 100];
        // Buggy impl says only [0, 50) is opaque — without validation, [50, 100)
        // would silently retain unprocessed values. With validation, fallback
        // to fill_mask_row covers the full row.
        apply_mask_spans(&mut fg, &mut buf, &hostile_mask::GappySpans, 0);
        // GappySpans's fill_mask_row returns AllOpaque → fg unchanged
        assert!(fg.iter().all(|v| (v - 0.7).abs() < 1e-6));
    }

    #[test]
    fn apply_mask_spans_overlapping_does_not_panic() {
        let mut fg = vec![0.5f32; 400];
        let mut buf = vec![0.0f32; 100];
        apply_mask_spans(&mut fg, &mut buf, &hostile_mask::OverlappingSpans, 0);
        // OverlappingSpans.fill_mask_row returns Partial with mask=0.5 → fg=0.25
        assert!(fg.iter().all(|v| (v - 0.25).abs() < 1e-6));
    }

    #[test]
    fn apply_mask_spans_empty_falls_back() {
        let mut fg = vec![0.7f32; 400];
        let mut buf = vec![0.0f32; 100];
        apply_mask_spans(&mut fg, &mut buf, &hostile_mask::EmptySpans, 0);
        // EmptySpans.fill_mask_row returns AllOpaque → fg unchanged
        assert!(fg.iter().all(|v| (v - 0.7).abs() < 1e-6));
    }

    #[test]
    fn apply_mask_spans_reversed_does_not_panic() {
        let mut fg = vec![0.5f32; 400];
        let mut buf = vec![0.0f32; 100];
        apply_mask_spans(&mut fg, &mut buf, &hostile_mask::ReversedSpan, 0);
        // ReversedSpan.fill_mask_row returns AllTransparent → fg = 0
        assert!(fg.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn artistic_blend_nan_input_does_not_propagate() {
        // Multiply mode: artistic blend. Inject NaN in fg alpha — output must
        // be sanitized to a finite value, not NaN.
        let nan = f32::NAN;
        let mut fg = vec![0.5, 0.5, 0.5, nan];
        let bg = vec![0.5, 0.5, 0.5, 1.0];
        blend_row(&mut fg, &bg, BlendMode::Multiply);
        for (i, &v) in fg.iter().enumerate() {
            assert!(v.is_finite(), "channel {i} is NaN/inf: {v}");
        }
    }

    #[test]
    fn artistic_blend_nan_color_does_not_propagate() {
        // NaN in fg color channel — must not propagate via alpha.
        let nan = f32::NAN;
        let mut fg = vec![nan, 0.5, 0.5, 0.5];
        let bg = vec![0.5, 0.5, 0.5, 0.5];
        blend_row(&mut fg, &bg, BlendMode::Screen);
        // Alpha must remain finite even if a color channel got zeroed
        assert!(fg[3].is_finite(), "alpha became non-finite: {}", fg[3]);
        for (i, &v) in fg.iter().enumerate() {
            assert!(v.is_finite(), "channel {i} is non-finite: {v}");
        }
    }

    #[test]
    fn artistic_blend_color_dodge_division_safe() {
        // ColorDodge: d / (1 - s). With sa near 0, inv_sa = 0, but for sa > 0
        // and unpremultiplied s = fg[i]/sa, if fg[i] is e.g. very large the
        // division could produce inf which our sanitizer must clamp.
        let mut fg = vec![1e30, 1e30, 1e30, 1.0];
        let bg = vec![0.5, 0.5, 0.5, 1.0];
        blend_row(&mut fg, &bg, BlendMode::ColorDodge);
        for (i, &v) in fg.iter().enumerate() {
            assert!(v.is_finite(), "channel {i} is non-finite: {v}");
        }
    }
}

#[cfg(test)]
mod align_margin_tests {
    use super::*;
    use crate::mask::{MaskSource, MaskSpan, MaskSpans, RoundedRectMask, SpanKind};

    /// A pixel value no mask source would ever produce, so reading one proves
    /// the applier touched a pixel nobody filled.
    const SENTINEL: f32 = -12345.0;

    /// `align_to` widens a Partial span over pixels the mask source left
    /// untouched. This pins the repair: those pixels must read as the constant
    /// of the span they were absorbed from.
    ///
    /// Arch-independent — it drives the helper directly rather than going
    /// through `mask_pixel_align()`, which is 1 on non-x86_64 and makes
    /// `align_to` an early-return there.
    #[test]
    fn aligned_margins_get_the_constant_of_the_span_they_came_from() {
        // The shape `RoundedRectMask` emits for a row with an *odd* corner
        // extent: [Partial 0..7] [Opaque 7..193] [Partial 193..200].
        let mut pre = MaskSpans::new();
        pre.push(MaskSpan {
            start: 0,
            end: 7,
            kind: SpanKind::Partial,
        });
        pre.push(MaskSpan {
            start: 7,
            end: 193,
            kind: SpanKind::Opaque,
        });
        pre.push(MaskSpan {
            start: 193,
            end: 200,
            kind: SpanKind::Partial,
        });

        // Only the two corner regions are filled; the opaque centre is
        // whatever the caller's scratch buffer held.
        let mut buf = vec![SENTINEL; 200];
        for v in buf[0..7].iter_mut() {
            *v = 0.25;
        }
        for v in buf[193..200].iter_mut() {
            *v = 0.75;
        }

        let mut post = pre.clone();
        post.align_to(2);
        materialize_aligned_margins(&pre, &post, &mut buf);

        for s in post.iter() {
            if s.kind != SpanKind::Partial {
                continue;
            }
            for (off, &v) in buf[s.start as usize..s.end as usize].iter().enumerate() {
                let x = s.start as usize + off;
                assert_ne!(
                    v, SENTINEL,
                    "aligned Partial span {s:?} still reads unwritten pixel {x}"
                );
            }
        }
        // Pixel 7 was absorbed out of the Opaque span: it must be an identity
        // multiply, which is exactly what `align_to`'s doc promises.
        assert_eq!(buf[7], 1.0, "margin absorbed from Opaque must read 1.0");
        // The source-filled pixels are untouched.
        assert_eq!(buf[0], 0.25);
        assert_eq!(buf[6], 0.25);
        assert_eq!(buf[199], 0.75);
    }

    /// The Transparent half of the same argument.
    #[test]
    fn aligned_margins_absorbed_from_transparent_read_zero() {
        let mut pre = MaskSpans::new();
        pre.push(MaskSpan {
            start: 0,
            end: 5,
            kind: SpanKind::Transparent,
        });
        pre.push(MaskSpan {
            start: 5,
            end: 11,
            kind: SpanKind::Partial,
        });
        pre.push(MaskSpan {
            start: 11,
            end: 16,
            kind: SpanKind::Transparent,
        });

        let mut buf = vec![SENTINEL; 16];
        for v in buf[5..11].iter_mut() {
            *v = 0.5;
        }

        let mut post = pre.clone();
        post.align_to(2);
        materialize_aligned_margins(&pre, &post, &mut buf);

        for s in post.iter() {
            if s.kind != SpanKind::Partial {
                continue;
            }
            for (off, &v) in buf[s.start as usize..s.end as usize].iter().enumerate() {
                let x = s.start as usize + off;
                assert_ne!(
                    v, SENTINEL,
                    "aligned Partial span {s:?} still reads unwritten pixel {x}"
                );
            }
        }
        // 11 was absorbed out of a Transparent span → must zero the pixel.
        assert_eq!(
            buf[11], 0.0,
            "margin absorbed from Transparent must read 0.0"
        );
    }

    /// The real generator, at odd radii, driven through the exact sequence
    /// `apply_mask_spans` uses — but with the alignment pinned to 2 instead of
    /// `mask_pixel_align()`, so this fires on every target, not just x86_64.
    ///
    /// `RoundedRectMask::mask_spans` fills only the two corner regions and
    /// leaves the opaque centre untouched. An odd corner extent puts the
    /// Partial/Opaque boundary on an odd column, so a 2-pixel alignment pulls
    /// an unwritten pixel into the Partial span. Poisoning the buffer makes
    /// that read observable.
    #[test]
    fn real_mask_odd_radii_never_leave_an_unwritten_pixel_in_a_partial_span() {
        let mut checked_any = false;
        for radius in [5.0f32, 7.0, 9.0, 11.0, 13.0, 15.0, 21.0, 23.0] {
            let mask = RoundedRectMask::uniform(200, 200, radius);
            for y in 0..200u32 {
                let mut buf = vec![SENTINEL; 200];
                let pre = mask.mask_spans(&mut buf, y);
                let mut post = pre.clone();
                post.align_to(2);
                materialize_aligned_margins(&pre, &post, &mut buf);

                for s in post.iter() {
                    if s.kind != SpanKind::Partial {
                        continue;
                    }
                    for (off, &v) in buf[s.start as usize..s.end as usize].iter().enumerate() {
                        let x = s.start as usize + off;
                        checked_any = true;
                        assert_ne!(
                            v, SENTINEL,
                            "radius {radius} y={y}: aligned Partial span {s:?} reads pixel {x}, \
                             which mask_spans never wrote"
                        );
                    }
                }
            }
        }
        assert!(checked_any, "test never inspected a Partial span");
    }

    /// End-to-end: `apply_mask_spans` must agree with the naive
    /// fill-then-multiply path for **odd** corner radii, with the scratch
    /// buffer poisoned so any pixel the applier reads without filling shows up.
    ///
    /// All three pre-existing `align_*` tests use even radii (40.0) and a
    /// zeroed scratch, so `align_to` either did not move a boundary or moved it
    /// onto a pixel that happened to read 0.0. Odd radii put the corner extent
    /// on an odd column, which is what makes the 2-pixel x86_64 alignment
    /// expand across it.
    ///
    /// Note this assertion only *fires* on x86_64: `mask_pixel_align()` is 1
    /// elsewhere and `align_to` early-returns, so on other targets it passes
    /// whether or not the margins are materialized. The two helper tests above
    /// carry the invariant on every target.
    #[test]
    fn apply_mask_spans_matches_naive_for_odd_radii() {
        for radius in [5.0f32, 7.0, 9.0, 11.0, 13.0, 15.0, 21.0] {
            let mask = RoundedRectMask::uniform(200, 200, radius);
            for y in [0, 1, 2, 3, 5, 8, 13, 20, 100, 179, 186, 194, 197, 199] {
                let mut pixels_naive = vec![0.7f32; 800];
                let mut buf_naive = vec![SENTINEL; 200];
                match mask.fill_mask_row(&mut buf_naive, y) {
                    mask::MaskFill::AllOpaque => {}
                    mask::MaskFill::AllTransparent => pixels_naive.fill(0.0),
                    mask::MaskFill::Partial => crate::mask_row(&mut pixels_naive, &buf_naive),
                }

                let mut pixels_spans = vec![0.7f32; 800];
                // Poisoned scratch: the span path must never read a pixel it
                // did not fill.
                let mut buf_spans = vec![SENTINEL; 200];
                crate::apply_mask_spans(&mut pixels_spans, &mut buf_spans, &mask, y);

                for (i, (a, b)) in pixels_naive.iter().zip(pixels_spans.iter()).enumerate() {
                    assert!(
                        (a - b).abs() < 1e-6,
                        "radius {radius} y={y} i={i}: naive={a}, spans={b}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod plus_clamp_tests {
    use super::*;

    /// `BlendMode::Plus` documents `clamp(S + D, 0, 1)`. The implementation
    /// only applied the upper bound, so a negative sum — reachable with the
    /// out-of-gamut f32 rows this crate accepts — passed through unclamped.
    #[test]
    fn plus_clamps_below_zero() {
        // Odd pixel count so any future SIMD port's scalar tail is exercised.
        let mut fg = vec![
            -0.5, -1.0, 0.25, 0.5, // sums go negative
            -2.0, 0.1, -0.1, 0.2, //
            0.9, 0.9, 0.9, 0.9, // sum exceeds 1 → upper clamp still applies
        ];
        let bg = vec![
            -0.25, 0.5, 0.25, 0.25, //
            0.5, -0.4, 0.0, 0.1, //
            0.9, 0.9, 0.9, 0.9, //
        ];
        blend_row(&mut fg, &bg, BlendMode::Plus);
        for (i, &v) in fg.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "channel {i} left the documented [0,1] range: {v}"
            );
        }
        // Upper clamp unchanged.
        assert_eq!(fg[8], 1.0);
        // Lower clamp now applied: -0.5 + -0.25 = -0.75 → 0.0.
        assert_eq!(fg[0], 0.0);
    }

    /// Same contract on the solid-pixel dispatch path.
    #[test]
    fn plus_solid_clamps_below_zero() {
        let mut fg = vec![-0.5, -1.0, 0.25, 0.5];
        blend_row_solid(&mut fg, &[-0.25, 0.5, 0.25, 0.25], BlendMode::Plus);
        for (i, &v) in fg.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "channel {i} left the documented [0,1] range: {v}"
            );
        }
    }
}
