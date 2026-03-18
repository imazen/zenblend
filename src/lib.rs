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

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

mod blend;
pub mod mask;
mod simd;

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
/// Uses [`MaskSource::mask_spans`] to identify opaque, transparent, and partial
/// regions. Opaque spans are skipped entirely. Transparent spans are zeroed.
/// Only partial spans invoke per-pixel mask multiplication — typically a small
/// fraction of the row (e.g., corner arcs for rounded rectangles).
///
/// `mask_buf` is a scratch buffer with one `f32` per pixel (`fg.len() / 4`).
/// Only the partial-span portions are filled by the mask source.
///
/// # Panics
///
/// Panics if `fg.len() != mask_buf.len() * 4` or `fg.len()` is not divisible by 4.
#[inline]
pub fn apply_mask_spans(
    fg: &mut [f32],
    mask_buf: &mut [f32],
    mask: &dyn mask::MaskSource,
    y: u32,
) {
    assert_eq!(
        fg.len(),
        mask_buf.len() * 4,
        "fg must have 4× as many elements as mask_buf"
    );
    assert_eq!(fg.len() % 4, 0, "fg length must be divisible by 4");

    let spans = mask.mask_spans(mask_buf, y);

    for span in spans.iter() {
        let px_start = span.start as usize;
        let px_end = span.end as usize;
        let ch_start = px_start * 4;
        let ch_end = px_end * 4;

        match span.kind {
            mask::SpanKind::Opaque => {} // skip
            mask::SpanKind::Transparent => {
                fg[ch_start..ch_end].fill(0.0);
            }
            mask::SpanKind::Partial => {
                simd::mask_row_apply(
                    &mut fg[ch_start..ch_end],
                    &mask_buf[px_start..px_end],
                );
            }
        }
    }
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
    assert_eq!(
        a.len(),
        t.len() * 4,
        "a must have 4× as many elements as t"
    );
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
            assert!((fg1[i] - fg2[i]).abs() < 1e-6, "Plus not commutative at {i}");
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
            assert!((fg[i] - original[i]).abs() < 1e-6, "Multiply(white) not identity at {i}");
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
        assert!(fg[0] == 0.0 || fg[0] == 1.0, "HardMix must be 0 or 1, got {}", fg[0]);
        assert!(fg[1] == 0.0 || fg[1] == 1.0, "HardMix must be 0 or 1, got {}", fg[1]);
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
        let mut fg = [
            0.4, 0.6, 0.8, 1.0,
            0.2, 0.4, 0.6, 0.5,
            0.1, 0.3, 0.5, 0.8,
        ];
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
        let a = [0.0, 0.0, 0.0, 0.0,  1.0, 1.0, 1.0, 1.0];
        let b = [1.0, 1.0, 1.0, 1.0,  0.0, 0.0, 0.0, 0.0];
        let t = [0.25, 0.75];
        let mut out = [0.0; 8];
        lerp_row(&a, &b, &t, &mut out);
        // Pixel 0: 0 + (1-0)*0.25 = 0.25
        for c in 0..4 {
            assert!((out[c] - 0.25).abs() < 1e-6);
        }
        // Pixel 1: 1 + (0-1)*0.75 = 0.25
        for c in 4..8 {
            assert!((out[c] - 0.25).abs() < 1e-6);
        }
    }
}
