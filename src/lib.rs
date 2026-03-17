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
    assert_eq!(fg.len(), mask.len() * 4, "fg must have 4× as many elements as mask");
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
            0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 1.0, 1.0,
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
}
