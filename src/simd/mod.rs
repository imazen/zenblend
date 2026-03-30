//! SIMD-accelerated blend kernels.
//!
//! Uses archmage `incant!` dispatch to select the best available implementation:
//! - x86_64: AVX2+FMA via magetypes `f32x8` (2 pixels/iter)
//! - AArch64 / WASM32 / scalar: portable `f32x4<T>` kernels (1 pixel/iter)

mod portable;

#[cfg(target_arch = "x86_64")]
mod x86;
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use x86::*;

// ---------------------------------------------------------------------------
// Portable kernel wrappers for incant! dispatch.
//
// Each token type needs named functions with the right suffix (_scalar, _neon,
// _wasm128). All delegate to the same generic kernels in `portable`.
// ---------------------------------------------------------------------------

use archmage::ScalarToken;

pub(crate) fn blend_src_over_row_scalar(t: ScalarToken, fg: &mut [f32], bg: &[f32]) {
    portable::blend_src_over_row(t, fg, bg);
}
pub(crate) fn blend_src_over_solid_scalar(t: ScalarToken, fg: &mut [f32], px: &[f32; 4]) {
    portable::blend_src_over_solid(t, fg, px);
}
pub(crate) fn blend_src_over_solid_opaque_scalar(t: ScalarToken, fg: &mut [f32], px: &[f32; 4]) {
    portable::blend_src_over_solid_opaque(t, fg, px);
}
pub(crate) fn mask_row_apply_scalar(t: ScalarToken, fg: &mut [f32], mask: &[f32]) {
    portable::mask_row_apply(t, fg, mask);
}
pub(crate) fn mask_row_rgb_apply_scalar(t: ScalarToken, fg: &mut [f32], mask: &[f32]) {
    portable::mask_row_rgb_apply(t, fg, mask);
}
pub(crate) fn lerp_row_apply_scalar(
    t: ScalarToken,
    a: &[f32],
    b: &[f32],
    tv: &[f32],
    out: &mut [f32],
) {
    portable::lerp_row_apply(t, a, b, tv, out);
}

#[cfg(target_arch = "aarch64")]
mod _neon_wrappers {
    use archmage::NeonToken;
    pub(crate) fn blend_src_over_row_neon(t: NeonToken, fg: &mut [f32], bg: &[f32]) {
        super::portable::blend_src_over_row(t, fg, bg);
    }
    pub(crate) fn blend_src_over_solid_neon(t: NeonToken, fg: &mut [f32], px: &[f32; 4]) {
        super::portable::blend_src_over_solid(t, fg, px);
    }
    pub(crate) fn blend_src_over_solid_opaque_neon(t: NeonToken, fg: &mut [f32], px: &[f32; 4]) {
        super::portable::blend_src_over_solid_opaque(t, fg, px);
    }
    pub(crate) fn mask_row_apply_neon(t: NeonToken, fg: &mut [f32], mask: &[f32]) {
        super::portable::mask_row_apply(t, fg, mask);
    }
    pub(crate) fn mask_row_rgb_apply_neon(t: NeonToken, fg: &mut [f32], mask: &[f32]) {
        super::portable::mask_row_rgb_apply(t, fg, mask);
    }
    pub(crate) fn lerp_row_apply_neon(
        t: NeonToken,
        a: &[f32],
        b: &[f32],
        tv: &[f32],
        out: &mut [f32],
    ) {
        super::portable::lerp_row_apply(t, a, b, tv, out);
    }
}
#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
use _neon_wrappers::*;

#[cfg(target_arch = "wasm32")]
mod _wasm_wrappers {
    use archmage::Wasm128Token;
    pub(crate) fn blend_src_over_row_wasm128(t: Wasm128Token, fg: &mut [f32], bg: &[f32]) {
        super::portable::blend_src_over_row(t, fg, bg);
    }
    pub(crate) fn blend_src_over_solid_wasm128(t: Wasm128Token, fg: &mut [f32], px: &[f32; 4]) {
        super::portable::blend_src_over_solid(t, fg, px);
    }
    pub(crate) fn blend_src_over_solid_opaque_wasm128(
        t: Wasm128Token,
        fg: &mut [f32],
        px: &[f32; 4],
    ) {
        super::portable::blend_src_over_solid_opaque(t, fg, px);
    }
    pub(crate) fn mask_row_apply_wasm128(t: Wasm128Token, fg: &mut [f32], mask: &[f32]) {
        super::portable::mask_row_apply(t, fg, mask);
    }
    pub(crate) fn mask_row_rgb_apply_wasm128(t: Wasm128Token, fg: &mut [f32], mask: &[f32]) {
        super::portable::mask_row_rgb_apply(t, fg, mask);
    }
    pub(crate) fn lerp_row_apply_wasm128(
        t: Wasm128Token,
        a: &[f32],
        b: &[f32],
        tv: &[f32],
        out: &mut [f32],
    ) {
        super::portable::lerp_row_apply(t, a, b, tv, out);
    }
}
#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
use _wasm_wrappers::*;

// ---------------------------------------------------------------------------
// Public dispatch (one runtime check per call via incant!)
// ---------------------------------------------------------------------------

/// SrcOver: fg[i] += bg[i] * (1 - fg_alpha). Row-based, 4ch RGBA.
pub(crate) fn blend_src_over_row(fg: &mut [f32], bg: &[f32]) {
    archmage::incant!(blend_src_over_row(fg, bg))
}

/// SrcOver solid: fg[i] += pixel[c] * (1 - fg_alpha). No row buffer.
pub(crate) fn blend_src_over_solid(fg: &mut [f32], pixel: &[f32; 4]) {
    archmage::incant!(blend_src_over_solid(fg, pixel))
}

/// SrcOver solid opaque: like solid but output alpha = 1.0.
pub(crate) fn blend_src_over_solid_opaque(fg: &mut [f32], pixel: &[f32; 4]) {
    archmage::incant!(blend_src_over_solid_opaque(fg, pixel))
}

/// Multiply premultiplied RGBA row by per-pixel mask (one f32 per pixel).
pub(crate) fn mask_row_apply(fg: &mut [f32], mask: &[f32]) {
    archmage::incant!(mask_row_apply(fg, mask))
}

/// Multiply RGB channels by per-pixel mask, leave alpha untouched.
pub(crate) fn mask_row_rgb_apply(fg: &mut [f32], mask: &[f32]) {
    archmage::incant!(mask_row_rgb_apply(fg, mask))
}

/// Linearly interpolate between two RGBA rows using per-pixel factor.
pub(crate) fn lerp_row_apply(a: &[f32], b: &[f32], t: &[f32], out: &mut [f32]) {
    archmage::incant!(lerp_row_apply(a, b, t, out))
}
