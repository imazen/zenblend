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
pub(crate) fn blend_dst_over_row_scalar(t: ScalarToken, fg: &mut [f32], bg: &[f32]) {
    portable::blend_dst_over_row(t, fg, bg);
}
pub(crate) fn blend_src_in_row_scalar(t: ScalarToken, fg: &mut [f32], bg: &[f32]) {
    portable::blend_src_in_row(t, fg, bg);
}
pub(crate) fn blend_dst_in_row_scalar(t: ScalarToken, fg: &mut [f32], bg: &[f32]) {
    portable::blend_dst_in_row(t, fg, bg);
}
pub(crate) fn blend_src_out_row_scalar(t: ScalarToken, fg: &mut [f32], bg: &[f32]) {
    portable::blend_src_out_row(t, fg, bg);
}
pub(crate) fn blend_dst_out_row_scalar(t: ScalarToken, fg: &mut [f32], bg: &[f32]) {
    portable::blend_dst_out_row(t, fg, bg);
}
pub(crate) fn blend_src_atop_row_scalar(t: ScalarToken, fg: &mut [f32], bg: &[f32]) {
    portable::blend_src_atop_row(t, fg, bg);
}
pub(crate) fn blend_dst_atop_row_scalar(t: ScalarToken, fg: &mut [f32], bg: &[f32]) {
    portable::blend_dst_atop_row(t, fg, bg);
}
pub(crate) fn blend_xor_row_scalar(t: ScalarToken, fg: &mut [f32], bg: &[f32]) {
    portable::blend_xor_row(t, fg, bg);
}
pub(crate) fn blend_plus_row_scalar(t: ScalarToken, fg: &mut [f32], bg: &[f32]) {
    portable::blend_plus_row(t, fg, bg);
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

// Artistic-mode SIMD kernels. Each gets a public `incant!`-dispatched entry plus
// per-tier token wrappers (scalar always; neon/wasm128 in the cfg modules below).
// The macro takes the public name, the per-tier wrapper names, and the portable
// kernel it delegates to, so there is one source of truth per mode.
macro_rules! artistic_kernel {
    ($pub_name:ident, $scalar:ident, $neon:ident, $wasm:ident, $portable:ident) => {
        pub(crate) fn $pub_name(fg: &mut [f32], bg: &[f32]) {
            // Only NEON / WASM128 SIMD kernels exist for the artistic modes;
            // x86 falls back to the scalar reference (already SIMD-competitive
            // there via the autovectorizer). The explicit tier list keeps the
            // macro from looking for nonexistent `_v3` / `_v4` x86 variants.
            archmage::incant!($pub_name(fg, bg), [neon, wasm128, scalar])
        }
        pub(crate) fn $scalar(t: ScalarToken, fg: &mut [f32], bg: &[f32]) {
            portable::$portable(t, fg, bg);
        }
    };
}

artistic_kernel!(
    blend_multiply,
    blend_multiply_scalar,
    blend_multiply_neon,
    blend_multiply_wasm128,
    blend_multiply_simd
);
artistic_kernel!(
    blend_overlay,
    blend_overlay_scalar,
    blend_overlay_neon,
    blend_overlay_wasm128,
    blend_overlay_simd
);
artistic_kernel!(
    blend_hard_light,
    blend_hard_light_scalar,
    blend_hard_light_neon,
    blend_hard_light_wasm128,
    blend_hard_light_simd
);
artistic_kernel!(
    blend_linear_light,
    blend_linear_light_scalar,
    blend_linear_light_neon,
    blend_linear_light_wasm128,
    blend_linear_light_simd
);
artistic_kernel!(
    blend_pin_light,
    blend_pin_light_scalar,
    blend_pin_light_neon,
    blend_pin_light_wasm128,
    blend_pin_light_simd
);
artistic_kernel!(
    blend_screen,
    blend_screen_scalar,
    blend_screen_neon,
    blend_screen_wasm128,
    blend_screen_simd
);
artistic_kernel!(
    blend_darken,
    blend_darken_scalar,
    blend_darken_neon,
    blend_darken_wasm128,
    blend_darken_simd
);
artistic_kernel!(
    blend_lighten,
    blend_lighten_scalar,
    blend_lighten_neon,
    blend_lighten_wasm128,
    blend_lighten_simd
);
artistic_kernel!(
    blend_difference,
    blend_difference_scalar,
    blend_difference_neon,
    blend_difference_wasm128,
    blend_difference_simd
);
artistic_kernel!(
    blend_linear_dodge,
    blend_linear_dodge_scalar,
    blend_linear_dodge_neon,
    blend_linear_dodge_wasm128,
    blend_linear_dodge_simd
);
artistic_kernel!(
    blend_linear_burn,
    blend_linear_burn_scalar,
    blend_linear_burn_neon,
    blend_linear_burn_wasm128,
    blend_linear_burn_simd
);
artistic_kernel!(
    blend_subtract,
    blend_subtract_scalar,
    blend_subtract_neon,
    blend_subtract_wasm128,
    blend_subtract_simd
);
artistic_kernel!(
    blend_exclusion,
    blend_exclusion_scalar,
    blend_exclusion_neon,
    blend_exclusion_wasm128,
    blend_exclusion_simd
);

#[cfg(target_arch = "aarch64")]
mod _neon_wrappers {
    use archmage::NeonToken;
    pub(crate) fn blend_src_over_row_neon(t: NeonToken, fg: &mut [f32], bg: &[f32]) {
        super::portable::blend_src_over_row(t, fg, bg);
    }
    pub(crate) fn blend_dst_over_row_neon(t: NeonToken, fg: &mut [f32], bg: &[f32]) {
        super::portable::blend_dst_over_row(t, fg, bg);
    }
    pub(crate) fn blend_src_in_row_neon(t: NeonToken, fg: &mut [f32], bg: &[f32]) {
        super::portable::blend_src_in_row(t, fg, bg);
    }
    pub(crate) fn blend_dst_in_row_neon(t: NeonToken, fg: &mut [f32], bg: &[f32]) {
        super::portable::blend_dst_in_row(t, fg, bg);
    }
    pub(crate) fn blend_src_out_row_neon(t: NeonToken, fg: &mut [f32], bg: &[f32]) {
        super::portable::blend_src_out_row(t, fg, bg);
    }
    pub(crate) fn blend_dst_out_row_neon(t: NeonToken, fg: &mut [f32], bg: &[f32]) {
        super::portable::blend_dst_out_row(t, fg, bg);
    }
    pub(crate) fn blend_src_atop_row_neon(t: NeonToken, fg: &mut [f32], bg: &[f32]) {
        super::portable::blend_src_atop_row(t, fg, bg);
    }
    pub(crate) fn blend_dst_atop_row_neon(t: NeonToken, fg: &mut [f32], bg: &[f32]) {
        super::portable::blend_dst_atop_row(t, fg, bg);
    }
    pub(crate) fn blend_xor_row_neon(t: NeonToken, fg: &mut [f32], bg: &[f32]) {
        super::portable::blend_xor_row(t, fg, bg);
    }
    pub(crate) fn blend_plus_row_neon(t: NeonToken, fg: &mut [f32], bg: &[f32]) {
        super::portable::blend_plus_row(t, fg, bg);
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

    macro_rules! artistic_neon {
        ($name:ident, $portable:ident) => {
            pub(crate) fn $name(t: NeonToken, fg: &mut [f32], bg: &[f32]) {
                super::portable::$portable(t, fg, bg);
            }
        };
    }
    artistic_neon!(blend_linear_dodge_neon, blend_linear_dodge_simd);
    artistic_neon!(blend_linear_burn_neon, blend_linear_burn_simd);
    artistic_neon!(blend_subtract_neon, blend_subtract_simd);
    artistic_neon!(blend_multiply_neon, blend_multiply_simd);
    artistic_neon!(blend_linear_light_neon, blend_linear_light_simd);
    artistic_neon!(blend_pin_light_neon, blend_pin_light_simd);
    artistic_neon!(blend_overlay_neon, blend_overlay_simd);
    artistic_neon!(blend_hard_light_neon, blend_hard_light_simd);
    artistic_neon!(blend_screen_neon, blend_screen_simd);
    artistic_neon!(blend_darken_neon, blend_darken_simd);
    artistic_neon!(blend_lighten_neon, blend_lighten_simd);
    artistic_neon!(blend_difference_neon, blend_difference_simd);
    artistic_neon!(blend_exclusion_neon, blend_exclusion_simd);
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
    pub(crate) fn blend_dst_over_row_wasm128(t: Wasm128Token, fg: &mut [f32], bg: &[f32]) {
        super::portable::blend_dst_over_row(t, fg, bg);
    }
    pub(crate) fn blend_src_in_row_wasm128(t: Wasm128Token, fg: &mut [f32], bg: &[f32]) {
        super::portable::blend_src_in_row(t, fg, bg);
    }
    pub(crate) fn blend_dst_in_row_wasm128(t: Wasm128Token, fg: &mut [f32], bg: &[f32]) {
        super::portable::blend_dst_in_row(t, fg, bg);
    }
    pub(crate) fn blend_src_out_row_wasm128(t: Wasm128Token, fg: &mut [f32], bg: &[f32]) {
        super::portable::blend_src_out_row(t, fg, bg);
    }
    pub(crate) fn blend_dst_out_row_wasm128(t: Wasm128Token, fg: &mut [f32], bg: &[f32]) {
        super::portable::blend_dst_out_row(t, fg, bg);
    }
    pub(crate) fn blend_src_atop_row_wasm128(t: Wasm128Token, fg: &mut [f32], bg: &[f32]) {
        super::portable::blend_src_atop_row(t, fg, bg);
    }
    pub(crate) fn blend_dst_atop_row_wasm128(t: Wasm128Token, fg: &mut [f32], bg: &[f32]) {
        super::portable::blend_dst_atop_row(t, fg, bg);
    }
    pub(crate) fn blend_xor_row_wasm128(t: Wasm128Token, fg: &mut [f32], bg: &[f32]) {
        super::portable::blend_xor_row(t, fg, bg);
    }
    pub(crate) fn blend_plus_row_wasm128(t: Wasm128Token, fg: &mut [f32], bg: &[f32]) {
        super::portable::blend_plus_row(t, fg, bg);
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

    macro_rules! artistic_wasm {
        ($name:ident, $portable:ident) => {
            pub(crate) fn $name(t: Wasm128Token, fg: &mut [f32], bg: &[f32]) {
                super::portable::$portable(t, fg, bg);
            }
        };
    }
    artistic_wasm!(blend_linear_dodge_wasm128, blend_linear_dodge_simd);
    artistic_wasm!(blend_linear_burn_wasm128, blend_linear_burn_simd);
    artistic_wasm!(blend_subtract_wasm128, blend_subtract_simd);
    artistic_wasm!(blend_multiply_wasm128, blend_multiply_simd);
    artistic_wasm!(blend_linear_light_wasm128, blend_linear_light_simd);
    artistic_wasm!(blend_pin_light_wasm128, blend_pin_light_simd);
    artistic_wasm!(blend_overlay_wasm128, blend_overlay_simd);
    artistic_wasm!(blend_hard_light_wasm128, blend_hard_light_simd);
    artistic_wasm!(blend_screen_wasm128, blend_screen_simd);
    artistic_wasm!(blend_darken_wasm128, blend_darken_simd);
    artistic_wasm!(blend_lighten_wasm128, blend_lighten_simd);
    artistic_wasm!(blend_difference_wasm128, blend_difference_simd);
    artistic_wasm!(blend_exclusion_wasm128, blend_exclusion_simd);
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

/// DstOver: `out = bg + fg * (1 - bg_alpha)`. Dispatched like SrcOver.
pub(crate) fn blend_dst_over_row(fg: &mut [f32], bg: &[f32]) {
    archmage::incant!(blend_dst_over_row(fg, bg))
}

pub(crate) fn blend_src_in_row(fg: &mut [f32], bg: &[f32]) {
    archmage::incant!(blend_src_in_row(fg, bg))
}

pub(crate) fn blend_dst_in_row(fg: &mut [f32], bg: &[f32]) {
    archmage::incant!(blend_dst_in_row(fg, bg))
}

pub(crate) fn blend_src_out_row(fg: &mut [f32], bg: &[f32]) {
    archmage::incant!(blend_src_out_row(fg, bg))
}

pub(crate) fn blend_dst_out_row(fg: &mut [f32], bg: &[f32]) {
    archmage::incant!(blend_dst_out_row(fg, bg))
}

pub(crate) fn blend_src_atop_row(fg: &mut [f32], bg: &[f32]) {
    archmage::incant!(blend_src_atop_row(fg, bg))
}

pub(crate) fn blend_dst_atop_row(fg: &mut [f32], bg: &[f32]) {
    archmage::incant!(blend_dst_atop_row(fg, bg))
}

pub(crate) fn blend_xor_row(fg: &mut [f32], bg: &[f32]) {
    archmage::incant!(blend_xor_row(fg, bg))
}

pub(crate) fn blend_plus_row(fg: &mut [f32], bg: &[f32]) {
    archmage::incant!(blend_plus_row(fg, bg))
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
