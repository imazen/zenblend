//! Blend mode dispatch: routes BlendMode enum to SIMD or scalar kernels.
//!
//! SrcOver uses SIMD-dispatched kernels. All other Porter-Duff and artistic
//! modes use scalar implementations. The match happens once per row, not per pixel.

use crate::BlendMode;
use crate::simd;

/// Dispatch blend_row to the appropriate kernel.
#[inline]
pub(crate) fn dispatch_blend_row(fg: &mut [f32], bg: &[f32], mode: BlendMode) {
    match mode {
        BlendMode::SrcOver => simd::blend_src_over_row(fg, bg),
        BlendMode::Clear => blend_clear(fg),
        BlendMode::Src => {} // fg already contains src
        BlendMode::Dst => fg.copy_from_slice(bg),
        BlendMode::DstOver => blend_dst_over(fg, bg),
        BlendMode::SrcIn => blend_src_in(fg, bg),
        BlendMode::DstIn => blend_dst_in(fg, bg),
        BlendMode::SrcOut => blend_src_out(fg, bg),
        BlendMode::DstOut => blend_dst_out(fg, bg),
        BlendMode::SrcAtop => blend_src_atop(fg, bg),
        BlendMode::DstAtop => blend_dst_atop(fg, bg),
        BlendMode::Xor => blend_xor(fg, bg),
        // SIMD-dispatched separable artistic modes (NEON/WASM128; scalar on x86).
        // These six reduce to a division-free polynomial in the PREMULTIPLIED
        // fg/bg, so the whole row is pure vector mul/add/min/max — a big win on
        // NEON where the scalar unpremultiply division is the bottleneck.
        BlendMode::Multiply => simd::blend_multiply(fg, bg),
        BlendMode::Screen => simd::blend_screen(fg, bg),
        BlendMode::Darken => simd::blend_darken(fg, bg),
        BlendMode::Lighten => simd::blend_lighten(fg, bg),
        BlendMode::Difference => simd::blend_difference(fg, bg),
        BlendMode::Exclusion => simd::blend_exclusion(fg, bg),
        // The remaining modes need a per-lane unpremultiply/clamp or a
        // both-branches lane select, and they stay scalar. That was first
        // recorded on a non-pipelined Graviton N1; RE-MEASURED 2026-07-28 on an
        // Apple M-series (wide, out-of-order) with a real vector kernel written
        // for Overlay and HardLight, and the conclusion HOLDS for a different
        // reason — LLVM already autovectorizes these branchy scalar loops
        // better than an explicit lane-select version does:
        //
        //   Overlay     scalar-fn arm 4333 ns/row   vector kernel 4981  (-15%)
        //   HardLight   scalar-fn arm 4350 ns/row   vector kernel 5012  (-15%)
        //
        // Note the trap in those runs: the vector kernel reported 3.05x "vs its
        // own scalar tier" while being 15% SLOWER in absolute terms, because it
        // degraded the forced-scalar tier (4344 -> 15210 ns) faster than it
        // helped NEON. A tier RATIO alone would have sold this as a 3x win.
        //
        // The division-free modes below are the opposite case and DID win: they
        // remove the unpremultiply entirely rather than vectorizing around it.
        BlendMode::Overlay => blend_overlay(fg, bg),
        BlendMode::HardLight => blend_hard_light(fg, bg),
        BlendMode::SoftLight => blend_soft_light(fg, bg),
        BlendMode::ColorDodge => blend_color_dodge(fg, bg),
        BlendMode::ColorBurn => blend_color_burn(fg, bg),
        BlendMode::LinearBurn => simd::blend_linear_burn(fg, bg),
        BlendMode::LinearDodge => simd::blend_linear_dodge(fg, bg),
        BlendMode::VividLight => blend_vivid_light(fg, bg),
        BlendMode::LinearLight => blend_linear_light(fg, bg),
        BlendMode::PinLight => blend_pin_light(fg, bg),
        BlendMode::HardMix => blend_hard_mix(fg, bg),
        BlendMode::Divide => blend_divide(fg, bg),
        BlendMode::Subtract => simd::blend_subtract(fg, bg),
        BlendMode::Plus => blend_plus(fg, bg),
        // non_exhaustive: future variants
        #[allow(unreachable_patterns)]
        _ => unimplemented!("blend mode not yet implemented"),
    }
}

/// Dispatch blend_row_solid to the appropriate kernel.
#[inline]
pub(crate) fn dispatch_blend_row_solid(fg: &mut [f32], pixel: &[f32; 4], mode: BlendMode) {
    match mode {
        BlendMode::SrcOver => simd::blend_src_over_solid(fg, pixel),
        // For non-SrcOver modes, construct a repeated bg row on the stack and delegate.
        // This is fine for solid backgrounds — the match is once per row.
        other => {
            // Fast path for trivial modes that don't need bg data
            match other {
                BlendMode::Clear => {
                    blend_clear(fg);
                    return;
                }
                BlendMode::Src => return, // fg unchanged
                _ => {}
            }
            // General path: repeat the solid pixel into a temp row and delegate
            for s in fg.chunks_exact_mut(4) {
                let mut px = [s[0], s[1], s[2], s[3]];
                let bg = *pixel;
                dispatch_blend_pixel(&mut px, &bg, other);
                s.copy_from_slice(&px);
            }
        }
    }
}

/// Dispatch blend_row_solid_opaque to the appropriate kernel.
#[inline]
pub(crate) fn dispatch_blend_row_solid_opaque(fg: &mut [f32], pixel: &[f32; 4], mode: BlendMode) {
    match mode {
        BlendMode::SrcOver => simd::blend_src_over_solid_opaque(fg, pixel),
        // For other modes, the opaque hint doesn't change the formula —
        // just lets us skip the alpha multiply. For now, delegate to solid.
        other => dispatch_blend_row_solid(fg, pixel, other),
    }
}

/// Single-pixel blend dispatch (for solid-pixel path of non-SrcOver modes).
#[inline]
fn dispatch_blend_pixel(fg: &mut [f32; 4], bg: &[f32; 4], mode: BlendMode) {
    match mode {
        BlendMode::Dst => *fg = *bg,
        BlendMode::DstOver => {
            let inv_da = 1.0 - bg[3];
            for i in 0..4 {
                fg[i] = bg[i] + fg[i] * inv_da;
            }
        }
        BlendMode::SrcIn => {
            let da = bg[3];
            for v in fg.iter_mut() {
                *v *= da;
            }
        }
        BlendMode::DstIn => {
            let sa = fg[3];
            for i in 0..4 {
                fg[i] = bg[i] * sa;
            }
        }
        BlendMode::SrcOut => {
            let inv_da = 1.0 - bg[3];
            for v in fg.iter_mut() {
                *v *= inv_da;
            }
        }
        BlendMode::DstOut => {
            let inv_sa = 1.0 - fg[3];
            for i in 0..4 {
                fg[i] = bg[i] * inv_sa;
            }
        }
        BlendMode::SrcAtop => {
            let sa = fg[3];
            let da = bg[3];
            for i in 0..4 {
                fg[i] = fg[i] * da + bg[i] * (1.0 - sa);
            }
        }
        BlendMode::DstAtop => {
            let sa = fg[3];
            let da = bg[3];
            for i in 0..4 {
                fg[i] = bg[i] * sa + fg[i] * (1.0 - da);
            }
        }
        BlendMode::Xor => {
            let sa = fg[3];
            let da = bg[3];
            for i in 0..4 {
                fg[i] = fg[i] * (1.0 - da) + bg[i] * (1.0 - sa);
            }
        }
        BlendMode::Multiply => blend_artistic_pixel(fg, bg, |s, d| s * d),
        BlendMode::Screen => blend_artistic_pixel(fg, bg, |s, d| s + d - s * d),
        BlendMode::Overlay => blend_artistic_pixel(fg, bg, |s, d| {
            if d < 0.5 {
                2.0 * s * d
            } else {
                1.0 - 2.0 * (1.0 - s) * (1.0 - d)
            }
        }),
        BlendMode::Darken => blend_artistic_pixel(fg, bg, |s, d| if s < d { s } else { d }),
        BlendMode::Lighten => blend_artistic_pixel(fg, bg, |s, d| if s > d { s } else { d }),
        BlendMode::HardLight => blend_artistic_pixel(fg, bg, |s, d| {
            if s < 0.5 {
                2.0 * s * d
            } else {
                1.0 - 2.0 * (1.0 - s) * (1.0 - d)
            }
        }),
        BlendMode::SoftLight => blend_artistic_pixel(fg, bg, |s, d| {
            // W3C formula
            if s <= 0.5 {
                d - (1.0 - 2.0 * s) * d * (1.0 - d)
            } else {
                let g = if d <= 0.25 {
                    ((16.0 * d - 12.0) * d + 4.0) * d
                } else {
                    d.sqrt()
                };
                d + (2.0 * s - 1.0) * (g - d)
            }
        }),
        BlendMode::ColorDodge => blend_artistic_pixel(fg, bg, |s, d| {
            if d <= 0.0 {
                0.0
            } else if s >= 1.0 {
                1.0
            } else {
                (d / (1.0 - s)).min(1.0)
            }
        }),
        BlendMode::ColorBurn => blend_artistic_pixel(fg, bg, |s, d| {
            if d >= 1.0 {
                1.0
            } else if s <= 0.0 {
                0.0
            } else {
                1.0 - ((1.0 - d) / s).min(1.0)
            }
        }),
        BlendMode::Difference => blend_artistic_pixel(fg, bg, |s, d| {
            let diff = s - d;
            if diff < 0.0 { -diff } else { diff }
        }),
        BlendMode::Exclusion => blend_artistic_pixel(fg, bg, |s, d| s + d - 2.0 * s * d),
        BlendMode::LinearBurn => {
            blend_artistic_pixel(fg, bg, |s, d| (s + d - 1.0).max(0.0));
        }
        BlendMode::LinearDodge => {
            blend_artistic_pixel(fg, bg, |s, d| (s + d).min(1.0));
        }
        BlendMode::VividLight => blend_artistic_pixel(fg, bg, vivid_light_fn),
        BlendMode::LinearLight => blend_artistic_pixel(fg, bg, |s, d| {
            if s < 0.5 {
                // LinearBurn(2·s, d) = max(0, 2s + d - 1)
                (2.0 * s + d - 1.0).max(0.0)
            } else {
                // LinearDodge(2s-1, d) = min(1, 2s - 1 + d)
                (2.0 * s - 1.0 + d).min(1.0)
            }
        }),
        BlendMode::PinLight => blend_artistic_pixel(fg, bg, |s, d| {
            if s < 0.5 {
                // Darken(2·s, d)
                d.min(2.0 * s)
            } else {
                // Lighten(2s-1, d)
                d.max(2.0 * s - 1.0)
            }
        }),
        BlendMode::HardMix => {
            blend_artistic_pixel(
                fg,
                bg,
                |s, d| {
                    if vivid_light_fn(s, d) < 0.5 { 0.0 } else { 1.0 }
                },
            )
        }
        BlendMode::Divide => blend_artistic_pixel(fg, bg, |s, d| {
            if s <= 0.0 {
                1.0 // d / 0 → 1.0 (clamped)
            } else {
                (d / s).min(1.0)
            }
        }),
        BlendMode::Subtract => {
            blend_artistic_pixel(fg, bg, |s, d| (d - s).max(0.0));
        }
        BlendMode::Plus => blend_plus_pixel(fg, bg),
        _ => {} // Clear, Src handled by caller; unknown modes are no-op
    }
}

// =============================================================================
// Row-level scalar kernels for Porter-Duff operators
// =============================================================================

#[inline]
fn blend_clear(fg: &mut [f32]) {
    for v in fg.iter_mut() {
        *v = 0.0;
    }
}

#[inline]
fn blend_dst_over(fg: &mut [f32], bg: &[f32]) {
    for (s, b) in fg.chunks_exact_mut(4).zip(bg.chunks_exact(4)) {
        let inv_da = 1.0 - b[3];
        let r = b[0] + s[0] * inv_da;
        let g = b[1] + s[1] * inv_da;
        let bl = b[2] + s[2] * inv_da;
        let a = b[3] + s[3] * inv_da;
        s[0] = r;
        s[1] = g;
        s[2] = bl;
        s[3] = a;
    }
}

#[inline]
fn blend_src_in(fg: &mut [f32], bg: &[f32]) {
    for (s, b) in fg.chunks_exact_mut(4).zip(bg.chunks_exact(4)) {
        let da = b[3];
        s[0] *= da;
        s[1] *= da;
        s[2] *= da;
        s[3] *= da;
    }
}

#[inline]
fn blend_dst_in(fg: &mut [f32], bg: &[f32]) {
    for (s, b) in fg.chunks_exact_mut(4).zip(bg.chunks_exact(4)) {
        let sa = s[3];
        s[0] = b[0] * sa;
        s[1] = b[1] * sa;
        s[2] = b[2] * sa;
        s[3] = b[3] * sa;
    }
}

#[inline]
fn blend_src_out(fg: &mut [f32], bg: &[f32]) {
    for (s, b) in fg.chunks_exact_mut(4).zip(bg.chunks_exact(4)) {
        let inv_da = 1.0 - b[3];
        s[0] *= inv_da;
        s[1] *= inv_da;
        s[2] *= inv_da;
        s[3] *= inv_da;
    }
}

#[inline]
fn blend_dst_out(fg: &mut [f32], bg: &[f32]) {
    for (s, b) in fg.chunks_exact_mut(4).zip(bg.chunks_exact(4)) {
        let inv_sa = 1.0 - s[3];
        s[0] = b[0] * inv_sa;
        s[1] = b[1] * inv_sa;
        s[2] = b[2] * inv_sa;
        s[3] = b[3] * inv_sa;
    }
}

#[inline]
fn blend_src_atop(fg: &mut [f32], bg: &[f32]) {
    for (s, b) in fg.chunks_exact_mut(4).zip(bg.chunks_exact(4)) {
        let sa = s[3];
        let da = b[3];
        let inv_sa = 1.0 - sa;
        let r = s[0] * da + b[0] * inv_sa;
        let g = s[1] * da + b[1] * inv_sa;
        let bl = s[2] * da + b[2] * inv_sa;
        let a = sa * da + b[3] * inv_sa;
        s[0] = r;
        s[1] = g;
        s[2] = bl;
        s[3] = a;
    }
}

#[inline]
fn blend_dst_atop(fg: &mut [f32], bg: &[f32]) {
    for (s, b) in fg.chunks_exact_mut(4).zip(bg.chunks_exact(4)) {
        let sa = s[3];
        let da = b[3];
        let inv_da = 1.0 - da;
        let r = b[0] * sa + s[0] * inv_da;
        let g = b[1] * sa + s[1] * inv_da;
        let bl = b[2] * sa + s[2] * inv_da;
        let a = b[3] * sa + s[3] * inv_da;
        s[0] = r;
        s[1] = g;
        s[2] = bl;
        s[3] = a;
    }
}

#[inline]
fn blend_xor(fg: &mut [f32], bg: &[f32]) {
    for (s, b) in fg.chunks_exact_mut(4).zip(bg.chunks_exact(4)) {
        let sa = s[3];
        let da = b[3];
        let inv_sa = 1.0 - sa;
        let inv_da = 1.0 - da;
        let r = s[0] * inv_da + b[0] * inv_sa;
        let g = s[1] * inv_da + b[1] * inv_sa;
        let bl = s[2] * inv_da + b[2] * inv_sa;
        let a = s[3] * inv_da + b[3] * inv_sa;
        s[0] = r;
        s[1] = g;
        s[2] = bl;
        s[3] = a;
    }
}

// =============================================================================
// Artistic blend modes (separable)
// =============================================================================

// Artistic blend modes in premultiplied space follow the general formula:
//   out_c = (1 - dst_a) * src_c + (1 - src_a) * dst_c + src_a * dst_a * f(src_c/src_a, dst_c/dst_a)
//   out_a = src_a + dst_a - src_a * dst_a  (same as SrcOver alpha)
//
// Where f(Cs, Cd) is the blend function on straight (unpremultiplied) color values.
// We unpremultiply per-pixel, apply f, then re-premultiply via the formula above.

/// Apply an artistic blend function per-pixel (scalar).
///
/// `f(src_straight, dst_straight) -> blended_straight` operates on unpremultiplied
/// color values per channel.
#[inline]
fn blend_artistic_pixel(fg: &mut [f32; 4], bg: &[f32; 4], f: impl Fn(f32, f32) -> f32) {
    let sa = fg[3];
    let da = bg[3];

    // Output alpha: same as SrcOver
    let out_a = sa + da - sa * da;

    // Treat non-finite inputs (NaN or inf, from corrupt or un-sanitized
    // upstream pixels) the same as the zero-alpha case. The plain
    // `<= 0.0` guard misses NaN (NaN comparisons are always false) and
    // misses +inf, letting non-finite values flow into out_a and poison
    // every downstream blend. `!is_finite()` catches both.
    if !out_a.is_finite() || out_a <= 0.0 {
        *fg = [0.0, 0.0, 0.0, 0.0];
        return;
    }

    // Unpremultiply to get straight color values
    let inv_sa = if sa > 0.0 { 1.0 / sa } else { 0.0 };
    let inv_da = if da > 0.0 { 1.0 / da } else { 0.0 };

    for i in 0..3 {
        let cs = fg[i] * inv_sa; // straight src color
        let cd = bg[i] * inv_da; // straight dst color
        let blended = f(cs, cd);
        // General premultiplied formula:
        // out_c = (1 - da) * fg[i] + (1 - sa) * bg[i] + sa * da * blended
        let out = (1.0 - da) * fg[i] + (1.0 - sa) * bg[i] + sa * da * blended;
        // Sanitize: NaN can leak from cs/cd if upstream pixels carry NaN, or
        // from blend kernels with division (color_dodge/color_burn/divide).
        // Replace non-finite outputs with 0.0 to avoid poisoning later passes.
        fg[i] = if out.is_finite() { out } else { 0.0 };
    }
    fg[3] = out_a;
}

// Row-level wrappers for artistic modes that process all pixels in a row.

macro_rules! artistic_row {
    ($name:ident, $f:expr) => {
        // Several of these scalar row kernels are now superseded by the
        // SIMD-dispatched versions in `crate::simd` for NEON/WASM128 targets
        // (they remain the scalar reference used by the equivalence tests and
        // the still-scalar modes: soft_light / color_dodge / color_burn /
        // vivid_light / hard_mix / divide). Allow dead_code so the unmigrated
        // duplicates don't warn.
        #[inline]
        #[allow(dead_code)]
        pub(crate) fn $name(fg: &mut [f32], bg: &[f32]) {
            for (s, b) in fg.chunks_exact_mut(4).zip(bg.chunks_exact(4)) {
                let mut pixel: [f32; 4] = [s[0], s[1], s[2], s[3]];
                let bg_pixel: [f32; 4] = [b[0], b[1], b[2], b[3]];
                blend_artistic_pixel(&mut pixel, &bg_pixel, $f);
                s.copy_from_slice(&pixel);
            }
        }
    };
}

artistic_row!(blend_multiply, |s: f32, d: f32| s * d);
artistic_row!(blend_screen, |s: f32, d: f32| s + d - s * d);
artistic_row!(blend_overlay, |s: f32, d: f32| {
    if d < 0.5 {
        2.0 * s * d
    } else {
        1.0 - 2.0 * (1.0 - s) * (1.0 - d)
    }
});
artistic_row!(blend_darken, |s: f32, d: f32| if s < d { s } else { d });
artistic_row!(blend_lighten, |s: f32, d: f32| if s > d { s } else { d });
artistic_row!(blend_hard_light, |s: f32, d: f32| {
    if s < 0.5 {
        2.0 * s * d
    } else {
        1.0 - 2.0 * (1.0 - s) * (1.0 - d)
    }
});
artistic_row!(blend_soft_light, |s: f32, d: f32| {
    if s <= 0.5 {
        d - (1.0 - 2.0 * s) * d * (1.0 - d)
    } else {
        let g = if d <= 0.25 {
            ((16.0 * d - 12.0) * d + 4.0) * d
        } else {
            d.sqrt()
        };
        d + (2.0 * s - 1.0) * (g - d)
    }
});
artistic_row!(blend_color_dodge, |s: f32, d: f32| {
    if d <= 0.0 {
        0.0
    } else if s >= 1.0 {
        1.0
    } else {
        (d / (1.0 - s)).min(1.0)
    }
});
artistic_row!(blend_color_burn, |s: f32, d: f32| {
    if d >= 1.0 {
        1.0
    } else if s <= 0.0 {
        0.0
    } else {
        1.0 - ((1.0 - d) / s).min(1.0)
    }
});
artistic_row!(blend_difference, |s: f32, d: f32| {
    let diff = s - d;
    if diff < 0.0 { -diff } else { diff }
});
artistic_row!(blend_exclusion, |s: f32, d: f32| s + d - 2.0 * s * d);
artistic_row!(blend_linear_burn, |s: f32, d: f32| (s + d - 1.0).max(0.0));
artistic_row!(blend_linear_dodge, |s: f32, d: f32| (s + d).min(1.0));
artistic_row!(blend_vivid_light, vivid_light_fn);
artistic_row!(blend_linear_light, |s: f32, d: f32| {
    if s < 0.5 {
        (2.0 * s + d - 1.0).max(0.0)
    } else {
        (2.0 * s - 1.0 + d).min(1.0)
    }
});
artistic_row!(blend_pin_light, |s: f32, d: f32| {
    if s < 0.5 {
        d.min(2.0 * s)
    } else {
        d.max(2.0 * s - 1.0)
    }
});
artistic_row!(blend_hard_mix, |s: f32, d: f32| {
    if vivid_light_fn(s, d) < 0.5 { 0.0 } else { 1.0 }
});
artistic_row!(blend_divide, |s: f32, d: f32| {
    if s <= 0.0 { 1.0 } else { (d / s).min(1.0) }
});
artistic_row!(blend_subtract, |s: f32, d: f32| (d - s).max(0.0));

/// VividLight channel function: shared by VividLight and HardMix.
#[inline]
fn vivid_light_fn(s: f32, d: f32) -> f32 {
    if s < 0.5 {
        // ColorBurn(2·s, d) = 1 - (1-d)/(2s)
        let s2 = 2.0 * s;
        if d >= 1.0 {
            1.0
        } else if s2 <= 0.0 {
            0.0
        } else {
            1.0 - ((1.0 - d) / s2).min(1.0)
        }
    } else {
        // ColorDodge(2s-1, d) = d / (1-(2s-1)) = d / (2-2s)
        let s2m1 = 2.0 * s - 1.0;
        if d <= 0.0 {
            0.0
        } else if s2m1 >= 1.0 {
            1.0
        } else {
            (d / (1.0 - s2m1)).min(1.0)
        }
    }
}

/// Plus: premultiplied add with clamp. Does NOT unpremultiply.
#[inline]
fn blend_plus(fg: &mut [f32], bg: &[f32]) {
    for (s, b) in fg.chunks_exact_mut(4).zip(bg.chunks_exact(4)) {
        s[0] = (s[0] + b[0]).min(1.0);
        s[1] = (s[1] + b[1]).min(1.0);
        s[2] = (s[2] + b[2]).min(1.0);
        s[3] = (s[3] + b[3]).min(1.0);
    }
}

/// Plus single pixel (for solid-pixel dispatch path).
#[inline]
fn blend_plus_pixel(fg: &mut [f32; 4], bg: &[f32; 4]) {
    fg[0] = (fg[0] + bg[0]).min(1.0);
    fg[1] = (fg[1] + bg[1]).min(1.0);
    fg[2] = (fg[2] + bg[2]).min(1.0);
    fg[3] = (fg[3] + bg[3]).min(1.0);
}

#[cfg(test)]
mod simd_equiv_tests {
    //! Verify that the SIMD-dispatched artistic kernels in `crate::simd`
    //! reproduce the scalar reference formula (`blend_artistic_pixel` via the
    //! `artistic_row!` kernels) within f32 rounding tolerance, across random
    //! premultiplied pixels plus alpha / saturation edge cases.

    use super::*;
    use crate::BlendMode;

    fn gen_row(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f32 / (1u64 << 53) as f32
        };
        let mut v = Vec::with_capacity(n * 4);
        for i in 0..n {
            // Inject edge cases at fixed indices: a=0, a=1, fully saturated.
            let a = match i % 17 {
                0 => 0.0,
                1 => 1.0,
                2 => 1e-7,
                _ => next(),
            };
            let r = (next() * a).min(a);
            let g = (next() * a).min(a);
            let b = (next() * a).min(a);
            v.extend_from_slice(&[r, g, b, a]);
        }
        v
    }

    /// Returns (max_abs_diff, count_of_pixels_with_diff_over_tol).
    fn compare(
        mode: BlendMode,
        scalar_ref: fn(&mut [f32], &[f32]),
        n: usize,
        seed: u64,
        tol: f32,
    ) -> (f32, usize) {
        let bg = gen_row(n, seed);
        let fg0 = gen_row(n, seed ^ 0xABCDEF);

        let mut fg_simd = fg0.clone();
        dispatch_blend_row(&mut fg_simd, &bg, mode);

        let mut fg_scalar = fg0.clone();
        scalar_ref(&mut fg_scalar, &bg);

        let mut max_diff = 0.0f32;
        let mut over = 0usize;
        for (px_s, px_r) in fg_simd.chunks_exact(4).zip(fg_scalar.chunks_exact(4)) {
            let mut pmax = 0.0f32;
            for (a, b) in px_s.iter().zip(px_r.iter()) {
                let d = (a - b).abs();
                pmax = pmax.max(d);
            }
            max_diff = max_diff.max(pmax);
            if pmax > tol {
                over += 1;
            }
        }
        (max_diff, over)
    }

    macro_rules! equiv_test {
        ($test:ident, $mode:expr, $scalar_ref:ident) => {
            #[test]
            fn $test() {
                let (max_diff, over) = compare($mode, $scalar_ref, 2048, 0x1234, 1e-6);
                assert!(
                    over == 0,
                    "{:?}: {over} pixels exceed 1e-6 tol (max_diff={max_diff})",
                    $mode
                );
            }
        };
    }

    // Only the six SIMD-dispatched modes (division-free premultiplied closed
    // forms). Each must match the scalar divide-then-remultiply reference to
    // within 1e-6 across random pixels + alpha/saturation edge cases.
    equiv_test!(
        multiply_simd_matches_scalar,
        BlendMode::Multiply,
        blend_multiply
    );
    equiv_test!(screen_simd_matches_scalar, BlendMode::Screen, blend_screen);
    equiv_test!(darken_simd_matches_scalar, BlendMode::Darken, blend_darken);
    equiv_test!(
        lighten_simd_matches_scalar,
        BlendMode::Lighten,
        blend_lighten
    );
    equiv_test!(
        difference_simd_matches_scalar,
        BlendMode::Difference,
        blend_difference
    );
    equiv_test!(
        exclusion_simd_matches_scalar,
        BlendMode::Exclusion,
        blend_exclusion
    );
}
