//! Portable SIMD blend kernels via magetypes `f32x4<T>` (shared by NEON, WASM128, and scalar).
//!
//! These operate on `f32x4` — one RGBA pixel per SIMD register.
//! The generic parameter `T` is a backend token (NeonToken, Wasm128Token, ScalarToken, etc.)
//! that determines the actual SIMD instructions used.

use magetypes::simd::backends::F32x4Backend;
use magetypes::simd::generic::f32x4;

/// SrcOver row blend using magetypes f32x4 (1 pixel per iteration).
#[inline]
pub(super) fn blend_src_over_row<T: F32x4Backend>(token: T, fg: &mut [f32], bg: &[f32]) {
    let (fg_chunks, _) = f32x4::<T>::partition_slice_mut(token, fg);
    let (bg_chunks, _) = f32x4::<T>::partition_slice(token, bg);

    for (fg_chunk, bg_chunk) in fg_chunks.iter_mut().zip(bg_chunks.iter()) {
        let fg_pixel = f32x4::load(token, fg_chunk);
        let bg_pixel = f32x4::load(token, bg_chunk);
        let inv_alpha = f32x4::splat(token, 1.0 - fg_chunk[3]);
        let result = fg_pixel + bg_pixel * inv_alpha;
        result.store(fg_chunk);
    }
}

/// DstOver row blend — `out = bg + fg * (1 - bg_alpha)`, 1 pixel per f32x4.
///
/// Mirror of [`blend_src_over_row`]. Every one of the four channels undergoes
/// the identical operation (unlike premultiply, where alpha is special), so all
/// four lanes do useful work with no masking or blending.
///
/// Added 2026-08-01: the nine Porter-Duff modes had NO SIMD path at all —
/// `blend_dst_over` and its siblings were plain scalar loops, while SrcOver and
/// the artistic modes were vectorized. The tier bench read ~1.00x for all nine,
/// which looked like "SIMD isn't helping" but actually meant "both arms run the
/// same scalar code because there is no SIMD arm to select."
#[inline]
pub(super) fn blend_dst_over_row<T: F32x4Backend>(token: T, fg: &mut [f32], bg: &[f32]) {
    let (fg_chunks, _) = f32x4::<T>::partition_slice_mut(token, fg);
    let (bg_chunks, _) = f32x4::<T>::partition_slice(token, bg);

    for (fg_chunk, bg_chunk) in fg_chunks.iter_mut().zip(bg_chunks.iter()) {
        let fg_pixel = f32x4::load(token, fg_chunk);
        let bg_pixel = f32x4::load(token, bg_chunk);
        // Note the asymmetry with SrcOver: the surviving alpha is the BG's.
        let inv_da = f32x4::splat(token, 1.0 - bg_chunk[3]);
        let result = bg_pixel + fg_pixel * inv_da;
        result.store(fg_chunk);
    }
}

/// The remaining Porter-Duff modes, one pixel per `f32x4`.
///
/// All of these apply the IDENTICAL operation to all four channels — alpha
/// included — so every lane does useful work with no masking, no select, and
/// no unpremultiply. That is what separates them from the 18 artistic modes,
/// where three hand-written decompositions all lost (see
/// `benchmarks/neon_unpremul_attempts_2026-07-31.md`); it is why these win and
/// those did not.
///
/// `$body` receives `(fg, bg, sa, da)` as vectors — `sa`/`da` pre-splatted from
/// the source and destination alpha respectively.
macro_rules! porter_duff_row {
    ($name:ident, $doc:expr, $body:expr) => {
        #[doc = $doc]
        #[inline]
        pub(super) fn $name<T: F32x4Backend>(token: T, fg: &mut [f32], bg: &[f32]) {
            let (fg_chunks, _) = f32x4::<T>::partition_slice_mut(token, fg);
            let (bg_chunks, _) = f32x4::<T>::partition_slice(token, bg);
            for (fg_chunk, bg_chunk) in fg_chunks.iter_mut().zip(bg_chunks.iter()) {
                let sa = f32x4::splat(token, fg_chunk[3]);
                let da = f32x4::splat(token, bg_chunk[3]);
                let f = f32x4::load(token, fg_chunk);
                let b = f32x4::load(token, bg_chunk);
                let out: f32x4<T> = $body(token, f, b, sa, da);
                out.store(fg_chunk);
            }
        }
    };
}

porter_duff_row!(blend_src_in_row, "SrcIn: `fg * da`.",
    |t: T, f: f32x4<T>, _b: f32x4<T>, _sa: f32x4<T>, da: f32x4<T>| { let _ = t; f * da });
porter_duff_row!(blend_dst_in_row, "DstIn: `bg * sa`.",
    |t: T, _f: f32x4<T>, b: f32x4<T>, sa: f32x4<T>, _da: f32x4<T>| { let _ = t; b * sa });
porter_duff_row!(blend_src_out_row, "SrcOut: `fg * (1 - da)`.",
    |t: T, f: f32x4<T>, _b: f32x4<T>, _sa: f32x4<T>, da: f32x4<T>| f * (f32x4::splat(t, 1.0) - da));
porter_duff_row!(blend_dst_out_row, "DstOut: `bg * (1 - sa)`.",
    |t: T, _f: f32x4<T>, b: f32x4<T>, sa: f32x4<T>, _da: f32x4<T>| b * (f32x4::splat(t, 1.0) - sa));
porter_duff_row!(blend_src_atop_row, "SrcAtop: `fg*da + bg*(1 - sa)`.",
    |t: T, f: f32x4<T>, b: f32x4<T>, sa: f32x4<T>, da: f32x4<T>| f * da + b * (f32x4::splat(t, 1.0) - sa));
porter_duff_row!(blend_dst_atop_row, "DstAtop: `bg*sa + fg*(1 - da)`.",
    |t: T, f: f32x4<T>, b: f32x4<T>, sa: f32x4<T>, da: f32x4<T>| b * sa + f * (f32x4::splat(t, 1.0) - da));
porter_duff_row!(blend_xor_row, "Xor: `fg*(1 - da) + bg*(1 - sa)`.",
    |t: T, f: f32x4<T>, b: f32x4<T>, sa: f32x4<T>, da: f32x4<T>| {
        let one = f32x4::splat(t, 1.0);
        f * (one - da) + b * (one - sa)
    });
porter_duff_row!(blend_plus_row, "Plus: `min(fg + bg, 1)`.",
    |t: T, f: f32x4<T>, b: f32x4<T>, _sa: f32x4<T>, _da: f32x4<T>| (f + b).min(f32x4::splat(t, 1.0)));

/// SrcOver solid pixel blend using magetypes f32x4.
#[inline]
pub(super) fn blend_src_over_solid<T: F32x4Backend>(token: T, fg: &mut [f32], pixel: &[f32; 4]) {
    let px = f32x4::load(token, pixel);
    let (fg_chunks, _) = f32x4::<T>::partition_slice_mut(token, fg);

    for fg_chunk in fg_chunks.iter_mut() {
        let fg_pixel = f32x4::load(token, fg_chunk);
        let inv_alpha = f32x4::splat(token, 1.0 - fg_chunk[3]);
        let result = fg_pixel + px * inv_alpha;
        result.store(fg_chunk);
    }
}

/// SrcOver solid opaque pixel blend using magetypes f32x4.
#[inline]
pub(super) fn blend_src_over_solid_opaque<T: F32x4Backend>(
    token: T,
    fg: &mut [f32],
    pixel: &[f32; 4],
) {
    let px = f32x4::load(token, pixel);
    let (fg_chunks, _) = f32x4::<T>::partition_slice_mut(token, fg);

    for fg_chunk in fg_chunks.iter_mut() {
        let fg_pixel = f32x4::load(token, fg_chunk);
        let inv_alpha = f32x4::splat(token, 1.0 - fg_chunk[3]);
        let result = fg_pixel + px * inv_alpha;
        let mut arr = result.to_array();
        arr[3] = 1.0;
        fg_chunk.copy_from_slice(&arr);
    }
}

/// Per-pixel mask multiply using magetypes f32x4.
#[inline]
pub(super) fn mask_row_apply<T: F32x4Backend>(token: T, fg: &mut [f32], mask: &[f32]) {
    let (fg_chunks, _) = f32x4::<T>::partition_slice_mut(token, fg);

    for (fg_chunk, &m) in fg_chunks.iter_mut().zip(mask.iter()) {
        let fg_pixel = f32x4::load(token, fg_chunk);
        let mask_vec = f32x4::splat(token, m);
        let result = fg_pixel * mask_vec;
        result.store(fg_chunk);
    }
}

/// Per-pixel mask multiply on RGB only, alpha untouched, magetypes f32x4.
#[inline]
pub(super) fn mask_row_rgb_apply<T: F32x4Backend>(token: T, fg: &mut [f32], mask: &[f32]) {
    let (fg_chunks, _) = f32x4::<T>::partition_slice_mut(token, fg);

    for (fg_chunk, &m) in fg_chunks.iter_mut().zip(mask.iter()) {
        let fg_pixel = f32x4::load(token, fg_chunk);
        let mask_vec = f32x4::from_array(token, [m, m, m, 1.0]);
        let result = fg_pixel * mask_vec;
        result.store(fg_chunk);
    }
}

// =============================================================================
// SIMD artistic blend modes (separable) via magetypes f32x4.
// =============================================================================
//
// Division-free premultiplied closed forms.
//
// For the separable modes whose `sa·da·f(cs,cd)` term cancels the unpremultiply
// reciprocals, the whole composite reduces to a polynomial in the PREMULTIPLIED
// fg/bg plus the scalar alphas. This eliminates BOTH per-pixel scalar divisions
// (the N1 FDIV bottleneck) and the cs/cd vectors. Derivations (Cs=fg/sa, Cd=bg/da):
//
//   out_c = (1-da)·fg + (1-sa)·bg + sa·da·f(Cs,Cd)
//
//   Multiply  f=Cs·Cd     -> sa·da·Cs·Cd = fg·bg            -> fg + bg − fg·bg ... no:
//             out = (1-da)fg+(1-sa)bg+fg·bg
//   Screen    f=Cs+Cd−CsCd-> da·fg+sa·bg−fg·bg              -> fg + bg − fg·bg
//   Darken    f=min(Cs,Cd)-> min(da·fg, sa·bg)              -> (1-da)fg+(1-sa)bg+min(da·fg,sa·bg)
//   Lighten   f=max(Cs,Cd)-> max(da·fg, sa·bg)
//   Exclusion f=Cs+Cd−2CsCd-> da·fg+sa·bg−2fg·bg            -> fg + bg − 2·fg·bg
//   Difference f=|Cs−Cd|  -> |da·fg − sa·bg|                -> (1-da)fg+(1-sa)bg+|da·fg−sa·bg|
//
// These agree with the scalar (divide-then-remultiply) reference to within f32
// rounding (verified by the equivalence tests in blend.rs) and are MORE robust
// at sa→0 (no 0/0 → NaN). The per-pixel scalar work is now just `out_a` + guard.
// =============================================================================

/// Premultiplied closed-form kernel: `combine(token, fg_v, bg_v, base, sa, da)`
/// returns the RGB output vector (lane 3 discarded, overwritten with out_a).
/// `base = (1-da)·fg + (1-sa)·bg` is precomputed (shared by several modes).
///
/// The 18 modes that need STRAIGHT colour do not get a hand-written kernel, and
/// that is a measured decision rather than an omission — on aarch64 NEON is
/// baseline, so the portable path is already vectorized (the tier bench reports
/// 1.00x neon-vs-forced-scalar for them). Three different hand-written
/// decompositions were built and benchmarked; all three lost. Numbers and the
/// reason each failed: `benchmarks/neon_unpremul_attempts_2026-07-31.md`. Read
/// that before attempting a fourth.
#[inline]
fn artistic_premul_simd<T, F>(token: T, fg: &mut [f32], bg: &[f32], combine: F)
where
    T: F32x4Backend,
    F: Fn(T, f32x4<T>, f32x4<T>, f32x4<T>, f32, f32) -> f32x4<T>,
{
    let (fg_chunks, _) = f32x4::<T>::partition_slice_mut(token, fg);
    let (bg_chunks, _) = f32x4::<T>::partition_slice(token, bg);

    for (fg_chunk, bg_chunk) in fg_chunks.iter_mut().zip(bg_chunks.iter()) {
        let sa = fg_chunk[3];
        let da = bg_chunk[3];
        let out_a = sa + da - sa * da;

        if !out_a.is_finite() || out_a <= 0.0 {
            *fg_chunk = [0.0, 0.0, 0.0, 0.0];
            continue;
        }

        let fg_v = f32x4::load(token, fg_chunk);
        let bg_v = f32x4::load(token, bg_chunk);
        // base = (1-da)·fg + (1-sa)·bg  (same operation order as scalar reference)
        let base = fg_v * f32x4::splat(token, 1.0 - da) + bg_v * f32x4::splat(token, 1.0 - sa);

        let out = combine(token, fg_v, bg_v, base, sa, da);

        // Vectorized non-finite sanitization (matches the scalar reference's
        // per-channel `is_finite` guard): `abs(x) < +inf` is false for both NaN
        // (abs(NaN) is NaN; NaN < inf is false) and ±inf — so one masked select
        // zeroes any non-finite lane. Avoids a per-pixel scalar branch loop.
        let inf = f32x4::splat(token, f32::INFINITY);
        let zero = f32x4::splat(token, 0.0);
        let finite = out.abs().simd_lt(inf);
        let clean = f32x4::blend(finite, out, zero);

        // Splice the correct out_a into lane 3, keeping sanitized RGB in 0..2,
        // without leaving SIMD. `rgb_sel` is true (all-bits) in lanes 0..2.
        let rgb_sel = f32x4::from_array(token, [1.0, 1.0, 1.0, 0.0]).simd_gt(zero);
        let result = f32x4::blend(rgb_sel, clean, f32x4::splat(token, out_a));

        result.store(fg_chunk);
    }
}

macro_rules! artistic_premul_fn {
    ($name:ident, $f:expr) => {
        #[inline]
        pub(super) fn $name<T: F32x4Backend>(token: T, fg: &mut [f32], bg: &[f32]) {
            artistic_premul_simd(token, fg, bg, $f);
        }
    };
}

// ── Overlay / HardLight: division-free premultiplied closed forms ──────────
//
// Added 2026-08-01. These two were previously in the "needs straight colour"
// group, where three hand-written decompositions were built and all LOST (see
// the note on `artistic_premul_simd`). Every one of those attempts kept the
// unpremultiply and tried to make it cheaper. That was the wrong target: the
// unpremultiply is not NEEDED here.
//
// Overlay is `Cd < 0.5 ? Multiply : Screen`, and BOTH of those already have
// division-free premultiplied forms in this file. Carrying the branch through
// the same algebra (Cs = fg/sa, Cd = bg/da):
//
//   condition  Cd < 0.5              <->  bg < 0.5·da
//   branch 1   sa·da·(2·Cs·Cd)       ->   2·fg·bg
//   branch 2   sa·da·(1−2(1−Cs)(1−Cd)) -> sa·da − 2(sa−fg)(da−bg)
//
// because sa(1−Cs) = sa−fg and da(1−Cd) = da−bg. No division survives, and the
// branch becomes a vector select on premultiplied values.
//
// HardLight is Overlay with the roles swapped (`Cs < 0.5`), so its condition is
// `fg < 0.5·sa` over the identical two branches.
artistic_premul_fn!(
    blend_overlay_simd,
    |t, fg: f32x4<T>, bg: f32x4<T>, base: f32x4<T>, sa: f32, da: f32| {
        let two = f32x4::splat(t, 2.0);
        let lo = base + two * fg * bg;
        let hi = base + f32x4::splat(t, sa * da)
            - two * (f32x4::splat(t, sa) - fg) * (f32x4::splat(t, da) - bg);
        f32x4::blend(bg.simd_lt(f32x4::splat(t, 0.5 * da)), lo, hi)
    }
);
artistic_premul_fn!(
    blend_hard_light_simd,
    |t, fg: f32x4<T>, bg: f32x4<T>, base: f32x4<T>, sa: f32, da: f32| {
        let two = f32x4::splat(t, 2.0);
        let lo = base + two * fg * bg;
        let hi = base + f32x4::splat(t, sa * da)
            - two * (f32x4::splat(t, sa) - fg) * (f32x4::splat(t, da) - bg);
        f32x4::blend(fg.simd_lt(f32x4::splat(t, 0.5 * sa)), lo, hi)
    }
);

// ── LinearLight / PinLight: also division-free in premultiplied space ──────
//
// Added 2026-08-01, after the Overlay/HardLight reduction. I had claimed the
// remaining eight straight-colour modes "do not reduce this way" as a blanket
// structural fact. That was too broad — it was a claim about a family, made
// without working each member. These two reduce cleanly.
//
// Both are min/max of affine functions, and `sa·da >= 0`, so the factor moves
// inside: `sa·da·min(x, y) = min(sa·da·x, sa·da·y)`. With Cs = fg/sa,
// Cd = bg/da:
//
//   LinearLight, Cs < 0.5:  sa·da·max(0, 2Cs+Cd-1)  -> max(0, 2·da·fg + sa·bg - sa·da)
//   LinearLight, else:      sa·da·min(1, 2Cs-1+Cd)  -> min(sa·da, 2·da·fg - sa·da + sa·bg)
//   PinLight,    Cs < 0.5:  sa·da·min(Cd, 2Cs)      -> min(sa·bg, 2·da·fg)
//   PinLight,    else:      sa·da·max(Cd, 2Cs-1)    -> max(sa·bg, 2·da·fg - sa·da)
//
// The `Cs < 0.5` condition is `fg < 0.5·sa` in both. No division anywhere.
artistic_premul_fn!(
    blend_linear_light_simd,
    |t, fg: f32x4<T>, bg: f32x4<T>, base: f32x4<T>, sa: f32, da: f32| {
        let two = f32x4::splat(t, 2.0);
        let sada = f32x4::splat(t, sa * da);
        let common = two * f32x4::splat(t, da) * fg + f32x4::splat(t, sa) * bg;
        let lo = base + (common - sada).max(f32x4::splat(t, 0.0));
        let hi = base + (common - sada).min(sada);
        f32x4::blend(fg.simd_lt(f32x4::splat(t, 0.5 * sa)), lo, hi)
    }
);
artistic_premul_fn!(
    blend_pin_light_simd,
    |t, fg: f32x4<T>, bg: f32x4<T>, base: f32x4<T>, sa: f32, da: f32| {
        let two = f32x4::splat(t, 2.0);
        let sada = f32x4::splat(t, sa * da);
        let sa_bg = f32x4::splat(t, sa) * bg;
        let two_da_fg = two * f32x4::splat(t, da) * fg;
        let lo = base + sa_bg.min(two_da_fg);
        let hi = base + sa_bg.max(two_da_fg - sada);
        f32x4::blend(fg.simd_lt(f32x4::splat(t, 0.5 * sa)), lo, hi)
    }
);

// ── ColorDodge / Divide: ONE vector divide instead of ~5 scalar ones ───────
//
// These do not go division-free — Cd/(1-Cs) has a genuine reciprocal. But the
// scalar path pays ~5 divides PER PIXEL (inv_sa, inv_da, then one per channel
// inside f), whereas the premultiplied form needs exactly ONE vector divide per
// pixel, which `vdivq_f32` does in a single instruction across 4 lanes.
//
//   ColorDodge  sa·da·min(Cd/(1-Cs), 1) -> min(sa²·bg/(sa-fg), sa·da)
//               Cd <= 0  -> 0            (i.e. bg <= 0)
//               Cs >= 1  -> sa·da        (i.e. fg >= sa)
//   Divide      sa·da·min(Cd/Cs, 1)     -> min(sa²·bg/fg, sa·da)
//               Cs <= 0  -> sa·da        (i.e. fg <= 0)
//
// The divide runs on every lane including the guarded ones — SIMD has no
// branches — so it can produce inf/NaN there. That is harmless because the
// guard is a `blend` that SELECTS a lane rather than multiplying by it, so the
// poisoned value is discarded rather than propagated.
artistic_premul_fn!(
    blend_color_dodge_simd,
    |t, fg: f32x4<T>, bg: f32x4<T>, base: f32x4<T>, sa: f32, da: f32| {
        let zero = f32x4::splat(t, 0.0);
        let sada = f32x4::splat(t, sa * da);
        let sa_v = f32x4::splat(t, sa);
        let num = f32x4::splat(t, sa * sa) * bg;
        let quot = (num / (sa_v - fg)).min(sada);
        // Cs >= 1 first, then Cd <= 0 — matching the scalar's guard order.
        let r = f32x4::blend(fg.simd_ge(sa_v), sada, quot);
        f32x4::blend(bg.simd_le(zero), zero, r) + base
    }
);
// ColorBurn, same shape as ColorDodge:
//   sa·da·(1 − min((1−Cd)/Cs, 1)) -> sa·da − min(sa²·(da−bg)/fg, sa·da)
//   guards: Cd >= 1 (bg >= da) -> sa·da ;  Cs <= 0 (fg <= 0) -> 0
artistic_premul_fn!(
    blend_color_burn_simd,
    |t, fg: f32x4<T>, bg: f32x4<T>, base: f32x4<T>, sa: f32, da: f32| {
        let zero = f32x4::splat(t, 0.0);
        let sada = f32x4::splat(t, sa * da);
        let da_v = f32x4::splat(t, da);
        let num = f32x4::splat(t, sa * sa) * (da_v - bg);
        let term = sada - (num / fg).min(sada);
        // Guard order matches the scalar: Cd >= 1 first, then Cs <= 0.
        let r = f32x4::blend(fg.simd_le(zero), zero, term);
        f32x4::blend(bg.simd_ge(da_v), sada, r) + base
    }
);

// ── VividLight / HardMix: compositions of the dodge/burn pair ──────────────
//
// VividLight is `Cs < 0.5 ? ColorBurn(2Cs, Cd) : ColorDodge(2Cs-1, Cd)`, so it
// reduces by substituting 2Cs (i.e. 2·fg) into the two forms above:
//
//   Cs < 0.5:  sa·da - min(sa²·(da-bg)/(2·fg), sa·da)
//   else:      min(sa²·bg/(2·(sa-fg)), sa·da)
//
// HardMix is `vivid < 0.5 ? 0 : 1`. The threshold is on the STRAIGHT value, but
// it can be compared without recovering it: `f < 0.5` <-> `sa·da·f < 0.5·sa·da`,
// and `sa·da·f` is exactly what the VividLight term already is. So HardMix costs
// one extra compare and select over VividLight, with no extra division.
#[inline]
fn vivid_term<T: F32x4Backend>(
    t: T,
    fg: f32x4<T>,
    bg: f32x4<T>,
    sa: f32,
    da: f32,
) -> f32x4<T> {
    let zero = f32x4::splat(t, 0.0);
    let two = f32x4::splat(t, 2.0);
    let sada = f32x4::splat(t, sa * da);
    let sa_v = f32x4::splat(t, sa);
    let da_v = f32x4::splat(t, da);
    let sa2 = f32x4::splat(t, sa * sa);

    // Cs < 0.5 branch: ColorBurn(2Cs, Cd).
    let burn = sada - (sa2 * (da_v - bg) / (two * fg)).min(sada);
    let burn = f32x4::blend(fg.simd_le(zero), zero, burn);
    let burn = f32x4::blend(bg.simd_ge(da_v), sada, burn);

    // Cs >= 0.5 branch: ColorDodge(2Cs-1, Cd).
    let dodge = (sa2 * bg / (two * (sa_v - fg))).min(sada);
    let dodge = f32x4::blend(fg.simd_ge(sa_v), sada, dodge);
    let dodge = f32x4::blend(bg.simd_le(zero), zero, dodge);

    f32x4::blend(fg.simd_lt(f32x4::splat(t, 0.5 * sa)), burn, dodge)
}

artistic_premul_fn!(
    blend_vivid_light_simd,
    |t, fg: f32x4<T>, bg: f32x4<T>, base: f32x4<T>, sa: f32, da: f32| {
        base + vivid_term(t, fg, bg, sa, da)
    }
);
artistic_premul_fn!(
    blend_hard_mix_simd,
    |t, fg: f32x4<T>, bg: f32x4<T>, base: f32x4<T>, sa: f32, da: f32| {
        let sada = f32x4::splat(t, sa * da);
        let v = vivid_term(t, fg, bg, sa, da);
        // f < 0.5  <->  sa·da·f < 0.5·sa·da
        base + f32x4::blend(v.simd_lt(f32x4::splat(t, 0.5 * sa * da)), f32x4::splat(t, 0.0), sada)
    }
);

// ── SoftLight: vectorized, but NOT division-free — and that is the point ───
//
// I claimed twice that this could not reduce because `D(Cd)` contains
// `sqrt(Cd)` and "sqrt does not factor out of the sa·da product". The premise
// was wrong twice over: `sqrt(Cd) = sqrt(bg)/sqrt(da)` with `da` a per-pixel
// SCALAR, and more importantly the goal is VECTORIZATION, not eliminating
// division.
//
// A fully premultiplied closed form does exist and is algebraically exact —
// verified to 2.2e-16 in f64 — but it is NUMERICALLY UNUSABLE in f32:
//
//   sa·bg - (sa - 2·fg)·bg·(da - bg)·(1/da)
//
// forms a tiny product and then scales it by a huge `1/da`, and measured 2.9e-3
// against the reference, ~6x the crate's 5e-4 tolerance. The reference is
// well-conditioned precisely because it NORMALIZES FIRST and keeps every
// intermediate in [0, 1].
//
// So this kernel does what the reference does — `cs = fg·inv_sa`,
// `cd = bg·inv_da` with the SAME two scalar reciprocals per pixel and the same
// guards — and vectorizes everything after that. Two scalar divides per pixel
// is what the scalar path already pays; the win is that the per-channel work
// (the polynomial, the sqrt, the branch) becomes one vector operation instead
// of three scalar ones.
artistic_premul_fn!(
    blend_soft_light_simd,
    |t, fg: f32x4<T>, bg: f32x4<T>, base: f32x4<T>, sa: f32, da: f32| {
        let two = f32x4::splat(t, 2.0);
        let one = f32x4::splat(t, 1.0);
        // Same guarded reciprocals the scalar reference computes.
        let inv_sa = if sa > 0.0 { 1.0 / sa } else { 0.0 };
        let inv_da = if da > 0.0 { 1.0 / da } else { 0.0 };
        let cs = fg * f32x4::splat(t, inv_sa);
        let cd = bg * f32x4::splat(t, inv_da);

        let lo = cd - (one - two * cs) * cd * (one - cd);

        let poly = ((f32x4::splat(t, 16.0) * cd - f32x4::splat(t, 12.0)) * cd
            + f32x4::splat(t, 4.0))
            * cd;
        let g = f32x4::blend(cd.simd_le(f32x4::splat(t, 0.25)), poly, cd.sqrt());
        let hi = cd + (two * cs - one) * (g - cd);

        let blended = f32x4::blend(cs.simd_le(f32x4::splat(t, 0.5)), lo, hi);
        base + f32x4::splat(t, sa * da) * blended
    }
);

artistic_premul_fn!(
    blend_divide_simd,
    |t, fg: f32x4<T>, bg: f32x4<T>, base: f32x4<T>, sa: f32, da: f32| {
        let zero = f32x4::splat(t, 0.0);
        let sada = f32x4::splat(t, sa * da);
        let num = f32x4::splat(t, sa * sa) * bg;
        let quot = (num / fg).min(sada);
        f32x4::blend(fg.simd_le(zero), sada, quot) + base
    }
);

// out = (1-da)fg + (1-sa)bg + fg·bg
artistic_premul_fn!(
    blend_multiply_simd,
    |_t, fg: f32x4<T>, bg: f32x4<T>, base: f32x4<T>, _sa, _da| { base + fg * bg }
);
// out = fg + bg − fg·bg
artistic_premul_fn!(
    blend_screen_simd,
    |_t, fg: f32x4<T>, bg: f32x4<T>, _base: f32x4<T>, _sa, _da| { (fg + bg) - fg * bg }
);
// out = fg + bg − 2·fg·bg
artistic_premul_fn!(
    blend_exclusion_simd,
    |_t, fg: f32x4<T>, bg: f32x4<T>, _base: f32x4<T>, _sa, _da| {
        let fb = fg * bg;
        (fg + bg) - (fb + fb)
    }
);
// out = base + min(da·fg, sa·bg)
artistic_premul_fn!(
    blend_darken_simd,
    |t: T, fg: f32x4<T>, bg: f32x4<T>, base: f32x4<T>, sa, da| {
        let dfg = fg * f32x4::splat(t, da);
        let sbg = bg * f32x4::splat(t, sa);
        base + dfg.min(sbg)
    }
);
// out = base + max(da·fg, sa·bg)
artistic_premul_fn!(
    blend_lighten_simd,
    |t: T, fg: f32x4<T>, bg: f32x4<T>, base: f32x4<T>, sa, da| {
        let dfg = fg * f32x4::splat(t, da);
        let sbg = bg * f32x4::splat(t, sa);
        base + dfg.max(sbg)
    }
);
// out = base + |da·fg − sa·bg|
artistic_premul_fn!(
    blend_difference_simd,
    |t: T, fg: f32x4<T>, bg: f32x4<T>, base: f32x4<T>, sa, da| {
        let dfg = fg * f32x4::splat(t, da);
        let sbg = bg * f32x4::splat(t, sa);
        base + (dfg - sbg).abs()
    }
);

// Three more modes that are ALSO division-free in premultiplied space, and so
// belong in this framework rather than in the generic unpremultiply path they
// were using. With `Cs = cs/sa` and `Cb = cb/da`, the `sa*da*B(Cs,Cb)` term of
// the compositing formula cancels both divisions exactly as it does for
// darken/lighten above:
//
//   LinearDodge  sa*da*min(1, Cs+Cb)      = min(sa*da, da*cs + sa*cb)
//   LinearBurn   sa*da*max(0, Cs+Cb-1)    = max(0, da*cs + sa*cb - sa*da)
//   Subtract     sa*da*max(0, Cb-Cs)      = max(0, sa*cb - da*cs)
//
// Being division-free is not just faster — it removes a division by a
// potentially tiny alpha, so these are also better conditioned than the
// unpremultiplying reference they replace.

// out = base + min(sa·da, da·fg + sa·bg)
artistic_premul_fn!(
    blend_linear_dodge_simd,
    |t: T, fg: f32x4<T>, bg: f32x4<T>, base: f32x4<T>, sa, da| {
        let dfg = fg * f32x4::splat(t, da);
        let sbg = bg * f32x4::splat(t, sa);
        base + f32x4::splat(t, sa * da).min(dfg + sbg)
    }
);
// out = base + max(0, da·fg + sa·bg − sa·da)
artistic_premul_fn!(
    blend_linear_burn_simd,
    |t: T, fg: f32x4<T>, bg: f32x4<T>, base: f32x4<T>, sa, da| {
        let dfg = fg * f32x4::splat(t, da);
        let sbg = bg * f32x4::splat(t, sa);
        base + ((dfg + sbg) - f32x4::splat(t, sa * da)).max(f32x4::splat(t, 0.0))
    }
);
// out = base + max(0, sa·bg − da·fg)
artistic_premul_fn!(
    blend_subtract_simd,
    |t: T, fg: f32x4<T>, bg: f32x4<T>, base: f32x4<T>, sa, da| {
        let dfg = fg * f32x4::splat(t, da);
        let sbg = bg * f32x4::splat(t, sa);
        base + (sbg - dfg).max(f32x4::splat(t, 0.0))
    }
);

/// Linearly interpolate between two RGBA rows, magetypes f32x4.
#[inline]
pub(super) fn lerp_row_apply<T: F32x4Backend>(
    token: T,
    a: &[f32],
    b: &[f32],
    t: &[f32],
    out: &mut [f32],
) {
    let (a_chunks, _) = f32x4::<T>::partition_slice(token, a);
    let (b_chunks, _) = f32x4::<T>::partition_slice(token, b);
    let (out_chunks, _) = f32x4::<T>::partition_slice_mut(token, out);

    for ((a_chunk, b_chunk), (&tv, out_chunk)) in a_chunks
        .iter()
        .zip(b_chunks.iter())
        .zip(t.iter().zip(out_chunks.iter_mut()))
    {
        let a_vec = f32x4::load(token, a_chunk);
        let b_vec = f32x4::load(token, b_chunk);
        let t_vec = f32x4::splat(token, tv);
        let result = a_vec + (b_vec - a_vec) * t_vec;
        result.store(out_chunk);
    }
}
