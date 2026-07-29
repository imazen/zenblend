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
