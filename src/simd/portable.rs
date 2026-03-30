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
