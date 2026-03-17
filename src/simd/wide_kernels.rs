//! Portable SIMD blend kernels via `wide` crate (shared by NEON and WASM128).
//!
//! These operate on `f32x4` — one RGBA pixel per SIMD register.
//! Less throughput than AVX2 (1 pixel vs 2 per iteration) but portable.

// These functions are called from neon.rs / wasm128.rs token wrappers.
// The token parameter proves feature availability at the call site.

/// SrcOver row blend using wide f32x4 (1 pixel per iteration).
#[inline]
pub(super) fn blend_src_over_row_wide(fg: &mut [f32], bg: &[f32]) {
    for (s, b) in fg.chunks_exact_mut(4).zip(bg.chunks_exact(4)) {
        let fg_pixel = wide::f32x4::from([s[0], s[1], s[2], s[3]]);
        let bg_pixel = wide::f32x4::from([b[0], b[1], b[2], b[3]]);
        let inv_alpha = wide::f32x4::splat(1.0 - s[3]);
        let result = fg_pixel + bg_pixel * inv_alpha;
        let arr: [f32; 4] = result.into();
        s.copy_from_slice(&arr);
    }
}

/// SrcOver solid pixel blend using wide f32x4.
#[inline]
pub(super) fn blend_src_over_solid_wide(fg: &mut [f32], pixel: &[f32; 4]) {
    let px = wide::f32x4::from(*pixel);
    for s in fg.chunks_exact_mut(4) {
        let fg_pixel = wide::f32x4::from([s[0], s[1], s[2], s[3]]);
        let inv_alpha = wide::f32x4::splat(1.0 - s[3]);
        let result = fg_pixel + px * inv_alpha;
        let arr: [f32; 4] = result.into();
        s.copy_from_slice(&arr);
    }
}

/// SrcOver solid opaque pixel blend using wide f32x4.
#[inline]
pub(super) fn blend_src_over_solid_opaque_wide(fg: &mut [f32], pixel: &[f32; 4]) {
    let px = wide::f32x4::from(*pixel);
    for s in fg.chunks_exact_mut(4) {
        let fg_pixel = wide::f32x4::from([s[0], s[1], s[2], s[3]]);
        let inv_alpha = wide::f32x4::splat(1.0 - s[3]);
        let result = fg_pixel + px * inv_alpha;
        let mut arr: [f32; 4] = result.into();
        arr[3] = 1.0;
        s.copy_from_slice(&arr);
    }
}

/// Per-pixel mask multiply using wide f32x4.
#[inline]
pub(super) fn mask_row_apply_wide(fg: &mut [f32], mask: &[f32]) {
    for (pixel, &m) in fg.chunks_exact_mut(4).zip(mask.iter()) {
        let fg_pixel = wide::f32x4::from([pixel[0], pixel[1], pixel[2], pixel[3]]);
        let mask_vec = wide::f32x4::splat(m);
        let result = fg_pixel * mask_vec;
        let arr: [f32; 4] = result.into();
        pixel.copy_from_slice(&arr);
    }
}
