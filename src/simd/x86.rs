//! x86-64 AVX2+FMA blend kernels using magetypes f32x8.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use archmage::X64V3Token;
use magetypes::simd::f32x8;

/// SrcOver row blend, AVX2+FMA.
///
/// Processes 2 RGBA pixels per 256-bit iteration (8 floats).
/// Uses FMA: `out = fg + bg * (1 - alpha)`.
#[archmage::arcane]
pub(crate) fn blend_src_over_row_v3(token: X64V3Token, fg: &mut [f32], bg: &[f32]) {
    let ones = f32x8::splat(token, 1.0);

    let (fg_chunks, _) = fg.as_chunks_mut::<8>();
    let (bg_chunks, _) = bg.as_chunks::<8>();

    for (fg_chunk, bg_chunk) in fg_chunks.iter_mut().zip(bg_chunks.iter()) {
        let fg_vec = f32x8::load(token, fg_chunk);
        let bg_vec = f32x8::load(token, bg_chunk);

        // Broadcast alpha within each pixel's 4 lanes using vshufps.
        // _MM_SHUFFLE(3,3,3,3) = 0xFF broadcasts lane 3 within each 128-bit half.
        let alpha = f32x8::from_m256(token, _mm256_shuffle_ps::<0xFF>(fg_vec.raw(), fg_vec.raw()));
        let inv_alpha = ones - alpha;

        // out = fg + bg * inv_alpha
        let result = bg_vec.mul_add(inv_alpha, fg_vec);
        result.store(fg_chunk);
    }

    // Scalar tail for remaining pixels (0 or 1 pixel)
    let done = fg_chunks.len() * 8;
    let remaining_fg = &mut fg[done..];
    let remaining_bg = &bg[done..];
    for (s, b) in remaining_fg
        .chunks_exact_mut(4)
        .zip(remaining_bg.chunks_exact(4))
    {
        let inv_a = 1.0 - s[3];
        s[0] += b[0] * inv_a;
        s[1] += b[1] * inv_a;
        s[2] += b[2] * inv_a;
        s[3] += b[3] * inv_a;
    }
}

/// SrcOver solid pixel blend, AVX2+FMA.
///
/// Broadcasts the solid pixel into both halves of a 256-bit register.
#[archmage::arcane]
pub(crate) fn blend_src_over_solid_v3(token: X64V3Token, fg: &mut [f32], pixel: &[f32; 4]) {
    let ones = f32x8::splat(token, 1.0);
    // Broadcast pixel into both 128-bit halves: [R,G,B,A,R,G,B,A]
    let px = f32x8::from_array(
        token,
        [
            pixel[0], pixel[1], pixel[2], pixel[3], pixel[0], pixel[1], pixel[2], pixel[3],
        ],
    );

    let (fg_chunks, _) = fg.as_chunks_mut::<8>();

    for fg_chunk in fg_chunks.iter_mut() {
        let fg_vec = f32x8::load(token, fg_chunk);
        let alpha = f32x8::from_m256(token, _mm256_shuffle_ps::<0xFF>(fg_vec.raw(), fg_vec.raw()));
        let inv_alpha = ones - alpha;
        let result = px.mul_add(inv_alpha, fg_vec);
        result.store(fg_chunk);
    }

    // Scalar tail
    let done = fg_chunks.len() * 8;
    let remaining = &mut fg[done..];
    for s in remaining.chunks_exact_mut(4) {
        let inv_a = 1.0 - s[3];
        s[0] += pixel[0] * inv_a;
        s[1] += pixel[1] * inv_a;
        s[2] += pixel[2] * inv_a;
        s[3] += pixel[3] * inv_a;
    }
}

/// SrcOver solid opaque pixel blend, AVX2+FMA.
///
/// Same as solid but forces output alpha to 1.0.
#[archmage::arcane]
pub(crate) fn blend_src_over_solid_opaque_v3(token: X64V3Token, fg: &mut [f32], pixel: &[f32; 4]) {
    let ones = f32x8::splat(token, 1.0);
    let px = f32x8::from_array(
        token,
        [
            pixel[0], pixel[1], pixel[2], pixel[3], pixel[0], pixel[1], pixel[2], pixel[3],
        ],
    );
    // Mask to force alpha lanes to 1.0: [0,0,0,1, 0,0,0,1]
    let alpha_mask = f32x8::from_array(token, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let alpha_blend = f32x8::from_array(token, [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]);

    let (fg_chunks, _) = fg.as_chunks_mut::<8>();

    for fg_chunk in fg_chunks.iter_mut() {
        let fg_vec = f32x8::load(token, fg_chunk);
        let alpha = f32x8::from_m256(token, _mm256_shuffle_ps::<0xFF>(fg_vec.raw(), fg_vec.raw()));
        let inv_alpha = ones - alpha;
        let blended = px.mul_add(inv_alpha, fg_vec);
        // Force alpha to 1.0: result = blended * alpha_blend + alpha_mask
        let result = blended.mul_add(alpha_blend, alpha_mask);
        result.store(fg_chunk);
    }

    // Scalar tail
    let done = fg_chunks.len() * 8;
    let remaining = &mut fg[done..];
    for s in remaining.chunks_exact_mut(4) {
        let inv_a = 1.0 - s[3];
        s[0] += pixel[0] * inv_a;
        s[1] += pixel[1] * inv_a;
        s[2] += pixel[2] * inv_a;
        s[3] = 1.0;
    }
}

/// Per-pixel mask multiply, AVX2.
///
/// Each mask value is broadcast to 4 RGBA channels: `fg[px*4+c] *= mask[px]`.
/// Processes 2 pixels per 256-bit iteration.
#[archmage::arcane]
pub(crate) fn mask_row_apply_v3(token: X64V3Token, fg: &mut [f32], mask: &[f32]) {
    let (fg_chunks, _) = fg.as_chunks_mut::<8>();
    let (mask_chunks, _) = mask.as_chunks::<2>();

    for (fg_chunk, mask_pair) in fg_chunks.iter_mut().zip(mask_chunks.iter()) {
        let fg_vec = f32x8::load(token, fg_chunk);
        // Broadcast mask[0] to lanes 0-3, mask[1] to lanes 4-7
        let mask_vec = f32x8::from_array(
            token,
            [
                mask_pair[0],
                mask_pair[0],
                mask_pair[0],
                mask_pair[0],
                mask_pair[1],
                mask_pair[1],
                mask_pair[1],
                mask_pair[1],
            ],
        );
        let result = fg_vec * mask_vec;
        result.store(fg_chunk);
    }

    // Scalar tail
    let done = fg_chunks.len() * 2;
    let remaining_fg = &mut fg[done * 4..];
    let remaining_mask = &mask[done..];
    for (pixel, &m) in remaining_fg.chunks_exact_mut(4).zip(remaining_mask.iter()) {
        pixel[0] *= m;
        pixel[1] *= m;
        pixel[2] *= m;
        pixel[3] *= m;
    }
}

/// Per-pixel mask multiply on RGB only (alpha untouched), AVX2.
///
/// Broadcasts mask to R, G, B lanes; uses blend mask to preserve alpha.
/// Processes 2 pixels per 256-bit iteration.
#[archmage::arcane]
pub(crate) fn mask_row_rgb_apply_v3(token: X64V3Token, fg: &mut [f32], mask: &[f32]) {
    // Blend mask: [1,1,1,0, 1,1,1,0] — select from result for RGB, from original for A
    let blend_select = f32x8::from_array(token, [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]);
    let alpha_select = f32x8::from_array(token, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

    let (fg_chunks, _) = fg.as_chunks_mut::<8>();
    let (mask_chunks, _) = mask.as_chunks::<2>();

    for (fg_chunk, mask_pair) in fg_chunks.iter_mut().zip(mask_chunks.iter()) {
        let fg_vec = f32x8::load(token, fg_chunk);
        let mask_vec = f32x8::from_array(
            token,
            [
                mask_pair[0],
                mask_pair[0],
                mask_pair[0],
                mask_pair[0],
                mask_pair[1],
                mask_pair[1],
                mask_pair[1],
                mask_pair[1],
            ],
        );
        let masked = fg_vec * mask_vec;
        // result = masked * blend_select + fg * alpha_select
        let result = fg_vec.mul_add(alpha_select, masked * blend_select);
        result.store(fg_chunk);
    }

    // Scalar tail
    let done = fg_chunks.len() * 2;
    let remaining_fg = &mut fg[done * 4..];
    let remaining_mask = &mask[done..];
    for (pixel, &m) in remaining_fg.chunks_exact_mut(4).zip(remaining_mask.iter()) {
        pixel[0] *= m;
        pixel[1] *= m;
        pixel[2] *= m;
    }
}

/// Linearly interpolate between two RGBA rows, AVX2+FMA.
///
/// out = a + (b - a) * t, with t broadcast from 1 value to 4 RGBA channels.
/// Processes 2 pixels per 256-bit iteration.
#[archmage::arcane]
pub(crate) fn lerp_row_apply_v3(
    token: X64V3Token,
    a: &[f32],
    b: &[f32],
    t: &[f32],
    out: &mut [f32],
) {
    let (a_chunks, _) = a.as_chunks::<8>();
    let (b_chunks, _) = b.as_chunks::<8>();
    let (out_chunks, _) = out.as_chunks_mut::<8>();
    let (t_chunks, _) = t.as_chunks::<2>();

    for (((a_chunk, b_chunk), t_pair), out_chunk) in a_chunks
        .iter()
        .zip(b_chunks.iter())
        .zip(t_chunks.iter())
        .zip(out_chunks.iter_mut())
    {
        let a_vec = f32x8::load(token, a_chunk);
        let b_vec = f32x8::load(token, b_chunk);
        let t_vec = f32x8::from_array(
            token,
            [
                t_pair[0], t_pair[0], t_pair[0], t_pair[0], t_pair[1], t_pair[1], t_pair[1],
                t_pair[1],
            ],
        );
        let diff = b_vec - a_vec;
        // out = a + diff * t = fmadd(diff, t, a)
        let result = diff.mul_add(t_vec, a_vec);
        result.store(out_chunk);
    }

    // Scalar tail
    let done = a_chunks.len() * 2;
    let a_rem = &a[done * 4..];
    let b_rem = &b[done * 4..];
    let t_rem = &t[done..];
    let out_rem = &mut out[done * 4..];
    for ((a_px, b_px), (&tv, out_px)) in a_rem
        .chunks_exact(4)
        .zip(b_rem.chunks_exact(4))
        .zip(t_rem.iter().zip(out_rem.chunks_exact_mut(4)))
    {
        out_px[0] = a_px[0] + (b_px[0] - a_px[0]) * tv;
        out_px[1] = a_px[1] + (b_px[1] - a_px[1]) * tv;
        out_px[2] = a_px[2] + (b_px[2] - a_px[2]) * tv;
        out_px[3] = a_px[3] + (b_px[3] - a_px[3]) * tv;
    }
}
