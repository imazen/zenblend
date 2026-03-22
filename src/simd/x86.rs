//! x86-64 AVX2+FMA blend kernels.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use archmage::X64V3Token;
use safe_unaligned_simd::x86_64::{_mm256_loadu_ps, _mm256_storeu_ps};

/// SrcOver row blend, AVX2+FMA.
///
/// Processes 2 RGBA pixels per 256-bit iteration (8 floats).
/// `vfnmadd231ps` computes `fg + bg * (1 - alpha)` in a single FMA:
///   result = fg + bg * inv_alpha = bg * inv_alpha + fg
///   Using fnmadd: result = -(bg * (-inv_alpha)) + fg  ... no, simpler:
///   out = fg + bg * inv_alpha → fmadd(bg, inv_alpha, fg)
#[archmage::arcane]
pub(crate) fn blend_src_over_row_v3(_token: X64V3Token, fg: &mut [f32], bg: &[f32]) {
    let ones = _mm256_set1_ps(1.0);

    let (fg_chunks, _) = fg.as_chunks_mut::<8>();
    let (bg_chunks, _) = bg.as_chunks::<8>();

    for (fg_chunk, bg_chunk) in fg_chunks.iter_mut().zip(bg_chunks.iter()) {
        let fg_vec = _mm256_loadu_ps(fg_chunk);
        let bg_vec = _mm256_loadu_ps(bg_chunk);

        // Extract alpha for each of the 2 pixels and broadcast within each pixel's lanes.
        // Pixel 0: lanes 0-3, alpha at lane 3
        // Pixel 1: lanes 4-7, alpha at lane 7
        // Use vshufps to broadcast alpha within each 128-bit half:
        // _MM_SHUFFLE(3,3,3,3) = 0xFF broadcasts lane 3 within each 128-bit half.
        let alpha = _mm256_shuffle_ps::<0xFF>(fg_vec, fg_vec);
        let inv_alpha = _mm256_sub_ps(ones, alpha);

        // out = fg + bg * inv_alpha
        let result = _mm256_fmadd_ps(bg_vec, inv_alpha, fg_vec);
        _mm256_storeu_ps(fg_chunk, result);
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
pub(crate) fn blend_src_over_solid_v3(_token: X64V3Token, fg: &mut [f32], pixel: &[f32; 4]) {
    let ones = _mm256_set1_ps(1.0);
    // Broadcast pixel into both 128-bit halves: [R,G,B,A,R,G,B,A]
    let px = _mm256_set_ps(
        pixel[3], pixel[2], pixel[1], pixel[0], pixel[3], pixel[2], pixel[1], pixel[0],
    );

    let (fg_chunks, _) = fg.as_chunks_mut::<8>();

    for fg_chunk in fg_chunks.iter_mut() {
        let fg_vec = _mm256_loadu_ps(fg_chunk);
        let alpha = _mm256_shuffle_ps::<0xFF>(fg_vec, fg_vec);
        let inv_alpha = _mm256_sub_ps(ones, alpha);
        let result = _mm256_fmadd_ps(px, inv_alpha, fg_vec);
        _mm256_storeu_ps(fg_chunk, result);
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
pub(crate) fn blend_src_over_solid_opaque_v3(_token: X64V3Token, fg: &mut [f32], pixel: &[f32; 4]) {
    let ones = _mm256_set1_ps(1.0);
    let px = _mm256_set_ps(
        pixel[3], pixel[2], pixel[1], pixel[0], pixel[3], pixel[2], pixel[1], pixel[0],
    );
    // Mask to force alpha lanes to 1.0: [0,0,0,1, 0,0,0,1] as blend mask
    // We'll just do the normal blend then overwrite alpha.
    let alpha_mask = _mm256_set_ps(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
    let alpha_blend = _mm256_set_ps(0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0);

    let (fg_chunks, _) = fg.as_chunks_mut::<8>();

    for fg_chunk in fg_chunks.iter_mut() {
        let fg_vec = _mm256_loadu_ps(fg_chunk);
        let alpha = _mm256_shuffle_ps::<0xFF>(fg_vec, fg_vec);
        let inv_alpha = _mm256_sub_ps(ones, alpha);
        let blended = _mm256_fmadd_ps(px, inv_alpha, fg_vec);
        // Force alpha to 1.0: result = blended * alpha_blend + alpha_mask
        let result = _mm256_fmadd_ps(blended, alpha_blend, alpha_mask);
        _mm256_storeu_ps(fg_chunk, result);
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
pub(crate) fn mask_row_apply_v3(_token: X64V3Token, fg: &mut [f32], mask: &[f32]) {
    let (fg_chunks, _) = fg.as_chunks_mut::<8>();
    let (mask_chunks, _) = mask.as_chunks::<2>();

    for (fg_chunk, mask_pair) in fg_chunks.iter_mut().zip(mask_chunks.iter()) {
        let fg_vec = _mm256_loadu_ps(fg_chunk);
        // Broadcast mask[0] to lanes 0-3, mask[1] to lanes 4-7
        let mask_vec = _mm256_set_ps(
            mask_pair[1],
            mask_pair[1],
            mask_pair[1],
            mask_pair[1],
            mask_pair[0],
            mask_pair[0],
            mask_pair[0],
            mask_pair[0],
        );
        let result = _mm256_mul_ps(fg_vec, mask_vec);
        _mm256_storeu_ps(fg_chunk, result);
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
pub(crate) fn mask_row_rgb_apply_v3(_token: X64V3Token, fg: &mut [f32], mask: &[f32]) {
    // Blend mask: [1,1,1,0, 1,1,1,0] — select from result for RGB, from original for A
    let blend_select = _mm256_set_ps(0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0);
    let alpha_select = _mm256_set_ps(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0);

    let (fg_chunks, _) = fg.as_chunks_mut::<8>();
    let (mask_chunks, _) = mask.as_chunks::<2>();

    for (fg_chunk, mask_pair) in fg_chunks.iter_mut().zip(mask_chunks.iter()) {
        let fg_vec = _mm256_loadu_ps(fg_chunk);
        let mask_vec = _mm256_set_ps(
            mask_pair[1],
            mask_pair[1],
            mask_pair[1],
            mask_pair[1],
            mask_pair[0],
            mask_pair[0],
            mask_pair[0],
            mask_pair[0],
        );
        let masked = _mm256_mul_ps(fg_vec, mask_vec);
        // result = masked * blend_select + fg * alpha_select
        let result = _mm256_fmadd_ps(fg_vec, alpha_select, _mm256_mul_ps(masked, blend_select));
        _mm256_storeu_ps(fg_chunk, result);
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
    _token: X64V3Token,
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
        let a_vec = _mm256_loadu_ps(a_chunk);
        let b_vec = _mm256_loadu_ps(b_chunk);
        let t_vec = _mm256_set_ps(
            t_pair[1], t_pair[1], t_pair[1], t_pair[1], t_pair[0], t_pair[0], t_pair[0], t_pair[0],
        );
        let diff = _mm256_sub_ps(b_vec, a_vec);
        // out = a + diff * t = fmadd(diff, t, a)
        let result = _mm256_fmadd_ps(diff, t_vec, a_vec);
        _mm256_storeu_ps(out_chunk, result);
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
