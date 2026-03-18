//! Scalar fallback blend kernels.

use archmage::ScalarToken;

/// SrcOver row blend, scalar fallback.
/// fg[i] += bg[i] * (1 - fg_alpha) for each RGBA pixel.
pub(crate) fn blend_src_over_row_scalar(_token: ScalarToken, fg: &mut [f32], bg: &[f32]) {
    for (s, b) in fg.chunks_exact_mut(4).zip(bg.chunks_exact(4)) {
        let inv_a = 1.0 - s[3];
        s[0] += b[0] * inv_a;
        s[1] += b[1] * inv_a;
        s[2] += b[2] * inv_a;
        s[3] += b[3] * inv_a;
    }
}

/// SrcOver with a solid pixel, scalar fallback.
pub(crate) fn blend_src_over_solid_scalar(_token: ScalarToken, fg: &mut [f32], pixel: &[f32; 4]) {
    for s in fg.chunks_exact_mut(4) {
        let inv_a = 1.0 - s[3];
        s[0] += pixel[0] * inv_a;
        s[1] += pixel[1] * inv_a;
        s[2] += pixel[2] * inv_a;
        s[3] += pixel[3] * inv_a;
    }
}

/// SrcOver with a solid opaque pixel, scalar fallback.
pub(crate) fn blend_src_over_solid_opaque_scalar(
    _token: ScalarToken,
    fg: &mut [f32],
    pixel: &[f32; 4],
) {
    for s in fg.chunks_exact_mut(4) {
        let inv_a = 1.0 - s[3];
        s[0] += pixel[0] * inv_a;
        s[1] += pixel[1] * inv_a;
        s[2] += pixel[2] * inv_a;
        s[3] = 1.0;
    }
}

/// Multiply premultiplied RGBA by per-pixel mask, scalar fallback.
/// mask has one f32 per pixel, fg has 4 f32 per pixel.
pub(crate) fn mask_row_apply_scalar(_token: ScalarToken, fg: &mut [f32], mask: &[f32]) {
    for (pixel, &m) in fg.chunks_exact_mut(4).zip(mask.iter()) {
        pixel[0] *= m;
        pixel[1] *= m;
        pixel[2] *= m;
        pixel[3] *= m;
    }
}

/// Multiply RGB by per-pixel mask, leave alpha untouched, scalar fallback.
pub(crate) fn mask_row_rgb_apply_scalar(_token: ScalarToken, fg: &mut [f32], mask: &[f32]) {
    for (pixel, &m) in fg.chunks_exact_mut(4).zip(mask.iter()) {
        pixel[0] *= m;
        pixel[1] *= m;
        pixel[2] *= m;
    }
}

/// Linearly interpolate between two RGBA rows, scalar fallback.
pub(crate) fn lerp_row_apply_scalar(
    _token: ScalarToken,
    a: &[f32],
    b: &[f32],
    t: &[f32],
    out: &mut [f32],
) {
    for ((a_px, b_px), (&t_val, out_px)) in a
        .chunks_exact(4)
        .zip(b.chunks_exact(4))
        .zip(t.iter().zip(out.chunks_exact_mut(4)))
    {
        out_px[0] = a_px[0] + (b_px[0] - a_px[0]) * t_val;
        out_px[1] = a_px[1] + (b_px[1] - a_px[1]) * t_val;
        out_px[2] = a_px[2] + (b_px[2] - a_px[2]) * t_val;
        out_px[3] = a_px[3] + (b_px[3] - a_px[3]) * t_val;
    }
}
