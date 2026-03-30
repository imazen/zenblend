//! NEON blend kernel wrappers (delegate to generic magetypes kernels).

use archmage::NeonToken;

pub(crate) fn blend_src_over_row_neon(token: NeonToken, fg: &mut [f32], bg: &[f32]) {
    super::wide_kernels::blend_src_over_row_generic(token, fg, bg);
}

pub(crate) fn blend_src_over_solid_neon(token: NeonToken, fg: &mut [f32], pixel: &[f32; 4]) {
    super::wide_kernels::blend_src_over_solid_generic(token, fg, pixel);
}

pub(crate) fn blend_src_over_solid_opaque_neon(token: NeonToken, fg: &mut [f32], pixel: &[f32; 4]) {
    super::wide_kernels::blend_src_over_solid_opaque_generic(token, fg, pixel);
}

pub(crate) fn mask_row_apply_neon(token: NeonToken, fg: &mut [f32], mask: &[f32]) {
    super::wide_kernels::mask_row_apply_generic(token, fg, mask);
}

pub(crate) fn mask_row_rgb_apply_neon(token: NeonToken, fg: &mut [f32], mask: &[f32]) {
    super::wide_kernels::mask_row_rgb_apply_generic(token, fg, mask);
}

pub(crate) fn lerp_row_apply_neon(
    token: NeonToken,
    a: &[f32],
    b: &[f32],
    t: &[f32],
    out: &mut [f32],
) {
    super::wide_kernels::lerp_row_apply_generic(token, a, b, t, out);
}
