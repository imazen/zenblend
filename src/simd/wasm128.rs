//! WASM128 blend kernel wrappers (delegate to wide_kernels).

use archmage::Wasm128Token;

pub(crate) fn blend_src_over_row_wasm128(
    _token: Wasm128Token,
    fg: &mut [f32],
    bg: &[f32],
) {
    super::wide_kernels::blend_src_over_row_wide(fg, bg);
}

pub(crate) fn blend_src_over_solid_wasm128(
    _token: Wasm128Token,
    fg: &mut [f32],
    pixel: &[f32; 4],
) {
    super::wide_kernels::blend_src_over_solid_wide(fg, pixel);
}

pub(crate) fn blend_src_over_solid_opaque_wasm128(
    _token: Wasm128Token,
    fg: &mut [f32],
    pixel: &[f32; 4],
) {
    super::wide_kernels::blend_src_over_solid_opaque_wide(fg, pixel);
}

pub(crate) fn mask_row_apply_wasm128(
    _token: Wasm128Token,
    fg: &mut [f32],
    mask: &[f32],
) {
    super::wide_kernels::mask_row_apply_wide(fg, mask);
}
