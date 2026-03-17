//! SIMD-accelerated blend kernels.
//!
//! Uses archmage incant! dispatch to select the best available implementation:
//! - x86_64: AVX2+FMA (X64V3Token)
//! - AArch64: NEON via wide (NeonToken)
//! - WASM32: SIMD128 via wide (Wasm128Token)
//! - Fallback: Scalar

mod scalar;
#[allow(unused_imports)]
use scalar::*;

#[cfg(target_arch = "x86_64")]
mod x86;
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use x86::*;

#[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
mod wide_kernels;

#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
use neon::*;

#[cfg(target_arch = "wasm32")]
mod wasm128;
#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
use wasm128::*;

/// SrcOver: fg[i] += bg[i] * (1 - fg_alpha). Row-based, 4ch RGBA.
pub(crate) fn blend_src_over_row(fg: &mut [f32], bg: &[f32]) {
    archmage::incant!(blend_src_over_row(fg, bg))
}

/// SrcOver solid: fg[i] += pixel[c] * (1 - fg_alpha). No row buffer.
pub(crate) fn blend_src_over_solid(fg: &mut [f32], pixel: &[f32; 4]) {
    archmage::incant!(blend_src_over_solid(fg, pixel))
}

/// SrcOver solid opaque: like solid but output alpha = 1.0.
pub(crate) fn blend_src_over_solid_opaque(fg: &mut [f32], pixel: &[f32; 4]) {
    archmage::incant!(blend_src_over_solid_opaque(fg, pixel))
}
