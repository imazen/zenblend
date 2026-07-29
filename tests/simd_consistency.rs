//! SIMD tier consistency tests for zenblend.
//!
//! Runs blend, mask, and lerp operations under every archmage SIMD tier
//! permutation and verifies all produce identical output (byte-exact via
//! FNV-1a hash, or within FMA tolerance for FMA-sensitive paths).

#![forbid(unsafe_code)]

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use zenblend::{BlendMode, blend_row, lerp_row, mask_row, mask_row_rgb};

/// Max absolute difference between two float slices.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Generate a row of premultiplied RGBA pixels (4 floats per pixel).
fn generate_rgba_row(pixel_count: usize, seed: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity(pixel_count * 4);
    for i in 0..pixel_count {
        let v = (i as u32).wrapping_mul(seed).wrapping_add(17);
        let a = ((v % 256) as f32) / 255.0;
        let r = (((v >> 8) % 256) as f32 / 255.0) * a;
        let g = (((v >> 4) % 256) as f32 / 255.0) * a;
        let b = (((v >> 2) % 256) as f32 / 255.0) * a;
        out.extend_from_slice(&[r, g, b, a]);
    }
    out
}

/// Generate a per-pixel mask (one f32 per pixel).
fn generate_mask(pixel_count: usize, seed: u32) -> Vec<f32> {
    (0..pixel_count)
        .map(|i| {
            let v = (i as u32).wrapping_mul(seed).wrapping_add(31);
            (v % 256) as f32 / 255.0
        })
        .collect()
}

const PIXELS: usize = 256;

#[test]
fn src_over_blend_all_tiers_match() {
    let bg = generate_rgba_row(PIXELS, 7);
    let fg_template = generate_rgba_row(PIXELS, 13);
    let mut reference: Option<Vec<f32>> = None;

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let mut fg = fg_template.clone();
        blend_row(&mut fg, &bg, BlendMode::SrcOver);

        if let Some(ref ref_data) = reference {
            let diff = max_abs_diff(ref_data, &fg);
            assert!(
                diff < 1e-6,
                "SrcOver blend differs under '{}': max_diff={diff}",
                perm.label,
            );
        } else {
            reference = Some(fg);
        }
    });
}

#[test]
fn multiply_blend_all_tiers_match() {
    let bg = generate_rgba_row(PIXELS, 19);
    let fg_template = generate_rgba_row(PIXELS, 23);
    let mut reference: Option<Vec<f32>> = None;

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let mut fg = fg_template.clone();
        blend_row(&mut fg, &bg, BlendMode::Multiply);

        if let Some(ref ref_data) = reference {
            let diff = max_abs_diff(ref_data, &fg);
            assert!(
                diff < 1e-6,
                "Multiply blend differs under '{}': max_diff={diff}",
                perm.label,
            );
        } else {
            reference = Some(fg);
        }
    });
}

#[test]
fn mask_row_all_tiers_match() {
    let fg_template = generate_rgba_row(PIXELS, 29);
    let mask = generate_mask(PIXELS, 37);
    let mut reference: Option<Vec<f32>> = None;

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let mut fg = fg_template.clone();
        mask_row(&mut fg, &mask);

        if let Some(ref ref_data) = reference {
            let diff = max_abs_diff(ref_data, &fg);
            assert!(
                diff < 1e-6,
                "mask_row differs under '{}': max_diff={diff}",
                perm.label,
            );
        } else {
            reference = Some(fg);
        }
    });
}

#[test]
fn mask_row_rgb_all_tiers_match() {
    let fg_template = generate_rgba_row(PIXELS, 41);
    let mask = generate_mask(PIXELS, 43);
    let mut reference: Option<Vec<f32>> = None;

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let mut fg = fg_template.clone();
        mask_row_rgb(&mut fg, &mask);

        if let Some(ref ref_data) = reference {
            let diff = max_abs_diff(ref_data, &fg);
            assert!(
                diff < 1e-6,
                "mask_row_rgb differs under '{}': max_diff={diff}",
                perm.label,
            );
        } else {
            reference = Some(fg);
        }
    });
}

#[test]
fn lerp_row_all_tiers_match() {
    let a = generate_rgba_row(PIXELS, 47);
    let b = generate_rgba_row(PIXELS, 53);
    let t = generate_mask(PIXELS, 59);
    let mut reference: Option<Vec<f32>> = None;

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let mut out = vec![0.0f32; PIXELS * 4];
        lerp_row(&a, &b, &t, &mut out);

        if let Some(ref ref_data) = reference {
            let diff = max_abs_diff(ref_data, &out);
            assert!(
                diff < 1e-6,
                "lerp_row differs under '{}': max_diff={diff}",
                perm.label,
            );
        } else {
            reference = Some(out);
        }
    });
}

/// Every blend mode, under every tier permutation.
///
/// The two tests above cover SrcOver and Multiply — 2 of 32 modes — so a tier
/// disagreement in any other mode was invisible here. That matters more now
/// that LinearDodge / LinearBurn / Subtract have moved onto the shared
/// premultiplied SIMD kernels: their scalar and NEON tiers must agree, and
/// this is what proves it rather than the fact that they share source.
#[test]
fn every_blend_mode_all_tiers_match() {
    const ALL_MODES: &[(&str, BlendMode)] = &[
        ("Clear", BlendMode::Clear),
        ("Src", BlendMode::Src),
        ("Dst", BlendMode::Dst),
        ("SrcOver", BlendMode::SrcOver),
        ("DstOver", BlendMode::DstOver),
        ("SrcIn", BlendMode::SrcIn),
        ("DstIn", BlendMode::DstIn),
        ("SrcOut", BlendMode::SrcOut),
        ("DstOut", BlendMode::DstOut),
        ("SrcAtop", BlendMode::SrcAtop),
        ("DstAtop", BlendMode::DstAtop),
        ("Xor", BlendMode::Xor),
        ("Multiply", BlendMode::Multiply),
        ("Screen", BlendMode::Screen),
        ("Overlay", BlendMode::Overlay),
        ("Darken", BlendMode::Darken),
        ("Lighten", BlendMode::Lighten),
        ("HardLight", BlendMode::HardLight),
        ("SoftLight", BlendMode::SoftLight),
        ("ColorDodge", BlendMode::ColorDodge),
        ("ColorBurn", BlendMode::ColorBurn),
        ("Difference", BlendMode::Difference),
        ("Exclusion", BlendMode::Exclusion),
        ("LinearBurn", BlendMode::LinearBurn),
        ("LinearDodge", BlendMode::LinearDodge),
        ("VividLight", BlendMode::VividLight),
        ("LinearLight", BlendMode::LinearLight),
        ("PinLight", BlendMode::PinLight),
        ("HardMix", BlendMode::HardMix),
        ("Divide", BlendMode::Divide),
        ("Subtract", BlendMode::Subtract),
        ("Plus", BlendMode::Plus),
    ];

    let bg = generate_rgba_row(PIXELS, 7);
    let fg_template = generate_rgba_row(PIXELS, 13);

    for &(name, mode) in ALL_MODES {
        let mut reference: Option<Vec<f32>> = None;
        let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let mut fg = fg_template.clone();
            blend_row(&mut fg, &bg, mode);
            if let Some(ref ref_data) = reference {
                let diff = max_abs_diff(ref_data, &fg);
                assert!(
                    diff < 1e-6,
                    "{name} differs under '{}': max_diff={diff}",
                    perm.label,
                );
            } else {
                reference = Some(fg);
            }
        });
    }
}
