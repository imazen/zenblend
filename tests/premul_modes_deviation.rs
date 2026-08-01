//! LinearDodge / LinearBurn / Subtract moved from the unpremultiplying generic
//! path to the division-free premultiplied kernels. This quantifies exactly how
//! much that changes their output.
//!
//! The move is algebraically an identity — with `Cs = cs/sa`, `Cb = cb/da`, the
//! `sa*da*B(Cs,Cb)` term cancels both divisions:
//!
//!   LinearDodge  sa*da*min(1, Cs+Cb)   = min(sa*da, da*cs + sa*cb)
//!   LinearBurn   sa*da*max(0, Cs+Cb-1) = max(0, da*cs + sa*cb - sa*da)
//!   Subtract     sa*da*max(0, Cb-Cs)   = max(0, sa*cb - da*cs)
//!
//! so any difference is pure floating-point rounding. It is not zero, though,
//! because the operation order differs — which is why this test MEASURES the
//! deviation rather than asserting bit-equality it would not get. The bound is
//! deliberately tight: if a future edit makes these disagree by more than
//! rounding, this fails.
//!
//! Direction matters too. The new form removes a division by alpha, so it is
//! better conditioned as alpha -> 0; the reference is the less accurate side
//! there. The small-alpha case is included explicitly for that reason.
//!
//! This mirrors the transformation already applied to Multiply / Screen /
//! Darken / Lighten / Difference / Exclusion, whose scalar tiers likewise use
//! the premultiplied form rather than the unpremultiplying reference.

use zenblend::{BlendMode, blend_row};

/// The unpremultiplying reference, transcribed from `blend_artistic_pixel`.
fn reference_pixel(fg: &mut [f32; 4], bg: &[f32; 4], f: impl Fn(f32, f32) -> f32) {
    let sa = fg[3];
    let da = bg[3];
    let out_a = sa + da - sa * da;
    if !out_a.is_finite() || out_a <= 0.0 {
        *fg = [0.0, 0.0, 0.0, 0.0];
        return;
    }
    let inv_sa = if sa > 0.0 { 1.0 / sa } else { 0.0 };
    let inv_da = if da > 0.0 { 1.0 / da } else { 0.0 };
    for i in 0..3 {
        let cs = fg[i] * inv_sa;
        let cd = bg[i] * inv_da;
        let blended = f(cs, cd);
        let out = (1.0 - da) * fg[i] + (1.0 - sa) * bg[i] + sa * da * blended;
        fg[i] = if out.is_finite() { out } else { 0.0 };
    }
    fg[3] = out_a;
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / 16_777_216.0
    }
}

/// Premultiplied RGBA rows, with alpha spanning the full range including the
/// small-alpha regime where the reference's `1/sa` is worst conditioned.
fn rows(n: usize, seed: u64, tiny_alpha: bool) -> (Vec<f32>, Vec<f32>) {
    let mut r = Rng(seed | 1);
    let mut fg = Vec::with_capacity(n * 4);
    let mut bg = Vec::with_capacity(n * 4);
    for i in 0..n {
        for out in [&mut fg, &mut bg] {
            let a = if tiny_alpha {
                // 1e-4 .. 1e-2 — where dividing by alpha hurts.
                1e-4 + r.unit() * 1e-2
            } else if i % 17 == 0 {
                1.0
            } else {
                r.unit()
            };
            out.push(r.unit() * a);
            out.push(r.unit() * a);
            out.push(r.unit() * a);
            out.push(a);
        }
    }
    (fg, bg)
}

fn max_dev(mode: BlendMode, f: impl Fn(f32, f32) -> f32, tiny_alpha: bool) -> f32 {
    let (fg, bg) = rows(4096, 0xC0FFEE ^ (mode as u64).wrapping_mul(31), tiny_alpha);

    let mut got = fg.clone();
    blend_row(&mut got, &bg, mode);

    let mut want = fg.clone();
    for (p, q) in want.chunks_exact_mut(4).zip(bg.chunks_exact(4)) {
        let mut px: [f32; 4] = p.try_into().unwrap();
        let bx: [f32; 4] = q.try_into().unwrap();
        reference_pixel(&mut px, &bx, &f);
        p.copy_from_slice(&px);
    }

    got.iter()
        .zip(want.iter())
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn premul_modes_match_reference_to_rounding() {
    // Channel values live in [0, 1]; 1e-6 is ~8 ULP there, comfortably above
    // reordering noise and far below anything visible at 8 or even 16 bits.
    const BOUND: f32 = 1e-6;
    let cases: Vec<(&str, BlendMode, fn(f32, f32) -> f32)> = vec![
        ("LinearDodge", BlendMode::LinearDodge, |s, d| (s + d).min(1.0)),
        ("LinearBurn", BlendMode::LinearBurn, |s, d| {
            (s + d - 1.0).max(0.0)
        }),
        ("Subtract", BlendMode::Subtract, |s, d| (d - s).max(0.0)),
    ];
    for (name, mode, f) in cases {
        for tiny in [false, true] {
            let dev = max_dev(mode, f, tiny);
            assert!(
                dev <= BOUND,
                "{name} (tiny_alpha={tiny}) deviates {dev:e} from the \
                 unpremultiplying reference, above {BOUND:e}"
            );
            println!("{name:12} tiny_alpha={tiny:<5} max deviation {dev:e}");
        }
    }
}

/// The identities are exact in real arithmetic, so opaque pixels (sa = da = 1,
/// where no division occurs in the reference either) must agree far more
/// tightly than the general bound.
#[test]
fn premul_modes_opaque_pixels_agree_tightly() {
    let mut r = Rng(0xABCDEF);
    let n = 2048;
    let mut fg = Vec::new();
    let mut bg = Vec::new();
    for _ in 0..n {
        for out in [&mut fg, &mut bg] {
            out.push(r.unit());
            out.push(r.unit());
            out.push(r.unit());
            out.push(1.0);
        }
    }
    let cases: Vec<(&str, BlendMode, fn(f32, f32) -> f32)> = vec![
        ("LinearDodge", BlendMode::LinearDodge, |s, d| (s + d).min(1.0)),
        ("LinearBurn", BlendMode::LinearBurn, |s, d| {
            (s + d - 1.0).max(0.0)
        }),
        ("Subtract", BlendMode::Subtract, |s, d| (d - s).max(0.0)),
    ];
    for (name, mode, f) in cases {
        let mut got = fg.clone();
        blend_row(&mut got, &bg, mode);
        let mut want = fg.clone();
        for (p, q) in want.chunks_exact_mut(4).zip(bg.chunks_exact(4)) {
            let mut px: [f32; 4] = p.try_into().unwrap();
            let bx: [f32; 4] = q.try_into().unwrap();
            reference_pixel(&mut px, &bx, &f);
            p.copy_from_slice(&px);
        }
        let dev = got
            .iter()
            .zip(want.iter())
            .map(|(g, w)| (g - w).abs())
            .fold(0.0f32, f32::max);
        assert!(dev <= 1e-7, "{name} opaque deviation {dev:e}");
        println!("{name:12} opaque         max deviation {dev:e}");
    }
}

/// Zero and non-finite alpha must still take the wipe-to-zero path.
#[test]
fn premul_modes_degenerate_alpha() {
    for mode in [
        BlendMode::LinearDodge,
        BlendMode::LinearBurn,
        BlendMode::Subtract,
    ] {
        let mut fg = vec![0.0f32, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0];
        let bg = vec![0.3f32, 0.3, 0.3, 0.0, 0.2, 0.2, 0.2, 1.0];
        blend_row(&mut fg, &bg, mode);
        assert_eq!(
            &fg[..4],
            &[0.0, 0.0, 0.0, 0.0],
            "{mode:?}: both-alpha-zero pixel must be cleared"
        );
        assert!(fg[7] > 0.0, "{mode:?}: opaque pixel must survive");
    }
}

/// Overlay and HardLight moved to division-free PREMULTIPLIED closed forms on
/// 2026-08-01. This checks that algebra against the ORIGINAL unpremultiplying
/// reference, which is the only thing that can catch it being wrong.
///
/// Neither existing test would: `simd_consistency` compares token permutations
/// against each other (all tiers would be consistently wrong), and the other
/// cases here cover different modes. The derivation being validated is
///
///   condition  Cd < 0.5                 <->  bg < 0.5*da
///   branch 1   sa*da*(2*Cs*Cd)          ->   2*fg*bg
///   branch 2   sa*da*(1-2(1-Cs)(1-Cd))  ->   sa*da - 2(sa-fg)(da-bg)
///
/// so an error in either branch, or in the translated comparison, shows up here
/// as a deviation far larger than f32 rounding.
#[test]
fn overlay_hardlight_premul_matches_unpremul_reference() {
    fn reference(fg: &mut [f32], bg: &[f32], f: impl Fn(f32, f32) -> f32) {
        for (s, b) in fg.chunks_exact_mut(4).zip(bg.chunks_exact(4)) {
            let (sa, da) = (s[3], b[3]);
            let out_a = sa + da - sa * da;
            if !out_a.is_finite() || out_a <= 0.0 {
                s.copy_from_slice(&[0.0; 4]);
                continue;
            }
            let inv_sa = if sa > 0.0 { 1.0 / sa } else { 0.0 };
            let inv_da = if da > 0.0 { 1.0 / da } else { 0.0 };
            for i in 0..3 {
                let out = (1.0 - da) * s[i]
                    + (1.0 - sa) * b[i]
                    + sa * da * f(s[i] * inv_sa, b[i] * inv_da);
                s[i] = if out.is_finite() { out } else { 0.0 };
            }
            s[3] = out_a;
        }
    }
    let overlay = |cs: f32, cd: f32| {
        if cd < 0.5 { 2.0 * cs * cd } else { 1.0 - 2.0 * (1.0 - cs) * (1.0 - cd) }
    };
    let hard_light = |cs: f32, cd: f32| {
        if cs < 0.5 { 2.0 * cs * cd } else { 1.0 - 2.0 * (1.0 - cs) * (1.0 - cd) }
    };

    // Sweep alphas AND colours across the Cd = 0.5 / Cs = 0.5 branch boundary,
    // which is where a mistranslated comparison would show.
    let mut s = 0x00A5_5A00u32;
    let n = 4096;
    let mk = |seed: &mut u32| -> Vec<f32> {
        (0..n * 4)
            .map(|i| {
                *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let r = (*seed >> 8) as f32 / 16_777_216.0;
                if i % 4 == 3 { r } else { r }
            })
            .collect()
    };
    for (name, mode, f) in [
        ("Overlay", BlendMode::Overlay, &overlay as &dyn Fn(f32, f32) -> f32),
        ("HardLight", BlendMode::HardLight, &hard_light as &dyn Fn(f32, f32) -> f32),
    ] {
        let fg0 = mk(&mut s);
        let bg = mk(&mut s);
        let mut got = fg0.clone();
        blend_row(&mut got, &bg, mode);
        let mut want = fg0.clone();
        reference(&mut want, &bg, f);
        let dev = got
            .iter()
            .zip(want.iter())
            .map(|(g, w)| (g - w).abs())
            .fold(0.0f32, f32::max);
        println!("{name:12} premul-vs-unpremul max deviation {dev:e}");
        assert!(dev <= 5e-4, "{name}: premultiplied form deviates {dev:e} from the reference");
    }
}
