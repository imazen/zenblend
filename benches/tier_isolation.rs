//! SIMD-tier isolation: the native top tier vs the same code forced to scalar.
//!
//! `blend_modes.rs` reports per-mode throughput, which tells you which modes
//! are fast but not whether the SIMD dispatch is earning its keep — a kernel
//! slower than its own scalar fallback is invisible there. This bench runs the
//! identical `blend_row` / `mask_row` / `lerp_row` calls with the native SIMD
//! token disabled. (The same gap in linear-srgb was hiding a real regression.)
//!
//! Run: `cargo bench --bench tier_isolation --features _dev`
//! Do NOT build with `-C target-cpu=native`: that pins the tier at compile
//! time, after which it cannot be disabled and this bench skips rather than
//! silently reporting the SIMD path under both labels.

use std::hint::black_box;
use std::time::Instant;

use zenblend::{BlendMode, blend_row, lerp_row, mask_row};

const WIDTH: usize = 1920;
const ROW_LEN: usize = WIDTH * 4;

#[cfg(target_arch = "aarch64")]
type TierToken = archmage::NeonToken;
#[cfg(target_arch = "x86_64")]
type TierToken = archmage::X64V3Token;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const TIER_NAME: &str = if cfg!(target_arch = "aarch64") {
    "neon"
} else {
    "v3(avx2)"
};

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn set_simd(enabled: bool) -> bool {
    use archmage::SimdToken;
    TierToken::dangerously_disable_token_process_wide(!enabled).is_ok()
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn set_simd(_enabled: bool) -> bool {
    false
}

fn make_row(seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut v = Vec::with_capacity(ROW_LEN);
    for _ in 0..WIDTH {
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f32 / (1u64 << 53) as f32
        };
        let a = next().clamp(0.0, 1.0);
        let r = next() * a;
        let g = next() * a;
        let b = next() * a;
        v.push(r);
        v.push(g);
        v.push(b);
        v.push(a);
    }
    v
}

/// Time `f` with SIMD on, then with SIMD off, and report the ratio.
fn ab(label: &str, iters: u64, mut f: impl FnMut()) {
    let mut run = |simd: bool| -> f64 {
        set_simd(simd);
        for _ in 0..(iters / 10).max(1) {
            f();
        }
        let start = Instant::now();
        for _ in 0..iters {
            f();
        }
        start.elapsed().as_nanos() as f64 / iters as f64
    };
    let simd_ns = run(true);
    let scalar_ns = run(false);
    set_simd(true);
    let speedup = scalar_ns / simd_ns;
    let flag = if speedup < 1.0 { "  <-- SLOWER" } else { "" };
    println!(
        "{label:<24} {TIER_NAME:>10}: {simd_ns:>8.1} ns/row   scalar: {scalar_ns:>8.1} ns/row   \
         {speedup:>5.2}x{flag}"
    );
}

fn main() {
    if !set_simd(true) || !set_simd(false) {
        eprintln!(
            "[tier_isolation] no toggleable SIMD tier on this target, or the tier is \
             compile-time guaranteed (drop -C target-cpu=native, build with --features _dev). \
             Skipping."
        );
        return;
    }
    set_simd(true);

    let iters = 100_000u64;
    let bg = make_row(1);
    let fg = make_row(2);
    let mask: Vec<f32> = (0..WIDTH).map(|i| (i % 61) as f32 / 60.0).collect();

    println!("=== zenblend SIMD tier isolation, {WIDTH}px RGBA rows, {iters} iters ===");

    // The premultiplied Overlay against the ORIGINAL unpremultiplying one, in
    // the SAME process. The tier arm above compares premul-vector against
    // premul-1-lane, which does not answer "is this faster than what shipped",
    // and cross-run absolutes drift up to 2x on this host so a before/after
    // from two separate runs proves nothing.
    {
        set_simd(true);
        let mut d = bg.clone();
        let time = |label: &str, mut f: Box<dyn FnMut(&mut Vec<f32>)>, d: &mut Vec<f32>| -> f64 {
            for _ in 0..(iters / 10).max(1) { d.copy_from_slice(&bg); f(d); }
            let start = Instant::now();
            for _ in 0..iters { d.copy_from_slice(&bg); f(d); }
            let ns = start.elapsed().as_nanos() as f64 / iters as f64;
            println!("{label:<32} {ns:>8.1} ns/row");
            ns
        };
        let fg1 = fg.clone();
        let new_ns = time("Overlay premul (new)",
            Box::new(move |x: &mut Vec<f32>| blend_row(black_box(x), black_box(&fg1), BlendMode::Overlay)), &mut d);
        let fg2 = fg.clone();
        let old_ns = time("Overlay unpremul (original)",
            Box::new(move |x: &mut Vec<f32>| zenblend::__bench_overlay_unpremul(black_box(x), black_box(&fg2))), &mut d);
        println!("Overlay premul-vs-original       {:>5.2}x", old_ns / new_ns);

        for (label, mode, orig) in [
            ("LinearLight", BlendMode::LinearLight,
             zenblend::__bench_linear_light_unpremul as fn(&mut [f32], &[f32])),
            ("PinLight", BlendMode::PinLight,
             zenblend::__bench_pin_light_unpremul as fn(&mut [f32], &[f32])),
            ("ColorDodge", BlendMode::ColorDodge,
             zenblend::__bench_color_dodge_unpremul as fn(&mut [f32], &[f32])),
            ("Divide", BlendMode::Divide,
             zenblend::__bench_divide_unpremul as fn(&mut [f32], &[f32])),
            ("ColorBurn", BlendMode::ColorBurn,
             zenblend::__bench_color_burn_unpremul as fn(&mut [f32], &[f32])),
            ("VividLight", BlendMode::VividLight,
             zenblend::__bench_vivid_light_unpremul as fn(&mut [f32], &[f32])),
            ("HardMix", BlendMode::HardMix,
             zenblend::__bench_hard_mix_unpremul as fn(&mut [f32], &[f32])),
            ("SoftLight", BlendMode::SoftLight,
             zenblend::__bench_soft_light_unpremul as fn(&mut [f32], &[f32])),
        ] {
            let f1 = fg.clone();
            let n_ns = time(&format!("{label} premul (new)"),
                Box::new(move |x: &mut Vec<f32>| blend_row(black_box(x), black_box(&f1), mode)), &mut d);
            let f2 = fg.clone();
            let o_ns = time(&format!("{label} unpremul (original)"),
                Box::new(move |x: &mut Vec<f32>| orig(black_box(x), black_box(&f2))), &mut d);
            println!("{label} premul-vs-original   {:>5.2}x", o_ns / n_ns);
        }
        println!();
    }

    // SrcOver is the SIMD-dispatched compositing path.
    let mut dst = bg.clone();
    ab("blend_row/SrcOver", iters, || {
        dst.copy_from_slice(&bg);
        blend_row(
            black_box(&mut dst),
            black_box(&fg),
            black_box(BlendMode::SrcOver),
        );
    });

    let mut dst2 = bg.clone();
    // t and mask are PER-PIXEL (WIDTH), not per-component (ROW_LEN).
    let t_row: Vec<f32> = (0..WIDTH).map(|i| (i % 97) as f32 / 96.0).collect();
    let mut out2 = vec![0.0f32; ROW_LEN];
    ab("lerp_row", iters, || {
        dst2.copy_from_slice(&bg);
        lerp_row(
            black_box(&dst2),
            black_box(&fg),
            black_box(&t_row),
            black_box(&mut out2),
        );
    });

    let mut dst3 = bg.clone();
    ab("mask_row", iters, || {
        dst3.copy_from_slice(&bg);
        mask_row(black_box(&mut dst3), black_box(&mask));
    });

    // Every blend mode. The bench previously covered SrcOver and Multiply
    // only — 2 of 32 — so a mode whose SIMD path loses to its own scalar
    // fallback was invisible for the other 30. That is exactly the gap that
    // was hiding real regressions in zenresize and zenpng.
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

    let mut dst4 = bg.clone();
    for &(name, mode) in ALL_MODES {
        ab(&format!("blend_row/{name}"), iters, || {
            dst4.copy_from_slice(&bg);
            blend_row(black_box(&mut dst4), black_box(&fg), black_box(mode));
        });
    }
}
