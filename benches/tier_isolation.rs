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

    // A representative non-SrcOver mode: if these dispatch to SIMD too, the
    // ratio shows it; if they are scalar-only, the ratio is ~1.00x and that is
    // the useful fact.
    let mut dst4 = bg.clone();
    ab("blend_row/Multiply", iters, || {
        dst4.copy_from_slice(&bg);
        blend_row(
            black_box(&mut dst4),
            black_box(&fg),
            black_box(BlendMode::Multiply),
        );
    });
}
