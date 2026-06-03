//! Benchmark: per-row blend kernel throughput across BlendMode variants.
//!
//! Covers the SIMD-dispatched SrcOver path plus representative scalar
//! Porter-Duff and artistic modes, over a realistic 1920-px-wide RGBA row.
//!
//! Run: cargo bench --bench blend_modes
//!
//! Reports ns/row and Mpix/s for each mode so ARM (Neoverse-N1) vs x86
//! ratios can be compared kernel-by-kernel.

use std::hint::black_box;
use std::time::Instant;

use zenblend::{BlendMode, blend_row, blend_row_solid, lerp_row, mask_row};

const WIDTH: usize = 1920;
const ROW_LEN: usize = WIDTH * 4;

/// Deterministic pseudo-random premultiplied RGBA row.
fn make_row(seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut v = Vec::with_capacity(ROW_LEN);
    for _ in 0..WIDTH {
        // xorshift -> [0,1)
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f32 / (1u64 << 53) as f32
        };
        let a = next().clamp(0.0, 1.0);
        // premultiplied: color <= alpha
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

fn bench_row(label: &str, iters: u64, mut f: impl FnMut()) {
    // warmup
    for _ in 0..(iters / 10).max(1) {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed().as_nanos() as f64;
    let ns_per_row = elapsed / iters as f64;
    let mpix_s = (WIDTH as f64) / ns_per_row * 1000.0;
    println!("{label:<28} {ns_per_row:>9.1} ns/row  {mpix_s:>8.1} Mpix/s");
}

fn main() {
    let iters = 200_000u64;
    let bg = make_row(1);
    let solid: [f32; 4] = [0.4, 0.3, 0.2, 0.8];

    println!("=== blend_row (fg over bg), 1920px RGBA, {iters} iters ===");
    for &(name, mode) in &[
        ("SrcOver (SIMD)", BlendMode::SrcOver),
        ("DstOver", BlendMode::DstOver),
        ("SrcAtop", BlendMode::SrcAtop),
        ("Xor", BlendMode::Xor),
        ("Multiply", BlendMode::Multiply),
        ("Screen", BlendMode::Screen),
        ("Overlay", BlendMode::Overlay),
        ("Darken", BlendMode::Darken),
        ("Lighten", BlendMode::Lighten),
        ("Exclusion", BlendMode::Exclusion),
        ("HardLight", BlendMode::HardLight),
        ("SoftLight", BlendMode::SoftLight),
        ("ColorDodge", BlendMode::ColorDodge),
        ("ColorBurn", BlendMode::ColorBurn),
        ("Difference", BlendMode::Difference),
        ("VividLight", BlendMode::VividLight),
        ("Plus", BlendMode::Plus),
    ] {
        let mut fg = make_row(2);
        let fg0 = fg.clone();
        bench_row(name, iters, || {
            fg.copy_from_slice(&fg0);
            blend_row(black_box(&mut fg), black_box(&bg), mode);
            black_box(&fg);
        });
    }

    println!("\n=== blend_row_solid (fg over solid pixel), 1920px ===");
    for &(name, mode) in &[
        ("SrcOver (SIMD)", BlendMode::SrcOver),
        ("Multiply", BlendMode::Multiply),
        ("Overlay", BlendMode::Overlay),
    ] {
        let mut fg = make_row(3);
        let fg0 = fg.clone();
        bench_row(name, iters, || {
            fg.copy_from_slice(&fg0);
            blend_row_solid(black_box(&mut fg), black_box(&solid), mode);
            black_box(&fg);
        });
    }

    println!("\n=== mask_row / lerp_row (SIMD), 1920px ===");
    {
        let mask: Vec<f32> = (0..WIDTH).map(|i| (i % 100) as f32 / 100.0).collect();
        let mut fg = make_row(4);
        let fg0 = fg.clone();
        bench_row("mask_row", iters, || {
            fg.copy_from_slice(&fg0);
            mask_row(black_box(&mut fg), black_box(&mask));
            black_box(&fg);
        });
    }
    {
        let a = make_row(5);
        let b = make_row(6);
        let t: Vec<f32> = (0..WIDTH).map(|i| (i % 100) as f32 / 100.0).collect();
        let mut out = vec![0.0f32; ROW_LEN];
        bench_row("lerp_row", iters, || {
            lerp_row(
                black_box(&a),
                black_box(&b),
                black_box(&t),
                black_box(&mut out),
            );
            black_box(&out);
        });
    }
}
