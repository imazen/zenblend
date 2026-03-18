//! Benchmark: mask_row (naive full-row) vs apply_mask_spans (span-based).
//!
//! Run: cargo bench --bench mask_spans

use std::hint::black_box;
use std::time::Instant;

use zenblend::mask::{
    LinearGradientMask, MaskFill, MaskSource, RadialGradientMask, RoundedRectMask,
};

fn main() {
    println!("=== RoundedRect (analytical spans) ===");
    for &width in &[200, 1000, 4000] {
        for &radius in &[20.0, 50.0] {
            bench_mask(&RoundedRectMask::uniform(width, width, radius), width, width,
                &format!("r={radius:<5}"));
        }
    }

    println!("\n=== LinearGradient (default spans = full-row passthrough) ===");
    for &width in &[200, 1000, 4000] {
        let mask = LinearGradientMask::new(width, width, (0.0, 0.0), (0.0, width as f32));
        bench_mask(&mask, width, width, "vert  ");
    }

    println!("\n=== RadialGradient (default spans = full-row passthrough) ===");
    for &width in &[200, 1000, 4000] {
        let c = width as f32 / 2.0;
        let mask = RadialGradientMask::new(width, width, (c, c), c * 0.3, c * 0.8);
        bench_mask(&mask, width, width, "spot  ");
    }
}

fn bench_mask(mask: &dyn MaskSource, width: u32, height: u32, label: &str) {
    let row_len = width as usize * 4;
    let iters = 500_000u64 / (width as u64).max(1);

    let mut pixels = vec![0.5f32; row_len];
    let mut mask_buf = vec![0.0f32; width as usize];

    // Rows: one near top corner, one center, one near bottom corner
    let rows = [5.min(height - 1), height / 2, (height - 6).max(0)];

    // --- Naive: fill_mask_row + mask_row on full row ---
    let start = Instant::now();
    for _ in 0..iters {
        for y in rows {
            let fill = mask.fill_mask_row(&mut mask_buf, y);
            pixels.fill(0.5);
            match fill {
                MaskFill::AllOpaque => {}
                MaskFill::AllTransparent => pixels.fill(0.0),
                MaskFill::Partial => zenblend::mask_row(&mut pixels, &mask_buf),
            }
            black_box(&pixels);
        }
    }
    let naive_ns = start.elapsed().as_nanos() as f64 / (iters as f64 * 3.0);

    // --- Span-based: apply_mask_spans ---
    let start = Instant::now();
    for _ in 0..iters {
        for y in rows {
            pixels.fill(0.5);
            zenblend::apply_mask_spans(&mut pixels, &mut mask_buf, mask, y);
            black_box(&pixels);
        }
    }
    let spans_ns = start.elapsed().as_nanos() as f64 / (iters as f64 * 3.0);

    let speedup = naive_ns / spans_ns;
    println!(
        "{width:>5}×{height:<5} {label}  naive={naive_ns:>8.1}ns  spans={spans_ns:>8.1}ns  {speedup:.2}×",
    );
}
