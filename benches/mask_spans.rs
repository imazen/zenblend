//! Benchmark: mask_row (naive full-row) vs apply_mask_spans (span-based).
//!
//! Run: cargo bench --bench mask_spans

use std::hint::black_box;
use std::time::Instant;

use zenblend::mask::{MaskFill, MaskSource, RoundedRectMask};

fn main() {
    for &width in &[200, 1000, 4000] {
        for &radius in &[20.0, 50.0] {
            bench_mask(width, radius);
        }
    }
}

fn bench_mask(width: u32, radius: f32) {
    let height = width; // square
    let mask = RoundedRectMask::uniform(width, height, radius);
    let row_len = width as usize * 4;
    let iters = 500_000u64 / (width as u64).max(1);

    // --- Naive: fill_mask_row + mask_row on full row ---
    let mut pixels = vec![0.5f32; row_len];
    let mut mask_buf = vec![0.0f32; width as usize];
    let start = Instant::now();
    for _ in 0..iters {
        // Use a corner row (row 5) that has partial content
        for y in [5, height / 2, height - 6] {
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
        for y in [5, height / 2, height - 6] {
            pixels.fill(0.5);
            zenblend::apply_mask_spans(
                &mut pixels,
                &mut mask_buf,
                &mask,
                y,
            );
            black_box(&pixels);
        }
    }
    let spans_ns = start.elapsed().as_nanos() as f64 / (iters as f64 * 3.0);

    let speedup = naive_ns / spans_ns;
    println!(
        "{width:>5}×{height:<5} r={radius:<5} naive={naive_ns:>8.1}ns  spans={spans_ns:>8.1}ns  {speedup:.2}×",
    );
}
