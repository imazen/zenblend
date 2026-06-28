# zenblend benchmarks — methodology & reproduction

How to run zenblend's benches fairly, and how to read the committed result files.
These are **internal kernel-throughput benches** (zenblend vs its own scalar/SIMD
paths and across architectures) — not cross-library comparisons, so there are no
competitor crates to pin.

## What the benches measure

| Bench | What it measures |
|-------|------------------|
| [`blend_modes`](../benches/blend_modes.rs) | Per-`BlendMode` blend-row throughput (ns/row and Mpix/s) on a 1920-px RGBA row, plus `blend_row_solid`, `mask_row`, and `lerp_row`. Lets you compare ARM (Neoverse-N1) vs x86 kernel-by-kernel. |
| [`mask_spans`](../benches/mask_spans.rs) | Span-based masking (`apply_mask_spans`) throughput vs the naive per-pixel path. |

Both are `harness = false` binaries run via `cargo bench`.

## Fairness guarantees

- **No I/O in the timed region.** Each input row is synthesized into a `Vec<f32>`
  (deterministic xorshift, premultiplied so `color ≤ alpha`) **before** timing
  starts. The timed loop only calls the blend/mask kernel. Inputs and outputs are
  fed through `std::hint::black_box` so nothing is optimized away.
- **Single-thread.** zenblend has no internal threading — every kernel runs on the
  calling thread. The benches are single-thread end to end; do not compare these
  numbers against a thread-pooled compositor.
- **No `-C target-cpu=native`.** Build with default `RUSTFLAGS` — runtime SIMD
  dispatch (archmage `incant!`) is what ships. On AArch64 the committed baseline
  confirms runtime-dispatch == `target-cpu=neoverse-n1` to within 1% (NEON is
  baseline aarch64), so **shipping numbers == measured numbers**.
- **Warmup before timing.** `blend_modes` runs `iters / 10` warmup iterations per
  kernel before the measured `iters` (200k) to settle the frequency/cache state.

## Reproduce

```sh
git clone https://github.com/imazen/zenblend && cd zenblend
git checkout <commit>          # the commit named in the result file you're reproducing

cargo bench --bench blend_modes   # per-mode ns/row + Mpix/s
cargo bench --bench mask_spans    # span vs per-pixel masking
```

There are **no competitor crates** — zenblend is the only thing under test, so
nothing external needs pinning. Re-run on each target architecture you care about
(x86-64 and AArch64 give very different artistic-mode ratios; see below).

## Result files

Each committed run lands as `benchmarks/<topic>_<YYYY-MM-DD>.md` and **must** state,
in its header: the git commit (or crate version), the CPU/box, RAM, `rustc -V`, the
exact command, and the threading mode. Current files:

- [`arm_neoverse_n1_baseline_2026-05-29.md`](arm_neoverse_n1_baseline_2026-05-29.md)
  — Ampere Altra **Neoverse-N1** (Hetzner CAX21) vs AMD Ryzen 9 7950X, `blend_modes`
  at 1920 px. Documents the division-free premultiplied closed forms that brought
  six separable artistic modes (Multiply, Screen, Darken, Lighten, Difference,
  Exclusion) to NEON/WASM SIMD: **+60–102%** on N1 (e.g. Screen 108→217 Mpix/s,
  Exclusion 103→207 Mpix/s), bit-faithful to the scalar reference within 1e-6.

Do not commit numbers you didn't generate, and don't extrapolate one size or one
architecture to another — measure each. Memory claims need heaptrack / `time -v`,
not estimates.

## Charts (what to plot for which decision)

| Question | Chart |
|----------|-------|
| "Which mode / tier is fastest?" | horizontal **bar**, sorted by throughput (Mpix/s); one bar per mode, separate series per architecture |
| "How does throughput scale with row width?" | **line**, x = pixels (log); fit `total = α + β·pixels` and report both the fixed per-call overhead and the per-pixel slope |
| "Is an A/B kernel change real / how noisy?" | **violin** or PDF of per-call times, or a paired 95% CI |

For new comparison charts (and any future cross-library comparison), prefer
[zenbench](https://github.com/imazen/zenbench) — it interleaves A/B runs to cancel
thermal/scheduler drift and emits a sorted throughput **bar chart**, a self-contained
**SVG** report (`--format=html`), and violin/PDF/regression plots. The current
`blend_modes` / `mask_spans` binaries predate that and use raw `Instant` timing;
porting them to zenbench is the way to get publishable charts with paired stats.
Avoid pie/3D/dual-axis plots.
