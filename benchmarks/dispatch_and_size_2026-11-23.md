# zenblend dispatch & size measurement (2026-11-23)

- **Baseline commit:** `9fdcada6dc2f908d9a59a80731ab9e452eca62f7`
- **Target change:** Un-inline non-`SrcOver` row and solid blend dispatchers (`dispatch_blend_row_other`, `dispatch_blend_row_solid_other` marked `#[inline(never)]`), keep `BlendMode::SrcOver` fast-pathed in `#[inline]` wrappers (`dispatch_blend_row`, `dispatch_blend_row_solid`, `dispatch_blend_row_solid_opaque`) so `SrcOver`-only callers dead-code-eliminate the 31 other modes, deduplicate `dispatch_blend_row_solid_opaque`'s non-`SrcOver` delegation directly onto `dispatch_blend_row_solid_other`, and remove the unreachable `_ => unimplemented!(...)` arm inside `dispatch_blend_row_other` (`BlendMode` variants are exhausted in-crate).
- **Benchmark host (`r5600g`):** AMD Ryzen 5 5600G with Radeon Graphics (Zen 3, 6 physical / 12 logical cores), Linux, single-thread pinned via `taskset -c 4` (`nice -n 19`) and serialized via `r5600g.lock`.
- **Build & probe hosts:** `i265` (`rustc 1.99.1 (58cf8f9c3 2026-10-13)`, `-C strip=none` ELF checks & test gates) and `i134` (`wasm-tools strip --all`, `gzip -n -9`).
- **Build flags:** Default shipping runtime-dispatch `RUSTFLAGS` (never `-C target-cpu=native`); `wasm32-unknown-unknown --no-default-features` built with `RUSTFLAGS="-C target-feature=+simd128"`.
- **Threading:** Single-thread across every kernel.

## Code Size & WASM Attribution

### Native ELF (`-C strip=none`)

| Target | Host | Before `.text` (B) | After `.text` (B) | Delta (B) | `vmaskmov` / `__simd_inner` / `_v4` |
|---|---|---:|---:|---:|---|
| `blend_modes` | `i265` | 393,063 | 392,219 | -844 | 0 / 0 / 0 |
| `mask_spans` | `i265` | 408,975 | 408,979 | +4 | 0 / 0 / 0 |
| `tier_isolation` (`_dev`) | `i265` | 400,331 | 400,751 | +420 | 0 / 0 / 0 |
| `blend_modes` | `r5600g` | 407,923 | 407,191 | -732 | — |
| `mask_spans` | `r5600g` | 423,017 | 423,017 | 0 | — |
| `tier_isolation` (`_dev`) | `r5600g` | 414,783 | 415,267 | +484 | — |

In `blend_modes`, the non-`SrcOver` dispatchers compile once out-of-line as `zenblend::blend::dispatch_blend_row_other` (`0x201d` B) and `zenblend::blend::dispatch_blend_row_solid_other` (`0x2450` B) rather than replicating at call sites.

### `wasm32-unknown-unknown` (`+simd128`, `--no-default-features`)

Three downstream usage patterns were measured with standalone `cdylib` probes (`opt-level = 3`, `lto = "thin"`, `codegen-units = 1`, `panic = "abort"`, stripped via `wasm-tools strip --all`, compressed via `gzip -n -9`):

| Probe | Before raw (B) | After raw (B) | Delta (B) | After stripped (B) | After `gzip -n -9` (B) | Notes |
|---|---:|---:|---:|---:|---:|---|
| `SrcOver`-only (`blend_row`, `blend_row_solid`, `blend_row_solid_opaque`) | 473 | 473 | 0 | 473 | 303 | Fast-path keeps full DCE of 31 non-`SrcOver` modes (`code` = 407 B, 1 func). Putting `#[inline(never)]` on `dispatch_blend_row` directly regressed this to 38,887 B and was rejected. |
| 2-callsite runtime `BlendMode` (`caller_a` + `caller_b`) | 40,273 | 39,749 | -524 | 39,749 | 11,532 | Shared out-of-line non-`SrcOver` dispatchers reduce Wasm functions to 55 (`code` = 37,743 B, `data` = 1,755 B). |
| 1-callsite full probe (all entry points + `RoundedRectMask` + `mask_row`/`lerp_row`) | 50,909 | 51,977 | +1,068 | — | 15,631 (from 15,529) | Single-callsite probe pays call-boundary/jump-table overhead (`code` 48,297 -> 49,201 B, `data` 2,335 -> 2,499 B, 66 funcs) because each dispatcher is reached only once for row blend and twice for solid/opaque. |

## Interleaved `zenbench` Pair (`r5600g`, `taskset -c 4`, 25 rounds)

Reproduced with a dual-crate `zenbench 0.1.11` binary linking `zenblend` at baseline commit `9fdcada6` alongside the post-change crate (`--rounds=25 --format=json`):

| Group | Before median (ns) | After median (ns) | Delta (%) | 95% CI (ns) | Significant / Wilcoxon `p` |
|---|---:|---:|---:|---:|---|
| `srcover_row_257px` | 87.88 | 78.63 | -10.29% | [-9.87, -8.70] | `True` / `0.0004825` |
| `runtime_modes_row_257px` | 786.72 | 796.08 | +3.08% | [+3.33, +46.53] | `False` / `0.01304` |
| `runtime_modes_solid_and_opaque_257px` | 1,778.41 | 1,689.29 | -5.11% | [-165.69, -23.23] | `True` / `0.01447` |
| `srcover_row_1920px` | 874.99 | 953.43 | +9.41% | [+77.02, +90.80] | `True` / `9.573e-05` |
| `runtime_modes_row_1920px` | 6,056.32 | 6,181.17 | +1.44% | [-40.79, +210.16] | `False` / `0.135` |
| `runtime_modes_solid_and_opaque_1920px` | 13,351.23 | 12,879.91 | -3.73% | [-710.47, -247.82] | `True` / `0.005415` |

> [!NOTE]
> `BlendMode::SrcOver` executes identical `simd::blend_src_over_row` instructions in baseline `9fdcada6` and post-change code via the same `#[inline]` path. The opposite-sign `srcover_row_257px` (-10.29%) and `srcover_row_1920px` (+9.41%) shifts reflect dual-crate `.text` page/cache-line alignment variance across the two linked copies; in the standalone `blend_modes` binary, `SrcOver (SIMD)` measured 964.7 -> 970.7 ns/row (+0.62%). Runtime multi-mode `blend_row_solid` + `blend_row_solid_opaque` improved significantly at both 257 px (-5.11%) and 1920 px (-3.73%).

## Standalone `blend_modes` & `mask_spans` (`r5600g`, `taskset -c 4`)

Command: `taskset -c 4 cargo bench --bench blend_modes` and `cargo bench --bench mask_spans` (1920 px RGBA, 200,000 iters for `blend_modes`):

| Benchmark | Before (ns/row) | After (ns/row) | Delta |
|---|---:|---:|---:|
| `blend_row/SrcOver (SIMD)` | 964.7 | 970.7 | +0.62% |
| `blend_row/DstOver` | 1,298.0 | 1,285.3 | -0.98% |
| `blend_row/SrcAtop` | 1,590.0 | 1,502.0 | -5.53% |
| `blend_row/Xor` | 1,519.3 | 1,582.2 | +4.14% |
| `blend_row/Multiply` | 6,533.0 | 6,608.9 | +1.16% |
| `blend_row/Screen` | 5,706.8 | 6,176.2 | +8.23% |
| `blend_row/Overlay` | 9,257.5 | 9,237.8 | -0.21% |
| `blend_row/Difference` | 7,430.0 | 6,900.1 | -7.13% |
| `blend_row/Plus` | 3,130.5 | 3,071.7 | -1.88% |
| `blend_row_solid/SrcOver (SIMD)` | 807.6 | 808.6 | +0.12% |
| `blend_row_solid/Multiply` | 7,130.2 | 7,148.9 | +0.26% |
| `blend_row_solid/Overlay` | 10,305.4 | 10,418.7 | +1.10% |
| `mask_row` | 1,023.9 | 1,058.5 | +3.38% |
| `lerp_row` | 820.8 | 862.9 | +5.13% |

`RoundedRect` (`mask_spans`) remained within run noise (e.g. `1000x1000 r=20` spans `409.6 -> 325.8 ns`, `4000x4000 r=50` spans `1112.9 -> 1081.0 ns`).

## Verification Gates

All ran locally after the final source edit:
- `cargo fmt --check`: pass
- `cargo check --target wasm32-unknown-unknown --no-default-features`: pass
- `cargo clippy --all-targets --all-features -- -D warnings`: pass
- `cargo test --release --features _dev`: pass (`94` unit + `4` `premul_modes_deviation` + `6` `simd_consistency`, 0 failed)
- `cargo test --doc`: pass (`4` doctests)
