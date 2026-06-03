# zenblend — ARM Neoverse-N1 perf baseline + SIMD artistic-mode win

**Date:** 2026-05-31
**Box:** `arm-zen` — Hetzner CAX21, Ampere Altra **Neoverse-N1**, 4c/8 GB, rustc 1.96
**x86 cross-check:** AMD Ryzen 9 7950X (runtime dispatch, no `target-cpu=native`)
**Bench:** `benches/blend_modes.rs` — 1920-px RGBA rows, 200k iters, per-mode ns/row + Mpix/s

This is the 3rd crate in the ARM-perf stream (after `linear-srgb`, `zenresize`).

## What was found

`zenblend`'s `SrcOver` path is 4-wide NEON (magetypes `f32x4`, 1 pixel/iter; x86
has a 2-pixel `f32x8` kernel). **All other Porter-Duff and artistic modes were
pure scalar** — and the artistic modes (Multiply, Screen, …) ran at 100–125
Mpix/s on N1 vs `SrcOver`'s ~635 Mpix/s, a ~5× gap, because each pixel did:

- two scalar reciprocals (`1/sa`, `1/da`) for the unpremultiply — N1's FDIV is
  non-pipelined (~10-cycle latency, no overlap), the dominant cost;
- a per-channel scalar loop with branches;
- an `[f32;4]` round-trip per pixel.

**f16-pathology check (the zenresize finding): ABSENT.** zenblend uses f32 RGBA
throughout — no f16 vector type, no f16↔f32 convert. The magetypes
scalar-fallback that cost zenresize 21× does not apply here.

## The win: division-free premultiplied closed forms

Six separable artistic modes algebraically cancel the unpremultiply reciprocals
when the `sa·da·f(Cs,Cd)` composite term is expanded in premultiplied space
(`Cs = fg/sa`, `Cd = bg/da`):

| Mode | Closed form (premultiplied) |
|---|---|
| Multiply | `(1−da)·fg + (1−sa)·bg + fg·bg` |
| Screen | `fg + bg − fg·bg` |
| Exclusion | `fg + bg − 2·fg·bg` |
| Darken | `(1−da)·fg + (1−sa)·bg + min(da·fg, sa·bg)` |
| Lighten | `(1−da)·fg + (1−sa)·bg + max(da·fg, sa·bg)` |
| Difference | `(1−da)·fg + (1−sa)·bg + \|da·fg − sa·bg\|` |

These become pure `f32x4` mul/add/min/max + a vectorized non-finite sanitize
(`abs(x) < +inf` masks both NaN and ±inf in one select) and a direct `store`.
No per-pixel division, no scalar round-trip. They match the scalar
divide-then-remultiply reference to within **1e-6** (13 random + edge-case
equivalence tests in `blend.rs`; the existing 94 per-mode ground-truth tests +
5 cross-tier consistency tests pass on the actual NEON path).

Modes that did NOT pay off and stay scalar (measured, not assumed):
- **Overlay/HardLight/LinearLight/PinLight** — need a both-branches lane select
  (`f32x4::blend`), which on N1 computes both arms + the select per pixel and
  **lost to** the scalar single-branch form (Overlay regressed 124→70 Mpix/s in
  a trial — reverted).
- **LinearBurn/LinearDodge/Subtract** — straight-color clamp doesn't cancel the
  reciprocals; the unpremultiply-SIMD path gave only marginal gains.
- **ColorDodge/ColorBurn/Divide/VividLight/HardMix/SoftLight** — per-lane
  division / sqrt; not addressed this pass.

## Numbers (blend_row, 1920px, Mpix/s — higher is better)

| Mode | x86 7950X | ARM N1 baseline | ARM N1 after | ARM Δ |
|---|--:|--:|--:|--:|
| **Screen** | 295.2 | 108.2 | **216.7** | **+100%** |
| **Exclusion** | — | 102.9 | **207.4** | **+102%** |
| **Multiply** | 322.9 | 119.8 | **191.8** | **+60%** |
| **Difference** | 290.5 | 101.0 | **179.9** | **+78%** |
| **Darken** | 325.4 | 112.0 | **179.8** | **+61%** |
| **Lighten** | — | 111.9 | **179.8** | **+61%** |
| SrcOver (SIMD, unchanged) | 2378 | 646 | 646 | — |
| Overlay (scalar, unchanged) | 352.9 | 123.9 | 123.9 | — |
| HardLight (scalar, unchanged) | 355.7 | 121.9 | 121.9 | — |
| DstOver (scalar, unchanged) | 857.8 | 442.8 | 442.8 | — |

Runtime-dispatch (shipping, `RUSTFLAGS` unset) == `target-cpu=neoverse-n1` to
within 1% noise for every kernel — NEON is baseline aarch64, so the N1 ISA flags
add nothing for f32 blends. **Shipping numbers == measured numbers.**

## Correctness

`#![forbid(unsafe_code)]` maintained. NEON path: 94 unit + 13 SIMD-equivalence +
5 cross-tier + 4 doctests all pass. Output is bit-faithful to the scalar
reference within the crate's 1e-6 tolerance, including alpha=0 / alpha=1 /
saturated / NaN-injected edge cases. The closed forms are strictly *more* robust
at `sa→0` (no `0/0 → NaN`).

## Next hypotheses (not done this pass)

1. 2-pixel `f32x8` artistic kernels for x86 (would mirror the SrcOver x86 path;
   x86 currently falls back to scalar for these modes — still ~3× faster than N1
   there, but a 2-pixel kernel could roughly double x86 artistic throughput).
2. The remaining per-pixel scalar `out_a` + guard is now the floor (~180–215
   Mpix/s); a 2-pixel NEON kernel processing two RGBA pixels per 128-bit pair of
   loads could amortize it further on N1.
