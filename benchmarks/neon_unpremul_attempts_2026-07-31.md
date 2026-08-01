# zenblend: three attempts to hand-vectorize the unpremultiplying blend modes

Apple M4 Pro, aarch64. `cargo bench --features _dev --bench tier_isolation`.
Overlay used as the representative branchy mode; the other 17 unpremultiplying
modes share the same driver shape.

**Result: all three lost to the shipped autovectorized scalar path. Nothing was
shipped.** This file exists so the fourth session doesn't retry them.

## Why this kept getting attempted

The 18 modes that need straight (unpremultiplied) colour run through a portable
driver, and a "no hand-written NEON here" reading of the source makes them look
like an obvious gap. They are not a gap: on aarch64 NEON is BASELINE, so the
portable driver is already vectorized by LLVM. The tier bench shows the shipped
Overlay at **1.00x neon-vs-forced-scalar** — both arms compile to equivalent
work, which is what "LLVM already did it" looks like.

## The attempts

| # | decomposition | measured |
|---|---|---|
| 1 | one RGBA pixel per `f32x4` | **-15%** |
| 2 | planar (4 px/vector) via scalar gather/scatter | **7715 vs 6088 ns/row (-27%)** |
| 3 | planar via `f32x4::deinterleave_4ch` (vector transpose) | **-2% to -6%** |

Attempt 1 wasted 3 of 4 lanes and had no cross-pixel parallelism, so the branchy
mode bodies became 1-pixel selects.

Attempt 2 fixed the lane utilisation — and got *worse*. The arithmetic was
genuinely faster (its own neon-vs-scalar ratio was 1.09x) but the hand-rolled
16-load/16-store transpose cost more than the arithmetic saved. This is the
useful datum: the transpose, not the blend math, is the whole budget.

Attempt 3 replaced that transpose with the real vector primitive
(`magetypes`' `deinterleave_4ch`, i.e. `transpose_4x4_copy`). That recovered
almost all of it — from -27% to roughly parity — but never got *ahead*, across
three consecutive runs:

```
run 1   Overlay 4359.8 ns/row   Overlay[planar] 4449.1   (-2.0%)
run 2   Overlay 4272.5 ns/row   Overlay[planar] 4514.3   (-5.7%)
run 3   Overlay 4342.5 ns/row   Overlay[planar] 4473.8   (-3.0%)
```

Attempt 3's planar arm also measured **1.00x** neon-vs-forced-scalar — LLVM
autovectorizes the planar form identically with or without the token, the same
way it already handles the shipped driver.

## Measurement note

Absolute timings on this host drift substantially between runs — the *same*
shipped Overlay kernel measured 6088 ns/row in one run and 4272 in another.
Only in-run paired comparisons are trustworthy; every number above compares two
arms measured in the same run. A cross-run comparison here would have "shown" a
30% win that does not exist.

## Conclusion

Three structurally different decompositions, none faster. The blend arithmetic
is not the bottleneck and the interleave/deinterleave overhead is not
recoverable beyond what LLVM already extracts. Shipping ~200 lines of
hand-written driver for parity-or-worse is a net loss in maintenance for no
speed, so the modes keep the portable path.

If a fourth attempt happens, it needs a *different bottleneck*, not a different
decomposition — e.g. avoiding the unpremultiply/repremultiply round trip
entirely by keeping a straight-alpha buffer across a chain of blends. That is an
API-level change, not a kernel-level one, and would need approval.
