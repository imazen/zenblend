# zenblend

Porter-Duff and artistic blend modes on premultiplied linear f32 RGBA rows. SIMD-accelerated.

## What it does

Row-level pixel blending for image compositing pipelines. All operations work on
**premultiplied linear f32 RGBA** data — 4 interleaved floats per pixel, alpha
pre-applied, in linear light.

Designed for integration with [zenpipe](https://github.com/imazen/zenpipe) and
[zenimage](https://github.com/imazen/zenimage). Not a standalone compositing
engine — it's the inner loop.

## Blend modes

**Porter-Duff (12):** Clear, Src, Dst, SrcOver, DstOver, SrcIn, DstIn, SrcOut,
DstOut, SrcAtop, DstAtop, Xor

**Artistic (21):** Multiply, Screen, Overlay, Darken, Lighten, HardLight, SoftLight,
ColorDodge, ColorBurn, Difference, Exclusion, LinearBurn, LinearDodge, VividLight,
LinearLight, PinLight, HardMix, Divide, Subtract, Plus

33 modes total. Artistic modes unpremultiply per-pixel, apply the blend function,
then re-premultiply. Plus operates directly on premultiplied data.

## Usage

```rust
use zenblend::{blend_row, BlendMode};

// fg and bg are &mut [f32] / &[f32], length divisible by 4
// fg is modified in place
blend_row(fg, bg, BlendMode::SrcOver);

// Blend against a solid color (no buffer needed for background)
blend_row_solid(fg, &[0.2, 0.0, 0.0, 0.5], BlendMode::Multiply);
```

### Masking

```rust
use zenblend::mask::{mask_row, mask_row_constant, RoundedRectMask, MaskSource};

// Per-pixel mask (4-channel, multiplies each channel)
mask_row(pixels, mask_values);

// Uniform opacity
mask_row_constant(pixels, 0.7);

// Span-optimized masking — skips fully opaque/transparent regions
let mask = RoundedRectMask::new(width, height, [10.0, 10.0, 10.0, 10.0]);
apply_mask_spans(pixels, &mut mask_buf, &mask, y);
```

Built-in masks: `RoundedRectMask`, `LinearGradientMask`, `RadialGradientMask`.
Implement `MaskSource` for custom masks.

### Interpolation

```rust
use zenblend::lerp_row;

// Per-pixel blend factor t ∈ [0, 1]
lerp_row(row_a, row_b, t_values, output);
```

## SIMD acceleration

SrcOver (the most common compositing operation) is SIMD-accelerated:

| Platform | ISA | Pixels/iter |
|----------|-----|-------------|
| x86_64 | AVX2 + FMA | 2 |
| AArch64 | NEON | 1 |
| WASM | simd128 | 1 |
| Fallback | scalar | 1 |

Other blend modes use scalar implementations. Masking span alignment
is SIMD-aware (snaps partial spans to block boundaries).

Runtime CPU dispatch via [archmage](https://crates.io/crates/archmage).
`#![forbid(unsafe_code)]` — all SIMD through safe abstractions.

## Features

- `default = ["std"]`
- `no_std + alloc` compatible (disable default features)

## License

Dual-licensed: [AGPL-3.0](LICENSE-AGPL3) or [commercial](LICENSE-COMMERCIAL).

I've maintained and developed open-source image server software — and the 40+
library ecosystem it depends on — full-time since 2011. Fifteen years of
continual maintenance, backwards compatibility, support, and the (very rare)
security patch. That kind of stability requires sustainable funding, and
dual-licensing is how we make it work without venture capital or rug-pulls.
Support sustainable and secure software; swap patch tuesday for patch leap-year.

[Our open-source products](https://www.imazen.io/open-source)

**Your options:**

- **Startup license** — $1 if your company has under $1M revenue and fewer
  than 5 employees. [Get a key →](https://www.imazen.io/pricing)
- **Commercial subscription** — Governed by the Imazen Site-wide Subscription
  License v1.1 or later. Apache 2.0-like terms, no source-sharing requirement.
  Sliding scale by company size.
  [Pricing & 60-day free trial →](https://www.imazen.io/pricing)
- **AGPL v3** — Free and open. Share your source if you distribute.

See [LICENSE-COMMERCIAL](LICENSE-COMMERCIAL) for details.
