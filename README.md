# zenblend [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenblend/ci.yml?style=flat-square)](https://github.com/imazen/zenblend/actions/workflows/ci.yml) [![MSRV](https://img.shields.io/badge/MSRV-1.93-blue?style=flat-square)](https://doc.rust-lang.org/cargo/reference/manifest.html#the-rust-version-field) [![License](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue?style=flat-square)](https://github.com/imazen/zenblend#license)

zenblend is a row-level pixel blending library for premultiplied linear f32 RGBA compositing pipelines.

All operations work on `&mut [f32]` slices -- 4 interleaved floats per pixel, alpha pre-applied, in linear light. Designed as the inner loop for [zenpipe](https://github.com/imazen/zenpipe) strip pipelines, not a standalone compositing engine.

## Blend modes

**Porter-Duff (12):** Clear, Src, Dst, SrcOver, DstOver, SrcIn, DstIn, SrcOut,
DstOut, SrcAtop, DstAtop, Xor

**Artistic (20):** Multiply, Screen, Overlay, Darken, Lighten, HardLight, SoftLight,
ColorDodge, ColorBurn, Difference, Exclusion, LinearBurn, LinearDodge, VividLight,
LinearLight, PinLight, HardMix, Divide, Subtract, Plus

32 modes total. Artistic modes unpremultiply per-pixel, apply the blend function,
then re-premultiply. Plus operates directly on premultiplied data.

## Getting started

### Blending two rows

```rust
use zenblend::{BlendMode, blend_row, blend_row_solid, blend_row_solid_opaque};

// fg and bg are premultiplied linear f32, 4ch RGBA, equal length, divisible by 4.
// fg is modified in place to contain the blended result.
let mut fg = vec![0.5, 0.0, 0.0, 0.5,  0.0, 0.3, 0.0, 1.0];
let bg =     vec![0.0, 0.3, 0.0, 1.0,  0.0, 0.0, 0.5, 0.5];
blend_row(&mut fg, &bg, BlendMode::SrcOver);

// Blend against a solid color -- no row buffer needed for background.
let mut row = vec![0.5, 0.0, 0.0, 0.5,  0.0, 0.3, 0.0, 1.0];
blend_row_solid(&mut row, &[0.2, 0.0, 0.0, 0.5], BlendMode::Multiply);

// Optimized path when the background is opaque (alpha = 1.0).
let mut row2 = vec![0.5, 0.0, 0.0, 0.5,  0.0, 0.3, 0.0, 1.0];
blend_row_solid_opaque(&mut row2, &[0.2, 0.1, 0.05, 1.0], BlendMode::SrcOver);
```

### Masking

```rust
use zenblend::mask::{RoundedRectMask, MaskSource};
use zenblend::{mask_row, mask_row_constant, mask_row_rgb, apply_mask_spans};

let mut pixels = vec![0.5, 0.0, 0.0, 0.5,  0.0, 0.3, 0.0, 1.0];

// Per-pixel mask: one f32 per pixel, broadcast to all 4 channels.
let mask_values = vec![0.8, 1.0];
mask_row(&mut pixels, &mask_values);

// Uniform opacity -- no mask buffer needed.
mask_row_constant(&mut pixels, 0.7);

// RGB-only mask: multiplies R, G, B but leaves alpha untouched.
// Use case: gain map application, vignette without opacity change.
let rgb_mask = vec![0.9, 1.0];
mask_row_rgb(&mut pixels, &rgb_mask);

// Span-optimized masking -- skips fully opaque/transparent regions.
let width = 64;
let height = 64;
let mask = RoundedRectMask::new(width, height, [10.0, 10.0, 10.0, 10.0]);
let mut row = vec![0.5f32; (width as usize) * 4];
let mut mask_buf = vec![0.0f32; width as usize];
apply_mask_spans(&mut row, &mut mask_buf, &mask, 0);
```

Built-in masks: `RoundedRectMask`, `LinearGradientMask`, `RadialGradientMask`.
Implement the `MaskSource` trait for custom masks.

### Interpolation

```rust
use zenblend::lerp_row;

// Per-pixel blend factor t in [0, 1]. One f32 per pixel.
// t=0 -> a, t=1 -> b.
let a   = vec![1.0, 0.0, 0.0, 1.0,  0.0, 0.0, 0.0, 0.0];
let b   = vec![0.0, 0.0, 1.0, 1.0,  0.0, 1.0, 0.0, 1.0];
let t   = vec![0.5, 0.5];
let mut out = vec![0.0f32; 8];
lerp_row(&a, &b, &t, &mut out);
```

## SIMD acceleration

SrcOver, masking (`mask_row`, `mask_row_rgb`), and `lerp_row` are SIMD-accelerated via runtime CPU dispatch through [archmage](https://crates.io/crates/archmage):

| Platform | ISA | Pixels/iter |
|----------|-----|-------------|
| x86_64 | AVX2 + FMA | 2 |
| AArch64 | NEON | 1 |
| WASM | simd128 | 1 |
| Fallback | scalar | 1 |

Other blend modes use scalar implementations. Mask span alignment is SIMD-aware (snaps partial spans to block boundaries).

`#![forbid(unsafe_code)]` -- all SIMD through safe abstractions.

## Limitations

- All data must be premultiplied linear f32 RGBA. There is no format conversion; bring your own linearization.
- No non-separable blend modes (Hue, Saturation, Color, Luminosity).
- Only SrcOver has a dedicated SIMD fast path for blending; the other 31 modes run scalar per-pixel loops.
- Row-level API only. There is no tile, buffer, or image-level compositing -- that belongs in zenpipe.

## Features

- `default = ["std"]`

## Image tech I maintain

| | |
|:--|:--|
| State of the art codecs* | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] ([rav1d-safe] · [zenrav1e] · [zenavif-parse] · [zenavif-serialize]) · [zenjxl] ([jxl-encoder] · [zenjxl-decoder]) · [zentiff] · [zenbitmaps] · [heic] · [zenraw] · [zenpdf] · [ultrahdr] · [mozjpeg-rs] · [webpx] |
| Compression | [zenflate] · [zenzop] |
| Processing | [zenresize] · [zenfilters] · [zenquant] · **zenblend** |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [resamplescope-rs] · [codec-eval] · [codec-corpus] |
| Pixel types & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] |
| ImageResizer | [ImageResizer] (C#) — 24M+ NuGet downloads across all packages |
| [Imageflow][] | Image optimization engine (Rust) — [.NET][imageflow-dotnet] · [node][imageflow-node] · [go][imageflow-go] — 9M+ NuGet downloads across all packages |
| [Imageflow Server][] | [The fast, safe image server](https://www.imazen.io/) (Rust+C#) — 552K+ NuGet downloads, deployed by Fortune 500s and major brands |

<sub>* as of 2026</sub>

### General Rust awesomeness

[archmage] · [magetypes] · [enough] · [whereat] · [zenbench] · [cargo-copter]

[And other projects](https://www.imazen.io/open-source) · [GitHub @imazen](https://github.com/imazen) · [GitHub @lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith) · [NuGet](https://www.nuget.org/profiles/imazen) (over 30 million downloads / 87 packages)

## License

Dual-licensed: [AGPL-3.0](LICENSE-AGPL3) or [commercial](LICENSE-COMMERCIAL).

I've maintained and developed open-source image server software -- and the 40+
library ecosystem it depends on -- full-time since 2011. Fifteen years of
continual maintenance, backwards compatibility, support, and the (very rare)
security patch. That kind of stability requires sustainable funding, and
dual-licensing is how I make it work without venture capital or rug-pulls.
Support sustainable and secure software; swap patch tuesday for patch leap-year.

[Our open-source products](https://www.imazen.io/open-source)

**Your options:**

- **Startup license** -- $1 if your company has under $1M revenue and fewer
  than 5 employees. [Get a key](https://www.imazen.io/pricing)
- **Commercial subscription** -- Governed by the Imazen Site-wide Subscription
  License v1.1 or later. Apache 2.0-like terms, no source-sharing requirement.
  Sliding scale by company size.
  [Pricing & 60-day free trial](https://www.imazen.io/pricing)
- **AGPL v3** -- Free and open. Share your source if you distribute.

See [LICENSE-COMMERCIAL](LICENSE-COMMERCIAL) for details.

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zentiff]: https://github.com/imazen/zentiff
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic-decoder-rs
[zenraw]: https://github.com/imazen/zenraw
[zenpdf]: https://github.com/imazen/zenpdf
[ultrahdr]: https://github.com/imazen/ultrahdr
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenrav1e]: https://github.com/imazen/zenrav1e
[mozjpeg-rs]: https://github.com/imazen/mozjpeg-rs
[zenavif-parse]: https://github.com/imazen/zenavif-parse
[zenavif-serialize]: https://github.com/imazen/zenavif-serialize
[webpx]: https://github.com/imazen/webpx
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenresize]: https://github.com/imazen/zenresize
[zenfilters]: https://github.com/imazen/zenfilters
[zenquant]: https://github.com/imazen/zenquant
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zencodecs
[zenlayout]: https://github.com/imazen/zenlayout
[zennode]: https://github.com/imazen/zennode
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-server
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
[ImageResizer]: https://github.com/imazen/resizer
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[zenbench]: https://github.com/imazen/zenbench
[cargo-copter]: https://github.com/imazen/cargo-copter
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[codec-eval]: https://github.com/imazen/codec-eval
[codec-corpus]: https://github.com/imazen/codec-corpus
