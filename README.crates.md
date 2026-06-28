<!-- GENERATED FROM README.md by zenutils gen-readme-crates.sh — DO NOT EDIT. -->

# zenblend [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenblend/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/zenblend/actions/workflows/ci.yml)

zenblend is a row-level pixel blending library for premultiplied linear f32 RGBA compositing pipelines: 32 Porter-Duff and artistic blend modes, masking, and interpolation. Pure Rust, `#![forbid(unsafe_code)]`, allocation-free, and SIMD-accelerated via runtime CPU dispatch.

All operations work on `&mut [f32]` slices -- 4 interleaved floats per pixel, alpha pre-applied, in linear light. Designed as the inner loop for [zenpipe](https://github.com/imazen/zenpipe) strip pipelines, not a standalone compositing engine.

## Quick start

```toml
[dependencies]
zenblend = "0.1"
```

```rust
use zenblend::{BlendMode, blend_row};

// `fg` is the TOP layer (Src), `bg` the BOTTOM layer (Dst). Both are
// premultiplied linear f32 RGBA (4 floats/pixel, equal length, divisible by 4).
// `fg` is blended over `bg` IN PLACE; `bg` is read-only.
let mut fg = vec![0.5, 0.0, 0.0, 0.5,  0.0, 0.3, 0.0, 1.0];
let bg     = vec![0.0, 0.3, 0.0, 1.0,  0.0, 0.0, 0.5, 0.5];
blend_row(&mut fg, &bg, BlendMode::SrcOver); // fg now holds the result
```

zenblend does **no** format conversion: linearize + premultiply on the way in, and
un-premultiply + encode back to sRGB on the way out (see the
[full round-trip example](#compositing-a-sprite-over-a-background-full-round-trip)).
For solid-color backgrounds, masking, gradients, and interpolation, read on.

## Blend modes

**Porter-Duff (12):** Clear, Src, Dst, SrcOver, DstOver, SrcIn, DstIn, SrcOut,
DstOut, SrcAtop, DstAtop, Xor

**Artistic (20):** Multiply, Screen, Overlay, Darken, Lighten, HardLight, SoftLight,
ColorDodge, ColorBurn, Difference, Exclusion, LinearBurn, LinearDodge, VividLight,
LinearLight, PinLight, HardMix, Divide, Subtract, Plus

32 modes total. Artistic modes unpremultiply per-pixel, apply the blend function,
then re-premultiply. Plus operates directly on premultiplied data. `BlendMode` is
`#[non_exhaustive]`, so match with a wildcard arm.

## Getting started

### Blending two rows

**Argument direction (read this first).** In `blend_row(fg, bg, mode)`, `fg` is the
**top** layer (Porter-Duff *source*, `Src`) and `bg` is the **bottom** layer
(Porter-Duff *destination*, `Dst`). `BlendMode::SrcOver` composites `fg` **over**
`bg` — i.e. `fg` is drawn on top, computing `Src + Dst·(1 - Src.alpha)`. The result
is written back into `fg` in place; `bg` is read-only. Swapping the two arguments
compiles cleanly but composites the layers in the wrong order (your overlay ends up
*underneath*), so make sure the top layer is `fg`.

```rust
use zenblend::{BlendMode, blend_row, blend_row_solid, blend_row_solid_opaque};

// fg = TOP layer (Src), bg = BOTTOM layer (Dst).
// Both are premultiplied linear f32, 4ch RGBA, equal length, divisible by 4.
// fg is modified in place to contain the blended result (bg is read-only).
let mut fg = vec![0.5, 0.0, 0.0, 0.5,  0.0, 0.3, 0.0, 1.0];
let bg =     vec![0.0, 0.3, 0.0, 1.0,  0.0, 0.0, 0.5, 0.5];
blend_row(&mut fg, &bg, BlendMode::SrcOver); // fg over bg

// Blend against a solid color -- no row buffer needed for background.
let mut row = vec![0.5, 0.0, 0.0, 0.5,  0.0, 0.3, 0.0, 1.0];
blend_row_solid(&mut row, &[0.2, 0.0, 0.0, 0.5], BlendMode::Multiply);

// Optimized path when the background is opaque (alpha = 1.0).
let mut row2 = vec![0.5, 0.0, 0.0, 0.5,  0.0, 0.3, 0.0, 1.0];
blend_row_solid_opaque(&mut row2, &[0.2, 0.1, 0.05, 1.0], BlendMode::SrcOver);
```

### Compositing a sprite over a background (full round-trip)

zenblend does **no format conversion**: it only blends premultiplied-linear-f32 rows.
The caller owns both ends of the trip — linearize + premultiply on the way *in*, and
un-premultiply + encode back to sRGB on the way *out*. There is no helper that turns
the result back into displayable 8-bit pixels; you must do the inverse yourself.

This example draws a sprite (the **top** layer, `fg`) over a background (the **bottom**
layer, `bg`) at an `(x, y)` offset, working one row at a time. The sprite is placed by
the caller's own row/column slicing — zenblend never sees coordinates.

```rust
use zenblend::{BlendMode, blend_row};

// sRGB transfer functions (zenblend does NOT provide these — bring your own).
fn srgb_to_linear(c: u8) -> f32 {
    let c = c as f32 / 255.0;
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}
fn linear_to_srgb(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0 + 0.5) as u8
}

/// One RGBA pixel: straight-alpha sRGB-u8  ->  premultiplied-linear-f32.
fn load_premul(px: &[u8; 4], out: &mut [f32; 4]) {
    let a = px[3] as f32 / 255.0;
    out[0] = srgb_to_linear(px[0]) * a; // premultiply RGB by alpha
    out[1] = srgb_to_linear(px[1]) * a;
    out[2] = srgb_to_linear(px[2]) * a;
    out[3] = a;
}

/// One RGBA pixel: premultiplied-linear-f32  ->  straight-alpha sRGB-u8.
fn store_unpremul(px: &[f32; 4], out: &mut [u8; 4]) {
    let a = px[3];
    let inv = if a > 0.0 { 1.0 / a } else { 0.0 }; // un-premultiply (guard a == 0)
    out[0] = linear_to_srgb(px[0] * inv);
    out[1] = linear_to_srgb(px[1] * inv);
    out[2] = linear_to_srgb(px[2] * inv);
    out[3] = (a * 255.0 + 0.5) as u8;
}

// --- Background (bottom layer) and sprite (top layer), both straight-alpha sRGB u8 ---
let (bg_w, bg_h) = (4usize, 4usize);
let mut background: Vec<u8> = vec![0; bg_w * bg_h * 4]; // your real bg pixels here
for px in background.chunks_exact_mut(4) {
    px.copy_from_slice(&[40, 80, 160, 255]); // opaque blue field
}

let (sp_w, sp_h) = (2usize, 2usize);
let sprite: Vec<u8> = vec![255, 0, 0, 128].repeat(sp_w * sp_h); // 50%-alpha red

// Caller-chosen placement. The sprite occupies bg columns dst_x..dst_x+sp_w
// and bg rows dst_y..dst_y+sp_h (assume it fits; clip these ranges if it doesn't).
let (dst_x, dst_y) = (1usize, 1usize);

// Scratch f32 rows, reused across rows (one pixel = 4 floats).
let mut fg_row = vec![0.0f32; sp_w * 4]; // sprite row, premultiplied-linear
let mut bg_row = vec![0.0f32; sp_w * 4]; // background segment, premultiplied-linear

for sy in 0..sp_h {
    let by = dst_y + sy; // background row this sprite row lands on

    // 1. Linearize + premultiply the sprite row (fg = TOP) ...
    let sprite_row = &sprite[sy * sp_w * 4 .. (sy + 1) * sp_w * 4];
    // ... and the matching background segment (bg = BOTTOM).
    let bg_start = (by * bg_w + dst_x) * 4;
    let bg_seg = &background[bg_start .. bg_start + sp_w * 4];
    for x in 0..sp_w {
        let s: &[u8; 4] = sprite_row[x * 4 .. x * 4 + 4].try_into().unwrap();
        let b: &[u8; 4] = bg_seg[x * 4 .. x * 4 + 4].try_into().unwrap();
        load_premul(s, (&mut fg_row[x * 4 .. x * 4 + 4]).try_into().unwrap());
        load_premul(b, (&mut bg_row[x * 4 .. x * 4 + 4]).try_into().unwrap());
    }

    // 2. Blend: fg (sprite, top) OVER bg (background, bottom). Result lands in fg_row.
    blend_row(&mut fg_row, &bg_row, BlendMode::SrcOver);

    // 3. Un-premultiply + encode the result back to sRGB-u8 INTO the background.
    let out_start = (by * bg_w + dst_x) * 4;
    let out_seg = &mut background[out_start .. out_start + sp_w * 4];
    for x in 0..sp_w {
        let r: &[f32; 4] = fg_row[x * 4 .. x * 4 + 4].try_into().unwrap();
        store_unpremul(r, (&mut out_seg[x * 4 .. x * 4 + 4]).try_into().unwrap());
    }
}
// `background` now holds the composited image as straight-alpha sRGB-u8.
```

The three steps — **linearize + premultiply**, **`blend_row` (fg over bg)**, then
**un-premultiply + encode** — are the full contract. Skip step 1 and you feed sRGB
gamma values into a linear-light blend (wrong, too-dark edges). Skip step 3 and you
hand back premultiplied-linear floats that no display expects. Both inverse halves
are the caller's responsibility.

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

Built-in masks: `RoundedRectMask` (`new` with per-corner radii, `circle`, `uniform`),
`LinearGradientMask`, `RadialGradientMask`. Implement the `MaskSource` trait for custom
masks. `apply_mask_spans` validates the spans a `MaskSource` returns (coverage, ordering,
in-range bounds) and falls back to a full-row fill if they're malformed, so a buggy or
hostile `MaskSource` can't panic via an out-of-bounds slice.

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

Hot paths dispatch to SIMD at runtime via [archmage](https://crates.io/crates/archmage)
— no `target-cpu` flags required, and the public build stays `#![forbid(unsafe_code)]`.

**SIMD on every tier** (x86-64 AVX2+FMA, AArch64 NEON, WASM `simd128`, scalar fallback):

| Tier | ISA | Pixels/iter (SrcOver) |
|------|-----|-----------------------|
| x86-64 | AVX2 + FMA | 2 |
| AArch64 | NEON | 1 |
| WASM | simd128 | 1 |
| fallback | scalar | 1 |

`BlendMode::SrcOver` blending, `mask_row`, `mask_row_rgb`, and `lerp_row` run on the
vector path above on all four tiers.

**SIMD on NEON and WASM `simd128`** — six separable artistic modes (Multiply, Screen,
Darken, Lighten, Difference, Exclusion) use division-free premultiplied closed forms,
measured at **+60–102%** throughput vs the scalar path on an Ampere Altra Neoverse-N1
(e.g. Screen 108→217 Mpix/s, Exclusion 103→207 Mpix/s). On x86-64 these six currently
run scalar (a 2-pixel kernel is a documented future hypothesis). The remaining
Porter-Duff and artistic modes run a scalar per-pixel loop everywhere.

Mask-span alignment is SIMD-aware: it snaps partial spans to block boundaries so the
kernel only touches whole vectors.


## Limitations

- All data must be premultiplied linear f32 RGBA. There is no format conversion; bring your own linearization.
- No non-separable blend modes (Hue, Saturation, Color, Luminosity).
- SIMD blend kernels cover `SrcOver` (all tiers) plus six separable artistic modes (Multiply, Screen, Darken, Lighten, Difference, Exclusion — NEON/WASM only); the remaining modes run scalar per-pixel loops.
- Row-level API only. There is no tile, buffer, or image-level compositing -- that belongs in zenpipe.

## Features

- `default = ["std"]` — the `std` feature gates nothing the public API needs; the crate is allocation-free and uses no `std`-only items, so it builds the same with `--no-default-features`.

## License

Dual-licensed: [AGPL-3.0](https://github.com/imazen/zenblend/blob/main/LICENSE-AGPL3) or [commercial](https://github.com/imazen/zenblend/blob/main/LICENSE-COMMERCIAL).

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

See [LICENSE-COMMERCIAL](https://github.com/imazen/zenblend/blob/main/LICENSE-COMMERCIAL) for details.

## Image tech I maintain

| | |
|:--|:--|
| **Codecs** ¹ | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] · [zenjxl] · [zenbitmaps] · [heic] · [zentiff] · [zenpdf] · [zensvg] · [zenjp2] · [zenraw] · [ultrahdr] |
| Codec internals | [zenjxl-decoder] · [jxl-encoder] · [zenrav1e] · [rav1d-safe] · [zenavif-parse] · [zenavif-serialize] |
| Compression | [zenflate] · [zenzop] · [zenzstd] |
| Processing | [zenresize] · [zenquant] · **zenblend** · [zenfilters] · [zensally] · [zentone] |
| Pixels & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline & framework | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] · [zenwasm] · [zentract] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [zenmetrics] · [resamplescope-rs] |
| Pickers & ML | [zenanalyze] · [zenpredict] · [zenpicker] |
| Products | [Imageflow] image engine ([.NET][imageflow-dotnet] · [Node][imageflow-node] · [Go][imageflow-go]) · [Imageflow Server] · [ImageResizer] (C#) |

<sub>¹ pure-Rust, `#![forbid(unsafe_code)]` codecs, as of 2026</sub>

### General Rust awesomeness

[zenbench] · [archmage] · [magetypes] · [enough] · [whereat] · [cargo-copter]

[Open source](https://www.imazen.io/open-source) · [@imazen](https://github.com/imazen) · [@lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith)

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic
[zentiff]: https://github.com/imazen/zentiff
[zenpdf]: https://github.com/imazen/zenpdf
[zensvg]: https://github.com/imazen/zenextras
[zenjp2]: https://github.com/imazen/zenextras
[zenraw]: https://github.com/imazen/zenraw
[ultrahdr]: https://github.com/imazen/ultrahdr
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenrav1e]: https://github.com/imazen/zenrav1e
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenavif-parse]: https://github.com/imazen/zenavif-parse
[zenavif-serialize]: https://github.com/imazen/zenavif-serialize
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenzstd]: https://github.com/imazen/zenzstd
[zenresize]: https://github.com/imazen/zenresize
[zenquant]: https://github.com/imazen/zenquant
[zenfilters]: https://github.com/imazen/zenfilters
[zensally]: https://github.com/imazen/zensally
[zentone]: https://github.com/imazen/zentone
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zencodecs
[zenlayout]: https://github.com/imazen/zenlayout
[zennode]: https://github.com/imazen/zennode
[zenwasm]: https://github.com/imazen/zenwasm
[zentract]: https://github.com/imazen/zentract
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[zenmetrics]: https://github.com/imazen/zenmetrics
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[zenanalyze]: https://github.com/imazen/zenanalyze
[zenpredict]: https://github.com/imazen/zenanalyze
[zenpicker]: https://github.com/imazen/zenanalyze
[zenbench]: https://github.com/imazen/zenbench
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[cargo-copter]: https://github.com/imazen/cargo-copter
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[ImageResizer]: https://github.com/imazen/resizer
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
