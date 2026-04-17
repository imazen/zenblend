# Changelog

## [Unreleased]

## [0.1.3] - 2026-04-17

### Added
- SIMD tier consistency tests verifying portable/SSE2/AVX2 produce identical results (cb92f72)

### Changed
- Migrated `Cargo.toml` from `exclude` to `include` whitelist for cleaner packages (8fb609d)
- Fixed nightly clippy `collapsible_match` warnings in `mask.rs`

## [0.1.2] - 2026-04-01

### Changed
- Version bump for release (9e0dcf8)

## [0.1.1] - 2026-04-01

### Changed
- Migrated SIMD from `wide` to `magetypes` (fa722bc)
- Updated archmage and magetypes to 0.9.16 (7af4b5a)
- Added i686-unknown-linux-gnu CI target via cross (9469a18)

### Fixed
- Stripped path overrides so crates resolve from registry (ed740d0, 3653a2b)
- Fixed broken intra-doc link (eda6313)

## [0.1.0] - 2026-03-29

### Added
- Porter-Duff and artistic blend modes on premultiplied linear f32 RGBA rows (1785bd4)
- 9 separable blend modes: Multiply, Screen, Overlay, Darken, Lighten, ColorDodge, ColorBurn, HardLight, SoftLight (a010c82)
- Mask system with per-pixel alpha modulation and rounded rect mask (32122a0)
- LinearGradientMask and RadialGradientMask (624abd9)
- MaskSpans for span-based mask application with alignment (b8af8d3, e865ef4)
- `mask_row_rgb` and `lerp_row` with full SIMD (206957c)
- SIMD acceleration via archmage: SSE2 + AVX2 tiers with portable fallback (fa722bc)
