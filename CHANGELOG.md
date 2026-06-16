# Changelog

## [Unreleased]

### Added
- Versioned public-API surface snapshot at `docs/public-api/zenblend.txt`,
  regenerated on every `cargo test` by `tests-dev/public_api_doc.rs`
  (`ZEN_API_DOC=check` verifies in CI's clippy job, `=off` skips elsewhere);
  `just api-doc` / `api-doc-check` recipes. Lives in `tests-dev/` so the
  include-whitelisted package tarball is unaffected.

### Changed
- SIMD-accelerated six separable artistic blend modes (Multiply, Screen, Darken, Lighten, Difference, Exclusion) on NEON/WASM128 via division-free premultiplied closed forms, replacing the scalar per-pixel unpremultiply path. Measured **+60–102%** throughput on Ampere Altra Neoverse-N1 (e.g. Screen 108→217 Mpix/s, Exclusion 103→207 Mpix/s). x86 keeps the scalar path. Bit-faithful to the scalar reference within 1e-6 (verified by new equivalence tests + existing per-mode/cross-tier tests). See `benchmarks/arm_neoverse_n1_baseline_2026-05-29.md`.

### Fixed
- docs(readme): state `blend_row` `fg`=top/Src, `bg`=bottom/Dst direction + add full sprite-over-background round-trip example — prevents silent compositing-order/premultiply corruption found by insulated-developer test.
- `apply_mask_spans` now validates spans returned by `MaskSource` impls (coverage, ordering, in-range bounds) and falls back to `fill_mask_row` when validation fails; a buggy or hostile `MaskSource` can no longer panic via slice OOB.
- Artistic blend modes (Multiply, Screen, Overlay, ColorDodge, ColorBurn, etc.) sanitize NaN/inf in input alpha and per-channel output, replacing non-finite results with 0 instead of poisoning downstream blends.

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
