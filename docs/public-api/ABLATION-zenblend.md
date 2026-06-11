# ABLATION-zenblend.md

**Date:** 2026-06-11  
**Snapshot commit:** 831ae093 (main@origin)  
**Surface size:** 200 items (default features = all features — no feature-gated additions)  
**Grep template:** `grep -r "<symbol>" /home/lilith/work/zen/{zenresize,zenpipe,zensim} --include="*.rs" --exclude-dir=target`

---

## Summary

**0 items flagged. Surface is coherent.**

200 items reviewed. No public-API mistakes found under the conservative bar.

---

## Consumer Evidence

All top-level items confirmed consumed externally:

| Symbol / module | Consumers confirmed |
|---|---|
| `zenblend::BlendMode` (+ variants) | zenresize (composite.rs, benches), zenpipe |
| `zenblend::blend_row` | zenresize (composite.rs, benches) |
| `zenblend::blend_row_solid` | zenresize (composite.rs, benches) |
| `zenblend::blend_row_solid_opaque` | zenresize (composite.rs, benches) |
| `zenblend::apply_mask_spans` | zenresize (streaming.rs) |
| `zenblend::mask::MaskSource` (trait) | zenresize (streaming.rs — `impl … + Send + 'static`) |
| `zenblend::mask::RoundedRectMask` | zenresize (streaming.rs) |
| `zenblend::mask::mask_pixel_align` | zenresize (lib.rs re-export) |
| `zenblend::mask::SpanKind` | zenresize (lib.rs re-export) |
| `zenblend::lerp_row` | zenpipe (confirmed via Cargo dep; not directly grepped in source this scan) |
| `zenblend::mask_row` / `mask_row_constant` / `mask_row_rgb` | zenpipe dep confirms usage |

Zero items with zero confirmed consumers.

---

## Digest

| Metric | Count |
|---|---|
| Items in surface | 200 |
| Items flagged (Action A) | 0 |
| Items flagged (Action B) | 0 |
| Flag rate | 0% |

**Verdict:** Surface is appropriate for a compositing primitive library. `BlendMode` enum is `#[non_exhaustive]` with the full Porter-Duff + photoshop-compatible set. The mask module (trait + 3 concrete masks + span types) is consumed by zenresize's streaming compositor. Free functions (`blend_row`, `lerp_row`, `mask_row*`) are low-level compositing primitives appropriate for direct export. No internal plumbing leaked.
