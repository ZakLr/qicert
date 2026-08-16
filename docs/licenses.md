# Licenses — backbone weights (recorded at download)

> Source of truth: `weights/_licenses.json` (exact HF `license` card strings,
> captured at download time 2026-08-16). Machine-readable, not paraphrased.
> **This file is the report §6 compliance table source — update it whenever a
> new weight repo lands, never after.**

## 1. Downloaded repos (exact HF card strings)

| Repo | Kind | License (exact HF string) | Downloaded |
|---|---|---|---|
| `Stanford-ILIAD/minivla-libero90-prismatic` | ckpt | `mit` | ✅ |
| `Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b` | backbone | `mit` | ✅ |
| `openvla/openvla` (github) | code | `MIT` | ✅ |

## 2. Embedded sub-model licenses ⚠️ (the card string is NOT the whole story)

The `minivla-libero90-prismatic` checkpoint **embeds full sub-model weights**
(verified 2026-08-16 by inspecting `checkpoints/step-122500-...pt`:

```
vision_backbone.dino_featurizer.{cls_token, reg_token, pos_embed,
  patch_embed.proj, blocks.*.attn.qkv, ...}   <- DINOv2 ViT weights INSIDE the ckpt
llm_backbone.*                                  <- Qwen2.5
vision_backbone.siglip_*                        <- SigLIP
```

| Sub-model | Source | License |
|---|---|---|
| **DINOv2** (ViT featurizer) | Meta, `facebook/dinov2` | **CC-BY-NC 4.0** (non-commercial) ⚠️ |
| SigLIP (vision tower) | Google, `google/siglip-...` | Apache-2.0 |
| Qwen2.5 (LLM backbone) | Alibaba, `Qwen/Qwen2.5-...` | Apache-2.0 |

The HF wrapper cards say `mit`; the wrapper's *own* code is MIT. But the
checkpoint weights incorporate DINOv2 under **CC-BY-NC 4.0**, which restricts
commercial use. Anyone redistributing fine-tuned weights derived from this
checkpoint inherits that restriction.

## 3. LIBERO dataset (N1 fine-tuning slice)

| Repo | Suite | License (exact HF string) | Downloaded |
|---|---|---|---|
| `openvla/modified_libero_rlds` | `libero_spatial_no_noops` (N1 slice) | `mit` | ✅ |

Source: `weights/_libero_licenses.json` (captured 2026-08-16). Same four
suites MiniVLA-1B itself was fine-tuned on; the spatial suite is N1's
pre-registered slice (submission/08-experiments.md).

## 4. Decision (Q11 — logged in council-log.md 2026-08-16)

- **DECIDED: MiniVLA for scored tables, documented.** qicert's own code ships
  MIT/Apache-2.0; **MiniVLA-derived fine-tuned checkpoints are NOT
  redistributed**; the exact strings above go in the report §6 compliance table
  verbatim, with the DINOv2 NC caveat stated. Fallback (SigLIP-only/π0) stays
  available only if a future audit makes it necessary.
