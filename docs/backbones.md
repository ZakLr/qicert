# Backbones & baselines: reference and download checklist

Decision (2026-08-11, chair): **robotics-first**. MiniVLA-1B is the plan's scored
robotics backbone, fits comfortably on the RTX 5060 (8 GB, sm_120) via the Docker
runtime, and yields *real* N1/N2/N3 tables at ~1/10 the compute of the AD track.
The library is track-agnostic (adapters + spec libraries); only the experiment
*order* changes. AD (LLaVA-1.5-7B) follows on Kaggle T4s via the N13 mirror.

---

## 1. Scored backbones (the report's tables must land on these)

| Backbone | Repo / checkpoints | Size (bf16/fp16) | License | Role |
|---|---|---|---|---|
| **MiniVLA-1B** (robotics) | `Stanford-ILIAD/openvla-mini` (code, MIT); checkpoints `Stanford-ILIAD/minivla-*` (see §2) | ~2.1–2.3 GB | code MIT; backbones Apache-2.0 (Qwen2.5, SigLIP) **+ DINOv2 CC-BY-NC 4.0 ⚠️** | **N1/N2/N3 robotics tables: first** |
| **LLaVA-1.5-7B** (AD) | `llava-hf/llava-1.5-7b-hf` | ~13.5–14 GB | Vicuna/LLaMA-2 terms | N13 AD mirror on Kaggle |

> ⚠️ **DINOv2 flag (Q11 evidence):** MiniVLA fuses DINOv2 (Meta) with SigLIP.
> DINOv2 weights are **CC-BY-NC 4.0 (non-commercial)**. The report §6 compliance
> row must state this explicitly. If the competition's terms conflict with
> non-commercial weights, the fallback is to keep MiniVLA for toolchain/hyper-
> parameter validation and land the *scored* robotics tables on a cleanly
> licensed backbone (e.g., OpenVLA's SigLIP-only variant or a π0-family model)
> Decide before Week-1 experiments, not after.

## 2. MiniVLA-1B checkpoints (exact HF IDs, from the official collection)

| Variant | HF repo | Use |
|---|---|---|
| LIBERO-90 (standard) | `Stanford-ILIAD/minivla-libero90-prismatic` | primary robotics benchmark |
| Bridge V2 (VQ action chunking) | `Stanford-ILIAD/minivla-vq-bridge-prismatic` | VQ action-chunking smoke |
| Wrist-image VQ | `Stanford-ILIAD/minivla-wrist-vq-libero90-prismatic` | multi-image (wrist) path |
| Image-history VQ (h=2) | `Stanford-ILIAD/minivla-history2-vq-libero90-prismatic` | image-history path |
| Base VLM backbone | `Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b` | raw backbone (no action head) |

**Load path:** `transformers` AutoClasses with **`trust_remote_code=True`**
(`AutoModelForVision2Seq` + `AutoProcessor`). Custom Prismatic/OpenVLA config code
ships inside the remote repo.

**Flash-attn caveat:** the upstream loop uses `attn_implementation="flash_attention_2"`.
Flash-attn builds on sm_120/CUDA 13 can be painful: keep the fallback
`"sdpa"`/`"eager"` path working first (MiniVLA at 1B scale is small enough that
SDPA costs are acceptable for validation runs; record any deviation).

## 3. Download + license checklist (0 GPU-h, do once, record everything)

Run in the Docker container (repo mounted at `/workspace`):

```bash
cd /workspace
python - <<'PY'
from huggingface_hub import snapshot_download, HfApi
api = HfApi()
repos = {
    "Stanford-ILIAD/openvla-mini": "code",
    "Stanford-ILIAD/minivla-libero90-prismatic": "ckpt",
    "Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b": "backbone",
}
for repo, kind in repos.items():
    info = api.model_info(repo)
    print(f"[{kind}] {repo}: license={info.cardData.get('license', 'UNKNOWN')}")
    snapshot_download(repo, local_dir=f"weights/{kind}")
PY
```

- [ ] **License rows recorded at download**: capture the `license` card field for
      every repo into `docs/licenses.md` (the report §6 compliance table needs the
      exact strings, not paraphrases). Include the DINOv2/SigLIP/Qwen2.5 sub-model
      licenses.
- [ ] **Version pins captured**: record `transformers`/`peft`/`accelerate`/
      `huggingface_hub` versions actually used (freeze into `environment.yml`).
- [ ] **Backbone smoke**: load `minivla-libero90-prismatic` in fp16 on the 5060,
      run one LIBERO prompt → action-chunk decode (VQ codewords); record VRAM peak
      and per-step latency. Kill-criterion check: if fp16 doesn't fit at batch 1
      with gradient checkpointing, the robotics-first plan needs revision (it will fit).
- [ ] **DINOv2 license decision**: chair + reproducibility-auditor: is CC-BY-NC
      acceptable for this competition? Log the decision in `council-log.md` (Q11).

## 4. Baselines (all tiny: run on the *same* backbone as qicert)

| Baseline | Cost | Needed for |
|---|---|---|
| INT8 (bitsandbytes 4/8-bit: **CPU fallback on Blackwell**, see below) | minutes | N1 reference, headline comparator |
| SVD-at-matched-ratio | numpy, seconds | accuracy-parity check (A1) |
| Performer / FAVOR+ | small torch module | compiler comparison at matched capacity (A2, E-comp-3) |
| RESTART+GEV, naive MC | pure Python | safety three-arm race (A3) |
| Autoencoder monitor | small | syndrome-shadow monitor ablation (A4) |

> **bitsandbytes on Blackwell:** bnb 0.44+ supports sm_120, but kernel builds on
> CUDA 13 images can silently fall back to CPU. v1 policy (Decision 2026-08-11):
> **no bitsandbytes**: MiniVLA-1B fits fp16/bf16 LoRA at 8 GB. INT8 rows on the
> robotics track use `torch.quantization` or a cu13-compatible bnb pin if/when
> verified; record the fallback in the bench output if used.

## 5. Warm-up models (optional; only if the 5060 needs a lighter smoke first)

| Model | Size (fp16) | License | Why |
|---|---|---|---|
| SmolVLM2-2.2B (`HuggingFaceTB/SmolVLM2-2.2B-Instruct`) | ~4.5 GB | Apache-2.0 | lightest real VL warm-up, native multi-image |
| Qwen2.5-VL-3B (`Qwen/Qwen2.5-VL-3B-Instruct`) | ~6.5–7 GB | Apache-2.0 | AD-track warm-up; same Qwen2.5 family as MiniVLA |

Warm-up models validate the *toolchain* only: the scored tables (N1/N2/N3/N13)
must land on MiniVLA-1B (robotics) and LLaVA-1.5-7B (AD), and the report must say
so explicitly. No substituting (N14 enforces whatever the tables claim).

## 6. Pipeline warm-up ladder (each step exercises one kernel stage)

1. **Step 0: single-layer smoke (0 GPU, CPU fine):** TT-cross one MiniVLA
   projector/MLP layer with the `tt_cross` kernel → exact Lipschitz product →
   INT8 + SVD on that same layer → compiler identity on one interaction block.
   Exercises stages 1/3/4 and the whole bench pipeline.
2. **Step 1: robotics smoke (5060):** full MiniVLA LoRA fine-tune on a LIBERO
   slice + INT8 reference (batch 1–2, grad-accum 8–16, gradient checkpointing).
3. **Step 2: N2′ bit-ordering sweep:** tag results by **layer type** (q/k/v
   projections, MLP, embeddings, head) so conclusions transfer to LLaVA-7B's
   same layer types (the AD-transfer watch-list item).
4. **Step 3: N1/N3 first scored tables:** real numbers into the report.
