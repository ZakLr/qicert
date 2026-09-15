# Run Data Register — everything we capture on every benchmark run

**Decision 2026-08-16 (user):** benchmarks and fine-tunes are NEVER re-run to fetch
missing data. Every run records everything now, so plots, ablations, tables, and
post-hoc analyses can be built from the artifacts afterwards. The rule that makes
this safe:

> **Every number in any table or plot traces back to a `run.json` + `config.json` +
> `metrics.jsonl` triple. If it isn't in the artifacts, it doesn't exist.**

Artifacts live in `results/<EXP_ID>/<RUN_ID>/` (JSON + JSONL, ASCII-safe, machine
readable). The human ledger stays in the plan workspace (`research/experiments.md`);
the machine ledger `results/ledger.csv` is generated from the artifacts and mirrors it.

Captured-by-default = the load-bearing set (categories 0–9 below). Cheap extras =
negligible cost, enabled with `--capture extra`. Heavy = real cost in time/disk,
enabled with `--capture heavy`. Excluded = deliberately not recorded, with reason.

---

## Captured by default (the load-bearing set)

### 0. Identity & provenance — every artifact starts with these

| Field | Example | Why it matters |
|---|---|---|
| `experiment_id` | `N1` | ties to the pre-registered plan row |
| `run_id` | uuid4 hex | unique, stable reference |
| `run_tag` | `minivla-libero90-sliceA` | human-meaningful name |
| `seed` + `seed_spec` | `0`, `{"torch":0,"numpy":0,"python":0,"hf":0,"dataloader":0}` | which RNGs were seeded, exactly |
| `bench_module`, `table` | `compress`, `kernel-smoke` | which bench row produced the run |
| `git_commit` (qicert) + `git_dirty` | `da3c8c0`, `false` | code version is the config's twin |
| `workspace_commit` | plan-workspace hash | research-side state at run time |
| `config_hash` | sha256 of canonical config | the run's fingerprint; ledger key |
| `status` | `completed/failed/killed/demoted` | honest lifecycle |
| `kill_criterion` + verdict | `R1`, `not triggered` | pre-registered gate evaluated |
| `start_utc`, `end_utc`, `wall_sec` | ISO-8601, seconds | timing is a metric, not a note |
| `command_line` | full `qicert.bench` invocation + args | exact reproducibility string |
| `qicert_version` | `0.1.0.dev0` | package version |

### 1. Hardware (host + GPU) — `system.json`

| Field | Source |
|---|---|
| GPU name, device index, bus id | torch / nvidia-smi |
| VRAM total, free-at-start, used-at-start | torch.cuda.mem_get_info |
| GPU driver version, CUDA driver version | nvidia-smi |
| compute capability (sm_120) | torch.cuda.get_device_capability |
| power cap W, temperature C at start | nvidia-smi / pynvml |
| GPU util % at start, power draw W at start | nvidia-smi / pynvml |
| CPU model, cores, threads, max freq | platform/os |
| RAM total, available at start | os / psutil-if-present |
| Disk free at start (weights cache, results dir) | shutil.disk_usage |
| OS/kernel (container), docker image tag + digest | os.release, image env |
| container flag (true for Docker runs) | env probe |
| full nvidia-smi JSON snapshot (all GPUs) | nvidia-smi |

### 2. Software environment — `env.json` (the pins)

| Field |
|---|
| python version + build |
| torch version + cuda build tag (`2.13.0+cu130`) + arch list |
| transformers, peft, accelerate, safetensors, sentencepiece, huggingface_hub |
| numpy, scipy, pytest |
| NumPy amplification-model parameters (IQAE scoring) — CUDA-Q pinned in the container for Phase-2, not used in Phase-1 |
| CUDA toolkit runtime version reachable in the runtime (nvml) |
| bitsandbytes presence (must be absent in v1 — record if it ever appears) |
| attn_implementation (sdpa/eager/flash_attention_2) availability |
| pip freeze (full dump), environment.yml hash, Dockerfile digest |
| cudnn version if reported by torch |

### 3. Model / backbone — part of `config.json`

| Field |
|---|
| backbone id + HF repo + checkpoint version + license string (recorded at download) |
| total / trainable / frozen params, % trainable |
| dtype (bf16/fp16/fp32), load-time VRAM peak |
| vision tower(s) (DINOv2, SigLIP), LLM backbone (Qwen2.5-0.5B), action head (VQ) |
| chunk size, image size(s), history length |
| LoRA config: r, alpha, dropout, target_modules, adapter name |
| quantization: none (v1) / INT8 method used for baseline rows |
| processor/tokenizer: name, padding, max_length |
| per-component load times (vision / LLM / head) |

### 4. Data / task — part of `config.json`

| Field |
|---|
| dataset id, split, version, license (exact string), download timestamp |
| slice definition (task subset / episode filter) — the exact filter that selects rows |
| n_train / n_val / n_test instances, n_episodes |
| shuffle/split seed |
| sequence stats: obs length, action dim, chunk size, image shape, dtype |
| normalization constants (mean/std) actually applied |
| num_workers, prefetch_factor, persistent_workers |
| data load time (first epoch), cache size |

### 5. Training configuration — part of `config.json` (resolved, no defaults hidden)

| Field |
|---|
| optimizer (AdamW): lr, betas, eps, weight_decay |
| scheduler (cosine): warmup_steps/ratio, min_lr, total_steps |
| batch: per-device, grad_accum, effective batch |
| epochs, max_steps |
| mixed precision: autocast dtype, GradScaler on/off |
| grad clipping max_norm, gradient checkpointing |
| loss: criterion, per-component weights, label smoothing |
| eval: interval, metrics, eval batch, eval seed |
| checkpointing: save_steps, keep_last, resume behavior |
| determinism flags (torch.use_deterministic_algorithms, cudnn.benchmark) |
| device placement, pin_memory, non_blocking |

### 6. Training dynamics — `metrics.jsonl`, one JSON line per logged step/interval

| Field | Frequency |
|---|---|
| step, global_step, epoch, wall_sec | every line |
| loss total + per-component (action, language, aux) | every step |
| grad_norm, param_norm | every step |
| per-layer grad norms (summary: max/mean per layer group) | every K steps |
| lr (current value) | every step |
| throughput: steps/s, samples/s, tokens/s | every step |
| VRAM current + peak, GPU util %, power draw W, temperature | every N seconds |
| cumulative energy kWh (power sampled × dt) | accumulated |
| eval metrics at interval: task success, action accuracy, val loss | every eval |
| samples seen, epochs elapsed, data-loading stall seconds | every step |
| NaN/inf events, grad spikes (value + step) | on occurrence |
| checkpoint events: path, size, timestamp | on save |
| warnings/errors captured with step tag | on occurrence |

### 7. Results / outcomes — `run.json` `results` block + bench table

| Field |
|---|
| final eval: per-task + aggregate task success (LIBERO), action MSE, val loss |
| per-seed results → mean ± std across ≥3 seeds + 95% CI |
| latency per inference step (decode), VRAM at inference, power at inference |
| INT8 comparison: accuracy delta at matched ratio (baseline rows) |
| certificate outputs (N3+): per-layer Lipschitz, tight norms, tightness κ, safe-set %, per-family pruning predicted vs measured |
| cross-seed variance decomposition (between-seed std, within-run noise) |

### 8. Resource & energy

| Field |
|---|
| wall-clock per phase (load / data / train / eval / save) |
| GPU energy kWh (sampled power × dt), CPU energy if measurable |
| energy per metric unit (e.g., kWh per success point) |
| CO₂e at declared grid intensity (record the intensity value used) |
| total GPU-h (quota accounting), session time, queue time |

### 9. Reproducibility cross-check (written at finalize)

| Field |
|---|
| resolved config (defaults expanded) + config_hash |
| git commit(s) + dirty flags + repo URLs |
| pip freeze dump + environment.yml hash + docker image digest |
| exact CLI invocation + args |
| dataset version/license/URL + download timestamp |
| canonical hardware string |
| tolerance note (what regeneration must match, e.g., "within 1e-6 for certificates, 1 point for success rate") |

---

## Cheap extras (`--capture extra`) — negligible cost, still valuable

- Per-layer grad norm summaries every K steps (bucketed by module group)
- Weight/activation histogram snapshots, downsampled (every ~500 steps, 16 bins)
- Forward/backward time breakdown per module, sampled (lightweight profiler)
- Attention statistics on one sample batch at interval (proxy scalars only)
- Data-loader throughput vs compute time split
- I/O stats: bytes read, cache hits, download sizes
- Checkpoint size + save time per save
- Top-k hardest samples (indices + loss only) per interval
- Optimizer internal state summary (mean/std of Adam m and v per layer group)
- Per-step learning-rate curve saved as raw array (for exact replotting)
- Any stderr warning lines with step tags

## Heavy opt-ins (`--capture heavy`) — real cost, only when explicitly wanted

- Full per-sample loss arrays per epoch
- Full optimizer state dumps at checkpoints
- torch.profiler full traces (exported Chrome trace)
- TensorBoard/MLflow-style event streams (converted to JSONL by recorder)
- Loss-surface samples (perturbed-weight evals) for landscape plots
- Per-layer weight snapshots at interval (in addition to normal checkpoints)
- Sampled input/output pairs (images + text + actions), capped and license-checked

## Explicitly NOT captured (and why)

| Data | Reason |
|---|---|
| Raw model weights beyond the checkpoint policy | disk; fully regenerable from config+code+seed |
| Raw video/image frames | license + privacy + disk; regenerable from dataset by seed |
| Full attention matrices | quadratic disk; only summary stats are useful post-hoc |
| Intermediate activations for every step | disk; regenerable on demand from checkpoints |
| MC rollout transcripts (safety) | only verdicts + counts stored; transcripts regenerable via seed |
| Full stdout/stderr of every process | only tagged warnings/errors; full logs are ephemeral |
| Sample texts/images beyond the capped set | avoids data leakage into artifacts |
| Anything outside the experiment scope | artifacts stay clean and auditable |

---

## The no-re-run guarantee (how this stays true)

1. `config.json` (resolved, hashed) + `git_commit` + `seed_spec` = exact reproduction triple.
2. `metrics.jsonl` is append-only and covers the full trajectory — any plot is a 5-line script away.
3. `system.json` + `env.json` + ledger row tie every number to hardware, pins, and the pre-registered experiment row.
4. If a field is missing, it's added to the *register and the recorder*, and the run is re-launched — never silently backfilled from memory.

**Ledger rule:** a bench module that runs a real experiment MUST open a recorder
(`bench._base.start_run`) and close it (`finish_run`); the machine ledger
`results/ledger.csv` is the audit trail, and `research/experiments.md` mirrors it
for humans. No recorder, no table.
