# Resource Declaration

_Generated 2026-09-11T14:25:46+00:00_

Per AGENTS.md: every claimed number must be traceable to a
recorded run with its full resource cost. This table IS that
trace, regenerated from `results/ledger.csv`.

## Machine

- Host: Zaki (Windows 11)
- CPU: AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- GPU: Tesla T4

## Scored runs (status=completed)

| experiment | run_id | seed | wall_sec | wall_min | energy_kwh | gpu | date |
|---|---|---|---|---|---|---|---|
| STEP1 | de6171520a0d | 0 | 213.8 | 3.6 | 0.0 | NVIDIA GeForce RTX 5060 Laptop GPU | 2026-08-16 |
| N2prime | 178c849b6525 | 0 | 18.1 | 0.3 | 0.0 | NVIDIA GeForce RTX 5060 Laptop GPU | 2026-08-16 |
| N3 | dec12aad89e5 | 0 | 43.3 | 0.7 | 0.0 |  | 2026-08-16 |
| N1 | a64ef93422ac | 0 | 671.7 | 11.2 | 0.0 | NVIDIA GeForce RTX 5060 Laptop GPU | 2026-08-16 |
| N1 | ad0d6b7e1d60 | 0 | 1175.7 | 19.6 | 0.0 | Tesla T4 | 2026-08-16 |
| N1 | f65cc0d9a214 | 0 | 1175.7 | 19.6 | 0.0 | Tesla T4 | 2026-08-16 |
| N1 | 93e4739c8baf | 0 | 1133.6 | 18.9 | 0.0 | Tesla T4 | 2026-08-16 |
| N1 | 2f06f2600d17 | 0 | 9152.8 | 152.5 | 0.0 | NVIDIA GeForce RTX 5060 Laptop GPU | 2026-08-16 |
| N1 | 486f7fdcc2f3 | 0 | 277.9 | 4.6 | 0.0 | NVIDIA GeForce RTX 5060 Laptop GPU | 2026-08-16 |
| N2 | dc5715f6a3f5 | 0 | 4.7 | 0.1 | 0.0 | NVIDIA GeForce RTX 5060 Laptop GPU | 2026-08-16 |
| N2 | 9f6f5e61d06b | 0 | 2.0 | 0.0 | 0.0 | NVIDIA GeForce RTX 5060 Laptop GPU | 2026-08-16 |
| N2 | cb296fa62e15 | 0 | 2.0 | 0.0 | 0.0 | NVIDIA GeForce RTX 5060 Laptop GPU | 2026-08-16 |
| N2 | 2a3343003fb8 | 0 | 2.2 | 0.0 | 0.0 | NVIDIA GeForce RTX 5060 Laptop GPU | 2026-08-16 |
| N3 | 15ceb3e7f716 | 0 | 529.0 | 8.8 | 0.0 | NVIDIA GeForce RTX 5060 Laptop GPU | 2026-08-25 |
| E6-latency | 9a63e288d766 | 0 | 25.8 | 0.4 | 0.0 | NVIDIA GeForce RTX 5060 Laptop GPU | 2026-08-25 |
| E7-int8-ablation | 0f8243dc1157 | 0 | 192.5 | 3.2 | 0.0 | NVIDIA GeForce RTX 5060 Laptop GPU | 2026-08-25 |
| E5-residual-pilot | 60b0229aff3b | 0 | 173.5 | 2.9 | 0.0 | NVIDIA GeForce RTX 5060 Laptop GPU | 2026-08-25 |
| N1local | dcff6386561f | 0 | 1851.2 | 30.9 | 0.0 | NVIDIA GeForce RTX 5060 Laptop GPU | 2026-08-25 |
| N2local | 8b174b3328d3 | 0 | 3.7 | 0.1 | 0.0 | NVIDIA GeForce RTX 5060 Laptop GPU | 2026-08-25 |
| N2local | e2c885f374bd | 0 | 2.0 | 0.0 | 0.0 | NVIDIA GeForce RTX 5060 Laptop GPU | 2026-08-25 |
| N2local | 5162e0b631ef | 0 | 1.8 | 0.0 | 0.0 | NVIDIA GeForce RTX 5060 Laptop GPU | 2026-08-25 |
| N2local | 1cea5b82eb9a | 0 | 1.7 | 0.0 | 0.0 | NVIDIA GeForce RTX 5060 Laptop GPU | 2026-08-25 |
| N2pre | c8b438f6fdaf | 0 | 6110.3 | 101.8 | 0.0 | NVIDIA GeForce RTX 5060 Laptop GPU | 2026-08-25 |
| N1v2smoke | 6948757482b1 | 0 | 75.2 | 1.3 | 0.0 | NVIDIA GeForce RTX 5060 Laptop GPU | 2026-09-11 |
| N1v2smoke | 96897acd2ab6 | 0 | 99.9 | 1.7 | 0.0 | NVIDIA GeForce RTX 5060 Laptop GPU | 2026-09-11 |

**Total scored wall-clock: 6.37 GPU-h**

## Failed / aborted runs (excluded from claims, kept for provenance)

| experiment | run_id | seed | wall_sec | error (truncated) |
|---|---|---|---|---|
| N1 | 6d638b41119a | 0 | 41.35 | {"error": "ModuleNotFoundError: No module named 'tensorflow_graphics'", "seed": 0} |
| N1 | 44159113eb38 | 0 | 301.29 | {"error": "NameError: name 'DataLoader' is not defined", "seed": 0} |
| N1 | da7520c9e4ac | 0 | 241.1 | {"error": "AttributeError: 'dict' object has no attribute 'to'", "seed": 0} |
| N1 | 14c82f6e5a36 | 0 | 563.57 | {"error": "NameError: name 'np' is not defined", "seed": 0} |
| N1local | 49ee7631157e | 0 | 17.68 | {"error": "ModuleNotFoundError: No module named 'dlimp'", "seed": 0} |
| N1local | 943f9dc73ed9 | 0 | 211.64 | {"error": "TypeError: must assign iterable to extended slice", "seed": 0} |
| N1v2 | d8bfaacb296e | 0 | 44.1 | {"error": "ModuleNotFoundError: No module named 'transformers.models.qwen2.tokenization_qwen2_fast'" |
| N1v2 | 36d8e302d4e7 | 0 | 8.59 | {"error": "ModuleNotFoundError: No module named 'dlimp'", "seed": 0} |
| N1v2smoke | 310b98bb1063 | 0 | 8.76 | {"error": "ModuleNotFoundError: No module named 'dlimp'", "seed": 0} |
| N1v2 | a8580a23cfa6 | 0 | 10.96 | {"error": "ModuleNotFoundError: No module named 'tensorflow_graphics'", "seed": 0} |
| N1v2 | 74414f7d803a | 0 | 66.87 | {"error": "ImportError: FlashAttention2 has been toggled on, but it cannot be used due to the follow |
| N1v2 | b6fe73dceff3 | 0 | 37.83 | {"error": "ImportError: FlashAttention2 has been toggled on, but it cannot be used due to the follow |
| N1v2smoke | 3d0eb218c406 | 0 | 31.62 | {"error": "TypeError: PreTrainedModel._check_and_enable_flash_attn_2() got multiple values for argum |
| N1v2smoke | db89c32a0189 | 0 | 55.54 | {"error": "ValueError: Only instances of PreTrainedModel support `target_modules='all-linear'`", "se |
| N1v2smoke | ec463e45bee2 | 0 | 61.99 | {"error": "ImportError: cannot import name 'LinearActivationQuantizedTensor' from 'torchao.quantizat |
| N1v2smoke | 789204ae13bd | 0 | 65.01 | {"error": "NameError: name 'np' is not defined", "seed": 0} |
| N1v2smoke | 62e6af501feb | 0 | 109.96 | {"error": "UnboundLocalError: cannot access local variable 'i8_sec' where it is not associated with  |

## Software stack (from latest completed run's env.json)

_env.json not found for any completed run_
