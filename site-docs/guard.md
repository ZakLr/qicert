# Runtime guard

Three checks, enforced at deployment — not in the paper alone.

## Load-time manifest check

`guard.check_certs_before_serve(run_dir)` verifies:

1. The manifest `manifest.json` hashes to its recorded self-hash
   (tampered manifest → refuse).
2. Every listed artifact `cert*/*lipschitz*/*robustness*` exists and
   hashes to the recorded value (missing/corrupted → refuse).
3. Every recorded bound is finite (NaN/Inf → refuse).

No manifest → refuse. No certificate artifacts → serve with a safe default.

## Per-step action gate

`guard.action_in_certified_set(action, reference, margin, n_actions)`
decides `|action - reference| ≤ margin` and `0 ≤ action < n_actions`.
The claim proved is: nothing out-of-ball is accepted; nothing in-ball and
in-range is rejected; a negative margin accepts nothing. A demo transcript
(5 cases: clean load SERVE, out-of-ball REFUSE, tampered REFUSE, NaN
REFUSE, missing-manifest REFUSE) ships in `results/guard-demo/`.

## Output-diversity entropy gate

`guard.ActionDiversityMonitor(window=64, min_entropy_bits=1.0)` watches the
action stream itself. Motivation (measured on this backbone): deep tensor
compression collapses the model to a single modal action token — a
certificate on such a model is sound but vacuous, because it never strays.
The monitor flags degenerate streams:

- Sliding window entropy over served tokens (bits); warn when below threshold
  for a full window.
- **Warn, never refuse** — a robot legitimately holding position repeats one
  token; refusing there would break correct behavior. The flag tells the
  fallback controller the certificate currently attests to a degenerate policy.
- `n_unique`, `entropy_bits`, `reset()` on episode boundary.

Tests cover the measured collapse signature (identical-token streams) and the
warn-vs-refuse design choice.

```python
from qicert.certify.guard import ActionDiversityMonitor
mon = ActionDiversityMonitor(window=64, min_entropy_bits=1.0)
verdict = mon.update(action_token)  # {"status": "ok|warn", ...}
```
