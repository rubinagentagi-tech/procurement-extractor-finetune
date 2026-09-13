# Rented-GPU runbook (RunPod) — quote-reader fine-tune

Written after the Colab run was killed at 60% on 2026-09-12. Every known failure mode from
that run is addressed here; the ones that cannot be eliminated are named explicitly.

## Why Colab failed (post-mortem)

| Fact | Detail |
|---|---|
| Killed at | **19:42 EDT**, session age 5.4 h — under half of the 12 h allowance |
| Colab's own message | "runtime has been disconnected due to inactivity or reaching its maximum duration" |
| Actual cause | dynamic free-tier GPU quota, not duration |
| Training reached | step 150 of 250 (60%) |
| What we lost | 50 steps ≈ 1.75 h of T4 compute |
| What we kept | `checkpoints/checkpoint-100` — global_step 100, 112 LoRA tensors, optimizer/scheduler state, 131 MB |
| Detection failure (mine) | I read "is it alive?" from the same browser page that had died, so 50 min of stale DOM read as live progress |
| Contingency failure (mine) | the fallback needed an API key that did not exist on the laptop |

Lesson applied: **detection must not depend on the thing being monitored, and the
contingency must be pre-credentialed.**

## Run sequence

```
# 1. once
mkdir -p ~/.config/runpod && printf '%s' 'KEY' > ~/.config/runpod/api_key && chmod 600 ~/.config/runpod/api_key
python3 runpod_ft.py preflight      # key valid? balance? anything already billing?

# 2. rent, ship, launch (1 epoch: finishes the epoch the checkpoint is inside)
python3 runpod_ft.py start
python3 runpod_ft.py upload          # configs, data, pod_job.sh, output/checkpoint-100
python3 runpod_ft.py run --epochs 1 --resume checkpoint-100 --max-hours 3
python3 runpod_ft.py tail            # live progress
python3 runpod_ft.py verify          # proves the resume took, artifacts exist

# 3. collect and close the meter
python3 runpod_ft.py fetch           # -> run-artifacts/runpod/
python3 runpod_ft.py status
python3 runpod_ft.py terminate       # stop storage billing too
```

## The three safety layers (why a failure cannot bill for 15 h)

1. **Wall-clock hard stop, armed on the pod before any work starts.** `sleep MAX_HOURS*3600`
   in a detached subshell: kills the training by PID and stops the pod even if everything
   else in the script has died. RunPod's own docs recommend exactly this pattern.
2. **Progress watchdog.** If the tqdm step counter does not advance for `STALL_MIN` (20 min),
   training is killed, whatever exists is packaged, and the pod stops.
3. **Exit trap.** However `pod_job.sh` ends — success, error, `kill` — it stops its own pod.

Plus an **independent laptop-side cron** (every 15 min, silent unless it acts): stops any pod
that has been RUNNING past 4 h, and reports finished/failed pods. This covers the case where
the pod-side layers themselves are broken.

## Residual risks (cannot be engineered away — name them, watch them)

| Risk | Impact | Mitigation in place |
|---|---|---|
| Balance hits $0 mid-run | **pod without a network volume is TERMINATED and data is unrecoverable** | top up $10 for a ~$1-2 job; artifacts also land on `/workspace` |
| Stopped volume keeps billing | 40 GB × $0.20/GB/mo ≈ **$8/mo** if left stopped | `fetch` then `terminate` as soon as artifacts are pulled |
| Restart may come back with zero GPUs | rare; only matters if we restart to fetch | fetch right after the run; a CPU pod is fine for `scp` |
| 4090 unavailable in all listed fallbacks | pod not created | retry later; GPU list has 6 fallbacks |
| Host failure mid-run | pod gone | checkout at `save_steps: 50` bounds the loss to ≤50 steps |
| Resume silently restarting at step 0 | pays ~1 extra hour | `verify` greps train.log for the resume line and prints checkpoint steps |
| Volume disk cleared on stop | anything outside `/workspace` lost | `HF_HOME=/workspace/hf`; all outputs under `/workspace/ft` |

## Cost and time expectation (measured baseline, not hope)

- Colab T4: 126 s/step at `max_length 1024`, batch 1 × accum 8, full gradient checkpointing.
- Rented 4090/A100, batch pinned to 4 × accum 2 (same effective batch, so **steps still 250/epoch**
  and the resume at step 100 stays valid): expect ~15-30 s/step.
- Remaining work: **150 steps** to finish epoch 1 (loss was already 0.0002 at step 100 —
  a second epoch adds little, so run 1 epoch and let the eval decide).
- Wall clock: ~10 min setup + 40-75 min training + ~15-25 min eval ≈ **1-1.5 h**.
- Cost at $0.74/h (Secure 4090): **≈ $1-1.5**. Worst case inside the 3 h cap: **≈ $2.5**.
- If the 4090 is slower than expected the 3 h wall-clock cap stops everything regardless.
