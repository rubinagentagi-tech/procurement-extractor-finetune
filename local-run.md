# Running this fine-tune on the Omarchy laptop (GTX 1050 Ti, 4 GB)

Verified facts about the machine (2026-09-12):

| | Value |
|---|---|
| GPU | NVIDIA GeForce GTX 1050 Ti, **4096 MiB**, compute capability **6.1** (Pascal) |
| Driver | 580.178.04 (CUDA 13 capable) |
| RAM / swap | 15 GiB / 30 GiB |
| Free disk | ~74 GB on `/home` |
| Python | 3.11.16 ✓ (Soup needs 3.10–3.12) |

## Verdict

- **7B is not possible here.** QLoRA 4-bit Qwen2.5-7B weights alone are ~4.4 GB,
  more than the card has. No config fixes that; it is a capacity wall.
- **1.5B is comfortable.** `local-1.5b.yaml` — validated with
  `soup train --dry-run -c local-1.5b.yaml` → *Config valid. Ready to train!*
- **3B is a coin flip.** `local-3b.yaml` — works or OOMs depending on sequence
  length; drop `data.max_length` to 512 if it OOMs.

## Setup (one time, ~10 min, ~3 GB download)

```bash
cd ~/procurement-ft-trial
python3 -m venv .venv                      # already exists
.venv/bin/pip install "soup-cli[train]" accelerate bitsandbytes
nvidia-smi                                 # confirm the 1050 Ti is visible
```

## Run (1.5B, the recommended local job)

```bash
cd ~/procurement-ft-trial
.venv/bin/python -m soup_cli.cli train -c local-1.5b.yaml --yes   # → ./output-local
.venv/bin/python eval_check.py --base Qwen/Qwen2.5-1.5B-Instruct --adapter ./output-local
```

To keep a long run alive and watch it: `nohup ... > local-run.log 2>&1 &` then
`tail -f local-run.log`. Closing the lid does not suspend this laptop
(`keep-awake.service` holds `sleep:idle:handle-lid-switch`).

## Honest time estimates (estimated, not yet measured on this card)

FLOPs per sample ≈ 6 × params × tokens; at ~500 tokens/sample the 1050 Ti
(Pascal: no tensor cores, FP16 runs at 1/64 rate so training effectively runs
at FP32 speed) lands around 1 effective TFLOP/s.

| Job | Steps (batch 1 × accum 8) | Estimate |
|---|---|---|
| 1.5B, 2000 rows, 1 epoch | 250 | **3–5 h** |
| 3B, 2000 rows, 1 epoch | 250 | 6–10 h, VRAM-marginal |
| 7B | — | not possible |

Halve the wall clock by training on a subset, e.g. 1000 rows. The dataset is
deterministic (seed 42), so a subset is reproducible and documentable.

## Issues to expect

1. **Pascal is slow by design.** No tensor cores; FP16 throughput is 1/64 of
   FP32, so mixed precision buys nothing. Everything runs at FP32-ish speed.
   The T4 in Colab is an order of magnitude faster for the same 7B job.
2. **No bf16.** Force fp16/auto precision; bf16 is unsupported on this card.
3. **4 GB VRAM caps sequence length.** `max_length: 768` at batch 1 with
   gradient checkpointing fits for 1.5B; 2048 (Soup's default) does not. This
   is exactly the OOM that stopped the Colab run until it was set to 1024.
4. **bitsandbytes 4-bit on Pascal** works but its fast kernels target newer
   architectures, so dequantization overhead shows up as wall-clock time.
5. **Thermals.** A multi-hour GPU run on a G3 17 will throttle; expect the
   second half of a long run to be slower than the first.
6. **Don't run it while driving the Colab tab.** The browser, TV and the
   training all compete for CPU/RAM on a 15 GiB machine.
7. **The eval costs real time too** — loading base + adapter and generating on
   100 samples adds ~30–60 min locally.
8. **Portfolio framing:** the 7B T4 run (Colab) is the flagship artifact. The
   local run's value is different and worth stating plainly — *a 4 GB laptop
   can fine-tune a usable extractor with the data never leaving the machine*,
   which is the on-premise story for client work.

## What the laptop run does NOT prove

It is a different (smaller) model on a different dataset split than the 7B
Colab run. Report them as two separate artifacts with their own eval tables —
never merge the numbers.
