#!/usr/bin/env bash
# Self-limiting fine-tune job for the quote-reader adapter on a rented RunPod GPU.
#
# Design goal: a failure must NEVER bill indefinitely and must NEVER lose more than
# save_steps of work. Three independent safety layers:
#   1. hard wall-clock stop armed BEFORE any work starts
#   2. progress watchdog -- no tqdm advance for STALL_MIN minutes => kill training, package, stop
#   3. trap on exit -- however the script ends (success, error, kill), the pod gets stopped
#
# Everything durable lives under /workspace because RunPod CLEARS the container disk on
# stop and preserves only the volume disk (/workspace). HF_HOME therefore points at the
# volume so a restart never re-downloads the 15 GB base model.
set -uo pipefail

FT=/workspace/ft
OUT=$FT/output
ART=$FT/artifacts
LOG=$FT/job.log
MAX_HOURS=${MAX_HOURS:-3}
STALL_MIN=${STALL_MIN:-20}
EPOCHS=${EPOCHS:-1}
CKPT=${CKPT:-}                       # e.g. /workspace/ft/output/checkpoint-100
export HF_HOME=/workspace/hf         # on the volume, survives stop
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

mkdir -p "$ART" "$OUT" "$HF_HOME"
: > "$ART/SUMMARY.txt"
STOPPED=0

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

stop_pod() {
  local reason="$1"
  [ "$STOPPED" = "1" ] && return 0
  STOPPED=1
  log "STOPPING POD -- reason: $reason"
  {
    echo "stop_reason=$reason"
    echo "stopped_at=$(date -u +%FT%TZ)"
  } >> "$ART/SUMMARY.txt"
  if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    if command -v curl >/dev/null 2>&1; then
      curl -s -X POST "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" | head -c 200
    fi
    # fallback that does not depend on curl existing in the image
    python3 - <<'PYSTOP' 2>/dev/null || log "self-stop failed (both curl and urllib)"
import os, urllib.request
pid, key = os.environ.get("RUNPOD_POD_ID"), os.environ.get("RUNPOD_API_KEY")
req = urllib.request.Request(f"https://rest.runpod.io/v1/pods/{pid}/stop", method="POST",
                             headers={"Authorization": f"Bearer {key}"})
try:
    print(urllib.request.urlopen(req, timeout=30).read()[:200].decode())
except Exception as e:
    print("stop error:", e)
PYSTOP
  else
    log "WARN no RUNPOD_POD_ID/RUNPOD_API_KEY in env -- STOP THIS POD MANUALLY"
  fi
}
trap 'stop_pod "script-exit"' EXIT

log "job start | EPOCHS=$EPOCHS CKPT=${CKPT:-none} MAX_HOURS=$MAX_HOURS STALL_MIN=$STALL_MIN"

# ---- layer 1: hard wall-clock stop, armed before anything can hang -------------
# kills only the training this script started (PID file), stops the pod, and does not
# depend on any pattern matching that could hit an unrelated process
( sleep $(( MAX_HOURS * 3600 )); \
  echo "wallclock_cap=${MAX_HOURS}h" >> "$ART/SUMMARY.txt"; \
  if [ -f "$FT/train.pid" ]; then kill -TERM "$(cat "$FT/train.pid")" 2>/dev/null; sleep 20; \
    kill -9 "$(cat "$FT/train.pid")" 2>/dev/null; fi; \
  if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then \
    curl -s -X POST "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}/stop" \
      -H "Authorization: Bearer ${RUNPOD_API_KEY}" >/dev/null 2>&1 || true; fi ) &

# ---- preflight: fail cheap, before hours of GPU time ---------------------------
if ! nvidia-smi -L >/dev/null 2>&1; then log "FAIL no GPU visible"; stop_pod "no-gpu"; exit 1; fi
log "gpu: $(nvidia-smi -L | head -1)"
log "python: $(python3 -V 2>&1) | disk: $(df -h /workspace | tail -1 | awk '{print $4" free"}')"
python3 -c "import torch;print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),'|',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU')" \
  || log "WARN torch import/preflight odd"

python3 -m pip install -q --upgrade pip
# PIN the version validated on Colab: an unpinned install could bring a different CLI
# with different flags/config keys and fail immediately (the Colab lesson).
python3 -m pip install -q "soup-cli[train]==0.75.0" accelerate bitsandbytes hf_transfer || {
  log "FAIL pip install"; stop_pod "pip-install-failed"; exit 1; }

if ! python3 -c "import torch;assert torch.cuda.is_available()" 2>/dev/null; then
  log "torch lost CUDA after install -- reinstalling cu124 wheels"
  python3 -m pip install -q --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124
fi
python3 -c "import torch,bitsandbytes as bnb;print('after install: cuda',torch.cuda.is_available(),'| bnb',bnb.__version__)" \
  || { log "FAIL bitsandbytes/torch check"; stop_pod "cuda-check-failed"; exit 1; }

# ---- config -------------------------------------------------------------------
python3 - <<'PY' || { echo "config edit failed"; exit 1; }
import json, os, yaml
p = "/workspace/ft/soup.yaml"
cfg = yaml.safe_load(open(p))
t = cfg.setdefault("training", {})
t["epochs"] = int(os.environ.get("EPOCHS", "1"))
t["save_steps"] = 50            # only save_* key that exists in this Soup schema
# PIN the batch math. On the T4 Soup chose batch_size=1 with accum=8 => 2000/8 = 250
# steps per epoch, and the rescued checkpoint is step 100 of 250. On a 24 GB card
# batch_size:auto would pick a bigger micro-batch, changing steps-per-epoch, and a resume
# at step 100 into a shorter epoch is invalid (HF raises or the run ends immediately).
# Keep the effective batch identical (4 x 2 = 8) so the step numbering still matches,
# while using the bigger card for real throughput.
t["batch_size"] = 4
t["gradient_accumulation_steps"] = 2
t.setdefault("gradient_checkpointing", "auto")
cfg["output"] = "./output"
yaml.safe_dump(cfg, open(p, "w"))
try:
    rows = sum(1 for _ in open(cfg["data"]["train"]))
except Exception:
    rows = 2000
eff = t["batch_size"] * t["gradient_accumulation_steps"]
print("config: epochs=%s steps/epoch=%d eff_batch=%d max_length=%s base=%s" % (
    t["epochs"], -(-rows // eff), eff, cfg["data"]["max_length"], cfg["base"]))
PY

cd "$FT" || { log "FAIL cd"; stop_pod "cd-failed"; exit 1; }
RESUME_ARG=""
if [ -n "$CKPT" ] && [ -f "$CKPT/trainer_state.json" ]; then
  RESUME_ARG="--resume $CKPT"
  log "resuming from $CKPT ($(python3 -c "import json;print('step',json.load(open('$CKPT/trainer_state.json'))['global_step'])" 2>/dev/null))"
fi

# ---- training, detached, with the progress watchdog ---------------------------
log "training start: soup train $RESUME_ARG --yes"
nohup python3 -m soup_cli.cli train -c soup.yaml $RESUME_ARG --yes > train.log 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$FT/train.pid"
START=$(date +%s)
LAST_STEP=""; LAST_CHANGE=$(date +%s)
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 60
  STEP=$(grep -oE '[0-9]+/[0-9]+ \[' train.log 2>/dev/null | tail -1 | cut -d/ -f1)
  NOW=$(date +%s)
  if [ -n "$STEP" ] && [ "$STEP" != "$LAST_STEP" ]; then LAST_STEP="$STEP"; LAST_CHANGE=$NOW; fi
  if [ $(( NOW - LAST_CHANGE )) -gt $(( STALL_MIN * 60 )) ]; then
    log "STALL: no step progress for ${STALL_MIN} min (last step ${LAST_STEP:-none}) -- killing training"
    echo "stall_killed=1 last_step=${LAST_STEP:-none}" >> "$ART/SUMMARY.txt"
    kill -TERM "$TRAIN_PID" 2>/dev/null; sleep 25; kill -9 "$TRAIN_PID" 2>/dev/null
    break
  fi
  if [ $(( NOW - START )) -gt $(( MAX_HOURS * 3600 )) ]; then
    log "wall clock cap hit inside watchdog -- killing training"
    kill -TERM "$TRAIN_PID" 2>/dev/null; sleep 15
    break
  fi
done
wait "$TRAIN_PID" 2>/dev/null
echo "TRAIN_EXIT=$?" >> train.log
if [ -n "$RESUME_ARG" ]; then
  if grep -qiE "resum(e|ing) (training )?from|resume_from_checkpoint|continuing training" train.log; then
    echo "resume_confirmed=yes" >> "$ART/SUMMARY.txt"
    log "resume confirmed in train.log"
  else
    echo "resume_confirmed=NO (checkpoint given but no resume line found)" >> "$ART/SUMMARY.txt"
    log "WARN checkpoint was given but train.log shows no resume line -- check verify output"
  fi
fi
log "training ended | $(grep -oE '[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+, *[0-9.]+s/it\]' train.log | tail -1)"

# ---- eval ---------------------------------------------------------------------
log "eval start (base vs adapter, 100 held-out quotes)"
python3 eval_check.py --base Qwen/Qwen2.5-7B-Instruct --adapter ./output > eval.log 2>&1
echo "EVAL_EXIT=$?" >> eval.log
log "eval done"

# ---- package artifacts on the volume ------------------------------------------
cp "$LOG" "$ART/" 2>/dev/null
cp train.log eval.log "$ART/" 2>/dev/null
# keep the tarball small: adapter + tokenizer + logs. Full checkpoints (with optimizer.pt,
# ~80 MB each) stay on the volume for resume and are not needed for the deliverable.
tar czf "$ART/results.tar.gz" -C /workspace/ft \
    output/adapter_config.json output/adapter_model.safetensors output/README.md \
    output/tokenizer.json output/tokenizer_config.json output/vocab.json output/merges.txt \
    output/added_tokens.json output/chat_template.jinja output/special_tokens_map.json \
    train.log eval.log 2>/dev/null
ls -la "$ART/results.tar.gz" 2>/dev/null || log "WARN tarball not created"
{
  echo "finished_at=$(date -u +%FT%TZ)"
  echo "epochs_requested=$EPOCHS"
  echo "resume_arg=${RESUME_ARG:-none}"
  echo "last_progress_line=$(grep -oE '[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+, *[0-9.]+s/it\]' train.log | tail -1)"
  echo "---- eval ----"
  tail -8 eval.log 2>/dev/null
  ls -la output/ | tail -8
} >> "$ART/SUMMARY.txt"
log "artifacts in $ART:"; ls -la "$ART"
