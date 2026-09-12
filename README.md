# 🏭 Fine-tune a procurement extractor on a rented GPU (trial runbook)

A real, reproducible LLM fine-tune: teach **Qwen2.5-7B-Instruct** to turn a
supplier quotation (plain text) into **valid structured JSON** — with
anomaly flags — using QLoRA, driven by [Soup](https://github.com/MakazhanAlpamys/Soup).
Everything is **synthetic data**: no real company, quote, price or person.
Run on a rented cloud GPU, ~2-4 hours, ~$2-5.

## Why this task
Format reliability. Base models understand quotes but drift out of JSON and
miss the two anomalies this task plants (line-total math mismatches and
missing unit prices). A fine-tune on 2,000 examples teaches schema adherence
— measurable as "valid JSON %" before vs after.

## Files
```
generator.py     # deterministic synthetic dataset (2,000 train + 300 val)
data/train.jsonl # alpaca rows: instruction + quote text → JSON
data/val.jsonl
soup.yaml        # Qwen2.5-7B-Instruct, QLoRA r=32, 4-bit, 2 epochs
eval_check.py    # baseline vs fine-tuned: valid-JSON / quote_no / subtotal / items
```

## Cost estimate (honest)
| GPU | ~$/hr | Est. run (7B, 2k rows, 2 ep) | Total |
|---|---|---|---|
| RTX 3090 / 4090 (24 GB) | $0.45-1.00 | 1.5-3 h | **$2-4** |
| A10 / L4 / T4-16 GB | $0.35-0.60 | 3-6 h | $2-4 |
Figures are estimates; the box is deleted when done, so costs stop at the
last second.

## Provider steps (one-time)
1. **RunPod** (recommended) or **Vast.ai**: create account, add ~$10 credit.
2. Grab an **API key** (RunPod: Settings → API Keys) or a rented instance's
   SSH address.

## Run (on the rented box, CUDA image)
```bash
pip install "soup-cli[train]"
# copy this folder to the box, then:
soup train --config soup.yaml          # trains → ./output (LoRA adapter)
soup merge --adapter ./output          # optional: merge LoRA into base
pip install transformers accelerate peft bitsandbytes
python eval_check.py --base Qwen/Qwen2.5-7B-Instruct --adapter ./output
```
Expect the baseline to score ~60-80% valid JSON and the fine-tuned model
~95%+ — that before/after table is your proof.

## Privacy note
The data in this repo is fictional, so sending it to a cloud GPU is safe.
**Never** point this pipeline at real quotations or client documents: rented
GPUs are third-party infrastructure. Real data stays on your own machine.

## What comes out
- `output/adapter_model.safetensors` (LoRA, ~20 MB: 10.1M trainable params at fp16,
  0.13% of the 7.6B base) + training log with loss
- The eval table (baseline vs fine-tuned)
- Optional: merged full model, or GGUF export for local chat

All synthetic. All reproducible (`python3 generator.py` re-creates the exact
data; seed 42). MIT licensed.
