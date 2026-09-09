#!/usr/bin/env python3
"""Eval: base model vs fine-tuned model on 100 held-out synthetic quotes.

Metrics: (1) % responses that are valid JSON, (2) exact-match of quote_no
and subtotal, (3) average item-count agreement. Run on the GPU box after
training (4-bit loads both models; no post-training merge needed to eval).

Usage:
    python eval_check.py --base Qwen/Qwen2.5-7B-Instruct --adapter ./output
    python eval_check.py --base Qwen/Qwen2.5-7B-Instruct          # baseline only
"""
import argparse, json, re, sys

def load_quotes(path, n=100):
    rows = []
    for line in open(path):
        r = json.loads(line)
        rows.append({"input": r["input"], "truth": json.loads(r["output"])})
        if len(rows) >= n:
            break
    return rows

def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def metric(model, tokenizer, quotes):
    ok_json = 0; ok_quote = 0; ok_sub = 0; ok_items = 0
    for q in quotes:
        msgs = [{"role": "user", "content": q["input"]}]
        t = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(t, return_tensors="pt").input_ids.to(model.device)
        out = model.generate(ids, max_new_tokens=512, do_sample=False, temperature=None, top_p=None)
        text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        pred = parse_json(text)
        if pred is None:
            continue
        ok_json += 1
        if pred.get("quote_no") == q["truth"]["quote_no"]:
            ok_quote += 1
        if abs((pred.get("subtotal") or 0) - (q["truth"]["subtotal"] or 0)) < 0.01:
            ok_sub += 1
        if pred.get("items") and len(pred["items"]) == len(q["truth"]["items"]):
            ok_items += 1
    n = len(quotes)
    print(f"  valid JSON      : {ok_json:3d}/{n}")
    print(f"  quote_no match  : {ok_quote:3d}/{n}")
    print(f"  subtotal match  : {ok_sub:3d}/{n}")
    print(f"  item count match: {ok_items:3d}/{n}")
    return ok_json

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--val", default="./data/val.jsonl")
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    quotes = load_quotes(a.val, a.n)
    print("== baseline:", a.base)
    m0 = AutoModelForCausalLM.from_pretrained(a.base, quantization_config=bnb, device_map="auto")
    metric(m0, tok, quotes)
    del m0; torch.cuda.empty_cache()
    if a.adapter:
        print("== fine-tuned:", a.adapter)
        m1 = AutoModelForCausalLM.from_pretrained(a.base, quantization_config=bnb, device_map="auto")
        from peft import PeftModel
        m1 = PeftModel.from_pretrained(m1, a.adapter)
        m1.eval()
        metric(m1, tok, quotes)
