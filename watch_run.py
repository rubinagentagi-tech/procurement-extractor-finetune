"""Overnight babysitter for the Colab fine-tune.

Polls the Colab tab over CDP until one of these outcomes, writes a durable report
to run-artifacts/, prints the reason to stdout (which Hermes relays as a
notification), and exits with a distinct code:

  exit 0  DONE      - run finished, eval table captured
  exit 2  FAILED    - training errored (OOM / EXIT: 1 / Error:)
  exit 3  STALLED   - no new cell output for STALL_MIN minutes
  exit 4  PAGE_LOST - could not reach the Colab tab (browser/runtime gone)
  exit 5  TIMEOUT   - hit the overall cap
"""
import datetime, json, os, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdp import CDP

ART = os.path.expanduser("~/procurement-ft-trial/run-artifacts")
LOG = "/tmp/colab-watch.log"
CELLS = (5, 6, 7, 9)
POLL_S = 45
STALL_MIN = 25
CAP_HOURS = 6


def ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


def log(msg, also_print=False):
    line = f"{ts()} {msg}"
    os.makedirs(ART, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")
    if also_print:
        print(line, flush=True)


def read_cells(c):
    out = {}
    for idx in CELLS:
        try:
            out[idx] = c.js(f"""(() => {{
              const cl = document.querySelectorAll('.cell')[{idx}];
              if (!cl) return '';
              const o = cl.querySelector('.output_area, .output');
              return o ? o.innerText : '';
            }})()""") or ""
        except Exception as e:
            out[idx] = f"<error {e}>"
    running = c.js("document.querySelectorAll('.running').length")
    page = ""
    try:
        page = (c.js("document.body.innerText") or "")
    except Exception:
        pass
    return out, int(running or 0), page


BROKEN_MARKERS = [
    "Session crashed",
    "You are not currently eligible for a GPU",
    "Cannot connect to GPU backend",
    "Could not connect to a runtime",
    "This runtime could not be started",
]


def chromium_alive():
    return subprocess.run(["pgrep", "-f", "remote-debugging-port=9333"],
                          capture_output=True).returncode == 0


def finish(status, out, extra="", code=0):
    os.makedirs(ART, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report = os.path.join(ART, f"{status.lower()}_{stamp}.json")
    with open(report, "w") as f:
        json.dump({"status": status, "when": stamp, "extra": extra, "cells": out}, f, indent=1)
    dl = os.path.expanduser("~/Downloads/colab-trial-results.zip")
    size = os.path.getsize(dl) if os.path.exists(dl) else 0
    txt = os.path.join(ART, f"{status.lower()}_{stamp}.txt")
    with open(txt, "w") as f:
        f.write(f"STATUS: {status}\nWHEN: {stamp}\nEXTRA: {extra}\nRESULTS ZIP: {dl} "
                f"({'%.1f MB' % (size/1e6) if size else 'NOT PRESENT'})\n\n")
        for k in CELLS:
            f.write(f"\n===== CELL {k} =====\n{out.get(k,'')}\n")
    print(f"STATUS={status} extra={extra} report={txt} zip={'%.1fMB' % (size/1e6) if size else 'missing'}",
          flush=True)
    sys.exit(code)


start = time.time()
os.makedirs(ART, exist_ok=True)
open(LOG, "a").write(f"\n{ts()} watcher start\n")

last_len = -1
last_change = time.time()
page_fails = 0

while (time.time() - start) / 3600 < CAP_HOURS:
    try:
        c = CDP()
        out, running, page = read_cells(c)
        c.close()
        page_fails = 0
    except Exception as e:
        page_fails += 1
        log(f"poll error #{page_fails}: {e}")
        if page_fails >= 3:
            if not chromium_alive():
                finish("PAGE_LOST", {}, f"chromium gone: {e}", 4)
            finish("PAGE_LOST", {}, f"cdp unreachable, chromium alive: {e}", 4)
        time.sleep(POLL_S)
        continue

    lengths = {k: len(out.get(k, "")) for k in CELLS}
    total = sum(lengths.values())
    log(f"running={running} lens={lengths}")

    if total != last_len:
        last_len, last_change = total, time.time()

    c5 = out.get(5, "")
    # --- failure signals -------------------------------------------------
    if "OOM at batch_size" in c5 or "CUDA out of memory" in c5:
        finish("FAILED", out, "OOM", 2)
    if "EXIT: 1" in c5:
        finish("FAILED", out, "training exit 1", 2)
    if running == 0 and "EXIT:" in c5 and "Error" in c5:
        finish("FAILED", out, "training error", 2)

    # --- success ---------------------------------------------------------
    if "Downloaded: colab-trial-results.zip" in out.get(9, ""):
        finish("DONE", out, "results zip downloaded", 0)
    if "EXIT: 0" in c5 and running == 0 and len(out.get(7, "")) > 400:
        finish("DONE", out, "training + eval completed", 0)

    # --- runtime/UI problems --------------------------------------------
    hit = [m for m in BROKEN_MARKERS if m in page]
    if hit:
        finish("BROKEN_RUNTIME", out, f"page shows: {hit}", 6)

    # --- stall -----------------------------------------------------------
    if running > 0 and (time.time() - last_change) / 60 > STALL_MIN:
        finish("STALLED", out, f"no output growth for {STALL_MIN} min", 3)

    time.sleep(POLL_S)

c = None
finish("TIMEOUT", {}, f"cap {CAP_HOURS}h reached", 5)
