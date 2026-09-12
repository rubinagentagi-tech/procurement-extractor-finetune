"""Poll the Colab tab over CDP until the run finishes; log status and the final report."""
import json, time, datetime, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdp import CDP

LOG = "/tmp/colab-run-status.log"
FINAL = "/tmp/colab-run-final.txt"
CELLS = (5, 6, 7, 8, 9)
MAX_MIN = 170


def stamp():
    return datetime.datetime.now().strftime("%H:%M:%S")


def snapshot(c):
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
            out[idx] = f"<err {e}>"
    running = c.js("document.querySelectorAll('.running').length")
    return out, running


def done(out, running):
    """Run finished (or failed) when nothing is running and the last stages have spoken."""
    if running and int(running) > 0:
        return False
    if "Downloaded: colab-trial-results.zip" in out.get(9, ""):
        return True
    if "EXIT:" in out.get(5, "") and (out.get(7, "").strip() or out.get(9, "").strip()):
        return True
    if "EXIT:" in out.get(5, "") and "no output dir" in out.get(6, ""):
        return True
    return False


start = time.time()
while (time.time() - start) / 60 < MAX_MIN:
    try:
        c = CDP()
        out, running = snapshot(c)
        c.close()
        line = f"{stamp()} running={running} " + " ".join(
            f"c{k}={'yes' if out.get(k, '').strip() else 'no'}({len(out.get(k, ''))}b)" for k in CELLS)
        with open(LOG, "a") as f:
            f.write(line + "\n")
        if done(out, running):
            with open(FINAL, "w") as f:
                f.write(json.dumps(out, indent=1))
            with open(LOG, "a") as f:
                f.write(f"{stamp()} FINISHED\n")
            print("finished", flush=True)
            break
    except Exception as e:
        with open(LOG, "a") as f:
            f.write(f"{stamp()} poll error: {e}\n")
    time.sleep(60)
else:
    with open(LOG, "a") as f:
        f.write(f"{stamp()} watcher hit {MAX_MIN} min cap\n")
