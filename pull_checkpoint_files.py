"""Pull Soup checkpoints off the Colab VM, file by file, via Colab's own file pane.

Verified mechanics (2026-09-12, in a 640px-wide Chromium window):
  * Colab's file-pane context menu has NO Download entry for FOLDERS -- only for files.
  * The menu opens ABOVE the clicked row, so the row must be scrolled low in the pane
    (`scrollIntoView({block:'end'})`) or the menu renders at negative y and cannot be clicked.
  * Chrome refuses scripted (untrusted) downloads: the Download item must be hit with a
    real CDP input click, not element.click().
  * Tree indentation encodes depth: children of a row sit after it with a larger x centre,
    until a row at the same or smaller x appears.
"""
import json, os, shutil, time, datetime

import cdp

DEST = os.path.expanduser("~/procurement-ft-trial/checkpoints")
DOWN = os.path.expanduser("~/Downloads")
LOG = "/tmp/checkpoint-pull.log"
REPO = "procurement-extractor-finetune"
TARGETS = ["checkpoint-100", "checkpoint-200"]
# Only these belong to a PEFT/HF Trainer checkpoint dir. The tree read can also pick up
# repo-level siblings that sit at the same indentation, so filter rather than trust it.
CKPT_FILES = {
    "adapter_config.json", "adapter_model.safetensors", "added_tokens.json",
    "chat_template.jinja", "merges.txt", "optimizer.pt", "rng_state.pth",
    "scheduler.pt", "special_tokens_map.json", "tokenizer.json",
    "tokenizer_config.json", "trainer_state.json", "training_args.bin", "vocab.json",
    "README.md",
}
CAP_HOURS = 7.0
PER_FILE_WAIT = 150  # seconds

os.makedirs(DEST, exist_ok=True)


def log(msg):
    line = f"{datetime.datetime.now().strftime('%H:%M:%S')} {msg}"
    with open(LOG, "a") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


class Pane:
    def __init__(self, c):
        self.c = c

    def rows(self):
        return json.loads(self.c.js("""(() => JSON.stringify(
          Array.from(document.querySelectorAll('.file-tree-name')).map(e => {
            const r = e.getBoundingClientRect();
            return [e.innerText.trim(), Math.round(r.x + r.width/2), Math.round(r.y + r.height/2)];
          })))()"""))

    def ensure_open(self):
        """Open Colab's file pane.

        Ground truth (verified 2026-09-12): the left rail is a column of `md-icon`s; the file
        browser is the one whose text is exactly `folder`, at x=36 (y varies with viewport
        height -- 360 in a 720-tall window). Clicking OTHER rail icons (vpn_key, table, ...)
        toggles unrelated panels and can close the pane again, so only ever click the folder
        icon and give the tree time to render.
        """
        if self.rows():
            return True
        for _ in range(3):
            pos = self.c.js("""(() => {
              const ic = Array.from(document.querySelectorAll('md-icon')).find(e => (e.innerText || '').trim() === 'folder');
              if (!ic) return null;
              const r = ic.getBoundingClientRect();
              return JSON.stringify([Math.round(r.x + r.width/2), Math.round(r.y + r.height/2)]);
            })()""")
            if not pos:
                time.sleep(3)
                continue
            x, y = json.loads(pos)
            self.c.click(x, y)
            for _ in range(4):
                time.sleep(2.5)
                if self.rows():
                    return True
        return bool(self.rows())

    def expand(self, name, expect_any=None):
        """Expand a directory row by clicking its label with a real CDP click.

        Ground truth (verified 2026-09-12): clicking the row LABEL toggles expansion
        (the `md-icon` reading arrow_drop_down/arrow_right is only an indicator; clicking
        the indicator itself does nothing, and fixed pixel offsets from the label toggle
        the wrong row). Children are detected by tree indentation: a child's label centres
        further right than its parent's.
        """
        for _ in range(4):
            kids = self.children(name)
            if kids and (expect_any is None or any(k in kids for k in expect_any)):
                return kids
            pos = self.c.js("""(() => {
              const el = Array.from(document.querySelectorAll('.file-tree-name'))
                .find(e => e.innerText.trim() === %s);
              if (!el) return null;
              el.scrollIntoView({block: 'center'});
              const r = el.getBoundingClientRect();
              return JSON.stringify([Math.round(r.x + r.width/2), Math.round(r.y + r.height/2)]);
            })()""" % json.dumps(name))
            if not pos:
                time.sleep(2)
                continue
            x, y = json.loads(pos)
            self.c.click(x, y)
            time.sleep(2.5)
        return self.children(name)

    def children(self, name):
        rows = self.rows()
        for i, (t, x, y) in enumerate(rows):
            if t == name:
                out = []
                for t2, x2, y2 in rows[i + 1:]:
                    if x2 <= x:
                        break
                    out.append(t2)
                return out
        return []

    def download(self, name):
        """Scroll the row low in the pane, right-click it, real-click the on-screen Download item."""
        pos = self.c.js("""(() => { const el = Array.from(document.querySelectorAll('.file-tree-name'))
            .find(e => e.innerText.trim() === %s);
          if (!el) return null;
          el.scrollIntoView({block: 'end'});
          const r = el.getBoundingClientRect();
          return JSON.stringify([Math.round(r.x + r.width/2), Math.round(r.y + r.height/2)]); })()"""
            % json.dumps(name))
        if not pos:
            return "row-missing"
        x, y = json.loads(pos)
        time.sleep(1.5)
        for ev in ("mousePressed", "mouseReleased"):
            self.c.send("Input.dispatchMouseEvent", type=ev, x=x, y=y, button="right", clickCount=1)
        time.sleep(2)
        items = json.loads(self.c.js("""(() => { const out = [];
          document.querySelectorAll('.goog-menuitem,[role=menuitem],.goog-menu *').forEach(e => {
            const r = e.getBoundingClientRect();
            if (r.height > 0 && (e.innerText || '').trim() === 'Download' && r.y > 0 && r.x > 0)
              out.push([Math.round(r.x + r.width/2), Math.round(r.y + r.height/2)]);
          });
          return JSON.stringify(out); })()"""))
        if not items:
            self.c.send("Input.dispatchKeyEvent", type="keyDown", key="Escape", code="Escape",
                        windowsVirtualKeyCode=27)
            return "no-on-screen-download-item"
        before = set(os.listdir(DOWN))
        self.c.click(*items[-1])
        waited = 0
        while waited < PER_FILE_WAIT:
            time.sleep(5)
            waited += 5
            new = [n for n in set(os.listdir(DOWN)) - before if not n.endswith(".crdownload")]
            if new:
                # a real download completed
                for n in new:
                    src = os.path.join(DOWN, n)
                    if os.path.getsize(src) == 0:
                        continue
                    return src
        return "download-timeout"


def fetched(tag):
    d = os.path.join(DEST, tag)
    return os.path.exists(os.path.join(d, "adapter_model.safetensors"))


start = time.time()
with open(LOG, "a") as fh:
    fh.write("\n--- file-level checkpoint puller start ---\n")
log("targets: " + ", ".join(TARGETS))

while (time.time() - start) / 3600 < CAP_HOURS:
    remaining = [t for t in TARGETS if not fetched(t)]
    if not remaining:
        log("all checkpoints captured — done")
        break
    tag = remaining[0]
    try:
        c = cdp.CDP()
        pane = Pane(c)
        if not pane.ensure_open():
            log("file pane unavailable; retry in 2 min")
            c.close()
            time.sleep(120)
            continue
        pane.expand(REPO)
        pane.expand("output")
        kids = [k for k in pane.expand(tag) if k in CKPT_FILES]
        if not kids:
            log(f"{tag}: not reachable yet; visible rows {[t for t, _, _ in pane.rows()][:10]}")
        log(f"{tag}: children {kids if kids else '(none yet)'}")
        if not kids:
            c.close()
            time.sleep(300)
            continue
        outdir = os.path.join(DEST, tag)
        os.makedirs(outdir, exist_ok=True)
        for f in kids:
            if os.path.exists(os.path.join(outdir, f)):
                continue
            res = pane.download(f)
            if res in ("download-timeout", "no-on-screen-download-item", "row-missing"):
                log(f"{tag}/{f}: {res}")
                break
            dst = os.path.join(outdir, f)
            shutil.move(res, dst)
            log(f"{tag}/{f}: {os.path.getsize(dst)/1e6:.2f} MB saved")
        c.close()
        if fetched(tag):
            log(f"{tag} COMPLETE ({sum(os.path.getsize(os.path.join(outdir, x)) for x in os.listdir(outdir))/1e6:.1f} MB)")
    except Exception as e:
        log(f"poll error: {type(e).__name__}: {e}")
    time.sleep(60)
else:
    log("hit the time cap")
