"""Minimal CDP driver for the debug-enabled Chromium (port 9333) — Colab control."""
import base64, json, subprocess, sys, time

import requests
import websockets.sync.client as wsc

PORT = 9333


class CDP:
    def __init__(self, url_match="colab", port=PORT):
        tl = requests.get(f"http://127.0.0.1:{port}/json", timeout=10).json()
        target = None
        for t in tl:
            if t.get("type") == "page" and url_match in t.get("url", ""):
                target = t
                break
        if target is None:
            for t in tl:
                if t.get("type") == "page":
                    target = t
                    break
        if target is None:
            raise RuntimeError("no page target")
        self.ws = wsc.connect(target["webSocketDebuggerUrl"], max_size=200 * 1024 * 1024, open_timeout=20)
        self.i = 0

    def send(self, method, **params):
        self.i += 1
        mid = self.i
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        deadline = time.time() + 90
        while time.time() < deadline:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})
        raise TimeoutError(method)

    def js(self, expr):
        r = self.send("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        res = r.get("result", {})
        if r.get("exceptionDetails"):
            return {"__exception": str(r["exceptionDetails"].get("text", "")) + " " + str(res.get("description", ""))[:400]}
        return res.get("value")

    def nav(self, url, wait=6):
        self.send("Page.enable")
        self.send("Page.navigate", url=url)
        time.sleep(wait)

    def key(self, key, code, vk, modifiers=0, text=None):
        base = {"modifiers": modifiers, "key": key, "code": code, "windowsVirtualKeyCode": vk, "nativeVirtualKeyCode": vk}
        self.send("Input.dispatchKeyEvent", type="keyDown", **base)
        if text:
            self.send("Input.dispatchKeyEvent", type="char", text=text, **base)
        self.send("Input.dispatchKeyEvent", type="keyUp", **base)
        time.sleep(0.3)

    def ctrl_f9(self):
        # Ctrl+F9 = Run all in Colab
        for t in ("keyDown", "keyUp"):
            self.send("Input.dispatchKeyEvent", type=t, key="F9", code="F9",
                      windowsVirtualKeyCode=120, nativeVirtualKeyCode=120, modifiers=2)

    def click(self, x, y):
        for t in ("mousePressed", "mouseReleased"):
            self.send("Input.dispatchMouseEvent", type=t, x=x, y=y, button="left", clickCount=1)
        time.sleep(0.5)

    def shot(self, path):
        r = self.send("Page.captureScreenshot", format="png")
        open(path, "wb").write(base64.b64decode(r["data"]))
        return path


def ocr(path, psm="6"):
    subprocess.run(["tesseract", path, path + ".ocr", "--psm", psm], capture_output=True)
    try:
        return open(path + ".ocr.txt").read()
    except FileNotFoundError:
        return ""


if __name__ == "__main__":
    c = CDP()
    cmd = sys.argv[1]
    if cmd == "js":
        print(json.dumps(c.js(sys.argv[2]), indent=1)[:4000])
    elif cmd == "ocr":
        p = c.shot(sys.argv[2] if len(sys.argv) > 2 else "/tmp/cdp-shot.png")
        print("shot:", p)
        print(ocr(p))
    elif cmd == "nav":
        c.nav(sys.argv[2])
        print("title:", c.js("document.title"))
    elif cmd == "runall":
        c.ctrl_f9()
        print("sent ctrl+f9")
