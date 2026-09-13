#!/usr/bin/env python3
"""Laptop-side watchdog for the rented RunPod pod.

Runs from Hermes cron every 15 minutes. Prints something ONLY when it acts or something
needs a human -- silent stdout means "all normal". Independent of any agent session, so a
dead chat, a closed laptop lid or a crashed script cannot leave a GPU billing.

Layers it enforces (pod_job.sh enforces the other two, on the pod itself):
  * stop a RUNNING pod whose wall clock exceeds the budget
  * report a finished/failed pod once, with what to do next
"""
import datetime, json, os, sys, urllib.request

PROJ = "/home/rubin/procurement-ft-trial"
STATE = os.path.join(PROJ, "runpod_state.json")
KEYFILE = os.path.expanduser("~/.config/runpod/api_key")
MARK = "/tmp/runpod_watchdog_seen.json"
API = "https://rest.runpod.io/v1"
BUDGET_HOURS = 4.0          # hard ceiling for the whole pod lifetime


def key():
    k = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not k and os.path.exists(KEYFILE):
        k = open(KEYFILE).read().strip()
    return k


def call(method, path, k):
    req = urllib.request.Request(f"{API}{path}", method=method,
                                 headers={"Authorization": f"Bearer {k}"})
    with urllib.request.urlopen(req, timeout=45) as r:
        body = r.read().decode()
    return json.loads(body) if body.strip() else {}


def main():
    k = key()
    if not k or not os.path.exists(STATE):
        return 0                                  # nothing to watch yet: stay silent
    st = json.load(open(STATE))
    pid = st.get("pod_id")
    if not pid:
        return 0
    try:
        d = call("GET", f"/pods/{pid}", k)
    except Exception as e:
        print(f"RunPod watchdog: could not read pod {pid}: {e}")
        return 0

    status = d.get("desiredStatus")
    started = d.get("lastStartedAt")
    hrs = None
    if started:
        try:
            t = datetime.datetime.fromisoformat(started.replace("Z", "+00:00"))
            hrs = (datetime.datetime.now(datetime.timezone.utc) - t).total_seconds() / 3600
        except Exception:
            hrs = None

    seen = json.load(open(MARK)) if os.path.exists(MARK) else {}

    if status == "RUNNING" and hrs is not None and hrs > BUDGET_HOURS:
        try:
            call("POST", f"/pods/{pid}/stop", k)
            print(f"RunPod watchdog: pod {pid} had been RUNNING {hrs:.1f} h "
                  f"(budget {BUDGET_HOURS} h) -- STOPPED it. Artifacts are on /workspace; "
                  f"run `runpod_ft.py refresh` then `fetch` to pull them.")
        except Exception as e:
            print(f"RunPod watchdog: tried to stop {pid} after {hrs:.1f} h but failed: {e}")
        seen[pid] = "stopped-over-budget"
        json.dump(seen, open(MARK, "w"))
        return 0

    if status in ("EXITED", "TERMINATED") and seen.get(pid) != status:
        cost = (d.get("costPerHr") or 0)
        print(f"RunPod watchdog: pod {pid} is {status} (was up ~{hrs:.1f} h, "
              f"{'~$%.2f' % (float(cost)*hrs) if hrs and cost else 'cost n/a'}). "
              f"Next: `python3 runpod_ft.py refresh && python3 runpod_ft.py fetch`, "
              f"then terminate to stop storage billing.")
        seen[pid] = status
        json.dump(seen, open(MARK, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
