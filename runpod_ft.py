"""RunPod driver for the procurement-extractor fine-tune.

Usage (key from $RUNPOD_API_KEY or ~/.config/runpod/api_key):
  python3 runpod_ft.py preflight        # validate key, list pods, show balance if available
  python3 runpod_ft.py start            # create a 4090/A6000 pod, wait for RUNNING, save state
  python3 runpod_ft.py sshcmd -- CMD    # run a shell command on the pod
  python3 runpod_ft.py upload           # push repo files (data, configs, eval) to the pod
  python3 runpod_ft.py run [--epochs 2] # install Soup, train, evaluate (writes runpod.log)
  python3 runpod_ft.py tail             # last lines of the remote log
  python3 runpod_ft.py fetch            # pull output/ + logs back to run-artifacts/runpod/
  python3 runpod_ft.py stop             # stop the pod (volume kept, ~$0.20/GB/month)
  python3 runpod_ft.py terminate        # destroy the pod (billing stops completely)

Cost guard: refuses to leave a pod running past MAX_HOURS; status always prints elapsed time.
"""
import argparse, json, os, random, string, subprocess, sys, time
from datetime import datetime, timezone

import requests

API = "https://rest.runpod.io/v1"
PROJ = "/home/rubin/procurement-ft-trial"
STATE = os.path.join(PROJ, "runpod_state.json")
ART = os.path.join(PROJ, "run-artifacts", "runpod")
LOG = os.path.join(ART, "runpod.log")
SSH_KEY = "/home/rubin/.ssh/id_ed25519"
MAX_HOURS = 5.0

# Python 3.10-3.12 is required by soup-cli, so a py3.11 torch image is the right base.
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
GPUS = ["NVIDIA GeForce RTX 4090", "NVIDIA RTX A6000", "NVIDIA L40S", "NVIDIA A100 80GB PCIe",
        "NVIDIA RTX A5000", "NVIDIA A40"]


def key():
    k = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not k:
        p = os.path.expanduser("~/.config/runpod/api_key")
        if os.path.exists(p):
            k = open(p).read().strip()
    if not k:
        sys.exit("No RunPod API key. Put it in ~/.config/runpod/api_key (chmod 600) or export RUNPOD_API_KEY.")
    return k


def hdr():
    return {"Authorization": f"Bearer {key()}", "Content-Type": "application/json"}


def api(method, path, **kw):
    r = requests.request(method, API + path, headers=hdr(), timeout=90, **kw)
    if r.status_code >= 400:
        raise SystemExit(f"{method} {path} -> {r.status_code}: {r.text[:500]}")
    return r.json() if r.text.strip() else {}


def save_state(d):
    os.makedirs(ART, exist_ok=True)
    d.setdefault("history", [])
    json.dump(d, open(STATE, "w"), indent=1)


def load_state():
    if not os.path.exists(STATE):
        sys.exit("no pod state yet — run `start` first")
    return json.load(open(STATE))


def preflight():
    pods = api("GET", "/pods")
    print(f"key OK. pods on account: {len(pods)}")
    for p in pods:
        print(f"  {p.get('id')} {p.get('name')} {p.get('desiredStatus')} gpu={p.get('machine',{}).get('gpuTypeId','?')}")
    try:
        q = {"query": "{ myself { clientBalance email } }"}
        r = requests.post(f"https://api.runpod.io/graphql?api_key={key()}", json=q, timeout=60)
        d = r.json().get("data", {}).get("myself", {})
        print(f"balance: ${d.get('clientBalance')}  account: {d.get('email')}")
        bal = float(d.get("clientBalance") or 0)
        if bal <= 0:
            print("!! balance is zero or negative — add credit before starting a pod (past-due account)")
    except Exception as e:
        print("balance check unavailable:", e)
    return pods


def start():
    body = {
        "name": "quote-reader-ft-" + "".join(random.choices(string.ascii_lowercase + string.digits, k=4)),
        "imageName": IMAGE,
        "gpuTypeIds": GPUS,
        "gpuTypePriority": "availability",
        "gpuCount": 1,
        "cloudType": "SECURE",
        "containerDiskInGb": 60,
        "volumeInGb": 40,   # stopped volume = $0.20/GB/mo, so keep it tight
        "volumeMountPath": "/workspace",
        "ports": "22/tcp,8888/http",
        "env": {"PUBLIC_KEY": open(SSH_KEY + ".pub").read().strip(),
                "JUPYTER_PASSWORD": "".join(random.choices(string.ascii_letters + string.digits, k=16)),
                # lets the JOB stop its own pod -- the whole point is that billing must not
                # depend on this laptop being awake
                "RUNPOD_API_KEY": key(),
                "HF_HOME": "/workspace/hf",
                "MAX_HOURS": str(MAX_HOURS),
                "STALL_MIN": "20"},
    }
    pod = api("POST", "/pods", json=body)
    pid = pod.get("id")
    print("created pod:", pid, "|", pod.get("name"))
    state = {"pod_id": pid, "name": pod.get("name"), "created": datetime.now(timezone.utc).isoformat(),
             "history": [f"{datetime.now(timezone.utc).isoformat()} created"]}
    save_state(state)
    for i in range(90):
        d = api("GET", f"/pods/{pid}")
        st = d.get("desiredStatus")
        ports = d.get("portMappings") or {}
        ssh = ports.get("22")
        print(f"  t+{i*10}s status={st} ssh={ssh}")
        if st == "RUNNING" and ssh:
            state.update({"status": st, "ssh_port": ssh, "raw": {
                k: d.get(k) for k in ("publicIp", "machineId", "costPerHr", "imageName")}})
            state["ssh"] = f"ssh -i {SSH_KEY} -p {ssh} -o StrictHostKeyChecking=no root@{d.get('publicIp')}" \
                if d.get("publicIp") else None
            state["ip"] = d.get("publicIp")
            save_state(state)
            print("READY:", state["ssh"])
            return state
        time.sleep(10)
    print("pod did not reach RUNNING within 15 min — stopping it so it cannot bill idle")
    try:
        api("POST", f"/pods/{pid}/stop")
        print("stopped pod", pid, "-- start it again with: runpod_ft.py start-pod")
    except SystemExit as e:
        print("could not stop automatically:", e)
    sys.exit(1)


def ssh_base(state=None):
    st = state or load_state()
    if not st.get("ip") or not st.get("ssh_port"):
        sys.exit("no SSH endpoint in state (pod stopped?) — start it again")
    return ["ssh", "-i", SSH_KEY, "-p", str(st["ssh_port"]),
            "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=20", f"root@{st['ip']}"]


def sshcmd(cmd, state=None, timeout=1800):
    return subprocess.run(ssh_base(state) + [cmd], capture_output=True, text=True, timeout=timeout)


def upload():
    st = load_state()
    os.makedirs(ART, exist_ok=True)
    subprocess.run(["ssh", "-i", SSH_KEY, "-p", str(st["ssh_port"]), "-o", "StrictHostKeyChecking=no",
                    f"root@{st['ip']}", "mkdir -p /workspace/ft/{data,deliverables}"], check=True)
    for f in ("soup.yaml", "pod_job.sh", "eval_check.py", "generator.py", "README.md"):
        p = os.path.join(PROJ, f)
        if os.path.exists(p):
            subprocess.run(["scp", "-i", SSH_KEY, "-P", str(st["ssh_port"]), "-o", "StrictHostKeyChecking=no",
                            p, f"root@{st['ip']}:/workspace/ft/"], check=True)
            print("uploaded", f)
    subprocess.run(["scp", "-i", SSH_KEY, "-P", str(st["ssh_port"]), "-o", "StrictHostKeyChecking=no", "-r",
                    os.path.join(PROJ, "data"), f"root@{st['ip']}:/workspace/ft/"], check=True)
    print("uploaded data/")
    subprocess.run(["ssh", "-i", SSH_KEY, "-p", str(st["ssh_port"]), "-o", "StrictHostKeyChecking=no",
                    f"root@{st['ip']}", "mkdir -p /workspace/ft/output /workspace/ft/artifacts /workspace/hf"],
                   check=True)
    # the rescued Colab checkpoint goes INSIDE output/ so --resume finds a valid checkpoint dir
    ck = os.path.join(PROJ, "checkpoints", "checkpoint-100")
    if os.path.isdir(ck):
        subprocess.run(["scp", "-i", SSH_KEY, "-P", str(st["ssh_port"]), "-o", "StrictHostKeyChecking=no", "-r",
                        ck, f"root@{st['ip']}:/workspace/ft/output/"], check=True)
        print("uploaded output/checkpoint-100 (resume point rescued from the Colab run)")


def run(epochs=1, resume=None, max_hours=MAX_HOURS):
    """Launch pod_job.sh detached. The job stops its own pod when it ends, so a laptop
    failure can no longer leave a GPU billing, and artifacts land on /workspace."""
    ckpt = f"/workspace/ft/output/{resume}" if resume else ""
    remote = f"""
set -e
cd /workspace/ft
chmod +x pod_job.sh
: > runpod.log
EPOCHS={epochs} MAX_HOURS={max_hours} STALL_MIN=20 CKPT="{ckpt}" \
  nohup bash /workspace/ft/pod_job.sh >> /workspace/ft/runpod.log 2>&1 &
sleep 10
echo '--- first lines of job.log ---'
tail -20 /workspace/ft/runpod.log 2>/dev/null || true
"""
    r = sshcmd(remote)
    print(r.stdout[-2500:])
    print(r.stderr[-800:])
    if r.returncode != 0:
        print("WARN: launch returned non-zero -- check `tail` output before assuming it started")


def verify():
    """Confirm the run RESUME took (not a silent restart at step 0) and artifacts exist."""
    r = sshcmd("""
cd /workspace/ft
echo '=== resume evidence (train.log) ==='
grep -m3 -iE "resum|global_step|continuing" train.log 2>/dev/null | head -6
echo '=== checkpoints written by this run ==='
for d in output/checkpoint-*; do
  [ -d "$d" ] && python3 -c "import json,sys;print('$d', 'global_step', json.load(open('$d/trainer_state.json')).get('global_step'))" 2>/dev/null
done
echo '=== artifacts ==='
ls -la artifacts/ 2>/dev/null
echo '=== summary ==='
cat artifacts/SUMMARY.txt 2>/dev/null
""")
    print(r.stdout[-4000:] or r.stderr[-1000:])


def refresh():
    """Re-read the pod record: IP and SSH port CHANGE across stop/start."""
    st = load_state()
    d = api("GET", f"/pods/{st['pod_id']}")
    ports = d.get("portMappings") or {}
    st.update({"status": d.get("desiredStatus"), "ip": d.get("publicIp"),
               "ssh_port": ports.get("22"),
               "raw": {k: d.get(k) for k in ("publicIp", "machineId", "costPerHr", "imageName")}})
    save_state(st)
    print(f"status={st['status']} ip={st['ip']} ssh_port={st['ssh_port']}")


def tail(n=40):
    r = sshcmd(f"cd /workspace/ft && tail -n {n} train.log 2>/dev/null; echo '--- eval ---'; tail -n 15 eval.log 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader")
    print(r.stdout[-3000:] or r.stderr[-800:])


def fetch():
    st = load_state()
    os.makedirs(ART, exist_ok=True)
    for item in ("output", "artifacts", "train.log", "eval.log", "runpod.log"):
        subprocess.run(["scp", "-i", SSH_KEY, "-P", str(st["ssh_port"]), "-o", "StrictHostKeyChecking=no", "-r",
                        f"root@{st['ip']}:/workspace/ft/{item}", ART + "/"], check=False)
    print("fetched into", ART)
    subprocess.run(["du", "-sh", *[os.path.join(ART, x) for x in os.listdir(ART)]], check=False)


def status():
    st = load_state()
    d = api("GET", f"/pods/{st['pod_id']}")
    created = datetime.fromisoformat(st["created"])
    hrs = (datetime.now(timezone.utc) - created).total_seconds() / 3600
    print(f"{st['name']} | {d.get('desiredStatus')} | up {hrs:.2f} h | cost ~${float(d.get('costPerHr') or 0)*hrs:.2f}")
    if hrs > MAX_HOURS:
        print(f"!! over the {MAX_HOURS} h guard — fetch and stop/terminate now")


def stop(terminate=False):
    st = load_state()
    pid = st["pod_id"]
    if terminate:
        api("DELETE", f"/pods/{pid}")
        st["history"].append("terminated")
        save_state(st)
        print("terminated", pid, "— billing stopped")
    else:
        api("POST", f"/pods/{pid}/stop")
        st["history"].append("stopped")
        save_state(st)
        print("stopped", pid, "— GPU billing stopped, volume kept (~$0.20/GB/month)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["preflight", "start", "upload", "run", "tail", "fetch",
                                    "status", "stop", "terminate", "sshcmd"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-hours", type=float, default=MAX_HOURS)
    ap.add_argument("--resume", default=None,
                    help="checkpoint dir name under /workspace/ft to resume from, e.g. checkpoint-100")
    ap.add_argument("rest", nargs="*")
    a = ap.parse_args()
    if a.cmd == "preflight":
        preflight()
    elif a.cmd == "start":
        start()
    elif a.cmd == "upload":
        upload()
    elif a.cmd == "run":
        run(a.epochs)
    elif a.cmd == "tail":
        tail()
    elif a.cmd == "fetch":
        fetch()
    elif a.cmd == "status":
        status()
    elif a.cmd == "stop":
        stop(False)
    elif a.cmd == "terminate":
        stop(True)
    elif a.cmd == "sshcmd":
        print(sshcmd(" ".join(a.rest)).stdout[-4000:])
