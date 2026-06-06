"""Mac-side poller: every 15s, fetch a compact status from each pod (via kubectl exec
pod_status.py) and write dashboard/status.json.  When BOTH slices finish, pull the full
results JSONs back and run `plot` (which also writes the CSV data tables), then stop.

Writes only the small status.json (overwritten); no growing logs.
    python3 dashboard/poll.py
"""
import json
import os
import subprocess
import time

DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DIR)
PODS = [("yye-scldpc-a", "A", "cpu-10-208-55-86"), ("yye-scldpc-b", "B", "cpu-10-208-55-85")]


def pod_status(pod, sl, node):
    try:
        r = subprocess.run(["kubectl", "exec", pod, "--", "python3", "/work/pod_status.py", sl],
                           capture_output=True, text=True, timeout=40)
        lines = [l for l in r.stdout.strip().splitlines() if l.startswith("{")]
        st = json.loads(lines[-1]) if lines else {"slice": sl, "error": "no output"}
    except Exception as e:
        st = {"slice": sl, "error": str(e)[:140]}
    st["pod"] = pod
    st["node"] = node
    return st


def finalize():
    for pod, sl, _ in PODS:
        subprocess.run(["kubectl", "cp", f"{pod}:/work/results_big_{sl}.json",
                        f"{ROOT}/results_big_{sl}.json"], timeout=180)
    subprocess.run(["python3", "experiments_construct_big.py", "plot"], cwd=ROOT, timeout=900)


def main():
    while True:
        pods = [pod_status(p, s, n) for p, s, n in PODS]
        # self-heal: if the bastion tunnel dropped, kubectl errors -> restart it
        if any("refus" in str(p.get("error", "")).lower() or
               "connect" in str(p.get("error", "")).lower() for p in pods):
            try:
                subprocess.run(["bash", "cluster_tunnel.sh"], cwd=ROOT, timeout=45)
            except Exception:
                pass
        alldone = all(p.get("done") for p in pods) and len(pods) == 2
        st = {"ts": int(time.time()), "alldone": alldone, "finalized": False, "pods": pods}
        json.dump(st, open(f"{DIR}/status.json", "w"))
        if alldone:
            try:
                finalize(); st["finalized"] = True
            except Exception as e:
                st["finalize_error"] = str(e)[:200]
            json.dump(st, open(f"{DIR}/status.json", "w"))
            return
        time.sleep(15)


if __name__ == "__main__":
    main()
