"""Mac-side poller for the PAS2 (matched-SE RL x shaping) run on the big-memory pod.
Every 30s fetch a compact status (kubectl exec pas2_status.py) -> dashboard/pas2_status.json.
When the run finishes, plot ON THE POD, then pull figures / results / CSVs back.
Self-heals the bastion tunnel if kubectl errors.
    python3 dashboard/pas2_poll.py
"""
import json
import os
import subprocess
import time

DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DIR)
POD = "yye-scldpc-pas2"


def status():
    try:
        r = subprocess.run(["kubectl", "exec", POD, "--", "python3", "/work/pas2_status.py"],
                           capture_output=True, text=True, timeout=60)
        lines = [l for l in r.stdout.strip().splitlines() if l.startswith("{")]
        return json.loads(lines[-1]) if lines else {"error": "no output", "raw": r.stderr[:200]}
    except Exception as e:
        return {"error": str(e)[:200]}


def finalize():
    subprocess.run(["kubectl", "exec", POD, "--", "sh", "-c",
                    "cd /work && python3 experiments_pas2.py plot"], timeout=1800)
    for name in ["results_pas2.json", "results_pas2_required_ebn0.csv",
                 "results_pas2_ablations.csv"]:
        subprocess.run(["kubectl", "cp", f"{POD}:/work/{name}", f"{ROOT}/{name}"], timeout=240)
    r = subprocess.run(["kubectl", "exec", POD, "--", "sh", "-c", "cd /work && ls exp_pas2_*.svg"],
                       capture_output=True, text=True, timeout=60)
    for f in r.stdout.split():
        subprocess.run(["kubectl", "cp", f"{POD}:/work/{f}", f"{ROOT}/{f}"], timeout=120)


def main():
    while True:
        st = status()
        err = str(st.get("error", "")).lower()
        if any(x in err for x in ("refus", "connect", "timed out", "i/o timeout", "unable")):
            try:
                subprocess.run(["bash", "cluster_tunnel.sh"], cwd=ROOT, timeout=45)
            except Exception:
                pass
        st["_poll_ts"] = int(time.time())
        st["finalized"] = False
        json.dump(st, open(f"{DIR}/pas2_status.json", "w"), indent=1)
        if st.get("alldone"):
            try:
                finalize()
                st["finalized"] = True
            except Exception as e:
                st["finalize_error"] = str(e)[:200]
            json.dump(st, open(f"{DIR}/pas2_status.json", "w"), indent=1)
            print("PAS2 run finalized.")
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
