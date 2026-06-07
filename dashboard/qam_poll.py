"""Mac-side poller for the QAM decoding-performance sweep.  Every 30s fetch a
compact status from the pod (kubectl exec qam_status.py) -> dashboard/qam_status.json.
When the run finishes, plot ON THE POD, then pull the figures / results / CSV back.
Self-heals the bastion tunnel if kubectl errors.
    python3 dashboard/qam_poll.py
"""
import json
import os
import subprocess
import time

DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DIR)
POD = "yye-scldpc-qam120"


def status():
    try:
        r = subprocess.run(["kubectl", "exec", POD, "--", "python3", "/work/qam_status.py"],
                           capture_output=True, text=True, timeout=60)
        lines = [l for l in r.stdout.strip().splitlines() if l.startswith("{")]
        return json.loads(lines[-1]) if lines else {"error": "no output", "raw": r.stderr[:200]}
    except Exception as e:
        return {"error": str(e)[:200]}


def finalize():
    subprocess.run(["kubectl", "exec", POD, "--", "sh", "-c",
                    "cd /work && python3 experiments_qam_sweep.py plot"], timeout=1800)
    for name in ["results_qam.json", "results_qam_finals.csv"]:
        subprocess.run(["kubectl", "cp", f"{POD}:/work/{name}", f"{ROOT}/{name}"], timeout=180)
    r = subprocess.run(["kubectl", "exec", POD, "--", "sh", "-c", "cd /work && ls exp_qam_*.svg"],
                       capture_output=True, text=True, timeout=60)
    for f in r.stdout.split():
        subprocess.run(["kubectl", "cp", f"{POD}:/work/{f}", f"{ROOT}/{f}"], timeout=120)


def main():
    while True:
        st = status()
        err = str(st.get("error", "")).lower()
        if any(x in err for x in ("refus", "connect", "timed out", "i/o timeout")):
            try:
                subprocess.run(["bash", "cluster_tunnel.sh"], cwd=ROOT, timeout=45)
            except Exception:
                pass
        st["_poll_ts"] = int(time.time())
        st["finalized"] = False
        json.dump(st, open(f"{DIR}/qam_status.json", "w"), indent=1)
        if st.get("alldone"):
            try:
                finalize()
                st["finalized"] = True
            except Exception as e:
                st["finalize_error"] = str(e)[:200]
            json.dump(st, open(f"{DIR}/qam_status.json", "w"), indent=1)
            print("QAM sweep finalized.")
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
