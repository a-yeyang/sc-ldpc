"""Mac-side poller for the PAM4 big run.  Every 30s fetch a compact status from the
pod (kubectl exec pam4_status.py) and write dashboard/pam4_status.json.  When all
slices finish, run `plot` ON THE POD (the Mac has no scipy), then pull the figures /
merged JSON / CSVs back, and stop.  Self-heals the bastion tunnel if kubectl errors.
    python3 dashboard/pam4_poll.py
"""
import json
import os
import subprocess
import time

DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DIR)
POD = "yye-scldpc-pam4"
N = 6


def status():
    try:
        r = subprocess.run(["kubectl", "exec", POD, "--", "python3", "/work/pam4_status.py"],
                           capture_output=True, text=True, timeout=60)
        lines = [l for l in r.stdout.strip().splitlines() if l.startswith("{")]
        return json.loads(lines[-1]) if lines else {"error": "no output", "raw": r.stderr[:200]}
    except Exception as e:
        return {"error": str(e)[:200]}


def finalize():
    # plot on the pod (has numpy+scipy), then pull artifacts back
    subprocess.run(["kubectl", "exec", POD, "--", "sh", "-c",
                    "cd /work && python3 experiments_construct_pam4.py plot"], timeout=1800)
    for k in range(N):
        subprocess.run(["kubectl", "cp", f"{POD}:/work/results_pam4big_{k}.json",
                        f"{ROOT}/results_pam4big_{k}.json"], timeout=180)
    for name in ["results_pam4big_merged.json", "results_pam4big_finals.csv",
                 "results_pam4big_hist.csv"]:
        subprocess.run(["kubectl", "cp", f"{POD}:/work/{name}", f"{ROOT}/{name}"], timeout=180)
    r = subprocess.run(["kubectl", "exec", POD, "--", "sh", "-c", "cd /work && ls exp_pam4big_*.svg"],
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
        json.dump(st, open(f"{DIR}/pam4_status.json", "w"), indent=1)
        if st.get("alldone"):
            try:
                finalize()
                st["finalized"] = True
            except Exception as e:
                st["finalize_error"] = str(e)[:200]
            json.dump(st, open(f"{DIR}/pam4_status.json", "w"), indent=1)
            print("PAM4 run finalized.")
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
