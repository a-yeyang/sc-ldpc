"""Runs ON the PAM4 pod: summarise all N slice logs + results_pam4big_*.json -> compact JSON.
Stdlib only; prints a few KB so the poller transfers almost nothing.
    python3 /work/pam4_status.py
"""
import json
import os
import re
import time

N = 6
CELL_RE = re.compile(r"=== R=([\d.]+) \(sc rate ([\d.]+)\) w=(\d) Z=(\d+).*?train@([\d.]+)dB")


def _cpu_cores():
    try:
        def usage():
            for ln in open("/sys/fs/cgroup/cpu.stat"):
                if ln.startswith("usage_usec"):
                    return int(ln.split()[1])
        a = usage(); t0 = time.time(); time.sleep(0.4); b = usage(); t1 = time.time()
        return round((b - a) / 1e6 / (t1 - t0), 1)
    except Exception:
        return None


def slice_status(k):
    log = f"/work/slice_{k}.log"
    res = f"/work/results_pam4big_{k}.json"
    pidf = f"/work/slice_{k}.pid"
    txt = open(log).read() if os.path.exists(log) else ""
    pid = open(pidf).read().strip() if os.path.exists(pidf) else ""
    alive = bool(pid) and os.path.isdir("/proc/" + pid)
    cells_done = txt.count("[cell done")
    done = ("slice %d done" % k) in txt
    cur = None
    cm = CELL_RE.findall(txt)
    if cm:
        r, scr, w, z, snr = cm[-1]
        cur = {"rate": float(r), "w": int(w), "Z": int(z), "snr": float(snr)}
    # planned cell count for this slice (parse the "slice k: cells [...]" header)
    total = None
    mh = re.search(r"slice %d: cells \[(.*?)\]\s+\(" % k, txt)
    if mh:
        total = len(re.findall(r"\(", mh.group(1)))
    rows = []
    if os.path.exists(res):
        try:
            d = json.load(open(res))
            for key, c in d.get("cells", {}).items():
                f = c["finals"]
                rows.append({"cell": key, "rate": c["rate"], "w": c["w"], "snr": c["train_snr"],
                             "rl": f["rl"]["ber"], "cem": f["cem"]["ber"],
                             "rnd": f["random_search"]["ber"], "seed0": f["seed0_default"]["ber"],
                             "rr": f["round_robin"]["ber"],
                             "gain": round(f["random_search"]["ber"] / max(f["rl"]["ber"], 1e-9), 2)})
        except Exception:
            pass
    tail = [l[:200] for l in txt.strip().splitlines()[-4:]]
    return {"slice": k, "alive": alive, "done": done, "cells_done": cells_done,
            "total": total, "current": cur, "rows": rows, "tail": tail}


def main():
    slices = [slice_status(k) for k in range(N)]
    alldone = all(s["done"] for s in slices)
    print(json.dumps({"ts": int(time.time()), "cpu": _cpu_cores(), "alldone": alldone,
                      "slices": slices}))


if __name__ == "__main__":
    main()
