"""Runs ON a pod: parse /work/big_run.log + results_big_<slice>.json -> compact JSON.
Stdlib only (fast, no numpy import); prints a few KB so the poller transfers almost nothing."""
import json
import os
import re
import sys
import time

sl = sys.argv[1] if len(sys.argv) > 1 else "?"
log = "/work/big_run.log"
res = f"/work/results_big_{sl}.json"
txt = open(log).read() if os.path.exists(log) else ""

total = 0
mh = re.search(r"rates \[([^\]]*)\] x w \[([^\]]*)\]", txt)
if mh:
    nr = len([x for x in mh.group(1).split(",") if x.strip()])
    nw = len([x for x in mh.group(2).split(",") if x.strip()])
    total = nr * nw

cells_done = txt.count("[cell done")
cur = None
cm = re.findall(r"=== R=([\d.]+) \(sc rate ([\d.]+)\) w=(\d)\s+E=(\d+).*?train@([\d.]+)dB", txt)
if cm:
    r, scr, w, e, snr = cm[-1]
    cur = {"rate": float(r), "sc_rate": float(scr), "w": int(w), "E": int(e), "snr": float(snr)}

def _cpu_cores():
    """Cores actually used by this pod (cgroup v2 usage_usec delta over a short sample)."""
    try:
        def usage():
            for ln in open("/sys/fs/cgroup/cpu.stat"):
                if ln.startswith("usage_usec"):
                    return int(ln.split()[1])
        a = usage(); t0 = time.time(); time.sleep(0.35); b = usage(); t1 = time.time()
        return round((b - a) / 1e6 / (t1 - t0), 1)
    except Exception:
        return None


def _cpu_quota():
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()
        return round(int(q) / int(p)) if q != "max" else None
    except Exception:
        return None


def _mem():
    """(used_GB, limit_GB) from the pod's cgroup."""
    try:
        cur = int(open("/sys/fs/cgroup/memory.current").read())
        mx = open("/sys/fs/cgroup/memory.max").read().strip()
        lim = int(mx) if mx != "max" else None
        return round(cur / 1e9, 1), (round(lim / 1e9) if lim else None)
    except Exception:
        return None, None


cpu = _cpu_cores()
quota = _cpu_quota()
mem_used, mem_lim = _mem()
pid = open("/work/run.pid").read().strip() if os.path.exists("/work/run.pid") else ""
alive = bool(pid) and os.path.isdir("/proc/" + pid)
elapsed = int(time.time() - os.path.getmtime("/work/run.pid")) if os.path.exists("/work/run.pid") else 0
done = (not alive) and ("done in" in txt)

rows = []
if os.path.exists(res):
    try:
        d = json.load(open(res))
        for k, c in d.get("cells", {}).items():
            f = c["finals"]
            rows.append({"cell": k, "rate": c["rate"], "w": c["w"], "E": c["E"],
                         "snr": c.get("train_snr"),
                         "rl": round(f["rl"]["fer"], 4), "cem": round(f["cem"]["fer"], 4),
                         "peredge": round(f["rl_peredge"]["fer"], 4),
                         "rand": round(f["random_search"]["fer"], 4),
                         "seed0": round(f["seed0_default"]["fer"], 4),
                         "rl_n4": c["stats"]["rl"]["n4"]})
    except Exception:
        pass

loglines = [l[:200] for l in txt.strip().splitlines()[-16:]]
print(json.dumps({"slice": sl, "total": total, "cells_done": cells_done, "current": cur,
                  "lsweep": "L-sweep" in txt, "alive": alive, "elapsed": elapsed,
                  "cpu": cpu, "cpu_quota": quota, "mem": mem_used, "mem_quota": mem_lim,
                  "done": done, "rows": rows, "log": loglines}))
