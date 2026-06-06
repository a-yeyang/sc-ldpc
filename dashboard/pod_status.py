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

print(json.dumps({"slice": sl, "total": total, "cells_done": cells_done, "current": cur,
                  "lsweep": "L-sweep" in txt, "alive": alive, "elapsed": elapsed,
                  "done": done, "rows": rows}))
