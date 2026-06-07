"""Runs ON the PAS pod: summarise pas_run.log + results_pas_*.json -> compact JSON.
Stdlib only.  python3 /work/pas_status.py
"""
import json
import os
import re
import time

LOG = "/work/pas_run.log"
PIDF = "/work/pas_run.pid"
CELL_RE = re.compile(r"=== JOINT R=([\d.]+) w=(\d+) M=(\d+) Z=(\d+).*?train@([\d.]+)dB")


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


def main():
    txt = open(LOG).read() if os.path.exists(LOG) else ""
    pid = open(PIDF).read().strip() if os.path.exists(PIDF) else ""
    alive = bool(pid) and os.path.isdir("/proc/" + pid)
    part1_done = "[PART 1 done" in txt
    alldone = "ALL DONE" in txt
    cells_done = txt.count("[cell done")
    cur = None
    cm = CELL_RE.findall(txt)
    if cm:
        r, w, M, Z, snr = cm[-1]
        cur = {"rate": float(r), "w": int(w), "M": int(M), "Z": int(Z), "snr": float(snr)}
    # PART 1 shaping-gain summary (if present)
    p1 = None
    if os.path.exists("/work/results_pas_part1.json"):
        try:
            d = json.load(open("/work/results_pas_part1.json"))
            p1 = {M: [{"target": g["target"], "gain_db": g["gain_db"]}
                      for g in d.get("dbgain", {}).get(M, [])] for M in d.get("dbgain", {})}
        except Exception:
            pass
    # PART 2 finals (rl_joint vs uniform per cell)
    p2 = []
    if os.path.exists("/work/results_pas_part2.json"):
        try:
            d = json.load(open("/work/results_pas_part2.json"))
            for key, c in d.get("cells", {}).items():
                f = c.get("finals", {})
                row = {"cell": key, "snr": c.get("train_snr")}
                for n in ("rl_joint", "shaped_rr", "uniform_rr"):
                    if n in f:
                        row[n] = {"ber": f[n]["ber"], "nu": f[n]["nu"], "net": f[n].get("net_rate")}
                p2.append(row)
        except Exception:
            pass
    tail = [l[:200] for l in txt.strip().splitlines()[-6:]]
    print(json.dumps({"ts": int(time.time()), "cpu": _cpu_cores(), "alive": alive,
                      "part1_done": part1_done, "alldone": alldone, "cells_done": cells_done,
                      "total_cells": 4, "current": cur, "shaping_gain_db": p1,
                      "part2_finals": p2, "tail": tail}))


if __name__ == "__main__":
    main()
