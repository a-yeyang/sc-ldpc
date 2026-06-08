"""Runs ON the PAS2 pod: summarise pas2_run.log + results_pas2.json -> compact JSON.
Stdlib only.  python3 /work/pas2_status.py
"""
import json
import os
import re
import time

LOG = "/work/pas2_run.log"
PIDF = "/work/pas2_run.pid"
RES = "/work/results_pas2.json"
CELL_RE = re.compile(
    r"=== CELL M=(\d+) Z=(\d+) w=(\d+) \| UNIFORM mp(\d+) R=([\d.]+) SE=([\d.]+)"
    r".*?SHAPED mp(\d+) R=([\d.]+) nu=([\d.]+)")
TOTAL = 8


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


def _mem_pct():
    try:
        cur = int(open("/sys/fs/cgroup/memory.current").read().strip())
        mx = open("/sys/fs/cgroup/memory.max").read().strip()
        if mx == "max":
            return None
        return round(100.0 * cur / int(mx), 1)
    except Exception:
        return None


def main():
    txt = open(LOG).read() if os.path.exists(LOG) else ""
    pid = open(PIDF).read().strip() if os.path.exists(PIDF) else ""
    alive = bool(pid) and os.path.isdir("/proc/" + pid)
    alldone = "ALL DONE" in txt
    cells_done = txt.count("[cell done")
    cur = None
    cm = CELL_RE.findall(txt)
    if cm:
        M, Z, w, mpu, ru, seu, mps, rs, nu = cm[-1]
        cur = {"M": int(M), "Z": int(Z), "w": int(w), "mp_u": int(mpu),
               "se": float(seu), "mp_s": int(mps), "nu": float(nu)}
    # last train SNR line
    snr = None
    sm = re.findall(r"train@([\d.]+)dB", txt)
    if sm:
        snr = float(sm[-1])
    # PART results: matched-SE Eb/N0@1e-5 gain (rl_joint vs uniform_cem) per cell, L=200
    gains = []
    if os.path.exists(RES):
        try:
            d = json.load(open(RES))
            for key, c in d.get("cells", {}).items():
                row = {"cell": key, "M": c.get("M"), "SE": round(c.get("train", {}).get("se_u", 0), 2),
                       "nu": round(c.get("train", {}).get("nu_matched", 0), 3)}
                rep = c.get("report", {}).get("200", {}).get("methods", {})
                base = (rep.get("uniform_cem") or rep.get("uniform_rr") or {}).get("ebn0_1e5")
                joint = (rep.get("rl_joint") or {}).get("ebn0_1e5")
                constr = (rep.get("constr_rl") or {}).get("ebn0_1e5")
                shp = (rep.get("shaped_rr") or {}).get("ebn0_1e5")
                row["ebn0_uniform"] = base
                row["ebn0_shaped"] = shp
                row["ebn0_constrRL"] = constr
                row["ebn0_joint"] = joint
                if base is not None and joint is not None:
                    row["joint_gain_dB"] = round(base - joint, 2)
                if base is not None and shp is not None:
                    row["shaping_gain_dB"] = round(base - shp, 2)
                # ablation: RL vs CEM vs random (train-L val BER)
                s = c.get("search_val_ber", {})
                row["val_rl"] = s.get("rl")
                row["val_cem"] = s.get("cem")
                row["val_random"] = s.get("random")
                row["val_joint"] = s.get("joint")
                row["val_separate"] = s.get("separate")
                gains.append(row)
        except Exception as e:
            gains = [{"err": str(e)[:120]}]
    tail = [l[:200] for l in txt.strip().splitlines()[-8:]]
    print(json.dumps({"ts": int(time.time()), "cpu": _cpu_cores(), "mem_pct": _mem_pct(),
                      "alive": alive, "alldone": alldone, "cells_done": cells_done,
                      "total_cells": TOTAL, "current": cur, "train_snr": snr,
                      "matched_se_gains": gains, "tail": tail}))


if __name__ == "__main__":
    main()
