"""Runs ON the QAM pod: summarise qam_run.log + results_qam.json -> compact JSON.
Stdlib only; prints a few KB so the poller transfers almost nothing.
    python3 /work/qam_status.py
"""
import json
import os
import re
import time

LOG = "/work/qam_run.log"
RES = "/work/results_qam.json"
PIDF = "/work/qam_run.pid"
HEAD_RE = re.compile(r"QAM sweep: (\d+) cells")
PROG_RE = re.compile(r"\[(\d+)/(\d+)\]")


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
    done_all = "DONE all cells" in txt
    total = None
    mh = HEAD_RE.search(txt)
    if mh:
        total = int(mh.group(1))
    progressed = [int(m.group(1)) for m in PROG_RE.finditer(txt)]
    cur = max(progressed) if progressed else 0
    cells_done = 0
    groups = {}
    sample = []
    if os.path.exists(RES):
        try:
            d = json.load(open(RES))
            cs = d.get("cells", [])
            cells_done = len(cs)
            for c in cs:
                groups[c["group"]] = groups.get(c["group"], 0) + 1
            for c in cs[-6:]:
                cu = c["curves"]["round_robin"]
                sample.append({"key": c["key"], "scR": round(c["sc_rate"], 3),
                               "snr0": c["snr0"], "ber_end": cu["y"][-1] if cu["y"] else None})
        except Exception:
            pass
    tail = [l[:200] for l in txt.strip().splitlines()[-5:]]
    print(json.dumps({"ts": int(time.time()), "cpu": _cpu_cores(), "alive": alive,
                      "alldone": done_all, "total": total, "current": cur,
                      "cells_done": cells_done, "groups": groups, "sample": sample,
                      "tail": tail}))


if __name__ == "__main__":
    main()
