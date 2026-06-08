"""RL optimisation of SC-LDPC construction across LARGE coupling memory w (3..30) and
chain length L (tens..hundreds), with a FAST structural-proxy reward.

Why a proxy: at large w the sliding window must span W>=w+1 positions, so end-to-end
Monte-Carlo BP is ~2.5 s/frame at w=30,L=40 and ~18 s/frame at L=300 -- infeasible for the
thousands of evaluations RL needs.  The lifted-graph 4-cycle count is ~0.3 s and w-independent,
and short-cycle count is the standard fast finite-length design objective.  So RL minimises the
4-cycle count of the coupled lifted graph (the construction = which of the w+1 components each
systematic base edge goes to); champions are then cross-checked by girth across L and by a few
end-to-end Monte-Carlo BER points at the small-w end (where MC is affordable).

Edge spreading is L-independent, so we optimise per w at a fixed OPT_L and report the proxy
(n4) and girth of the learned construction across L.

    python3 experiments_construct_wl.py run        # full w-grid (resume-capable)
    python3 experiments_construct_wl.py plot
"""
from __future__ import annotations
import json
import os
import sys
import time
import multiprocessing as mp
from dataclasses import asdict
import numpy as np

import outpaths as OP
import plotting
import rl_construct as R

BG, ILS, Z, MP = 1, 0, 16, 13            # 5G NR BG1, R~2/3 base graph, Z=16 lift
OPT_L = 60                                # chain length used during optimisation
WS = [3, 5, 8, 12, 16, 20, 30]            # coupling memory sweep (the new regime)
LS = [30, 60, 120, 240]                   # chain lengths for validation
STEPS, BATCH = 14, 100                     # PG steps x batch (= pod cores); proxy is cheap
RESULT = "results_wl.json"


def cfg(w, L):
    W = min(w + 1, L)                      # window must span the coupling (>=w+1)
    return R.Config(bg=BG, ils=ILS, Z=Z, mp=MP, w=w, L=L, W=W, max_iter=10)


def _arr(a):
    return np.asarray(a, dtype=np.int64).tolist()


def _save(out):
    json.dump(out, open(OP.route(RESULT) + ".tmp", "w")); os.replace(OP.route(RESULT) + ".tmp", OP.route(RESULT))


# --------------------------------------------------------------------------- #
#  fast proxy: 4-cycle count of the coupled lifted graph (parallel over cores)
# --------------------------------------------------------------------------- #
_CACHE = {}


def _proxy_worker(task):
    cfg_d, assign = task
    c = R.Config(**cfg_d)
    key = (c.bg, c.ils, c.Z, c.mp)
    comp = _CACHE.get(key) or _CACHE.setdefault(key, c.component())
    from sc_ldpc import SCLDPCCode
    sc = SCLDPCCode(comp, w=c.w, L=c.L, assign=np.asarray(assign, dtype=np.int64))
    return R.count_4cycles(sc)


def proxy_batch(c, assigns, pool):
    cfg_d = asdict(c)
    tasks = [(cfg_d, np.asarray(a, dtype=np.int64)) for a in assigns]
    n4 = pool.map(_proxy_worker, tasks) if pool else [_proxy_worker(t) for t in tasks]
    return [int(x) for x in n4]


# --------------------------------------------------------------------------- #
#  RL (REINFORCE) + random search + default, all minimising the proxy n4
# --------------------------------------------------------------------------- #
def train_rl(c, policy, pool, base_seed, log_every=5, label="PG"):
    feature = isinstance(policy, R.FeaturePolicy)
    best = {"n4": 10 ** 18, "assign": None}
    hist = {"step": [], "mean_n4": [], "best_n4": []}
    for step in range(1, STEPS + 1):
        traces, assigns, ps = [], [], []
        for _ in range(BATCH):
            if feature:
                a, tr = policy.rollout(); traces.append(tr)
            else:
                a, p = policy.sample(); ps.append(p)
            assigns.append(a)
        n4 = proxy_batch(c, assigns, pool)
        rew = -np.array(n4, dtype=np.float64)
        adv = rew - rew.mean()
        if adv.std() > 1e-9:
            adv = adv / adv.std()
        if feature:
            policy.update(list(zip(traces, adv)))
        else:
            policy.update([(assigns[i], ps[i], adv[i]) for i in range(BATCH)])
        bi = int(np.argmin(n4))
        if n4[bi] < best["n4"]:
            best = {"n4": n4[bi], "assign": np.asarray(assigns[bi]).copy()}
        hist["step"].append(step); hist["mean_n4"].append(float(np.mean(n4)))
        hist["best_n4"].append(best["n4"])
        if step % log_every == 0 or step == 1:
            print(f"  [{label}] step {step:2d}  meanN4={np.mean(n4):8.0f}  bestN4={best['n4']}",
                  flush=True)
    return hist, best


def search_random(c, pool, seed=11):
    rng = np.random.default_rng(seed)
    _, _, E = R.edge_meta(c); Ck = c.w + 1
    best = {"n4": 10 ** 18, "assign": None}; hist = {"step": [], "best_n4": []}
    for step in range(1, STEPS + 1):
        assigns = [rng.integers(0, Ck, size=E) for _ in range(BATCH)]
        n4 = proxy_batch(c, assigns, pool)
        bi = int(np.argmin(n4))
        if n4[bi] < best["n4"]:
            best = {"n4": n4[bi], "assign": np.asarray(assigns[bi]).copy()}
        hist["step"].append(step); hist["best_n4"].append(best["n4"])
    return hist, best


def run():
    out = None
    if os.path.exists(OP.route(RESULT)):
        try:
            out = json.load(open(OP.route(RESULT))); out.setdefault("cells", {})
            print(f"[resume] {RESULT}: {len(out['cells'])} w-cells done {sorted(out['cells'])}",
                  flush=True)
        except Exception:
            out = None
    if out is None:
        out = {"config": {"bg": BG, "Z": Z, "mp": MP, "opt_L": OPT_L, "WS": WS, "LS": LS},
               "cells": {}}
    t0 = time.time()
    with mp.Pool(R.n_workers()) as pool:
        print(f"w-sweep {WS}  L-valid {LS}  (proxy=4-cycle, {R.n_workers()} workers)", flush=True)
        for w in WS:
            key = f"w{w}"
            if key in out["cells"]:
                print(f"  [skip] {key}", flush=True); continue
            c = cfg(w, OPT_L)
            _, _, E = R.edge_meta(c)
            print(f"\n=== w={w}  L={OPT_L}  E={E}  space {w+1}^{E}  rate={c.build().rate:.3f} ===",
                  flush=True)
            polf = R.FeaturePolicy(c, lr=0.15, ent=0.02, seed=0)
            h_f, best_f = train_rl(c, polf, pool, 1000, label="PG-feat")
            pole = R.PerEdgePolicy(E, w + 1, lr=0.2, ent=0.01, seed=0)
            h_e, best_e = train_rl(c, pole, pool, 4000, label="PG-edge")
            h_r, best_r = search_random(c, pool)
            seed0 = R.random_assign(c, np.random.default_rng(0))
            champs = {"rl": best_f["assign"], "rl_peredge": best_e["assign"],
                      "random": best_r["assign"], "seed0": seed0}
            # validate n4 + girth across L for each champion
            valid = {}
            for name, a in champs.items():
                row = {}
                for L in LS:
                    cL = cfg(w, L)
                    sc = cL.build(assign=np.asarray(a, dtype=np.int64))
                    row[str(L)] = {"n4": R.count_4cycles(sc), "rate": cL.build().rate}
                row["girth_L60"] = R.girth(cfg(w, 60).build(assign=np.asarray(a, dtype=np.int64)),
                                           max_g=10, n_seeds=300, seed=1)
                valid[name] = row
            n4_opt = {k: valid[k][str(OPT_L)]["n4"] for k in champs}
            print(f"  n4@L{OPT_L}: RL={n4_opt['rl']} edge={n4_opt['rl_peredge']} "
                  f"rand={n4_opt['random']} seed0={n4_opt['seed0']}", flush=True)
            out["cells"][key] = {"w": w, "E": int(E), "rate": c.build().rate,
                                 "n4_opt": n4_opt, "valid": valid,
                                 "hist": {"rl": h_f, "rl_peredge": h_e, "random": h_r},
                                 "theta_rl": polf.theta.tolist(),
                                 "champions": {k: _arr(v) for k, v in champs.items()}}
            _save(out)
    print(f"\nw-sweep done in {time.time()-t0:.0f}s -> {RESULT}", flush=True)
    make_plots(out)


def make_plots(out=None):
    if out is None:
        out = json.load(open(OP.route(RESULT)))
    cells = out["cells"]
    ws = sorted(int(k[1:]) for k in cells)
    # n4 @ OPT_L vs w, one line per method
    LAB = {"rl": "RL feature [ours]", "rl_peredge": "RL per-edge [ours]",
           "random": "random search", "seed0": "default seed0"}
    COL = {"rl": "#d62728", "rl_peredge": "#ff7f0e", "random": "#1f77b4", "seed0": "#7f7f7f"}
    series = []
    for m in ["rl", "rl_peredge", "random", "seed0"]:
        y = [max(cells[f"w{w}"]["n4_opt"][m], 0.5) for w in ws]
        series.append({"x": ws, "y": y, "label": LAB[m], "color": COL[m]})
    plotting.semilogy(series, xlabel="coupling memory w", ylabel=f"4-cycles @ L={OPT_L} (lower=better)",
                      title="RL minimises coupled-graph 4-cycles across w (BG1 Z=16, R~2/3)",
                      path="exp_wl_n4_vs_w.svg")
    # n4 vs L for the RL champion at a few w
    for w in ws:
        v = cells[f"w{w}"]["valid"]
        ser = [{"x": LS, "y": [max(v[m][str(L)]["n4"], 0.5) for L in LS], "label": LAB[m], "color": COL[m]}
               for m in ["rl", "random", "seed0"]]
        plotting.semilogy(ser, xlabel="chain length L", ylabel="4-cycles (lower=better)",
                          title=f"4-cycles vs chain length L (w={w})", path=f"exp_wl_n4_vs_L_w{w}.svg")
    print("wrote exp_wl_*.svg")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"run": run, "plot": make_plots}[cmd]()
