"""Experiment B -- ZERO-SHOT GENERALISATION of a learned SC-LDPC construction policy.

Thesis (AAAI "neural combinatorial optimisation with generalisation")
---------------------------------------------------------------------
Classical per-instance construction search (PEG / CEM) must RESTART from scratch
for every code configuration (Z, w, mp, L): the optimiser's state is the per-edge
categorical of *that* code, with no notion of a reusable design rule.  An RL policy
over *config-agnostic* per-(edge, component) features instead learns a single
construction RULE that, by construction, applies to ANY code size.  If that rule
TRANSFERS, a single forward rollout on an UNSEEN config (cost ~0) should rival a
full per-config CEM search (cost = a whole optimisation).  This is the amortisation
super-power that per-instance search cannot have.

Objective = ERROR-FLOOR structure, not the waterfall.  The reward of a finished
construction is -(elementary absorbing-set count) of the lifted coupled Tanner
graph (floor_gate.absorbing_set_spectrum) -- a CHEAP combinatorial surrogate with
NO channel simulation, so REINFORCE is fast on CPU.  (floor_gate already showed
absorbing-set count correlates with the measured floor BER for this code family.)

Transferable policy (TransferPolicy, this file)
-----------------------------------------------
A linear-softmax policy over per-(edge, component) features built from the running
construction state.  It has a FIXED number of parameters (independent of code size
AND of w) so the SAME theta scores a w=2 code and a w=5 code:
  shared edge/state features (5): row-load frac, col-load frac, global-load frac,
      "component still empty for this edge's row", ditto for its column.
  component-ROLE features (3): is_parity_carrier (component 0 holds B_par),
      normalised coupling offset a/w, and (centred offset)^2 -- these replace the
      per-absolute-index biases of rl_construct.FeaturePolicy (which cannot transfer
      because the #components C = w+1 changes with w).  Coupling offset a in 0..w is
      a genuine, w-independent physical coordinate (B_{i-t} sits at band offset a).
Total: 8 parameters, shared across every config in the training set and every
unseen test config.

Protocol
--------
TRAIN  one TransferPolicy by REINFORCE: each step samples a random TRAIN config and
       a batch of constructions from the policy, rewards each by -(AS count) on
       common scoring, normalises the advantage, updates the shared theta.
TEST   on HELD-OUT UNSEEN configs.  For each, compare on AS count (and, for a couple,
       the measured floor BER) :
         (a) zero_shot : the trained policy's GREEDY rollout  (~0 cost at test time)
         (b) cem       : per-instance CEM optimised FOR THAT config (expensive)
         (c) round_robin
         (d) best_of_random (same #evals as CEM -- a fair random-search budget)
HEADLINE : fraction of the (round_robin -> CEM) improvement that the zero-shot
           policy recovers, per config and pooled.  Plus the learned theta as
           interpretable design rules.

Pure NumPy + multiprocessing, checkpointed (JSON after training and after each test
config) so partial results are pullable mid-run.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import time
import argparse
from dataclasses import asdict
import numpy as np
import multiprocessing as mp

from rl_construct import (Config, Adam, _softmax, edge_meta, n_components,
                          round_robin_assign, n_workers, count_4cycles)
from floor_gate import absorbing_set_spectrum, short_cycle_spectrum, measure_ber_curve


# --------------------------------------------------------------------------- #
#  cheap combinatorial reward: elementary absorbing-set count (parallel worker)
# --------------------------------------------------------------------------- #
def _as_count(cfg: Config, assign, a_max, beam):
    sc = cfg.build(assign=np.asarray(assign, dtype=np.int64))
    ab = absorbing_set_spectrum(sc, a_max=a_max, beam=beam)
    n4 = count_4cycles(sc)             # smooth tie-break when AS counts collide / are 0
    return ab["as_count"], n4, ab["as_min_a"], ab["as_min_b"]


def _score_worker(task):
    """(config_dict, assign, a_max, beam) -> scalar cost = AS*1e6 + n4 (lower better)."""
    cfg_d, assign, a_max, beam = task
    cfg = Config(**cfg_d)
    asn, n4, _, _ = _as_count(cfg, assign, a_max, beam)
    return float(asn) * 1e6 + float(n4)


def score_batch(cfg: Config, assigns, a_max, beam, pool):
    tasks = [(asdict(cfg), np.asarray(a, np.int64), a_max, beam) for a in assigns]
    raw = pool.map(_score_worker, tasks) if pool is not None else [_score_worker(t) for t in tasks]
    return np.asarray(raw, float)


def as_detail(cfg: Config, assign, a_max, beam):
    """Full structural descriptor of one finished construction (for reporting)."""
    sc = cfg.build(assign=np.asarray(assign, dtype=np.int64))
    ab = absorbing_set_spectrum(sc, a_max=a_max, beam=beam)
    cyc = short_cycle_spectrum(sc)
    return {"as_count": int(ab["as_count"]), "as_min_a": int(ab["as_min_a"]),
            "as_min_b": int(ab["as_min_b"]), "as_by_a": ab["as_by_a"],
            "n4": int(cyc["n4"]), "n6": int(cyc["n6"]), "girth": int(cyc["girth"]),
            "rate": float(sc.rate)}


# --------------------------------------------------------------------------- #
#  TransferPolicy: linear-softmax over CONFIG- AND w-INDEPENDENT features
# --------------------------------------------------------------------------- #
class TransferPolicy:
    """Single shared theta (8 params) usable for ANY (Z, w, mp, L).

    Feature vector phi[a] for assigning the current edge to component a in 0..w:
      [0] row-load fraction   cnt_row[r,a] / deg_row[r]
      [1] col-load fraction   cnt_col[c,a] / deg_col[c]
      [2] global-load frac    cnt_glob[a]  / (#placed so far)
      [3] row-empty flag      1 if comp a currently has no edge in row r
      [4] col-empty flag      1 if comp a currently has no edge in col c
      [5] parity-carrier      1 if a == 0   (component 0 holds the parity part B_par)
      [6] coupling offset     a / w                 (normalised band position)
      [7] centred-offset^2    (a/w - 0.5)^2         (lets the policy prefer mid/edge bands)
    Sizes 5..7 depend only on the *relative* component identity, so a 3-component
    (w=2) and a 6-component (w=5) construction share the SAME parameters.
    """
    N_FEAT = 8

    def __init__(self, lr=0.15, ent=0.02, seed=0):
        self.theta = np.zeros(self.N_FEAT)
        self.opt = Adam(self.theta.shape, lr=lr)
        self.ent = ent
        self.rng = np.random.default_rng(seed)

    # ---- per-config geometry cache (rows/cols/degrees), keyed by cfg tuple ---- #
    _geom_cache: dict = {}

    @classmethod
    def geom(cls, cfg: Config):
        key = (cfg.bg, cfg.ils, cfg.Z, cfg.mp, cfg.w)
        g = cls._geom_cache.get(key)
        if g is None:
            rows, cols, E = edge_meta(cfg)
            C = n_components(cfg)
            nrow = int(rows.max()) + 1
            ncol = int(cols.max()) + 1
            row_deg = np.bincount(rows, minlength=nrow)
            col_deg = np.bincount(cols, minlength=ncol)
            # static per-component role features (offset / parity / centred^2)
            a = np.arange(C, dtype=float)
            w = max(cfg.w, 1)
            role = np.zeros((C, 3))
            role[:, 0] = (a == 0).astype(float)        # parity carrier
            role[:, 1] = a / w                          # normalised offset
            role[:, 2] = (a / w - 0.5) ** 2             # centred offset^2
            g = dict(rows=rows, cols=cols, E=E, C=C, nrow=nrow, ncol=ncol,
                     row_deg=row_deg, col_deg=col_deg, role=role)
            cls._geom_cache[key] = g
        return g

    def _features(self, g, e, cnt_row, cnt_col, cnt_glob, placed):
        C = g["C"]
        r, c = g["rows"][e], g["cols"][e]
        phi = np.zeros((C, self.N_FEAT))
        phi[:, 0] = cnt_row[r] / max(g["row_deg"][r], 1)
        phi[:, 1] = cnt_col[c] / max(g["col_deg"][c], 1)
        phi[:, 2] = cnt_glob / max(placed, 1)
        phi[:, 3] = (cnt_row[r] == 0).astype(float)
        phi[:, 4] = (cnt_col[c] == 0).astype(float)
        phi[:, 5:8] = g["role"]
        return phi

    def rollout(self, cfg: Config, greedy=False):
        g = self.geom(cfg)
        C, E = g["C"], g["E"]
        cnt_row = np.zeros((g["nrow"], C))
        cnt_col = np.zeros((g["ncol"], C))
        cnt_glob = np.zeros(C)
        a = np.zeros(E, dtype=np.int64)
        trace = []
        for e in range(E):
            phi = self._features(g, e, cnt_row, cnt_col, cnt_glob, e)
            p = _softmax(phi @ self.theta)
            act = int(np.argmax(p)) if greedy else int(self.rng.choice(C, p=p))
            a[e] = act
            trace.append((phi, p, act))
            r, c = g["rows"][e], g["cols"][e]
            cnt_row[r, act] += 1; cnt_col[c, act] += 1; cnt_glob[act] += 1
        return a, trace

    def greedy_assign(self, cfg: Config):
        a, _ = self.rollout(cfg, greedy=True)
        return a

    # ---- REINFORCE update (advantage * score-grad + entropy bonus) ----------- #
    def update(self, batch):
        grad = np.zeros(self.N_FEAT)
        for trace, adv in batch:
            for phi, p, act in trace:
                grad += adv * (phi[act] - p @ phi)
                H = -np.sum(p * np.log(p + 1e-12))
                grad += self.ent * ((-p * (np.log(p + 1e-12) + H)) @ phi)
        grad /= max(len(batch), 1)
        self.theta = self.opt.step(self.theta, grad)


# --------------------------------------------------------------------------- #
#  per-instance CEM over per-edge categoricals minimising AS count (the baseline)
# --------------------------------------------------------------------------- #
def cem_floor(cfg: Config, n_steps, batch, a_max, beam, pool, elite_frac=0.3,
              smooth=0.7, seed=77, log=None):
    """Classical per-config construction search.  Restarts from a uniform per-edge
    categorical (NO transfer) -- the expensive optimum we benchmark zero-shot against."""
    rows, cols, E = edge_meta(cfg)
    C = n_components(cfg)
    rng = np.random.default_rng(seed)
    p = np.full((E, C), 1.0 / C)
    n_elite = max(2, int(batch * elite_frac))
    best = (np.inf, None)
    hist = []
    for step in range(1, n_steps + 1):
        assigns = [np.array([rng.choice(C, p=p[e]) for e in range(E)]) for _ in range(batch)]
        scores = score_batch(cfg, assigns, a_max, beam, pool)
        elite = np.argsort(scores)[:n_elite]
        freq = np.zeros((E, C))
        for idx in elite:
            freq[np.arange(E), assigns[idx]] += 1.0
        freq /= n_elite
        p = smooth * p + (1 - smooth) * freq
        p = np.clip(p, 1e-3, None); p /= p.sum(axis=1, keepdims=True)
        bi = int(np.argmin(scores))
        if scores[bi] < best[0]:
            best = (float(scores[bi]), np.asarray(assigns[bi]).copy())
        hist.append(float(best[0] // 1e6))
        if log and (step % 5 == 0 or step == 1):
            log(f"    [CEM] step {step:3d}  batch minAS={int(scores.min()//1e6):4d} "
                f"bestAS={int(best[0]//1e6):4d}")
    return best[1], int(best[0] // 1e6), hist


def best_of_random(cfg: Config, n_evals, a_max, beam, pool, seed=123):
    """Pure random search at a matched evaluation budget (#evals == CEM's)."""
    rows, cols, E = edge_meta(cfg)
    C = n_components(cfg)
    rng = np.random.default_rng(seed)
    assigns = [rng.integers(0, C, size=E) for _ in range(n_evals)]
    scores = score_batch(cfg, assigns, a_max, beam, pool)
    bi = int(np.argmin(scores))
    return np.asarray(assigns[bi]).copy(), int(scores[bi] // 1e6)


# --------------------------------------------------------------------------- #
#  TRAIN: REINFORCE over a SET of training configs (shared policy)
# --------------------------------------------------------------------------- #
def train_transfer(policy: TransferPolicy, train_cfgs, n_steps, batch, a_max, beam,
                   pool, log, base_seed=1000):
    hist = {"step": [], "cfg": [], "mean_as": [], "min_as": [], "theta": []}
    cfg_names = list(train_cfgs.keys())
    sel_rng = np.random.default_rng(base_seed)
    for step in range(1, n_steps + 1):
        name = cfg_names[int(sel_rng.integers(0, len(cfg_names)))]
        cfg = train_cfgs[name]
        traces, assigns = [], []
        for _ in range(batch):
            a, tr = policy.rollout(cfg)
            assigns.append(a); traces.append(tr)
        scores = score_batch(cfg, assigns, a_max, beam, pool)
        as_counts = scores // 1e6                 # integer AS counts
        # reward = -AS (use the smooth score for the advantage so n4 breaks ties)
        rewards = -scores / 1e6
        adv = rewards - rewards.mean()
        if adv.std() > 1e-9:
            adv = adv / (adv.std() + 1e-9)
        policy.update(list(zip(traces, adv)))
        hist["step"].append(step); hist["cfg"].append(name)
        hist["mean_as"].append(float(as_counts.mean()))
        hist["min_as"].append(int(as_counts.min()))
        hist["theta"].append([float(t) for t in policy.theta])
        if step % 5 == 0 or step == 1:
            th = ",".join(f"{t:+.2f}" for t in policy.theta)
            log(f"  [PG] step {step:3d}/{n_steps}  cfg={name:13s} "
                f"meanAS={as_counts.mean():7.1f} minAS={int(as_counts.min()):4d}  "
                f"theta=[{th}]")
    return hist


# --------------------------------------------------------------------------- #
#  configs
# --------------------------------------------------------------------------- #
def make_train_cfgs(L):
    # w in {2,3,4} -> C in {3,4,5}; Z in {16,32}; mp spanning low..mid-high rate.
    return {
        "Z16_w2_mp24": Config(bg=2, ils=0, Z=16, mp=24, w=2, L=L, W=max(6, 2 + 2)),
        "Z16_w3_mp13": Config(bg=2, ils=0, Z=16, mp=13, w=3, L=L, W=max(6, 3 + 2)),
        "Z32_w2_mp9":  Config(bg=2, ils=0, Z=32, mp=9,  w=2, L=L, W=max(6, 2 + 2)),
        "Z16_w4_mp24": Config(bg=2, ils=0, Z=16, mp=24, w=4, L=L, W=max(6, 4 + 2)),
    }


def make_test_cfgs(L):
    # HELD-OUT: unseen (Z, w, mp) combos.  w in {3,5} -> C in {4,6}; Z16 & Z32.
    # Each is novel vs the train set (new w*Z*mp triple).
    return {
        "Z16_w3_mp9":  Config(bg=2, ils=0, Z=16, mp=9,  w=3, L=L, W=max(6, 3 + 2)),
        "Z32_w3_mp13": Config(bg=2, ils=0, Z=32, mp=13, w=3, L=L, W=max(6, 3 + 2)),
        "Z16_w5_mp24": Config(bg=2, ils=0, Z=16, mp=24, w=5, L=L, W=max(6, 5 + 2)),
        "Z16_w2_mp13": Config(bg=2, ils=0, Z=16, mp=13, w=2, L=L, W=max(6, 2 + 2)),
    }


# --------------------------------------------------------------------------- #
#  MAIN
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120, help="REINFORCE steps")
    ap.add_argument("--batch", type=int, default=32, help="constructions / step")
    ap.add_argument("--a_max", type=int, default=6)
    ap.add_argument("--beam", type=int, default=1500)
    ap.add_argument("--cem_steps", type=int, default=25)
    ap.add_argument("--cem_batch", type=int, default=28)
    ap.add_argument("--L", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.15)
    ap.add_argument("--ent", type=float, default=0.02)
    # floor-BER verification on the first `floor_k` test configs
    ap.add_argument("--floor_k", type=int, default=2)
    ap.add_argument("--floor_frames", type=int, default=4000)
    ap.add_argument("--floor_snrs", type=float, nargs="+", default=[4.0, 4.5, 5.0])
    ap.add_argument("--out_prefix", type=str, default="transfer")
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    args = ap.parse_args()

    if args.quick:
        args.steps = 6; args.batch = 8; args.cem_steps = 3; args.cem_batch = 8
        args.beam = 800; args.floor_k = 1; args.floor_frames = 200
        args.floor_snrs = [4.0, 4.5]

    logf = open(f"{args.out_prefix}_run.log", "a", buffering=1)
    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")

    nw = n_workers()
    train_cfgs = make_train_cfgs(args.L)
    test_cfgs = make_test_cfgs(args.L)
    t_start = time.time()

    log("=" * 70)
    log("EXPERIMENT B : zero-shot generalisation of a learned construction policy")
    log(f"workers={nw}  steps={args.steps} batch={args.batch} a_max={args.a_max} "
        f"beam={args.beam}  cem={args.cem_steps}x{args.cem_batch}  L={args.L}")
    log("TRAIN configs: " + ", ".join(
        f"{k}(C={n_components(v)},E={edge_meta(v)[2]},R={v.build(seed=0).rate:.2f})"
        for k, v in train_cfgs.items()))
    log("TEST  configs: " + ", ".join(
        f"{k}(C={n_components(v)},E={edge_meta(v)[2]},R={v.build(seed=0).rate:.2f})"
        for k, v in test_cfgs.items()))

    ckpt = {
        "meta": {"steps": args.steps, "batch": args.batch, "a_max": args.a_max,
                 "beam": args.beam, "cem_steps": args.cem_steps,
                 "cem_batch": args.cem_batch, "L": args.L, "workers": nw,
                 "feature_names": ["row_load", "col_load", "glob_load", "row_empty",
                                   "col_empty", "parity_carrier", "offset", "offset_c2"]},
        "train_cfgs": {k: {"Z": v.Z, "w": v.w, "mp": v.mp, "L": v.L,
                           "C": n_components(v), "E": int(edge_meta(v)[2]),
                           "rate": float(v.build(seed=0).rate)}
                       for k, v in train_cfgs.items()},
        "test_cfgs": {k: {"Z": v.Z, "w": v.w, "mp": v.mp, "L": v.L,
                          "C": n_components(v), "E": int(edge_meta(v)[2]),
                          "rate": float(v.build(seed=0).rate)}
                      for k, v in test_cfgs.items()},
    }

    def save():
        with open(f"{args.out_prefix}.json", "w") as f:
            json.dump(ckpt, f, indent=1)

    pool = mp.get_context("spawn").Pool(nw)
    try:
        # ================= TRAIN =================
        log("\n----- TRAIN (REINFORCE on shared TransferPolicy) -----")
        policy = TransferPolicy(lr=args.lr, ent=args.ent, seed=0)
        t0 = time.time()
        thist = train_transfer(policy, train_cfgs, args.steps, args.batch, args.a_max,
                               args.beam, pool, log)
        train_time = time.time() - t0
        theta = [float(t) for t in policy.theta]
        log(f"  trained in {train_time:.0f}s.  learned theta = {theta}")
        ckpt["train_hist"] = thist
        ckpt["theta"] = theta
        ckpt["train_time_s"] = train_time
        # interpretable rules
        fn = ckpt["meta"]["feature_names"]
        rules = sorted(zip(fn, theta), key=lambda t: -abs(t[1]))
        log("  learned design rules (|weight| desc):")
        for name, val in rules:
            log(f"      {name:16s} {val:+.3f}")
        ckpt["rules_sorted"] = [[n, v] for n, v in rules]
        save()
        log(f"  [checkpoint] {args.out_prefix}.json saved after training")

        # ================= TEST (zero-shot vs CEM vs RR vs random) =================
        log("\n----- TEST (zero-shot vs per-instance CEM) -----")
        cem_evals = args.cem_steps * args.cem_batch
        results = {}
        for ti, (name, cfg) in enumerate(test_cfgs.items()):
            log(f"\n  === unseen config {name} "
                f"(Z={cfg.Z} w={cfg.w} mp={cfg.mp} C={n_components(cfg)} "
                f"E={edge_meta(cfg)[2]} R={cfg.build(seed=0).rate:.3f}) ===")
            tcfg0 = time.time()

            # (a) ZERO-SHOT : single greedy rollout of the shared policy
            zs = policy.greedy_assign(cfg)
            zs_d = as_detail(cfg, zs, args.a_max, args.beam)
            log(f"    (a) zero_shot     AS={zs_d['as_count']:4d}  "
                f"minA={zs_d['as_min_a']} n4={zs_d['n4']} g={zs_d['girth']}  (~0 cost)")

            # (c) round_robin
            rr = round_robin_assign(cfg)
            rr_d = as_detail(cfg, rr, args.a_max, args.beam)
            log(f"    (c) round_robin   AS={rr_d['as_count']:4d}  "
                f"minA={rr_d['as_min_a']} n4={rr_d['n4']} g={rr_d['girth']}")

            # (d) best-of-random at CEM's evaluation budget
            ra, ra_as = best_of_random(cfg, cem_evals, args.a_max, args.beam, pool,
                                       seed=2024 + ti)
            ra_d = as_detail(cfg, ra, args.a_max, args.beam)
            log(f"    (d) best_random   AS={ra_d['as_count']:4d}  "
                f"minA={ra_d['as_min_a']} n4={ra_d['n4']}  ({cem_evals} evals)")

            # (b) per-instance CEM (the expensive optimum)
            log(f"    (b) CEM (per-instance, {cem_evals} evals) ...")
            ca, ca_as, cem_h = cem_floor(cfg, args.cem_steps, args.cem_batch,
                                         args.a_max, args.beam, pool, seed=700 + ti,
                                         log=log)
            ca_d = as_detail(cfg, ca, args.a_max, args.beam)
            log(f"    (b) cem           AS={ca_d['as_count']:4d}  "
                f"minA={ca_d['as_min_a']} n4={ca_d['n4']} g={ca_d['girth']}")

            # headline: fraction of the (round_robin -> CEM) improvement that the
            # zero-shot policy recovers.  gap>0 (CEM beats RR) is the informative
            # regime; we also report the raw zs-vs-cem excess so a degenerate gap
            # (CEM ~ RR, or RR already optimal) is still interpretable.
            rr_as, zs_as = rr_d["as_count"], zs_d["as_count"]
            gap = rr_as - ca_as                      # CEM's improvement over RR (AS)
            recov = (rr_as - zs_as) / gap if gap > 0 else None
            # zero-shot's residual excess over CEM, normalised by RR's level
            # (0 => matches CEM ; 1 => no better than RR ; <0 => beats CEM).
            zs_excess_over_cem = (zs_as - ca_as) / rr_as if rr_as > 0 else \
                (0.0 if zs_as == ca_as else float(zs_as - ca_as))
            recov_str = f"{100*recov:.0f}% of it" if recov is not None else \
                "(RR>=CEM gap <=0; see zs-vs-cem)"
            log(f"    >>> RR->CEM gap = {gap}  ;  zero-shot recovers {recov_str}  "
                f"(zs {zs_as} vs cem {ca_as} vs rr {rr_as} vs rand {ra_d['as_count']})")

            entry = {
                "cfg": {"Z": cfg.Z, "w": cfg.w, "mp": cfg.mp, "L": cfg.L,
                        "C": n_components(cfg), "E": int(edge_meta(cfg)[2]),
                        "rate": float(cfg.build(seed=0).rate)},
                "zero_shot": zs_d, "round_robin": rr_d, "best_random": ra_d, "cem": ca_d,
                "zero_shot_assign": [int(x) for x in zs],
                "cem_assign": [int(x) for x in ca],
                "round_robin_assign": [int(x) for x in rr],
                "cem_hist_as": cem_h,
                "gap_rr_minus_cem": int(gap),
                "frac_recovered": float(recov) if recov is not None else None,
                "zs_excess_over_cem_norm": float(zs_excess_over_cem),
                "cem_evals": int(cem_evals),
            }
            results[name] = entry
            ckpt["results"] = results
            save()
            log(f"    [checkpoint] {name} saved  ({time.time()-tcfg0:.0f}s for this config)")

        # ================= FLOOR-BER verification on first floor_k test cfgs =====
        log("\n----- FLOOR-BER verification (measured) -----")
        floor_names = list(test_cfgs.keys())[:max(0, args.floor_k)]
        floor_out = {}
        for name in floor_names:
            cfg = test_cfgs[name]
            entry = results[name]
            assigns = [np.asarray(entry["round_robin_assign"]),
                       np.asarray(entry["zero_shot_assign"]),
                       np.asarray(entry["cem_assign"])]
            anames = ["round_robin", "zero_shot", "cem"]
            log(f"  {name}: measuring floor BER @ {args.floor_snrs} dB "
                f"({args.floor_frames} frames/SNR) for {anames} ...")
            t0 = time.time()
            ber, fer, ferr, nfr = measure_ber_curve(
                cfg, assigns, args.floor_snrs, args.floor_frames,
                frame_seed=7000, pool=pool, frame_chunks=nw)
            log(f"    done in {time.time()-t0:.0f}s")
            for ai, an in enumerate(anames):
                row = "  ".join(f"{args.floor_snrs[si]}dB:BER={ber[ai,si]:.2e}"
                                f"(fe={int(ferr[ai,si])})" for si in range(len(args.floor_snrs)))
                log(f"      {an:12s} {row}")
            floor_out[name] = {
                "snrs": args.floor_snrs, "names": anames,
                "ber": ber.tolist(), "fer": fer.tolist(),
                "ferr": ferr.tolist(), "nfr": nfr.tolist(),
            }
            ckpt["floor"] = floor_out
            save()
            log(f"    [checkpoint] floor-BER for {name} saved")

        # ================= POOLED SUMMARY =================
        recs = [results[n]["frac_recovered"] for n in test_cfgs
                if results[n]["frac_recovered"] is not None]
        excess = [results[n]["zs_excess_over_cem_norm"] for n in test_cfgs]
        wins_vs_rr = sum(results[n]["zero_shot"]["as_count"]
                         <= results[n]["round_robin"]["as_count"] for n in test_cfgs)
        beats_random = sum(results[n]["zero_shot"]["as_count"]
                           <= results[n]["best_random"]["as_count"] for n in test_cfgs)
        matches_cem = sum(results[n]["zero_shot"]["as_count"]
                          <= results[n]["cem"]["as_count"] for n in test_cfgs)
        # geo-mean-ish AS ratios zs/cem and rr/cem (lower zs/cem -> closer to optimum)
        def _ratios(num, den):
            r = []
            for n in test_cfgs:
                a = results[n][num]["as_count"]; b = results[n][den]["as_count"]
                if b > 0:
                    r.append(a / b)
            return r
        zs_over_cem = _ratios("zero_shot", "cem")
        rr_over_cem = _ratios("round_robin", "cem")
        summary = {
            "n_test": len(test_cfgs),
            "mean_frac_recovered": float(np.mean(recs)) if recs else None,
            "median_frac_recovered": float(np.median(recs)) if recs else None,
            "n_informative_gap": len(recs),
            "median_zs_excess_over_cem_norm": float(np.median(excess)) if excess else None,
            "median_zs_over_cem_ratio": float(np.median(zs_over_cem)) if zs_over_cem else None,
            "median_rr_over_cem_ratio": float(np.median(rr_over_cem)) if rr_over_cem else None,
            "zs_beats_or_ties_round_robin": int(wins_vs_rr),
            "zs_beats_or_ties_best_random": int(beats_random),
            "zs_matches_or_beats_cem": int(matches_cem),
            "theta": theta,
            "train_time_s": train_time,
            "cem_evals_per_config": int(cem_evals),
        }
        ckpt["summary"] = summary
        ckpt["total_time_s"] = time.time() - t_start
        save()

        log("\n================ POOLED SUMMARY ================")
        log(f"test configs: {len(test_cfgs)}")
        if recs:
            log(f"zero-shot recovers MEAN {100*np.mean(recs):.0f}% / "
                f"MEDIAN {100*np.median(recs):.0f}% of the round_robin->CEM gap "
                f"({len(recs)}/{len(test_cfgs)} configs where CEM beat RR)")
        if zs_over_cem:
            log(f"AS ratio vs per-instance CEM (1.0 = matches optimum): "
                f"zero-shot median {np.median(zs_over_cem):.2f}x  vs  "
                f"round_robin median {np.median(rr_over_cem):.2f}x")
        log(f"zero-shot <= round_robin (AS):  {wins_vs_rr}/{len(test_cfgs)}")
        log(f"zero-shot <= best_random (AS):  {beats_random}/{len(test_cfgs)}")
        log(f"zero-shot <= CEM (AS):          {matches_cem}/{len(test_cfgs)}")
        log(f"TRAIN cost: one policy in {train_time:.0f}s for ALL configs; "
            f"per-config CEM cost: {cem_evals} evals EACH (x{len(test_cfgs)} = "
            f"{cem_evals*len(test_cfgs)} evals) -- zero-shot test cost = 1 rollout each")
        log(f"learned theta = {theta}")
        log(f"total runtime {time.time()-t_start:.0f}s")
        log(f"[saved] {args.out_prefix}.json")

    finally:
        pool.close(); pool.join()
        logf.close()


if __name__ == "__main__":
    main()
