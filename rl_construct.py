"""Reinforcement-learning *construction* optimisation for SC-LDPC codes.

Research idea
-------------
The companion module ``rl_decoder.py`` learns *how to decode* a fixed SC-LDPC
code.  This module learns *how to build* the code.  The construction knob is the
**edge spreading**: the systematic edges of the 5G-NR base matrix B are split
across the w+1 component matrices B_0..B_w (variable position t connects to check
positions t, t+1, ..., t+w through B_0, B_1, ..., B_w).  The repo default assigns
every systematic edge to a *random* component; that single random choice swings
the end-to-end frame-error rate by ~7x at a fixed SNR (see ``_explore.py``).  We
search the 3^E assignment space (E = #systematic base edges, ~40 -> ~1e19 codes)
with policy-gradient RL guided by the *real* encode -> AWGN -> windowed-BP -> BER
pipeline.

Construction as a Markov Decision Process
-----------------------------------------
* episode      = build one complete code by assigning the E systematic edges in a
                 fixed (row-major) order, then evaluate it end-to-end.
* time step t  = assign edge e_t to a component a_t in {0..w}.
* state  s_t   = features of edge e_t given the partial construction: running
                 per-(row, component) and per-(column, component) edge counts and
                 the global per-component load.  The counts make this a genuine
                 sequential MDP -- each placement changes the state seen by later
                 placements -- and let the policy learn structural rules (e.g.
                 "spread a row's edges evenly across components", which avoids the
                 clustering that creates short cycles / weak coupling).
* action a_t   = component index in {0..w}.
* reward       = 0 for t < E, then a single terminal reward R = -FER (frame-error
                 rate of the finished code over a batch of common-random-number
                 frames at the training SNR).  Ground truth is used only to score
                 the finished code (design time); the learned *construction* (the
                 assignment vector) is deployed as-is, exactly like any code design.

Maximising E[R] is REINFORCE (Williams 1992 / Bello 2016 neural combinatorial
optimisation): with a factorised policy pi_theta(a) = prod_t pi(a_t|s_t),
    grad J = E[(R - b) * sum_t grad log pi(a_t|s_t)] .
A within-batch baseline b plus *common random numbers* (every code in a batch is
scored on the *same* frames) makes the advantage estimate low-variance even with
few frames per code, which is what makes this affordable on a CPU.

Everything is pure NumPy + multiprocessing, in the dependency-free spirit of the
repo.  The two non-RL search baselines (random search, cross-entropy method) and
a short-cycle / girth analysis of the learned graphs live here too.
"""
from __future__ import annotations
import os
# pin BLAS/Accelerate to 1 thread per process BEFORE importing numpy: we get our
# parallelism from multiprocessing over constructions, and internal threading would
# only oversubscribe the cores.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from dataclasses import dataclass, asdict
import numpy as np

from nr_ldpc import NRLDPCCode
from sc_ldpc import SCLDPCCode
import channel as ch


# --------------------------------------------------------------------------- #
#  configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    bg: int = 2
    ils: int = 0
    Z: int = 16
    mp: int = 8          # rate-matched parity rows (None -> full base graph)
    w: int = 2
    L: int = 30
    W: int = 6           # decoding window
    max_iter: int = 12
    alpha: float = 0.8

    def component(self):
        return NRLDPCCode(self.bg, self.ils, self.Z, mp=self.mp)

    def build(self, assign=None, seed=0):
        return SCLDPCCode(self.component(), w=self.w, L=self.L, seed=seed, assign=assign)


def n_components(cfg: Config) -> int:
    return cfg.w + 1


def edge_meta(cfg: Config):
    """(rows, cols, n_edges) of the systematic base edges in canonical order."""
    sc = cfg.build(seed=0)
    ri, cj = sc.sys_edge_rc
    return ri.copy(), cj.copy(), sc.n_sys_edges


# --------------------------------------------------------------------------- #
#  end-to-end evaluation of one construction  (common random numbers)
# --------------------------------------------------------------------------- #
def _gen_frames(K, N_tx, frame_seed, n_frames):
    """Deterministic CRN frame bank: identical across constructions of one config."""
    rng = np.random.default_rng(frame_seed)
    info = [rng.integers(0, 2, size=K).astype(np.uint8) for _ in range(n_frames)]
    noise = [rng.standard_normal(N_tx) for _ in range(n_frames)]
    return info, noise


# ---- multiprocessing worker (caches the component code per worker) ---------- #
_CACHE = {}


def _worker(task):
    """Evaluate one assignment over a *frame range* [lo, hi) of the CRN bank."""
    cfg_d, assign, ebn0, frame_seed, n_frames, lo, hi = task
    cfg = Config(**cfg_d)
    key = (cfg.bg, cfg.ils, cfg.Z, cfg.mp)
    comp = _CACHE.get(key)
    if comp is None:
        comp = cfg.component()
        _CACHE[key] = comp
    sc = SCLDPCCode(comp, w=cfg.w, L=cfg.L, assign=np.asarray(assign, dtype=np.int64))
    sigma = ch.ebn0_to_sigma(ebn0, sc.rate)
    info_list, noise_list = _gen_frames(sc.K, sc.N_tx, frame_seed, n_frames)
    be = bits = fe = 0
    for idx in range(lo, hi):
        info, noise = info_list[idx], noise_list[idx]
        cw, _ = sc.encode(info)
        llr = np.zeros(sc.num_var)
        llr[sc.tx_mask] = ch.llr_awgn(ch.bpsk(cw[sc.tx_mask]) + sigma * noise, sigma)
        llr[sc.known_mask] = 30.0
        hard = sc.decode_windowed(llr, W=cfg.W, max_iter=cfg.max_iter, alpha=cfg.alpha)
        err = int((sc.extract_info(hard) != info).sum())
        be += err; bits += info.size; fe += int(err > 0)
    return be, bits, fe, hi - lo


def n_workers():
    """Use all cores: the main process is blocked in pool.map during evaluation,
    and each worker is pinned to one thread, so there is no oversubscription."""
    return max(1, os.cpu_count() or 8)


def eval_assignments(cfg: Config, assigns, ebn0, frame_seed, n_frames, pool=None,
                     frame_chunks=1):
    """Evaluate assignments on the *same* CRN frames.  `frame_chunks` splits each
    assignment's frames into that many parallel tasks (use >1 to parallelise a
    single-assignment validation across cores)."""
    cfg_d = asdict(cfg)
    bounds = np.linspace(0, n_frames, frame_chunks + 1).astype(int)
    tasks, owner = [], []
    for ai, a in enumerate(assigns):
        aa = np.asarray(a, dtype=np.int64)
        for ci in range(frame_chunks):
            lo, hi = int(bounds[ci]), int(bounds[ci + 1])
            if hi > lo:
                tasks.append((cfg_d, aa, ebn0, frame_seed, n_frames, lo, hi))
                owner.append(ai)
    raw = pool.map(_worker, tasks) if pool is not None else [_worker(t) for t in tasks]
    agg = [[0, 0, 0, 0] for _ in range(len(assigns))]
    for o, (be, bits, fe, nf) in zip(owner, raw):
        agg[o][0] += be; agg[o][1] += bits; agg[o][2] += fe; agg[o][3] += nf
    out = []
    for be, bits, fe, nf in agg:
        out.append({"ber": be / max(bits, 1), "fer": fe / max(nf, 1), "bit_err": be,
                    "n_info": bits, "frame_err": fe, "n_frames": nf})
    return out


def eval_batch(cfg: Config, assigns, ebn0, frame_seed, n_frames, pool=None):
    """One task per assignment (the common case in a policy-gradient batch)."""
    return eval_assignments(cfg, assigns, ebn0, frame_seed, n_frames, pool=pool,
                            frame_chunks=1)


def reward_of(metric, kind="fer"):
    """Map an evaluation metric dict to a scalar reward (higher = better)."""
    if kind == "fer":
        return -metric["fer"]
    if kind == "logber":
        return -np.log10(max(metric["ber"], 1e-6))
    if kind == "biterr":                    # mean bit errors per frame (finer, noisier)
        return -metric["bit_err"] / metric["n_frames"]
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
#  Adam optimiser (pure NumPy)
# --------------------------------------------------------------------------- #
class Adam:
    def __init__(self, shape, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = np.zeros(shape); self.v = np.zeros(shape); self.t = 0

    def step(self, theta, grad):
        """Gradient *ascent* (maximise objective)."""
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad * grad
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return theta + self.lr * mh / (np.sqrt(vh) + self.eps)


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# --------------------------------------------------------------------------- #
#  Policy 1: per-edge logits  (a "structured bandit" -- no generalisation)
# --------------------------------------------------------------------------- #
class PerEdgePolicy:
    """Independent categorical per edge: theta[E, C].  Most expressive (E*C params),
    but cannot transfer to a different code (one logit row per concrete edge)."""

    def __init__(self, n_edges, n_comp, lr=0.1, ent=0.01, seed=0):
        self.E, self.C = n_edges, n_comp
        self.theta = np.zeros((n_edges, n_comp))
        self.opt = Adam(self.theta.shape, lr=lr)
        self.ent = ent
        self.rng = np.random.default_rng(seed)

    def probs(self):
        return _softmax(self.theta)

    def sample(self):
        p = self.probs()
        a = np.array([self.rng.choice(self.C, p=p[e]) for e in range(self.E)])
        return a, p

    def grad_logp(self, a, p):
        """d log pi(a)/d theta  (E x C):  onehot(a) - p."""
        g = -p.copy()
        g[np.arange(self.E), a] += 1.0
        return g

    def entropy_grad(self, p):
        H = -np.sum(p * np.log(p + 1e-12), axis=1, keepdims=True)
        return -p * (np.log(p + 1e-12) + H)            # dH/dtheta

    def update(self, batch):
        """batch: list of (a, p, advantage)."""
        grad = np.zeros_like(self.theta)
        for a, p, adv in batch:
            grad += adv * self.grad_logp(a, p) + self.ent * self.entropy_grad(p)
        grad /= len(batch)
        self.theta = self.opt.step(self.theta, grad)

    def best_assign(self):
        return np.argmax(self.theta, axis=1)


# --------------------------------------------------------------------------- #
#  Policy 2: feature-based linear softmax  (the sequential constructive MDP)
# --------------------------------------------------------------------------- #
class FeaturePolicy:
    """Linear-softmax policy over per-(edge, component) features built from the
    *running* construction state.  Parameters: C per-component biases + 3 shared
    weights (row-balance, col-balance, global-balance) = C+3 numbers, independent of
    code size -> the learned policy transfers to other Z / L / mp.  This is the
    genuine sequential MDP: counts updated after each placement change later states.
    """

    # per-component shared features: row/col/global load fractions + two binary
    # "this component is still empty for this edge's row / column" flags (which give
    # the policy a sharp knob on the clustering that creates short cycles).
    N_SHARED = 5

    def __init__(self, cfg: Config, lr=0.15, ent=0.02, seed=0):
        self.rows, self.cols, self.E = edge_meta(cfg)
        self.C = n_components(cfg)
        self.nrow = int(self.rows.max()) + 1
        self.ncol = int(self.cols.max()) + 1
        self.row_deg = np.bincount(self.rows, minlength=self.nrow)
        self.col_deg = np.bincount(self.cols, minlength=self.ncol)
        self.F = self.C + self.N_SHARED
        self.theta = np.zeros(self.F)
        self.theta[:self.C] = 0.0
        self.opt = Adam(self.theta.shape, lr=lr)
        self.ent = ent
        self.rng = np.random.default_rng(seed)

    # ---- per-(edge,action) feature matrix given running counts --------------- #
    def _features(self, e, cnt_row, cnt_col, cnt_glob, placed):
        r, c = self.rows[e], self.cols[e]
        phi = np.zeros((self.C, self.F))
        phi[np.arange(self.C), np.arange(self.C)] = 1.0                 # per-comp bias
        phi[:, self.C + 0] = cnt_row[r] / max(self.row_deg[r], 1)       # row load fraction
        phi[:, self.C + 1] = cnt_col[c] / max(self.col_deg[c], 1)       # col load fraction
        phi[:, self.C + 2] = cnt_glob / max(placed, 1)                  # global load fraction
        phi[:, self.C + 3] = (cnt_row[r] == 0).astype(float)           # comp empty for row r
        phi[:, self.C + 4] = (cnt_col[c] == 0).astype(float)           # comp empty for col c
        return phi

    def rollout(self, greedy=False):
        """Construct one assignment; return (assign, trace) where trace lets us form
        the policy-gradient without re-deriving features."""
        cnt_row = np.zeros((self.nrow, self.C))
        cnt_col = np.zeros((self.ncol, self.C))
        cnt_glob = np.zeros(self.C)
        a = np.zeros(self.E, dtype=np.int64)
        trace = []
        for e in range(self.E):
            phi = self._features(e, cnt_row, cnt_col, cnt_glob, e)
            z = phi @ self.theta
            p = _softmax(z)
            act = int(np.argmax(p)) if greedy else int(self.rng.choice(self.C, p=p))
            a[e] = act
            trace.append((phi, p, act))
            r, c = self.rows[e], self.cols[e]
            cnt_row[r, act] += 1; cnt_col[c, act] += 1; cnt_glob[act] += 1
        return a, trace

    def update(self, batch):
        """batch: list of (trace, advantage).  The advantage multiplies only the
        score gradient; the entropy bonus is unconditional (exploration)."""
        grad = np.zeros(self.F)
        for trace, adv in batch:
            grad += adv * self._episode_grad_logp(trace) + self._episode_ent_grad(trace)
        grad /= len(batch)
        self.theta = self.opt.step(self.theta, grad)

    def _episode_grad_logp(self, trace):
        g = np.zeros(self.F)
        for phi, p, act in trace:
            g += phi[act] - p @ phi
        return g

    def _episode_ent_grad(self, trace):
        g = np.zeros(self.F)
        for phi, p, act in trace:
            H = -np.sum(p * np.log(p + 1e-12))
            g += self.ent * ((-p * (np.log(p + 1e-12) + H)) @ phi)
        return g

    def greedy_assign(self):
        a, _ = self.rollout(greedy=True)
        return a


# --------------------------------------------------------------------------- #
#  structural / deterministic baselines
# --------------------------------------------------------------------------- #
def round_robin_assign(cfg: Config):
    """Deterministic 'balanced' spreading: edge e -> e mod (w+1) (a natural,
    human heuristic that perfectly balances the global component loads)."""
    _, _, E = edge_meta(cfg)
    return np.arange(E) % n_components(cfg)


def random_assign(cfg: Config, rng):
    _, _, E = edge_meta(cfg)
    return rng.integers(0, n_components(cfg), size=E)


# --------------------------------------------------------------------------- #
#  training loops  (REINFORCE, CEM, random search) -- equal evaluation budget
# --------------------------------------------------------------------------- #
def _validate(cfg, assign, ebn0, n_frames, frame_seed, pool):
    """Re-score a single champion on a large frame bank, split across all cores."""
    chunks = n_workers() if pool is not None else 1
    return eval_assignments(cfg, [assign], ebn0, frame_seed, n_frames, pool=pool,
                            frame_chunks=chunks)[0]


def train_reinforce(cfg: Config, policy, ebn0, n_steps, batch, frames,
                    pool=None, reward_kind="fer", val_frames=300, val_seed=99,
                    base_seed=1000, log_every=10, normalize_adv=True, verbose=True):
    """Policy-gradient construction search.  Returns history dict + best assignment."""
    feature = isinstance(policy, FeaturePolicy)
    hist = {"step": [], "mean_reward": [], "best_val_fer": [], "evals": []}
    best = {"fer": np.inf, "ber": np.inf, "assign": None}
    seen_evals = 0
    for step in range(1, n_steps + 1):
        frame_seed = base_seed + step          # resample frames each step (no overfit)
        traces, assigns, ps = [], [], []
        for _ in range(batch):
            if feature:
                a, tr = policy.rollout()
                traces.append(tr)
            else:
                a, p = policy.sample(); ps.append(p)
            assigns.append(a)
        metrics = eval_batch(cfg, assigns, ebn0, frame_seed, frames, pool=pool)
        seen_evals += batch
        rewards = np.array([reward_of(m, reward_kind) for m in metrics])
        b = rewards.mean()
        adv = rewards - b
        if normalize_adv and adv.std() > 1e-9:
            adv = adv / (adv.std() + 1e-9)
        if feature:
            policy.update(list(zip(traces, adv)))
        else:
            policy.update([(assigns[i], ps[i], adv[i]) for i in range(batch)])
        # track best by *batch* fer, refine the champion on a big validation bank
        bi = int(np.argmin([m["fer"] for m in metrics]))
        cand = assigns[bi]
        vm = _validate(cfg, cand, ebn0, val_frames, val_seed, pool)
        if vm["fer"] < best["fer"] or (vm["fer"] == best["fer"] and vm["ber"] < best["ber"]):
            best = {"fer": vm["fer"], "ber": vm["ber"], "assign": np.asarray(cand).copy()}
        hist["step"].append(step); hist["mean_reward"].append(float(rewards.mean()))
        hist["best_val_fer"].append(best["fer"]); hist["evals"].append(seen_evals)
        if verbose and (step % log_every == 0 or step == 1):
            extra = ""
            if feature:
                extra = "  theta=[" + ",".join(f"{t:+.2f}" for t in policy.theta) + "]"
            print(f"  [PG] step {step:3d}  meanR={rewards.mean():+.3f} "
                  f"batchFER[min={min(m['fer'] for m in metrics):.3f}] "
                  f"bestVAL_FER={best['fer']:.3f} (BER={best['ber']:.2e}){extra}", flush=True)
    return hist, best


def train_cem(cfg: Config, ebn0, n_steps, batch, frames, elite_frac=0.3,
              pool=None, smooth=0.7, val_frames=300, val_seed=99, base_seed=2000,
              log_every=10, seed=7, verbose=True):
    """Cross-entropy method over per-edge categoricals (strong non-RL baseline)."""
    _, _, E = edge_meta(cfg); C = n_components(cfg)
    rng = np.random.default_rng(seed)
    p = np.full((E, C), 1.0 / C)
    n_elite = max(2, int(batch * elite_frac))
    hist = {"step": [], "best_val_fer": [], "evals": []}
    best = {"fer": np.inf, "ber": np.inf, "assign": None}
    seen = 0
    for step in range(1, n_steps + 1):
        frame_seed = base_seed + step
        assigns = [np.array([rng.choice(C, p=p[e]) for e in range(E)]) for _ in range(batch)]
        metrics = eval_batch(cfg, assigns, ebn0, frame_seed, frames, pool=pool)
        seen += batch
        fers = np.array([m["fer"] for m in metrics])
        elite = np.argsort(fers)[:n_elite]
        freq = np.zeros((E, C))
        for idx in elite:
            a = assigns[idx]
            freq[np.arange(E), a] += 1.0
        freq /= n_elite
        p = smooth * p + (1 - smooth) * freq
        p = np.clip(p, 1e-3, None); p /= p.sum(axis=1, keepdims=True)
        cand = assigns[int(np.argmin(fers))]
        vm = _validate(cfg, cand, ebn0, val_frames, val_seed, pool)
        if vm["fer"] < best["fer"] or (vm["fer"] == best["fer"] and vm["ber"] < best["ber"]):
            best = {"fer": vm["fer"], "ber": vm["ber"], "assign": np.asarray(cand).copy()}
        hist["step"].append(step); hist["best_val_fer"].append(best["fer"]); hist["evals"].append(seen)
        if verbose and (step % log_every == 0 or step == 1):
            print(f"  [CEM] step {step:3d}  batchFER[min={fers.min():.3f}] "
                  f"bestVAL_FER={best['fer']:.3f} (BER={best['ber']:.2e})", flush=True)
    return hist, best


def train_random(cfg: Config, ebn0, n_steps, batch, frames, pool=None,
                 val_frames=300, val_seed=99, base_seed=3000, log_every=10,
                 seed=11, verbose=True):
    """Pure random search at the SAME evaluation budget as RL/CEM (the key baseline)."""
    rng = np.random.default_rng(seed)
    _, _, E = edge_meta(cfg); C = n_components(cfg)
    hist = {"step": [], "best_val_fer": [], "evals": []}
    best = {"fer": np.inf, "ber": np.inf, "assign": None}
    seen = 0
    for step in range(1, n_steps + 1):
        frame_seed = base_seed + step
        assigns = [rng.integers(0, C, size=E) for _ in range(batch)]
        metrics = eval_batch(cfg, assigns, ebn0, frame_seed, frames, pool=pool)
        seen += batch
        fers = np.array([m["fer"] for m in metrics])
        cand = assigns[int(np.argmin(fers))]
        vm = _validate(cfg, cand, ebn0, val_frames, val_seed, pool)
        if vm["fer"] < best["fer"] or (vm["fer"] == best["fer"] and vm["ber"] < best["ber"]):
            best = {"fer": vm["fer"], "ber": vm["ber"], "assign": np.asarray(cand).copy()}
        hist["step"].append(step); hist["best_val_fer"].append(best["fer"]); hist["evals"].append(seen)
        if verbose and (step % log_every == 0 or step == 1):
            print(f"  [RND] step {step:3d}  batchFER[min={fers.min():.3f}] "
                  f"bestVAL_FER={best['fer']:.3f} (BER={best['ber']:.2e})", flush=True)
    return hist, best


# --------------------------------------------------------------------------- #
#  graph analysis: short cycles & girth of the coupled lifted Tanner graph
# --------------------------------------------------------------------------- #
def _var_neighbors(tan):
    """List of check-neighbour sets per variable node."""
    nbr = [[] for _ in range(tan.num_var)]
    for c, v in zip(tan.e_chk.tolist(), tan.e_var.tolist()):
        nbr[v].append(c)
    return nbr


def count_4cycles(sc) -> int:
    """Exact number of 4-cycles in the lifted coupled Tanner graph.
    N4 = sum over variable pairs of C(#shared checks, 2)."""
    tan = sc.full_tanner()
    # accumulate, per check, the variable neighbours; count co-occurring var pairs
    chk_vars = [[] for _ in range(tan.num_chk)]
    for c, v in zip(tan.e_chk.tolist(), tan.e_var.tolist()):
        chk_vars[c].append(v)
    pair_shared = {}
    for vs in chk_vars:
        vs = sorted(set(vs))
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                k = (vs[i], vs[j])
                pair_shared[k] = pair_shared.get(k, 0) + 1
    return int(sum(s * (s - 1) // 2 for s in pair_shared.values()))


def girth(sc, max_g=12, n_seeds=None, seed=0) -> int:
    """Girth (shortest cycle length) of the lifted Tanner graph via BFS from variable
    nodes.  Returns max_g+ if no cycle <= max_g found from the sampled seeds."""
    tan = sc.full_tanner()
    nbr_v = _var_neighbors(tan)
    nbr_c = [[] for _ in range(tan.num_chk)]
    for c, v in zip(tan.e_chk.tolist(), tan.e_var.tolist()):
        nbr_c[c].append(v)
    seeds = range(tan.num_var) if n_seeds is None else \
        np.random.default_rng(seed).choice(tan.num_var, size=min(n_seeds, tan.num_var),
                                            replace=False)
    g = max_g + 2
    for s in seeds:
        # BFS over bipartite graph; nodes encoded (kind, idx): kind 0=var,1=chk
        dist = {("v", s): 0}; parent = {("v", s): None}
        from collections import deque
        q = deque([("v", s)])
        while q:
            kind, idx = q.popleft()
            d = dist[(kind, idx)]
            if d > g // 2:
                continue
            nbrs = [("c", c) for c in nbr_v[idx]] if kind == "v" else [("v", v) for v in nbr_c[idx]]
            for nb in nbrs:
                if nb == parent[(kind, idx)]:
                    continue
                if nb in dist:
                    cyc = d + dist[nb] + 1
                    # a true cycle (even length) needs the two BFS branches to differ
                    if cyc < g and cyc % 2 == 0:
                        g = cyc
                else:
                    dist[nb] = d + 1; parent[nb] = (kind, idx); q.append(nb)
        if g == 4:
            break
    return g


def construction_stats(cfg: Config, assign):
    """Cheap descriptors of a construction for the mechanism analysis."""
    sc = cfg.build(assign=assign)
    C = n_components(cfg)
    comp_load = np.bincount(np.asarray(assign), minlength=C)
    # per coupled check-block-row degree (boundary/bulk profile)
    tan = sc.full_tanner()
    chk_deg = np.bincount(tan.e_chk, minlength=tan.num_chk)
    return {
        "rate": sc.rate,
        "comp_load": comp_load.tolist(),
        "comp_load_std": float(comp_load.std()),
        "n4": count_4cycles(sc),
        "chk_deg_mean": float(chk_deg.mean()),
        "chk_deg_max": int(chk_deg.max()),
    }
