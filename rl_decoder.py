"""Reinforcement-learning-controlled sliding-window decoder for SC-LDPC codes.

Research idea
-------------
Belief propagation is a Markov process: the message state at iteration k depends
only on the state at k-1.  For a *spatially-coupled* code decoded with a sliding
window, there is a second, coarser Markov chain on top of it -- the window walks
the chain t = 0,1,...,L-1 in the natural "decoding-wave" order, and at every
position the decoder must decide *how hard to work* before it commits the target
symbols and advances.  The fixed-schedule window decoder spends the *same*
`max_iter` on every position; but the decoding wave makes positions wildly
heterogeneous (boundary positions converge in 1-2 iterations, the wavefront needs
many).  That heterogeneity is exactly what an RL agent can exploit.

We therefore cast windowed SC-LDPC decoding as a Markov Decision Process and learn
a control policy with tabular Q-learning (pure NumPy, in the dependency-free spirit
of this repo).  Baseline and RL share the *same* BP engine (`WindowBP`); they differ
only in the controller, so any gain is attributable to the learned policy alone.

MDP
---
* one episode      = decode one received frame, position by position
* a "time step"    = one BP iteration on the current window
* state  s_t       = compact, deployable features of the current window
                     (no ground truth): unsatisfied-check fraction, last-iteration
                     progress, target-symbol reliability, iterations used, an
                     error-propagation flag, and the normalised position t/L
* action a_t       = COMMIT (finalise the target position, advance the window) or
                     CONTINUE with a normalised-min-sum scaling alpha drawn from a
                     small discrete set  ->  the agent learns both *when to stop*
                     and *which alpha to use*, neither of which the fixed decoder
                     can adapt
* reward           = -iter_cost            per CONTINUE      (complexity / latency)
                     -err_cost * (target bit errors)  on COMMIT  (reliability)
                     ground truth is used only for the *training* reward, never in
                     the state, so the learned policy is deployable as-is.

Maximising the return  -sum(iters) - err_cost * sum(errors)  trades decoding
complexity against error rate; sweeping `err_cost` traces a Pareto front that we
show dominates the fixed-iteration window decoder.
"""
from __future__ import annotations
import numpy as np

from decoder import Tanner, MSG_CAP
import channel as ch

# action 0 is reserved for COMMIT; actions 1.. are CONTINUE@alpha_set[k-1]
COMMIT = 0


# --------------------------------------------------------------------------- #
#  stateful single-iteration BP on one window  (mirrors Tanner.decode exactly)
# --------------------------------------------------------------------------- #
class WindowBP:
    """Runs normalised min-sum BP on a fixed window Tanner one iteration at a time.

    Stepping this N times (without early stop) reproduces `Tanner.decode(max_iter=N)`
    bit-for-bit, so the RL controller and the fixed baseline operate on an identical
    decoder -- only the stopping / alpha decisions differ.
    """

    def __init__(self, tan: Tanner):
        self.tan = tan

    def reset(self, llr: np.ndarray):
        t = self.tan
        self.llr = np.asarray(llr, dtype=np.float64)
        self.m_vc = self.llr[t.e_var].copy()         # var -> chk messages (per edge)
        self.total = self.llr.copy()                  # current posterior LLR
        self.hard = (self.llr < 0).astype(np.uint8)
        self.iters = 0
        self.unsat = self._unsat()                    # current # unsatisfied checks
        self.prev_unsat = self.unsat

    def _unsat(self) -> int:
        t = self.tan
        syn = np.bincount(t.e_chk, weights=self.hard[t.e_var].astype(np.float64),
                          minlength=t.num_chk).astype(np.int64) & 1
        return int(syn.sum())

    def step(self, alpha: float):
        """One flooding BP iteration with scaling `alpha` (check update + var update)."""
        t = self.tan
        m_cv = t._check_minsum(self.m_vc, alpha)
        gathered = np.where(t.V_mask, m_cv[t.V_safe], 0.0)
        sum_cv = gathered.sum(axis=1)
        self.total = self.llr + sum_cv
        self.hard = (self.total < 0).astype(np.uint8)
        new_vc = np.clip(self.total[:, None] - gathered, -MSG_CAP, MSG_CAP)
        self.m_vc[t.V_tab[t.V_mask]] = new_vc[t.V_mask]
        self.iters += 1
        self.prev_unsat = self.unsat
        self.unsat = self._unsat()


# --------------------------------------------------------------------------- #
#  the MDP environment
# --------------------------------------------------------------------------- #
class SCWindowEnv:
    """Windowed SC-LDPC decoding as an MDP (see module docstring)."""

    def __init__(self, sc, W=6, alpha_set=(0.5, 0.7, 0.9),
                 ebn0_range=(2.0, 3.5), max_iter_cap=20,
                 iter_cost=1.0, err_cost=60.0, rel_tau=1.0):
        self.sc = sc
        self.W = W
        self.alpha_set = tuple(alpha_set)
        self.n_actions = 1 + len(self.alpha_set)         # COMMIT + CONTINUE@alpha
        self.ebn0_range = ebn0_range
        self.max_iter_cap = max_iter_cap
        self.iter_cost = iter_cost
        self.err_cost = err_cost
        self.rel_tau = rel_tau
        self.blk = sc.nb * sc.Z
        self.windows = sc._window_layout(W)              # cached (tan, v_lo, v_hi) per t
        self.engines = [WindowBP(self.windows[t][0]) for t in range(sc.L)]
        self.n_state_features = 6

    # ---- frame setup ------------------------------------------------------- #
    def reset(self, rng, ebn0=None):
        sc = self.sc
        if ebn0 is None:
            ebn0 = rng.uniform(*self.ebn0_range)
        self.ebn0 = float(ebn0)
        self.sigma = ch.ebn0_to_sigma(ebn0, sc.rate)
        self.cw, self.info_true = sc.encode(rng)
        self.llr_full = sc.make_llr(self.cw, self.sigma, rng)
        self.dec_bits = np.zeros(sc.num_var, dtype=np.uint8)
        self.t = -1
        self.ep_flag = 0.0           # error-propagation risk inherited from previous commit
        self.total_iters = 0
        self._begin_position(0)
        return self._state()

    def _begin_position(self, t):
        """Build the window LLR for position t (finalised bits become known) and
        reset that window's BP engine."""
        sc, blk = self.sc, self.blk
        self.t = t
        tan, v_lo, v_hi = self.windows[t]
        self.v_lo, self.v_hi = v_lo, v_hi
        nloc = (v_hi - v_lo + 1) * blk
        llr_loc = np.empty(nloc, dtype=np.float64)
        for tp in range(v_lo, v_hi + 1):
            g0, g1 = tp * blk, (tp + 1) * blk
            l0 = (tp - v_lo) * blk
            if tp < t:                                   # already finalised -> known
                llr_loc[l0:l0 + blk] = np.where(self.dec_bits[g0:g1] == 0, 30.0, -30.0)
            else:
                llr_loc[l0:l0 + blk] = self.llr_full[g0:g1]
        self.target_lo = (t - v_lo) * blk                # target block in local index
        self.target_hi = self.target_lo + blk
        self.eng = self.engines[t]
        self.eng.reset(llr_loc)

    # ---- state ------------------------------------------------------------- #
    def _state(self):
        eng, tan = self.eng, self.windows[self.t][0]
        nchk = max(1, tan.num_chk)
        f_unsat = eng.unsat / nchk
        f_prog = (eng.prev_unsat - eng.unsat) / nchk     # >0 = improving last iter
        tgt = eng.total[self.target_lo:self.target_hi]
        f_targrel = float(np.mean(np.abs(tgt) < self.rel_tau))   # fraction unreliable
        f_iter = eng.iters / self.max_iter_cap
        f_ep = self.ep_flag
        f_pos = self.t / self.sc.L
        return np.array([f_unsat, f_prog, f_targrel, f_iter, f_ep, f_pos],
                        dtype=np.float64)

    # ---- transition -------------------------------------------------------- #
    def step(self, action):
        """action: COMMIT(0) or CONTINUE with alpha_set[action-1]."""
        sc, blk = self.sc, self.blk
        force_commit = self.eng.iters >= self.max_iter_cap
        if action != COMMIT and not force_commit:
            # ---- CONTINUE: one more BP iteration -----------------------------
            alpha = self.alpha_set[action - 1]
            self.eng.step(alpha)
            self.total_iters += 1
            return self._state(), -self.iter_cost, False, {"commit": False}

        # ---- COMMIT: finalise the target position and advance ----------------
        hard_loc = self.eng.hard
        tgt_hat = hard_loc[self.target_lo:self.target_hi]
        g0 = self.t * blk
        self.dec_bits[g0:g0 + blk] = tgt_hat
        # training reward: bit errors over the target block (ground truth)
        tgt_true = self.cw[g0:g0 + blk]
        n_err = int((tgt_hat != tgt_true).sum())
        reward = -self.err_cost * (n_err / blk)
        # error-propagation flag handed to the next position: did we commit while
        # the window still had unsatisfied checks?  (cheap, ground-truth-free)
        self.ep_flag = 1.0 if self.eng.unsat > 0 else 0.0
        # advance
        if self.t + 1 >= sc.L:
            return self._state(), reward, True, {"commit": True, "n_err": n_err}
        self._begin_position(self.t + 1)
        return self._state(), reward, False, {"commit": True, "n_err": n_err}

    # ---- evaluation helpers ------------------------------------------------ #
    def info_bit_errors(self):
        return int((self.sc.extract_info(self.dec_bits) != self.info_true).sum())

    def rollout(self, controller, rng, ebn0=None):
        """Run one frame under a controller(state, env)->action. Returns metrics."""
        s = self.reset(rng, ebn0=ebn0)
        done = False
        while not done:
            a = controller(s, self)
            s, _, done, _ = self.step(a)
        be = self.info_bit_errors()
        return {"iters": self.total_iters, "bit_err": be,
                "n_info": self.info_true.size, "frame_err": int(be > 0)}


# --------------------------------------------------------------------------- #
#  baseline controllers (share the same BP engine via the env)
# --------------------------------------------------------------------------- #
def fixed_controller(max_iter, alpha=0.8):
    """The classic window decoder: iterate at fixed alpha until the window syndrome
    is satisfied (early stop) or `max_iter` is reached, then commit."""
    def ctrl(state, env):
        if env.eng.unsat == 0 or env.eng.iters >= max_iter:
            return COMMIT
        # pick the CONTINUE action whose alpha is closest to the requested alpha
        k = int(np.argmin([abs(a - alpha) for a in env.alpha_set]))
        return k + 1
    return ctrl


# --------------------------------------------------------------------------- #
#  tabular Q-learning agent  (pure NumPy)
# --------------------------------------------------------------------------- #
class TabularQAgent:
    """Q-learning over a discretised state, epsilon-greedy behaviour policy."""

    def __init__(self, n_actions, bins, lr=0.2, gamma=0.97,
                 eps_start=1.0, eps_end=0.05, seed=0):
        self.n_actions = n_actions
        self.bins = [np.asarray(b, dtype=np.float64) for b in bins]
        self.dims = tuple(len(b) + 1 for b in self.bins)
        self.Q = np.zeros(self.dims + (n_actions,), dtype=np.float64)
        self.N = np.zeros(self.dims + (n_actions,), dtype=np.int64)   # visit counts
        self.lr = lr
        self.gamma = gamma
        self.eps = eps_start
        self.eps_start, self.eps_end = eps_start, eps_end
        self.rng = np.random.default_rng(seed)

    def _disc(self, state):
        return tuple(int(np.digitize(state[i], self.bins[i]))
                     for i in range(len(self.bins)))

    def act(self, state, greedy=False):
        if (not greedy) and self.rng.random() < self.eps:
            return int(self.rng.integers(self.n_actions))
        idx = self._disc(state)
        q = self.Q[idx]
        m = q.max()
        # random tie-break among maximal actions (important early on)
        best = np.flatnonzero(q == m)
        return int(best[self.rng.integers(best.size)])

    def update(self, s, a, r, s2, done):
        i = self._disc(s) + (a,)
        target = r if done else r + self.gamma * self.Q[self._disc(s2)].max()
        self.N[i] += 1
        self.Q[i] += self.lr * (target - self.Q[i])

    def greedy_controller(self):
        def ctrl(state, env):
            return self.act(state, greedy=True)
        return ctrl


def default_bins():
    """Discretisation of the 6 state features (see SCWindowEnv._state)."""
    return [
        np.array([1e-6, 0.02, 0.06, 0.15]),   # f_unsat   -> {0,>0 small,...,large}
        np.array([-1e-9, 0.01, 0.05]),         # f_prog    -> {worse, flat, small+, big+}
        np.array([0.05, 0.2, 0.4]),            # f_targrel -> reliability of target
        np.array([0.15, 0.35, 0.6, 0.85]),     # f_iter    -> budget used
        np.array([0.5]),                       # f_ep      -> {0,1}
        np.array([0.1, 0.5, 0.9]),             # f_pos     -> {head, bulk, tail-ish, tail}
    ]


# --------------------------------------------------------------------------- #
#  training loop
# --------------------------------------------------------------------------- #
def train(env, agent, n_episodes=4000, eps_decay_frac=0.7, seed=0, log_every=500):
    """Tabular Q-learning. Returns a list of (episode, mean_recent_return)."""
    rng = np.random.default_rng(seed)
    decay_eps = max(1, int(n_episodes * eps_decay_frac))
    history, recent = [], []
    for ep in range(1, n_episodes + 1):
        agent.eps = max(agent.eps_end,
                        agent.eps_start - (agent.eps_start - agent.eps_end) * ep / decay_eps)
        s = env.reset(rng)
        done = False
        ret = 0.0
        while not done:
            a = agent.act(s, greedy=False)
            s2, r, done, _ = env.step(a)
            agent.update(s, a, r, s2, done)
            s = s2
            ret += r
        recent.append(ret)
        if ep % log_every == 0:
            mr = float(np.mean(recent[-log_every:]))
            history.append((ep, mr))
            print(f"  ep {ep:5d}  eps={agent.eps:.3f}  mean_return(last {log_every})={mr:8.2f}",
                  flush=True)
    return history


# --------------------------------------------------------------------------- #
#  evaluation
# --------------------------------------------------------------------------- #
def evaluate(env, controller, ebn0_list, n_frames=200, seed=12345,
             target_ferr=40, min_frames=40):
    """BER/FER and average BP-iteration complexity per Eb/N0 under a controller."""
    rng = np.random.default_rng(seed)
    out = {"x": [], "ber": [], "fer": [], "iters": []}
    for ebn0 in ebn0_list:
        be = bits = fe = nf = it = 0
        for _ in range(n_frames):
            m = env.rollout(controller, rng, ebn0=ebn0)
            be += m["bit_err"]; bits += m["n_info"]; fe += m["frame_err"]
            it += m["iters"]; nf += 1
            if nf >= min_frames and fe >= target_ferr:
                break
        out["x"].append(float(ebn0))
        out["ber"].append(be / bits)
        out["fer"].append(fe / nf)
        out["iters"].append(it / nf)
    return out
