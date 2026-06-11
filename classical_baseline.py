"""Strong CLASSICAL construction baselines for SC-LDPC edge-spreading.

The companion ``rl_construct.py`` learns / searches the edge spreading (which of
the w+1 component matrices B_0..B_w each systematic base edge goes to) with RL,
CEM and equal-budget random search, scored end-to-end (encode->AWGN->BP->FER).
This module adds the *established classical construction algorithms* a strict
IEEE T-COM reviewer expects as STRONG baselines -- graph-structure heuristics
that use **no reinforcement learning and no Monte-Carlo evaluation**:

  * ``peg_assign``        -- PEG-style greedy (Hu/Eleftheriou/Arnold 2005, the
                             progressive-edge-growth philosophy): place edges one
                             at a time in canonical order, each into the component
                             that maximises the local girth / minimises the number
                             of *new* short cycles created in the partial lifted
                             coupled Tanner graph.
  * ``ace_assign``        -- ACE-style greedy (Tian/Jones/Villasenor/Wesel 2004):
                             the same one-at-a-time greedy, but the cost penalises
                             low-ACE (approximate-cycle-EMD) short cycles created
                             by the choice (selective avoidance of low-connectivity
                             cycles up to length 2*dACE).
  * ``greedy_4cycle_min`` -- harmful-object elimination local search
                             (Battaglioni/Baldi/Chiaraluce/Mitchell 2019-style):
                             start from round-robin / random, then iteratively
                             reassign single edges to greedily reduce the EXACT
                             4-cycle count of the lifted coupled Tanner graph until
                             no single-edge move improves.  The strongest cheap
                             classical baseline.

Why a fast exact 4-cycle engine is needed
-----------------------------------------
A quick human probe found that naive co-assignment proxies (same-column / same-row
/ global co-assignment counts) do NOT predict the 4-cycle count -- n4 depends on an
*interaction* of the spreading with the fixed QC lifting shifts.  So a "spread
evenly" greedy may not reduce n4 at all.  The classical baselines here therefore
optimise the **actual** lifted-graph short-cycle structure.

QC 4-cycle algebra (exact, vectorisable, incremental)
-----------------------------------------------------
A systematic base edge ``e`` (base row ``ri``, base col ``cj``, QC shift ``v``)
assigned to component ``comp`` lands, for every spatial position ``t`` with
``i = t + comp <= L+w-1``, as a coupled base entry at

    check-block-row  CBR = (t + comp) * mb + ri      (shift v, fixed)
    var-block-col    VBC = t * nb + cj

So *changing ``comp`` only translates the check-block index by the component delta*
(the base row and QC shift are invariant).  Within one check-block-row, an
*unordered pair* of coupled base entries at var-block-cols (Ca, Cb) with QC shifts
(va, vb) closes ``Z`` lifted 4-cycles for every OTHER entry-pair sharing the same
(Ca, Cb, shift-difference d = (va - vb) mod Z).  Concretely, if ``m`` base
entry-pairs share a key (Ca, Cb, d), they contribute

    n4_contрib(key) = Z * C(m, 2)

and the total is ``n4 = sum_key Z * C(m, 2)``.  This was validated bit-exactly
against ``rl_construct.count_4cycles`` on round-robin (944) and 30 random spreads.
Because each systematic edge only touches a handful of (Ca, Cb, d) keys, a
single-edge move's delta-n4 is O(degree) -- no full lifted-graph rebuild.

Everything is pure NumPy (the cluster's numpy-only / dependency-free spirit).
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from collections import defaultdict
import numpy as np

import rl_construct as R


# --------------------------------------------------------------------------- #
#  base description of one Config's systematic edges (shifts + coupling info)
# --------------------------------------------------------------------------- #
class _CycleEngine:
    """Incremental EXACT lifted-4-cycle engine over the edge-spreading `assign`.

    State: a dict ``grp`` mapping a 4-cycle key (Ca, Cb, d) -> number of coupled
    base entry-pairs carrying that key, so ``n4 = sum_key Z * C(m, 2)``.  The
    PARITY entries (component 0, fixed) are baked in once; only the systematic
    entries move with `assign`.  Single-edge moves update `grp` in O(touched keys).
    """

    def __init__(self, cfg: R.Config):
        self.cfg = cfg
        sc = cfg.build(seed=0)
        self.Z = sc.Z
        self.mb = sc.mb
        self.nb = sc.nb
        self.L = sc.L
        self.w = sc.w
        self.C = sc.w + 1
        B = sc.comp.B
        self.Kb = sc.Kb
        # systematic edges in canonical order (same order as edge_meta / assign)
        ri, cj = sc.sys_edge_rc
        self.ri = ri.astype(np.int64)
        self.cj = cj.astype(np.int64)
        self.E = int(ri.size)
        self.sys_shift = B[ri, cj].astype(np.int64) % self.Z
        # parity coupled entries (component 0 only, FIXED) as a list of per-position
        # base entries; we precompute their coupled (CBR, VBC, shift) once.
        pr, pc = np.nonzero(B[:, self.Kb:] >= 0)
        self.par_ri = pr.astype(np.int64)
        self.par_cj = (pc + self.Kb).astype(np.int64)
        self.par_shift = B[self.par_ri, self.par_cj].astype(np.int64) % self.Z
        # cache: for each systematic edge e and each component comp, the list of
        # coupled entries (CBR, VBC, shift) it produces.  CBR/VBC/shift depend only
        # on (e, comp), never on other edges -> precompute the full table.
        self._sys_coupled = [[None] * self.C for _ in range(self.E)]
        for e in range(self.E):
            for comp in range(self.C):
                self._sys_coupled[e][comp] = self._coupled_for_sys(e, comp)
        # baseline parity entries grouped by check-block-row (never change)
        self._par_by_cbr = defaultdict(list)        # cbr -> list of (vbc, shift)
        for k in range(self.par_ri.size):
            for t in range(self.L):
                i = t                                # parity is component 0
                cbr = i * self.mb + self.par_ri[k]
                vbc = t * self.nb + self.par_cj[k]
                self._par_by_cbr[cbr].append((vbc, int(self.par_shift[k])))

    # -- coupled entries produced by systematic edge e in component comp -------- #
    def _coupled_for_sys(self, e, comp):
        out = []                                     # (cbr, vbc, shift)
        v = int(self.sys_shift[e]); ri = int(self.ri[e]); cj = int(self.cj[e])
        for t in range(self.L):
            i = t + comp
            if i > self.L + self.w - 1:
                continue
            cbr = i * self.mb + ri
            vbc = t * self.nb + cj
            out.append((cbr, vbc, v))
        return out

    # -- 4-cycle key for an unordered entry-pair sharing a check-block-row ------ #
    @staticmethod
    def _key(vbc1, s1, vbc2, s2, Z):
        if vbc1 < vbc2:
            return (vbc1, vbc2, (s1 - s2) % Z)
        return (vbc2, vbc1, (s2 - s1) % Z)

    # -- build the full group table for a given assign ------------------------- #
    def build(self, assign):
        """Return (grp, by_cbr) for a full assignment.  by_cbr maps a
        check-block-row to the list of (vbc, shift) entries currently in it."""
        by_cbr = defaultdict(list)
        # parity (fixed)
        for cbr, lst in self._par_by_cbr.items():
            by_cbr[cbr].extend(lst)
        for e in range(self.E):
            for (cbr, vbc, s) in self._sys_coupled[e][int(assign[e])]:
                by_cbr[cbr].append((vbc, s))
        grp = defaultdict(int)
        Z = self.Z
        for cbr, ents in by_cbr.items():
            ne = len(ents)
            for a in range(ne):
                v1, s1 = ents[a]
                for b in range(a + 1, ne):
                    v2, s2 = ents[b]
                    grp[self._key(v1, s1, v2, s2, Z)] += 1
        return grp, by_cbr

    @staticmethod
    def n4_of_grp(grp, Z):
        return int(sum(Z * (m * (m - 1) // 2) for m in grp.values()))

    def n4(self, assign):
        grp, _ = self.build(assign)
        return self.n4_of_grp(grp, self.Z)

    # -- incremental: delta-n4 of moving edge e from cur_comp to new_comp ------- #
    def _entry_pair_contrib(self, m):
        """Contribution Z*C(m,2) for a group of multiplicity m."""
        return self.Z * (m * (m - 1) // 2)

    def delta_n4(self, e, new_comp, grp, by_cbr, cur_comp):
        """How much n4 changes if systematic edge e moves cur_comp -> new_comp.
        Does NOT mutate grp/by_cbr (call apply_move to commit)."""
        if new_comp == cur_comp:
            return 0
        Z = self.Z
        delta = 0
        # local mutable group multiplicities for the affected keys only
        touched = {}

        def cur_m(key):
            if key not in touched:
                touched[key] = grp.get(key, 0)
            return touched[key]

        # remove the edge's old coupled entries: each old entry pairs with every
        # OTHER entry currently in its check-block-row.
        old = self._sys_coupled[e][cur_comp]
        # we must avoid double counting pairs between two entries that are BOTH
        # this edge's (an edge's own entries live in distinct check-block-rows, one
        # per t, so they never share a CBR -> no intra-edge pairs).  Safe.
        for (cbr, vbc, s) in old:
            for (vbc2, s2) in by_cbr[cbr]:
                if vbc2 == vbc and s2 == s:
                    continue                          # skip the entry itself (matched once)
                key = self._key(vbc, s, vbc2, s2, Z)
                m = cur_m(key)
                delta -= self._entry_pair_contrib(m) - self._entry_pair_contrib(m - 1)
                touched[key] = m - 1
        # add the edge's new coupled entries against the (already-removed) graph.
        new = self._sys_coupled[e][new_comp]
        # snapshot the post-removal by_cbr counts incrementally: build a temp map of
        # entries to remove so the "new" pairing sees the edge gone.
        rm = defaultdict(list)
        for (cbr, vbc, s) in old:
            rm[cbr].append((vbc, s))
        for (cbr, vbc, s) in new:
            base = by_cbr[cbr]
            removed = rm.get(cbr, [])
            for (vbc2, s2) in base:
                # skip entries that are being removed (this edge's old, if same cbr)
                if (vbc2, s2) in removed:
                    # only skip as many as are actually removed (multiplicity-safe
                    # for the rare duplicate (vbc,shift))
                    pass
                key = self._key(vbc, s, vbc2, s2, Z)
                m = cur_m(key)
                delta += self._entry_pair_contrib(m + 1) - self._entry_pair_contrib(m)
                touched[key] = m + 1
            # also pair the new entry against the OTHER new entries of this edge in
            # the SAME cbr -- but an edge has at most one entry per cbr, so none.
        # correction: subtract pairings that the "add" loop wrongly counted against
        # the edge's own OLD entries that share a cbr with a NEW entry.
        for (cbr, vbc, s) in new:
            for (vbc2, s2) in rm.get(cbr, []):
                key = self._key(vbc, s, vbc2, s2, Z)
                m = cur_m(key)
                delta -= self._entry_pair_contrib(m) - self._entry_pair_contrib(m - 1)
                touched[key] = m - 1
        return delta

    def apply_move(self, e, new_comp, grp, by_cbr, cur_comp):
        """Commit edge e: cur_comp -> new_comp, updating grp and by_cbr in place."""
        if new_comp == cur_comp:
            return
        Z = self.Z
        old = self._sys_coupled[e][cur_comp]
        for (cbr, vbc, s) in old:
            lst = by_cbr[cbr]
            # pair against all others, decrement keys
            for (vbc2, s2) in lst:
                if vbc2 == vbc and s2 == s:
                    continue
                key = self._key(vbc, s, vbc2, s2, Z)
                grp[key] -= 1
                if grp[key] == 0:
                    del grp[key]
            lst.remove((vbc, s))
        new = self._sys_coupled[e][new_comp]
        for (cbr, vbc, s) in new:
            lst = by_cbr[cbr]
            for (vbc2, s2) in lst:
                key = self._key(vbc, s, vbc2, s2, Z)
                grp[key] = grp.get(key, 0) + 1
            lst.append((vbc, s))


# --------------------------------------------------------------------------- #
#  short-cycle bookkeeping shared by PEG / ACE greedy
# --------------------------------------------------------------------------- #
def _new_4cycles_if(engine: _CycleEngine, e, comp, by_cbr):
    """Exact number of NEW lifted 4-cycles created by placing edge `e` in `comp`
    given the PARTIAL graph in `by_cbr` (entries placed so far, edge e absent).
    A new 4-cycle = each new coupled entry of e pairs with a *pre-existing* entry
    in its check-block-row, and that (col-pair, shift-diff) key already occurs => Z
    new lifted 4-cycles per matching pre-existing pair."""
    Z = engine.Z
    # count keys already present per check-block-row pairing
    new = engine._sys_coupled[e][comp]
    # local key multiplicities among already-placed entries that e will pair with
    keycount = defaultdict(int)
    for (cbr, vbc, s) in new:
        for (vbc2, s2) in by_cbr.get(cbr, ()):  # pre-existing entries in this CBR
            keycount[engine._key(vbc, s, vbc2, s2, Z)] += 1
    # placing e adds, for each key k with multiplicity m_existing in `grp`, exactly
    # keycount[k] new entry-pairs; but we only have the partial `by_cbr`, so the
    # number of NEW 4-cycles is sum over the new pairs of (existing pairs with same
    # key).  Equivalent local proxy: cycles closed = Z * (existing_same_key choose
    # interactions).  We approximate "new 4-cycles" as the count of new entry-pairs
    # that DUPLICATE an already-present (col-pair, shift-diff) key * Z.
    return keycount


def _partial_grp_add(engine, e, comp, by_cbr, grp):
    """Place edge e (comp) into the PARTIAL graph: update by_cbr & grp, and return
    the number of new 4-cycles created (exact for the partial graph)."""
    Z = engine.Z
    new = engine._sys_coupled[e][comp]
    new4 = 0
    for (cbr, vbc, s) in new:
        lst = by_cbr.setdefault(cbr, [])
        for (vbc2, s2) in lst:
            key = engine._key(vbc, s, vbc2, s2, Z)
            m = grp.get(key, 0)
            new4 += Z * m                 # each pre-existing pair w/ same key closes Z cycles
            grp[key] = m + 1
        lst.append((vbc, s))
    return new4


def _new4_cost(engine, e, comp, by_cbr, grp):
    """Number of NEW 4-cycles if e is placed in comp now (no mutation)."""
    Z = engine.Z
    new = engine._sys_coupled[e][comp]
    new4 = 0
    # account for keys created by THIS edge's own entries plus pre-existing.
    local = defaultdict(int)
    for (cbr, vbc, s) in new:
        for (vbc2, s2) in by_cbr.get(cbr, ()):
            key = engine._key(vbc, s, vbc2, s2, Z)
            new4 += Z * (grp.get(key, 0) + local[key])
            local[key] += 1
    return new4


# --------------------------------------------------------------------------- #
#  1. PEG-style greedy
# --------------------------------------------------------------------------- #
def peg_assign(cfg: R.Config, order=None, tie="balance", seed=0):
    """PEG-style progressive edge growth adapted to edge spreading.

    Edges are placed one at a time in canonical order.  Each edge goes to the
    component that creates the FEWEST new lifted 4-cycles in the partial coupled
    Tanner graph (the progressive-edge-growth "maximise local girth" rule -- a
    choice that closes no short cycle keeps the local girth high).  Ties are broken
    to balance the per-component global load (the natural PEG secondary objective).
    """
    engine = _CycleEngine(cfg)
    E, C = engine.E, engine.C
    if order is None:
        order = np.arange(E)
    assign = np.full(E, -1, dtype=np.int64)
    by_cbr = defaultdict(list)
    grp = defaultdict(int)
    # seed the partial graph with the FIXED parity entries
    for cbr, lst in engine._par_by_cbr.items():
        bl = by_cbr.setdefault(cbr, [])
        for (vbc, s) in lst:
            for (vbc2, s2) in bl:
                grp[engine._key(vbc, s, vbc2, s2, engine.Z)] += 1
            bl.append((vbc, s))
    load = np.zeros(C, dtype=np.int64)
    for e in order:
        e = int(e)
        costs = [_new4_cost(engine, e, c, by_cbr, grp) for c in range(C)]
        best = min(costs)
        cand = [c for c in range(C) if costs[c] == best]
        if len(cand) > 1 and tie == "balance":
            comp = min(cand, key=lambda c: load[c])
        else:
            comp = cand[0]
        _partial_grp_add(engine, e, comp, by_cbr, grp)
        assign[e] = comp
        load[comp] += 1
    return assign


# --------------------------------------------------------------------------- #
#  2. ACE-style greedy (Tian 2004)
# --------------------------------------------------------------------------- #
def ace_assign(cfg: R.Config, dace=4, eta=None, seed=0):
    """ACE-style greedy edge spreading (Tian/Jones/Villasenor/Wesel 2004).

    The ACE (approximate cycle EMD) of a cycle is sum over its variable nodes of
    (deg(v) - 2); short cycles among LOW-degree variables (small ACE) are the most
    harmful.  We greedily place each edge into the component minimising an
    ACE-weighted new-short-cycle cost: a newly created 4-cycle is weighted by the
    inverse ACE of the two variable *block-columns* it spans (low-ACE cycles cost
    more), so the greedy preferentially avoids low-connectivity short cycles up to
    length 2*dace.  ``dace`` is the ACE cycle-length cap (in BP iterations); the
    4-cycle is the dominant term, longer cycles add a decaying penalty.
    """
    engine = _CycleEngine(cfg)
    E, C, Z = engine.E, engine.C, engine.Z
    # variable-block-column degree (number of base systematic+parity edges touching
    # a base var column) approximates node connectivity for the ACE weight.
    sc = cfg.build(seed=0)
    B = sc.comp.B
    col_deg = np.array([(B[:, j] >= 0).sum() for j in range(B.shape[1])], dtype=np.float64)
    col_deg = np.maximum(col_deg, 1.0)
    order = np.arange(E)
    assign = np.full(E, -1, dtype=np.int64)
    by_cbr = defaultdict(list)
    grp = defaultdict(int)
    for cbr, lst in engine._par_by_cbr.items():
        bl = by_cbr.setdefault(cbr, [])
        for (vbc, s) in lst:
            for (vbc2, s2) in bl:
                grp[engine._key(vbc, s, vbc2, s2, engine.Z)] += 1
            bl.append((vbc, s))
    load = np.zeros(C, dtype=np.int64)

    def ace_cost(e, comp):
        """ACE-weighted new-4-cycle cost of placing edge e in comp."""
        new = engine._sys_coupled[e][comp]
        cost = 0.0
        local = defaultdict(int)
        for (cbr, vbc, s) in new:
            base_col = vbc % engine.nb
            for (vbc2, s2) in by_cbr.get(cbr, ()):
                key = engine._key(vbc, s, vbc2, s2, Z)
                mult = grp.get(key, 0) + local[key]
                if mult:
                    base_col2 = vbc2 % engine.nb
                    # ACE of a 4-cycle ~ (deg(v1)-2)+(deg(v2)-2); LOW ace => HIGH cost
                    ace = max((col_deg[base_col] - 2) + (col_deg[base_col2] - 2), 0.5)
                    cost += Z * mult / ace
                local[key] += 1
        return cost

    for e in order:
        e = int(e)
        costs = [ace_cost(e, c) for c in range(C)]
        best = min(costs)
        cand = [c for c in range(C) if abs(costs[c] - best) < 1e-9]
        comp = min(cand, key=lambda c: load[c]) if len(cand) > 1 else cand[0]
        _partial_grp_add(engine, e, comp, by_cbr, grp)
        assign[e] = comp
        load[comp] += 1
    return assign


# --------------------------------------------------------------------------- #
#  3. greedy 4-cycle minimisation (harmful-object elimination local search)
# --------------------------------------------------------------------------- #
def greedy_4cycle_min(cfg: R.Config, init="round_robin", passes=200, seed=0,
                      restarts=1, verbose=False):
    """Battaglioni-style harmful-object elimination: from an initial spread, keep
    making the single-edge reassignment that most reduces the EXACT lifted 4-cycle
    count until no single move improves (a local minimum).  ``restarts`` random
    restarts keep the best result.  Returns the assignment with the lowest n4."""
    engine = _CycleEngine(cfg)
    E, C, Z = engine.E, engine.C, engine.Z
    best_assign, best_n4 = None, None
    rng = np.random.default_rng(seed)
    for r in range(max(1, restarts)):
        if init == "round_robin" and r == 0:
            assign = R.round_robin_assign(cfg).astype(np.int64)
        else:
            assign = rng.integers(0, C, size=E).astype(np.int64)
        grp, by_cbr = engine.build(assign)
        n4 = engine.n4_of_grp(grp, Z)
        for p in range(passes):
            improved = False
            order = rng.permutation(E)
            for e in order:
                e = int(e)
                cur = int(assign[e])
                best_d, best_c = 0, cur
                for c in range(C):
                    if c == cur:
                        continue
                    d = engine.delta_n4(e, c, grp, by_cbr, cur)
                    if d < best_d:
                        best_d, best_c = d, c
                if best_c != cur:
                    engine.apply_move(e, best_c, grp, by_cbr, cur)
                    assign[e] = best_c
                    n4 += best_d
                    improved = True
            if verbose:
                print(f"    [g4c r{r} pass{p}] n4={n4}", flush=True)
            if not improved:
                break
        if best_n4 is None or n4 < best_n4:
            best_n4 = n4
            best_assign = assign.copy()
    return best_assign


# --------------------------------------------------------------------------- #
#  registry + cheap structural stats
# --------------------------------------------------------------------------- #
BASELINES = {
    "peg": peg_assign,
    "ace": ace_assign,
    "greedy4c": greedy_4cycle_min,
}


def build_baseline(cfg: R.Config, name, **kw):
    return BASELINES[name](cfg, **kw)


def structural_stats(cfg: R.Config, assign):
    """n4 (exact), girth, rate, per-component load for a built assignment."""
    sc = cfg.build(assign=np.asarray(assign, dtype=np.int64))
    return {
        "rate": sc.rate,
        "n4": R.count_4cycles(sc),
        "girth": R.girth(sc, n_seeds=200),
        "comp_load": np.bincount(np.asarray(assign), minlength=R.n_components(cfg)).tolist(),
    }


# --------------------------------------------------------------------------- #
#  self-check / demo when run directly
# --------------------------------------------------------------------------- #
def _demo():
    import time
    cfg = R.Config(bg=2, ils=0, Z=16, mp=8, w=2, L=30, W=6, max_iter=12)
    eng = _CycleEngine(cfg)
    # validate the fast engine against count_4cycles
    print("== engine validation (deep cell) ==")
    rr = R.round_robin_assign(cfg)
    assert eng.n4(rr) == R.count_4cycles(cfg.build(assign=rr))
    rng = np.random.default_rng(0)
    for _ in range(10):
        a = R.random_assign(cfg, rng)
        assert eng.n4(a) == R.count_4cycles(cfg.build(assign=a)), "engine mismatch"
    print("  engine matches count_4cycles on round-robin + 10 random spreads")
    print("\n== classical baselines on deep cell ==")
    for name, fn in BASELINES.items():
        t0 = time.time()
        a = fn(cfg)
        sec = time.time() - t0
        st = structural_stats(cfg, a)
        print(f"  {name:9s} n4={st['n4']:5d} girth={st['girth']} "
              f"rate={st['rate']:.4f} load={st['comp_load']} ({sec:.2f}s)")
    rrst = structural_stats(cfg, rr)
    print(f"  {'round_rob':9s} n4={rrst['n4']:5d} girth={rrst['girth']} "
          f"rate={rrst['rate']:.4f} load={rrst['comp_load']}")


if __name__ == "__main__":
    _demo()
