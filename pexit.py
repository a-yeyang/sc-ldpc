"""Protograph EXIT / PEXIT density-evolution threshold analyzer for SC-LDPC codes.

This is the *analysis-grounded, frame-free* objective for the RL edge-spreading
construction paper (gap-filler (1) in docs/TCOM_PLAN.md S5): it gives a principled
decoding-threshold predictor that does not require any Monte-Carlo frames, to
answer the "results are only Monte-Carlo, no theory" reviewer concern.

What it computes
----------------
The protograph-based EXIT (PEXIT) density evolution of Liva & Chiani
(GLOBECOM 2007) for the BIAWGN channel.  A protograph is a small bipartite
graph with `nb` variable-node *types* and `mb` check-node *types*; the base
matrix `b[i][j]` gives the number of parallel edges between check-type i and
var-type j (5G NR base graphs are simple, so entries are 0/1).  PEXIT tracks a
mutual-information value on every edge and iterates variable->check and
check->variable updates; the ensemble *decodes* at a channel parameter sigma_ch
if the a-posteriori MI of every (non-known) variable type converges to 1.

The threshold is found by bisecting Eb/N0 (converted to the channel noise sigma
with the SC code rate, matching `channel.ebn0_to_sigma`).

The SC protograph
-----------------
For an `assign` vector (length E = #systematic base edges, values 0..w) we build
the *band-diagonal* protograph: variable position t in 0..L-1 connects to check
position i = t+a through component B_a (a=0..w).  Systematic base edge e lives in
component B_assign[e]; the parity part lives entirely in B_0.  The chain is
TERMINATED: the last w variable positions have their systematic bits known
(forced to zero), which is exactly the boundary that makes finite-L SC beat the
uncoupled block code (threshold saturation).  Getting this boundary right is the
whole point -- without it the SC threshold collapses to the component threshold.

Channel / consistent-Gaussian model
------------------------------------
BPSK over AWGN with noise std `sigma` (Es=1) gives a channel LLR L = 2y/sigma^2
that is consistent-Gaussian with mean 2/sigma^2 and variance 4/sigma^2, i.e.
LLR ~ N(m/2, m) with m = 4/sigma^2.  The "EXIT" channel parameter is therefore
sigma_ch = sqrt(m) = 2/sigma, and the channel mutual information is J(2/sigma).
Punctured variable types get sigma_ch = 0 (no observation); known/terminated
variable types get sigma_ch = +inf (I_ch = 1).

Pure NumPy, no scipy: J/Jinv use the standard ten-Brink closed-form polynomial
approximation.
"""
from __future__ import annotations
import numpy as np

from rl_construct import Config, edge_meta, n_components
import channel as ch


# --------------------------------------------------------------------------- #
#  J(.) and J^{-1}(.)  --  ten Brink closed-form approximation (no scipy)
# --------------------------------------------------------------------------- #
# J(sigma) = mutual information of a BIAWGN-consistent LLR with variance
# sigma^2 (mean sigma^2/2).  The ten Brink (2004) piecewise-polynomial fit is the
# standard dependency-free choice; max abs error ~ 1e-4 over the full range.
_H1, _H2, _H3 = 0.3073, 0.8935, 1.1064
_W1, _W2, _W3 = 0.8935, 0.8935, 1.1064  # (unused alt constants kept for clarity)

_A_J1, _B_J1, _C_J1 = -0.0421061, 0.209252, -0.00640081
_A_J2, _B_J2, _C_J2, _D_J2 = 0.00181491, -0.142675, -0.0822054, 0.0549608
_SIGMA_STAR = 1.6363

# Jinv coefficients (sigma as a function of mutual information I)
_A_S1, _B_S1, _C_S1 = 1.09542, 0.214217, 2.33727
_A_S2, _B_S2, _C_S2 = 0.706692, 0.386013, -1.75017
_I_STAR = 0.3646


def J(sigma):
    """Mutual information of a consistent-Gaussian LLR with std `sigma`.

    Vectorised; clips to [0, 1].  J(0)=0, J(+inf)=1."""
    s = np.asarray(sigma, dtype=np.float64)
    out = np.empty_like(s)
    # handle the +inf / very-large tail explicitly -> MI = 1
    big = ~np.isfinite(s) | (s > 10.0)
    sml = s <= _SIGMA_STAR
    mid = (~sml) & (~big)
    # small-sigma branch
    ss = s[sml]
    out[sml] = (_A_J1 * ss ** 3 + _B_J1 * ss ** 2 + _C_J1 * ss)
    # mid branch
    sm = s[mid]
    out[mid] = 1.0 - np.exp(_A_J2 * sm ** 3 + _B_J2 * sm ** 2 + _C_J2 * sm + _D_J2)
    out[big] = 1.0
    return np.clip(out, 0.0, 1.0)


def Jinv(I):
    """Inverse of J: the LLR std that achieves mutual information `I`.

    Vectorised; I in [0,1] -> sigma in [0, +inf)."""
    Ii = np.clip(np.asarray(I, dtype=np.float64), 0.0, 1.0 - 1e-12)
    out = np.empty_like(Ii)
    lo = Ii <= _I_STAR
    hi = ~lo
    Il = Ii[lo]
    out[lo] = _A_S1 * Il ** 2 + _B_S1 * Il + _C_S1 * np.sqrt(Il)
    Ih = Ii[hi]
    out[hi] = -_A_S2 * np.log(_B_S2 * (1.0 - Ih)) - _C_S2 * Ih
    return out


def _as_scalar(x):
    a = np.asarray(x, dtype=np.float64)
    return a if a.ndim else a.reshape(1)


def J_scalar(sigma):
    return float(J(_as_scalar(sigma))[0]) if np.ndim(sigma) == 0 else J(sigma)


def Jinv_scalar(I):
    return float(Jinv(_as_scalar(I))[0]) if np.ndim(I) == 0 else Jinv(I)


# --------------------------------------------------------------------------- #
#  protograph construction
# --------------------------------------------------------------------------- #
def component_protograph(cfg: Config):
    """Protograph (base matrix as 0/1 multiplicity) of the *uncoupled* component
    code, plus the per-column channel kind.

    Returns (Bproto[mb,nb], var_kind[nb]) where var_kind in {'rx','punct'}.
    The first two systematic columns are punctured (no channel observation)."""
    comp = cfg.component()
    B = comp.B
    mb, nb = B.shape
    Bproto = (B >= 0).astype(np.int64)
    var_kind = ["rx"] * nb
    var_kind[0] = "punct"
    var_kind[1] = "punct"
    return Bproto, var_kind


def sc_protograph(cfg: Config, assign):
    """Band-diagonal SC protograph with termination boundary, for `assign`.

    Layout mirrors sc_ldpc.SCLDPCCode exactly:
      - variable positions t = 0..L-1, each a block of `nb` base columns
      - check positions i = 0..L+w-1, each a block of `mb` base rows
      - component B_a connects var position t to check position i=t+a
      - systematic base edge e is in component B_assign[e]; parity part in B_0
      - the first 2 systematic cols of every position are punctured
      - the last w variable positions are *terminated* (systematic bits known=0)

    Returns a dict describing the global protograph:
      Bproto  : (MB, NB) int multiplicity matrix  (MB=(L+w)*mb, NB=L*nb)
      var_kind: list of length NB, each in {'rx','punct','known'}
      nb, mb, L, w
    """
    comp = cfg.component()
    B = comp.B
    mb, nb, Kb, w, L = comp.mb, comp.nb, comp.Kb, cfg.w, cfg.L
    assign = np.asarray(assign, dtype=np.int64)

    # base-edge -> component split (identical to SCLDPCCode._edge_spread)
    ri, cj = np.nonzero(B[:, :Kb] >= 0)
    assert assign.shape == (ri.size,), f"assign length {ri.size} expected"
    comps = [np.zeros((mb, nb), dtype=np.int64) for _ in range(w + 1)]
    comps[0][:, Kb:] = (B[:, Kb:] >= 0).astype(np.int64)          # parity -> B_0
    for e in range(ri.size):
        comps[assign[e]][ri[e], cj[e]] = 1                        # systematic -> B_a

    MB = (L + w) * mb
    NB = L * nb
    Bproto = np.zeros((MB, NB), dtype=np.int64)
    for a in range(w + 1):
        ca = comps[a]
        rr, cc = np.nonzero(ca > 0)
        for t in range(L):
            i = t + a
            if i > L + w - 1:
                continue
            gr = i * mb + rr
            gc = t * nb + cc
            Bproto[gr, gc] += ca[rr, cc]

    # per global variable-type channel kind
    var_kind = []
    for t in range(L):
        terminated = (t >= L - w)
        for j in range(nb):
            if j < 2:
                var_kind.append("punct")          # punctured systematic cols 0,1
            elif terminated and j < Kb:
                var_kind.append("known")           # terminated systematic bits (=0)
            else:
                var_kind.append("rx")
    return {"Bproto": Bproto, "var_kind": var_kind,
            "nb": nb, "mb": mb, "L": L, "w": w}


def sc_rate(cfg: Config, assign=None):
    """SC transmitted code rate (same as SCLDPCCode.rate), needed for Eb/N0->sigma."""
    if assign is None:
        from rl_construct import round_robin_assign
        assign = round_robin_assign(cfg)
    sc = cfg.build(assign=np.asarray(assign, dtype=np.int64))
    return sc.rate


# --------------------------------------------------------------------------- #
#  core PEXIT density evolution
# --------------------------------------------------------------------------- #
def protograph_pexit(Bproto, var_sigma_ch, max_iter=400, tol=1e-7,
                     known_mask=None, return_trace=False):
    """Protograph EXIT density evolution (Liva-Chiani 2007), BIAWGN.

    Parameters
    ----------
    Bproto : (MB, NB) int  edge-multiplicity matrix (check-type x var-type).
    var_sigma_ch : (NB,) float  channel LLR std per variable type
        (0 for punctured, +inf for perfectly-known/terminated).
    known_mask : (NB,) bool or None  variable types whose bits are *known*
        (excluded from the convergence test; they start and stay at MI 1).
        If None it is inferred from var_sigma_ch == +inf.
    max_iter : int
    tol : convergence tolerance on min a-posteriori MI.

    Returns
    -------
    converged : bool   True if every non-known variable type reaches MI ~ 1.
    Iapp_min  : float  the final minimum a-posteriori MI over non-known var types
        (a soft "how close to decoding" score; 1.0 == success).
    (optional) trace : list of Iapp_min per iteration.
    """
    Bproto = np.asarray(Bproto)
    MB, NB = Bproto.shape
    sig = np.asarray(var_sigma_ch, dtype=np.float64)
    if known_mask is None:
        known_mask = ~np.isfinite(sig)
    known_mask = np.asarray(known_mask, dtype=bool)

    # channel MI per variable type (known -> 1, punctured(sigma=0) -> 0)
    Ich = J(sig)                       # J(+inf)=1, J(0)=0
    Ich[known_mask] = 1.0
    ch_var = Jinv(Ich) ** 2            # (NB,) channel LLR variance per var-type
    non_known = ~known_mask

    # --- sparse edge list (the efficient representation) ------------------- #
    # one entry per *parallel edge*: a base entry of multiplicity m -> m edges.
    ci, vj = np.nonzero(Bproto > 0)
    mlt = Bproto[ci, vj]
    if mlt.max() > 1:                  # expand multi-edges (5G NR base is simple)
        ci = np.repeat(ci, mlt)
        vj = np.repeat(vj, mlt)
    ci = ci.astype(np.int64)
    vj = vj.astype(np.int64)
    E = ci.size
    if E == 0:
        return (True, 1.0, [1.0]) if return_trace else (True, 1.0)

    # I_ev[e] : MI var->chk on edge e ;  I_ec[e] : MI chk->var on edge e
    I_ev = np.zeros(E)
    I_ec = np.zeros(E)

    trace = []
    for _ in range(max_iter):
        # ---- variable-node update (extrinsic, sum-in-variance domain) ---- #
        ec_var = Jinv(I_ec) ** 2                              # per-edge incoming var
        col_sum = np.zeros(NB)
        np.add.at(col_sum, vj, ec_var)                       # sum over all edges of var j
        # extrinsic to edge e: channel + (col_sum_j - this edge)
        extr_var = ch_var[vj] + (col_sum[vj] - ec_var)
        I_ev = J(np.sqrt(np.maximum(extr_var, 0.0)))
        I_ev[known_mask[vj]] = 1.0                            # known bits: full MI out

        # ---- check-node update (dual, complementary message) ------------- #
        comp_var = Jinv(1.0 - I_ev) ** 2
        row_sum = np.zeros(MB)
        np.add.at(row_sum, ci, comp_var)                     # sum over all edges of check i
        extr_cvar = row_sum[ci] - comp_var
        I_ec = 1.0 - J(np.sqrt(np.maximum(extr_cvar, 0.0)))

        # ---- a-posteriori MI per variable type (all incoming) ------------ #
        ec_var2 = Jinv(I_ec) ** 2
        col_sum2 = np.zeros(NB)
        np.add.at(col_sum2, vj, ec_var2)
        app_var = ch_var + col_sum2
        Iapp = J(np.sqrt(np.maximum(app_var, 0.0)))
        Iapp[known_mask] = 1.0
        Iapp_min = float(Iapp[non_known].min()) if non_known.any() else 1.0
        trace.append(Iapp_min)
        if Iapp_min > 1.0 - tol:
            break
        # detect stall (no progress) -> failure
        if len(trace) > 8 and abs(trace[-1] - trace[-9]) < 1e-9 and Iapp_min < 0.999:
            break

    converged = trace[-1] > 1.0 - 1e-4
    if return_trace:
        return converged, trace[-1], trace
    return converged, trace[-1]


# --------------------------------------------------------------------------- #
#  channel-parameter helpers and threshold bisection
# --------------------------------------------------------------------------- #
def _var_sigma_ch(var_kind, sigma_awgn):
    """Map var_kind labels + AWGN noise std to per-type channel LLR std sigma_ch.

    Consistent-Gaussian: LLR has variance 4/sigma_awgn^2 => sigma_ch = 2/sigma_awgn.
    Punctured -> 0 ; known -> +inf."""
    sch = 2.0 / sigma_awgn
    out = np.empty(len(var_kind))
    for k, kind in enumerate(var_kind):
        if kind == "punct":
            out[k] = 0.0
        elif kind == "known":
            out[k] = np.inf
        else:
            out[k] = sch
    return out


def pexit_converges_at(proto, sigma_awgn, max_iter=400):
    """Does the protograph decode at AWGN noise std `sigma_awgn`?"""
    sch = _var_sigma_ch(proto["var_kind"], sigma_awgn)
    known = np.array([k == "known" for k in proto["var_kind"]])
    conv, _ = protograph_pexit(proto["Bproto"], sch, max_iter=max_iter,
                               known_mask=known)
    return conv


def _bisect_sigma_threshold(proto, sig_lo=0.2, sig_hi=2.0, n_iter=24, max_iter=400):
    """Bisect over AWGN noise std sigma: find max sigma that still decodes.

    Larger sigma = worse channel.  Returns the threshold sigma (decodes for
    sigma <= sigma_thr).  Expands the bracket if needed."""
    # ensure sig_lo decodes and sig_hi fails; expand otherwise
    lo, hi = sig_lo, sig_hi
    # push lo down until it converges (good channel)
    tries = 0
    while not pexit_converges_at(proto, lo, max_iter) and tries < 12:
        lo *= 0.6
        tries += 1
    if not pexit_converges_at(proto, lo, max_iter):
        return None        # never converges even at very low noise -> bad ensemble
    tries = 0
    while pexit_converges_at(proto, hi, max_iter) and tries < 12:
        hi *= 1.4
        tries += 1
    # bisection
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if pexit_converges_at(proto, mid, max_iter):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sigma_to_ebn0(sigma_awgn, rate):
    """Inverse of channel.ebn0_to_sigma: sigma = sqrt(1/(2*R*ebn0_lin))."""
    esn0 = 1.0 / (2.0 * sigma_awgn ** 2)
    ebn0_lin = esn0 / rate
    return 10.0 * np.log10(ebn0_lin)


def pexit_threshold(cfg: Config, assign, max_iter=400, n_bisect=26):
    """Protograph PEXIT decoding threshold of the SC code, in dB Eb/N0.

    Returns the minimum Eb/N0 (dB) at which the SC protograph (with termination
    boundary) converges to MI=1.  None if it never converges."""
    proto = sc_protograph(cfg, assign)
    rate = sc_rate(cfg, assign)
    sig_thr = _bisect_sigma_threshold(proto, n_iter=n_bisect, max_iter=max_iter)
    if sig_thr is None:
        return None
    return sigma_to_ebn0(sig_thr, rate)


# --------------------------------------------------------------------------- #
#  regular-protograph threshold (for the correctness gate)
# --------------------------------------------------------------------------- #
def regular_protograph(dv, dc):
    """Smallest (dv, dc)-regular protograph: one var-type of degree dv, one
    check-type of degree dc.  Requires dv, dc such that a single-row/single-col
    base with multiplicities matches.  We use a base with the right ratio:
      n var-types = dc, m check-types = dv, all-ones base -> every var-type has
      degree dv (one edge to each check-type) and every check-type degree dc.
    Design rate = 1 - dv/dc (matches a (dv,dc)-regular LDPC)."""
    nb, mb = dc, dv
    Bproto = np.ones((mb, nb), dtype=np.int64)
    var_kind = ["rx"] * nb
    return {"Bproto": Bproto, "var_kind": var_kind, "nb": nb, "mb": mb,
            "rate": 1.0 - dv / dc}


def regular_threshold_db(dv, dc, max_iter=2000, n_bisect=30):
    """BIAWGN PEXIT threshold (Eb/N0 dB) of the (dv,dc)-regular ensemble."""
    proto = regular_protograph(dv, dc)
    rate = proto["rate"]
    sig_thr = _bisect_sigma_threshold(proto, sig_lo=0.5, sig_hi=1.2,
                                      n_iter=n_bisect, max_iter=max_iter)
    if sig_thr is None:
        return None, None
    return sigma_to_ebn0(sig_thr, rate), sig_thr
