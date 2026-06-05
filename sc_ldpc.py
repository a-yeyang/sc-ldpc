"""Spatially-coupled LDPC built on a 5G NR component code.

Construction ("systematic edge spreading", coupling memory w, coupling length L)
------------------------------------------------------------------------------
The 5G NR base matrix B = [B_sys | B_par] is split into w+1 component matrices

        B = B_0 + B_1 + ... + B_w        (edge spreading)

where *only the systematic edges* are spread (each systematic base entry is
randomly assigned to one component), while the whole parity part B_par is kept
in B_0.  The coupled parity-check matrix is the band-diagonal

        H_SC[i, t] = B_{i-t}             0 <= i-t <= w
                                          i = 0..L+w-1  (check positions)
                                          t = 0..L-1    (variable positions)

Keeping B_par entirely in B_0 means every spatial position is encoded with the
*exact* 5G NR parity structure, so the standard 5G encoder solves each position
recursively (`NRLDPCCode.solve_parity`).  The chain is terminated by forcing the
systematic bits of the last w positions to zero, which satisfies the w extra
"termination" check rows and creates the boundary that triggers the SC decoding
wave / threshold saturation.

Variable-node global index : t*(nb*Z) + j*Z + r
Check-node    global index : i*(mb*Z) + ri*Z + r
"""
from __future__ import annotations
import numpy as np
from nr_ldpc import NRLDPCCode, shift_vec, edges_from_entries
from decoder import Tanner


class SCLDPCCode:
    def __init__(self, component: NRLDPCCode, w: int = 2, L: int = 40, seed: int = 0,
                 assign=None):
        """`assign` (length = #systematic base edges, entries in [0, w]) explicitly
        chooses the component B_0..B_w for each systematic edge -- this is the
        construction knob optimised by RL.  If None, a random spreading (seeded by
        `seed`) is used, reproducing the original behaviour exactly."""
        self.comp = component
        self.w = w
        self.L = L
        self.Z = component.Z
        self.mb = component.mb
        self.nb = component.nb
        self.Kb = component.Kb
        self.seed = seed
        self._edge_spread(seed, assign)
        self._build_coupled_entries()
        self._build_masks()

    # ------------------------------------------------------------------ #
    #  edge spreading
    # ------------------------------------------------------------------ #
    def _edge_spread(self, seed, assign=None):
        B, Kb, w = self.comp.B, self.Kb, self.w
        # canonical systematic-edge order (row-major); RL `assign` indexes into this
        ri, cj = np.nonzero(B[:, :Kb] >= 0)
        self.sys_edge_rc = (ri, cj)
        self.n_sys_edges = int(ri.size)
        if assign is None:
            rng = np.random.default_rng(seed)
            assign = rng.integers(0, w + 1, size=ri.size)
        else:
            assign = np.asarray(assign, dtype=np.int64)
            assert assign.shape == (ri.size,), \
                f"assign must have length {ri.size} (got {assign.shape})"
            assert assign.min() >= 0 and assign.max() <= w, \
                f"assign entries must be in [0, {w}]"
        self.assign = assign
        comps = [np.full_like(B, -1) for _ in range(w + 1)]
        # parity part -> component 0
        comps[0][:, Kb:] = B[:, Kb:]
        # systematic part -> spread over components 0..w per `assign`
        for e in range(ri.size):
            comps[assign[e]][ri[e], cj[e]] = B[ri[e], cj[e]]
        # sanity: spreading reconstructs B's systematic part exactly
        recon = np.full_like(B, -1)
        recon[:, Kb:] = B[:, Kb:]
        merged = np.full_like(B, -1)
        for c in comps:
            merged = np.where(c >= 0, c, merged)
        assert np.array_equal(merged, B), "edge spreading does not reconstruct B"
        self.comps = comps
        # per-component systematic entries (row, col, shift) for fast encoding
        self.sys_entries = []
        for c in comps:
            r, j = np.nonzero((c >= 0))
            keep = j < Kb
            self.sys_entries.append((r[keep], j[keep], c[r[keep], j[keep]]))

    # ------------------------------------------------------------------ #
    #  coupled entry table (used to build full / windowed Tanner graphs)
    # ------------------------------------------------------------------ #
    def _build_coupled_entries(self):
        L, w, mb, nb = self.L, self.w, self.mb, self.nb
        I, RI, T, CJ, V = [], [], [], [], []
        for a in range(w + 1):
            r, c = np.nonzero(self.comps[a] >= 0)
            v = self.comps[a][r, c]
            for t in range(L):
                i = t + a                      # check block index
                if i > L + w - 1:
                    continue
                I.append(np.full(r.size, i)); RI.append(r)
                T.append(np.full(r.size, t)); CJ.append(c); V.append(v)
        self.ent_i = np.concatenate(I)
        self.ent_ri = np.concatenate(RI)
        self.ent_t = np.concatenate(T)
        self.ent_cj = np.concatenate(CJ)
        self.ent_v = np.concatenate(V)
        self.num_chk = (L + w) * mb * self.Z
        self.num_var = L * nb * self.Z

    def full_tanner(self) -> Tanner:
        Z, mb, nb = self.Z, self.mb, self.nb
        bigrow = self.ent_i * mb + self.ent_ri
        bigcol = self.ent_t * nb + self.ent_cj
        chk, var = edges_from_entries(bigrow, bigcol, self.ent_v, Z,
                                      (self.L + self.w) * mb, self.L * nb)
        return Tanner(chk, var, self.num_chk, self.num_var)

    def window_tanner(self, c_lo, c_hi, v_lo, v_hi):
        """Sub-Tanner over check blocks [c_lo,c_hi] and var blocks [v_lo,v_hi]."""
        Z, mb, nb = self.Z, self.mb, self.nb
        m = ((self.ent_i >= c_lo) & (self.ent_i <= c_hi) &
             (self.ent_t >= v_lo) & (self.ent_t <= v_hi))
        bigrow = (self.ent_i[m] - c_lo) * mb + self.ent_ri[m]
        bigcol = (self.ent_t[m] - v_lo) * nb + self.ent_cj[m]
        nC = (c_hi - c_lo + 1) * mb
        nV = (v_hi - v_lo + 1) * nb
        chk, var = edges_from_entries(bigrow, bigcol, self.ent_v[m], Z, nC, nV)
        return Tanner(chk, var, nC * Z, nV * Z)

    # ------------------------------------------------------------------ #
    #  punctured / known masks and rate
    # ------------------------------------------------------------------ #
    def _build_masks(self):
        L, w, Kb, nb, Z = self.L, self.w, self.Kb, self.nb, self.Z
        N = L * nb * Z
        punct = np.zeros(N, dtype=bool)     # punctured systematic cols 0,1 (no channel info)
        known = np.zeros(N, dtype=bool)      # terminated systematic bits (known zero)
        info = np.zeros(N, dtype=bool)       # free information bits
        for t in range(L):
            base = t * nb * Z
            punct[base: base + 2 * Z] = True            # cols 0,1
            if t >= L - w:                              # terminated positions
                known[base: base + Kb * Z] = True
            else:                                        # information positions
                info[base: base + Kb * Z] = True
        known &= ~punct       # punctured already excluded from "known transmit"
        self.punct_mask = punct
        self.known_mask = known
        self.info_mask = info
        # transmitted = everything except punctured and terminated-known systematic
        self.tx_mask = ~(punct | known)
        self.K = int(info.sum())
        self.N_tx = int(self.tx_mask.sum())

    @property
    def rate(self):
        return self.K / self.N_tx

    @property
    def n_info_positions(self):
        return self.L - self.w

    # ------------------------------------------------------------------ #
    #  encoder
    # ------------------------------------------------------------------ #
    def encode(self, rng_or_msg):
        """Encode.  Accepts an RNG (random info) or a length-K info bit array.
        Returns (codeword bits [N], info bits [K])."""
        L, w, Kb, mb, nb, Z = self.L, self.w, self.Kb, self.mb, self.nb, self.Z
        if isinstance(rng_or_msg, np.random.Generator):
            info = rng_or_msg.integers(0, 2, size=self.K).astype(np.uint8)
        else:
            info = np.asarray(rng_or_msg, dtype=np.uint8)
            assert info.size == self.K
        cw = np.zeros(L * nb * Z, dtype=np.uint8)
        sys = np.zeros((L, Kb, Z), dtype=np.uint8)      # systematic blocks per position
        # place info into the first L-w positions
        info_blocks = info.reshape(self.n_info_positions, Kb, Z)
        sys[:L - w] = info_blocks
        # encode position by position
        for i in range(L):
            S = np.zeros((mb, Z), dtype=np.uint8)
            for a in range(w + 1):
                t = i - a
                if t < 0:
                    continue
                rows, cols, shifts = self.sys_entries[a]
                for e in range(rows.size):
                    S[rows[e]] ^= shift_vec(sys[t, cols[e]], int(shifts[e]))
            p = self.comp.solve_parity(S)               # 5G recursive parity solve
            base = i * nb * Z
            cw[base: base + Kb * Z] = sys[i].reshape(-1)
            cw[base + Kb * Z: base + nb * Z] = p.reshape(-1)
        return cw, info

    # ------------------------------------------------------------------ #
    #  channel-side helpers
    # ------------------------------------------------------------------ #
    def make_llr(self, cw, sigma, rng, large=30.0):
        """Modulate the transmitted bits over BPSK+AWGN and return full LLR vec."""
        import channel as ch
        tx = ch.bpsk(cw[self.tx_mask])
        y = ch.awgn(tx, sigma, rng)
        llr = np.zeros(self.num_var, dtype=np.float64)
        llr[self.tx_mask] = ch.llr_awgn(y, sigma)
        llr[self.known_mask] = large        # terminated bits are known to be 0
        # punctured bits keep llr = 0
        return llr

    def extract_info(self, hard):
        return hard[self.info_mask]

    # ------------------------------------------------------------------ #
    #  sliding-window decoder (the low-latency SC-LDPC decoder)
    # ------------------------------------------------------------------ #
    def _window_layout(self, W):
        """Build and cache the per-target window sub-Tanners (fixed across frames)."""
        if not hasattr(self, "_win_cache"):
            self._win_cache = {}
        if W in self._win_cache:
            return self._win_cache[W]
        L, w = self.L, self.w
        windows = []
        for t in range(L):
            v_lo = max(0, t - w)
            v_hi = min(t + W - 1, L - 1)
            c_lo = t
            c_hi = v_hi if v_hi < L - 1 else (L + w - 1)   # include termination checks at the tail
            tan = self.window_tanner(c_lo, c_hi, v_lo, v_hi)
            windows.append((tan, v_lo, v_hi))
        self._win_cache[W] = windows
        return windows

    def decode_windowed(self, llr_full, W=6, max_iter=50, method="minsum",
                        alpha=0.8, large=30.0):
        """Sliding-window BP.  A window of W spatial positions slides from t=0
        to L-1; after BP only the leftmost ("target") position is finalised and
        output, then the window advances by one position.  Memory/latency scale
        with W instead of L.  `llr_full` is the length-num_var channel LLR."""
        L, w, nb, Z = self.L, self.w, self.nb, self.Z
        blk = nb * Z
        dec_bits = np.zeros(self.num_var, dtype=np.uint8)
        windows = self._window_layout(W)        # cached across frames/SNRs
        for t in range(L):
            tan, v_lo, v_hi = windows[t]
            # local LLR
            nloc = (v_hi - v_lo + 1) * blk
            llr_loc = np.empty(nloc, dtype=np.float64)
            for tp in range(v_lo, v_hi + 1):
                g0, g1 = tp * blk, (tp + 1) * blk
                l0 = (tp - v_lo) * blk
                if tp < t:    # already decided -> known
                    llr_loc[l0:l0 + blk] = np.where(dec_bits[g0:g1] == 0, large, -large)
                else:
                    llr_loc[l0:l0 + blk] = llr_full[g0:g1]
            hard_loc = tan.decode(llr_loc, max_iter=max_iter, method=method, alpha=alpha)
            # finalise the target position t
            l0 = (t - v_lo) * blk
            dec_bits[t * blk:(t + 1) * blk] = hard_loc[l0:l0 + blk]
        return dec_bits
