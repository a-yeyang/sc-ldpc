"""Spatially-coupled polar codes via Partially Information Coupling (PIC).

Construction (Wu, Yang, Xie, Yuan, "Partially Information Coupled Polar Codes,"
IEEE Access 2018; generalized coupling depth J: Yu et al., IEEE T-COMM 2021)
------------------------------------------------------------------------------
A chain of L code blocks (CBs), each a 5G NR polar component code
(`nr_polar.NRPolarCode`).  Adjacent blocks are *coupled* by sharing systematic
information bits: the "coupled-out" bits of block t are reused as the
"coupled-in" bits of block t+k, for every coupling level k = 1..J (J is the
coupling depth, the analogue of the SC-LDPC coupling memory w).

Per-block payload layout (the A payload coordinates of the component, in the
order they sit on the most-reliable information positions):

        [ in^(1) .. in^(J) | fresh | out^(1) .. out^(J) ]   (+ CRC)
          \____ J*c ____/    Kfresh   \____ J*c ____/

    in^(k)_t  = out^(k)_{t-k}          (shared bits, the coupling memory)
    out^(k)_t = fresh information      (new, also fed to block t+k)

Termination (the boundary that seeds the SC decoding wave):
  * left  : in^(k)_t with t-k < 0    are dummy 0  (known)
  * right : out^(k)_t with t+k > L-1 are dummy 0  (known)
This mirrors the zero-termination of the SC-LDPC chain in `sc_ldpc.py`.

Decoding (feed-forward windowed SC / CA-SCL)
--------------------------------------------
Blocks are decoded left to right.  When decoding block t, every coupled-in
group whose source block already decoded *and passed CRC* is clamped to those
known bits ("dynamic freezing"), turning block t into a lower-rate, better
protected code -- exactly the coupling gain.  Only CRC-verified coupled bits
are propagated, which avoids the severe SC error propagation.  The list size
selects the per-block algorithm: list_size=1 is SC, list_size>1 is CA-SCL.
"""
from __future__ import annotations
import numpy as np

import channel as ch
import polar_decoder as pd
import polar_bp as bp
from nr_polar import NRPolarCode, crc_ok


class SCPolarCode:
    def __init__(self, component: NRPolarCode, L: int = 20, J: int = 1,
                 c: int = 8, seed: int = 0):
        self.comp = component
        self.L = int(L)
        self.J = int(J)
        self.c = int(c)
        self.seed = int(seed)
        self.A = component.A
        self.E = component.E
        self.N = component.N
        self.Jc = self.J * self.c
        assert self.A >= 2 * self.Jc + 1, \
            f"payload A={self.A} too small for J*c={self.Jc} on each side"
        self.Kfresh = self.A - 2 * self.Jc
        self._build_slot_maps()
        self._count_info()

    # ------------------------------------------------------------------ #
    #  payload <-> u-coordinate slot maps
    # ------------------------------------------------------------------ #
    def _build_slot_maps(self):
        ip, A, c, J, Jc = self.comp.info_positions, self.A, self.c, self.J, self.Jc
        self.in_slots = [slice(k * c, (k + 1) * c) for k in range(J)]            # in^(k+1)
        self.fresh_slot = slice(Jc, Jc + self.Kfresh)
        self.out_slots = [slice(A - Jc + k * c, A - Jc + (k + 1) * c) for k in range(J)]
        # u-domain coordinates of each coupling group (same for every block)
        self.in_upos = [ip[s] for s in self.in_slots]
        self.out_upos = [ip[s] for s in self.out_slots]

    def _count_info(self):
        n_out = sum(1 for t in range(self.L) for k in range(1, self.J + 1)
                    if t + k <= self.L - 1)
        self.K = self.L * self.Kfresh + n_out * self.c     # free information bits

    @property
    def rate(self) -> float:
        """Information rate of the whole coupled chain."""
        return self.K / (self.L * self.E)

    @property
    def n_coded(self) -> int:
        return self.L * self.E

    # ------------------------------------------------------------------ #
    #  encoder
    # ------------------------------------------------------------------ #
    def encode(self, rng_or_msg):
        """Encode.  Accepts an RNG (random info) or a length-K info bit array.
        Returns (codeword bits [L*E], info bits [K])."""
        if isinstance(rng_or_msg, np.random.Generator):
            info = rng_or_msg.integers(0, 2, size=self.K).astype(np.uint8)
        else:
            info = np.asarray(rng_or_msg, dtype=np.uint8)
            assert info.size == self.K
        cw = np.zeros(self.L * self.E, dtype=np.uint8)
        stored_out = {}                      # (block, level) -> c coupled bits
        ptr = 0
        for t in range(self.L):
            p = np.zeros(self.A, dtype=np.uint8)
            for k in range(1, self.J + 1):   # coupled-in (shared with earlier block)
                if t - k >= 0:
                    p[self.in_slots[k - 1]] = stored_out[(t - k, k)]
                # else left dummy 0
            p[self.fresh_slot] = info[ptr:ptr + self.Kfresh]
            ptr += self.Kfresh
            for k in range(1, self.J + 1):   # coupled-out (shared with later block)
                if t + k <= self.L - 1:
                    o = info[ptr:ptr + self.c]
                    ptr += self.c
                    p[self.out_slots[k - 1]] = o
                    stored_out[(t, k)] = o
                else:
                    stored_out[(t, k)] = np.zeros(self.c, dtype=np.uint8)   # right dummy
            cw[t * self.E:(t + 1) * self.E] = self.comp.encode(p)
        assert ptr == self.K
        return cw, info

    # ------------------------------------------------------------------ #
    #  channel
    # ------------------------------------------------------------------ #
    def make_llr(self, cw, sigma, rng):
        """Modulate every coded bit over BPSK+AWGN -> length-(L*E) channel LLRs."""
        y = ch.awgn(ch.bpsk(cw), sigma, rng)
        return ch.llr_awgn(y, sigma)

    # ------------------------------------------------------------------ #
    #  bidirectional iterative windowed SC / CA-SCL decoder
    # ------------------------------------------------------------------ #
    def _known(self, t, k, side, payload_hat, crc_pass):
        """Known value of block t's coupled-`side` group at level k, or None.

        in^(k)_t  == out^(k)_{t-k}  (left neighbour)   side='in'
        out^(k)_t == in^(k)_{t+k}   (right neighbour)  side='out'
        Boundary groups are dummy zeros (the chain termination)."""
        if side == "in":
            if t - k < 0:
                return np.zeros(self.c, dtype=np.uint8)            # left termination
            if crc_pass[t - k]:
                return payload_hat[t - k][self.out_slots[k - 1]]    # verified neighbour
        else:
            if t + k > self.L - 1:
                return np.zeros(self.c, dtype=np.uint8)            # right termination
            if crc_pass[t + k]:
                return payload_hat[t + k][self.in_slots[k - 1]]
        return None

    def decode(self, llr_full, list_size: int = 8, max_iter: int = None):
        """Windowed bidirectional decoder.  Sweeps the chain forward then
        backward; on each visit a block clamps every coupled group whose
        neighbour is a termination or has passed CRC ("dynamic freezing"),
        so the decoding wave propagates inward from both ends.  Returns
        (info_hat [K], crc_pass [L]).  list_size=1 is SC, list_size>1 CA-SCL."""
        E, comp, J = self.E, self.comp, self.J
        crc_len = comp.crc_len
        check = (lambda info: crc_ok(info, crc_len)) if crc_len else None
        if max_iter is None:
            max_iter = 2 * (J + 1)
        crc_pass = np.zeros(self.L, dtype=bool)
        payload_hat = np.zeros((self.L, self.A), dtype=np.uint8)
        llr_blocks = [comp.rate_dematch(llr_full[t * E:(t + 1) * E]) for t in range(self.L)]
        for it in range(max_iter):
            order = range(self.L) if it % 2 == 0 else range(self.L - 1, -1, -1)
            changed = False
            for t in order:
                if crc_pass[t]:
                    continue                                 # already resolved
                fmask = comp.frozen_mask.copy()
                fval = np.zeros(comp.N, dtype=np.uint8)
                for k in range(1, J + 1):
                    kin = self._known(t, k, "in", payload_hat, crc_pass)
                    if kin is not None:
                        fmask[self.in_upos[k - 1]] = True
                        fval[self.in_upos[k - 1]] = kin
                    kout = self._known(t, k, "out", payload_hat, crc_pass)
                    if kout is not None:
                        fmask[self.out_upos[k - 1]] = True
                        fval[self.out_upos[k - 1]] = kout
                u_hat = pd.scl_decode(llr_blocks[t], fmask, fval, L=list_size,
                                      info_positions=comp.info_positions, crc_check=check)
                info_crc = u_hat[comp.info_positions]
                payload_hat[t] = info_crc[:self.A]
                passed = crc_ok(info_crc, crc_len) if crc_len else True
                if passed and not crc_pass[t]:
                    changed = True
                crc_pass[t] = passed
            if crc_pass.all() or not changed:
                break
        return self._extract_info(payload_hat), crc_pass

    # ------------------------------------------------------------------ #
    #  windowed belief-propagation decoder (soft coupling -> diversity gain)
    # ------------------------------------------------------------------ #
    def decode_bp(self, llr_full, bp_iter: int = 30, outer_iter: int = None):
        """Soft windowed BP.  Each block is decoded by polar BP; the extrinsic
        LLRs on the shared coordinates are exchanged with neighbouring blocks
        and fed back as a-priori LLRs over several outer sweeps, so a shared
        bit accumulates channel evidence from *both* blocks (the coupling /
        diversity gain).  Returns (info_hat [K], crc_pass [L])."""
        E, comp, J = self.E, self.comp, self.J
        if outer_iter is None:
            outer_iter = 2 * (J + 1)
        crc_len = comp.crc_len
        chan = [comp.rate_dematch(llr_full[t * E:(t + 1) * E]) for t in range(self.L)]
        ext = [np.zeros(comp.N, dtype=np.float64) for _ in range(self.L)]   # extrinsic on u
        info_crc = np.zeros((self.L, comp.K), dtype=np.uint8)               # per-block info+CRC
        for outer in range(outer_iter):
            order = range(self.L) if outer % 2 == 0 else range(self.L - 1, -1, -1)
            for t in order:
                fmask = comp.frozen_mask.copy()
                fval = np.zeros(comp.N, dtype=np.uint8)
                prior = np.zeros(comp.N, dtype=np.float64)
                for k in range(1, J + 1):
                    up_in, up_out = self.in_upos[k - 1], self.out_upos[k - 1]
                    if t - k < 0:
                        fmask[up_in] = True                          # left termination
                    else:
                        prior[up_in] += ext[t - k][up_out]           # neighbour belief on g
                    if t + k > self.L - 1:
                        fmask[up_out] = True                         # right termination
                    else:
                        prior[up_out] += ext[t + k][up_in]
                u_hat, soft = bp.bp_decode(chan[t], fmask, fval, max_iter=bp_iter,
                                           prior=prior, return_soft=True)
                ext[t] = soft
                info_crc[t] = u_hat[comp.info_positions]
        payload_hat = info_crc[:, :self.A]
        crc_pass = (np.array([crc_ok(info_crc[t], crc_len) for t in range(self.L)])
                    if crc_len else np.ones(self.L, dtype=bool))
        return self._extract_info(payload_hat), crc_pass

    def _extract_info(self, payload_hat) -> np.ndarray:
        """Collect free info bits in the same order the encoder consumed them."""
        out = []
        for t in range(self.L):
            out.append(payload_hat[t][self.fresh_slot])
            for k in range(1, self.J + 1):
                if t + k <= self.L - 1:
                    out.append(payload_hat[t][self.out_slots[k - 1]])
        return np.concatenate(out)

    def __repr__(self):
        return (f"SCPolarCode(L={self.L}, J={self.J}, c={self.c}, "
                f"comp N={self.N} E={self.E} A={self.A}, K={self.K}, R={self.rate:.3f})")
