"""Belief-propagation decoder for QC-LDPC / SC-LDPC Tanner graphs.

`Tanner` is built from a flat edge list (e_chk, e_var).  It stores the graph in
a padded [node x max_degree] layout so that both the check-node and
variable-node updates are fully vectorised in NumPy.

Two flooding-schedule decoders are provided:
  * normalised min-sum  (default, robust, the 5G workhorse)
  * sum-product (tanh)   (exact BP)

A syndrome-based early stop is used: as soon as the hard decision satisfies all
parity checks the iteration stops.
"""
from __future__ import annotations
import numpy as np

MSG_CAP = 1e2   # clip LLR message magnitudes (degree-1 boundary checks -> +inf otherwise)


class Tanner:
    def __init__(self, e_chk, e_var, num_chk, num_var):
        e_chk = np.asarray(e_chk, dtype=np.int64)
        e_var = np.asarray(e_var, dtype=np.int64)
        self.E = e_chk.size
        self.num_chk = int(num_chk)
        self.num_var = int(num_var)
        self.e_chk = e_chk
        self.e_var = e_var
        self._build_padded()

    def _slots(self, node_of_edge, num_nodes):
        """Padded [num_nodes x maxdeg] edge-index table (pad = -1) + mask."""
        deg = np.bincount(node_of_edge, minlength=num_nodes)
        maxd = int(deg.max()) if num_nodes else 0
        order = np.argsort(node_of_edge, kind="stable")
        ptr = np.concatenate([[0], np.cumsum(deg)])
        slot = np.arange(node_of_edge.size) - ptr[node_of_edge[order]]
        table = np.full((num_nodes, maxd), -1, dtype=np.int64)
        table[node_of_edge[order], slot] = order
        mask = table >= 0
        safe = np.where(mask, table, 0)
        return table, mask, safe, deg

    def _build_padded(self):
        self.C_tab, self.C_mask, self.C_safe, self.dc = self._slots(self.e_chk, self.num_chk)
        self.V_tab, self.V_mask, self.V_safe, self.dv = self._slots(self.e_var, self.num_var)
        self._c_idx = np.arange(self.num_chk)

    # ------------------------------------------------------------------ #
    def decode(self, llr_ch, max_iter=50, method="minsum", alpha=0.8,
               return_iters=False):
        """llr_ch: length num_var channel LLRs (>0 favours bit 0).
        Returns hard-decision bits (length num_var)."""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        m_vc = llr_ch[self.e_var].copy()          # var -> chk messages (per edge)
        m_cv = np.zeros(self.E, dtype=np.float64)  # chk -> var messages (per edge)
        hard = (llr_ch < 0).astype(np.uint8)
        it = 0
        for it in range(1, max_iter + 1):
            # ---- check update ----
            if method == "minsum":
                m_cv = self._check_minsum(m_vc, alpha)
            else:
                m_cv = self._check_sumproduct(m_vc)
            # ---- variable update + decision ----
            gathered = np.where(self.V_mask, m_cv[self.V_safe], 0.0)
            sum_cv = gathered.sum(axis=1)
            total = llr_ch + sum_cv
            hard = (total < 0).astype(np.uint8)
            # syndrome early stop
            syn = np.bincount(self.e_chk, weights=hard[self.e_var].astype(np.float64),
                              minlength=self.num_chk).astype(np.int64) & 1
            if not syn.any():
                break
            # outgoing var->chk = total - incoming (per edge)
            new_vc = np.clip(total[:, None] - gathered, -MSG_CAP, MSG_CAP)
            m_vc[self.V_tab[self.V_mask]] = new_vc[self.V_mask]
        if return_iters:
            return hard, it
        return hard

    def _check_minsum(self, m_vc, alpha):
        vals = np.where(self.C_mask, m_vc[self.C_safe], np.nan)
        sign = np.where(np.isnan(vals), 1.0, np.sign(np.where(np.isnan(vals), 1.0, vals)))
        sign[sign == 0] = 1.0
        signprod = np.prod(sign, axis=1)
        absval = np.where(self.C_mask, np.abs(m_vc[self.C_safe]), np.inf)
        arg = np.argmin(absval, axis=1)
        min1 = np.minimum(absval[self._c_idx, arg], MSG_CAP)
        absval2 = absval.copy()
        absval2[self._c_idx, arg] = np.inf
        min2 = np.minimum(absval2.min(axis=1), MSG_CAP)   # +inf for degree-1 checks -> capped
        is_arg = np.zeros(absval.shape, dtype=bool)
        is_arg[self._c_idx, arg] = True
        out_mag = np.where(is_arg, min2[:, None], min1[:, None])
        out_sign = signprod[:, None] * sign
        out = alpha * out_sign * out_mag
        m_cv = np.zeros(self.E, dtype=np.float64)
        m_cv[self.C_tab[self.C_mask]] = out[self.C_mask]
        return m_cv

    def _check_sumproduct(self, m_vc):
        t = np.tanh(np.clip(m_vc, -30, 30) / 2.0)
        tt = np.where(self.C_mask, t[self.C_safe], 1.0)
        prod = np.prod(tt, axis=1)
        # exclude self: divide, guarding against ~0
        with np.errstate(divide="ignore", invalid="ignore"):
            others = prod[:, None] / tt
        others = np.clip(others, -1 + 1e-12, 1 - 1e-12)
        out = np.clip(2.0 * np.arctanh(others), -MSG_CAP, MSG_CAP)
        m_cv = np.zeros(self.E, dtype=np.float64)
        m_cv[self.C_tab[self.C_mask]] = out[self.C_mask]
        return m_cv

    def syndrome_weight(self, hard):
        syn = np.bincount(self.e_chk, weights=hard[self.e_var].astype(np.float64),
                          minlength=self.num_chk).astype(np.int64) & 1
        return int(syn.sum())
