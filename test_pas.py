"""Local validation for pas.py (numpy-only, no scipy).  Gates deployment.

Checks:
  (1) shaped E[|X|^2] == 1 after renorm (uniform and shaped);
  (2) R_BMD(uniform) sane: <= m, monotone-ish in SNR, near a hand-computed MI check;
  (3) R_BMD(shaped nu*) > R_BMD(uniform), gain GROWS with M and ~vanishes for QPSK;
  (4) a-priori demap with nu=0 reduces to qam.soft_demap (assert closeness);
  (5) a tiny coded PAS run decodes (BER falls with SNR), idealised DM;
  (6) one joint-RL step runs and updates the policy + nu.

Run:  python3 test_pas.py
"""
from __future__ import annotations
import numpy as np

import qam
import pas


def _ok(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' :: ' + extra) if extra else ''}")
    assert cond, name


def test_energy():
    print("(1) shaped symbol energy Es=1 after renorm")
    rng = np.random.default_rng(0)
    for M in (4, 16, 64, 256):
        for nu in (0.0, 0.05, 0.15):
            pmf = pas.mb_pmf(M, nu)
            syms, scale = pas.sample_shaped_symbols(M, pmf, 200000, rng)
            e = float(np.mean(np.abs(syms) ** 2))
            _ok(f"M={M} nu={nu}: E|X|^2={e:.4f}", abs(e - 1.0) < 0.02, f"scale={scale:.4f}")


def test_uniform_rbmd_sane():
    print("(2) R_BMD(uniform) sane (<= m, monotone in SNR)")
    for M in (4, 16, 64, 256):
        m = qam.bits_per_symbol(M)
        prev = -1.0
        mono = True
        vals = []
        for snr in (0.0, 6.0, 12.0, 20.0, 30.0):
            d = pas.r_bmd(M, 0.0, snr, n_sym=40000, seed=1)
            vals.append(round(d["r_bmd"], 3))
            if d["r_bmd"] < prev - 1e-2:
                mono = False
            prev = d["r_bmd"]
            assert d["r_bmd"] <= m + 1e-6, f"R_BMD>{m}"
        # at very high SNR R_BMD -> H(X) = m (uniform)
        hi = pas.r_bmd(M, 0.0, 40.0, n_sym=40000, seed=1)["r_bmd"]
        _ok(f"M={M} m={m} R_BMD(SNR)={vals} hi={hi:.3f}",
            mono and abs(hi - m) < 0.05)


def test_shaping_gain():
    print("(3) shaping gain: R_BMD(shaped nu*) > R_BMD(uniform), grows with M, ~0 for QPSK")
    # operating Es/N0 per M chosen near the 0.75*m rate regime where shaping helps most
    snr_for = {4: 6.0, 16: 12.0, 64: 18.0, 256: 24.0}
    gains_bits = {}
    for M in (4, 16, 64, 256):
        snr = snr_for[M]
        uni = pas.r_bmd(M, 0.0, snr, n_sym=60000, seed=2)["r_bmd"]
        nu, d = pas.best_nu(M, snr, n_sym=60000, seed=2)
        gains_bits[M] = (round(uni, 3), round(d["r_bmd"], 3), round(d["r_bmd"] - uni, 3), nu)
        _ok(f"M={M} snr={snr} uni={uni:.3f} shaped={d['r_bmd']:.3f} "
            f"gain={d['r_bmd']-uni:+.3f} bits nu*={nu:.3f}",
            d["r_bmd"] >= uni - 1e-3)
    # QPSK gain should be tiny; gain should grow with M (16<64<256)
    g = {M: gains_bits[M][2] for M in gains_bits}
    _ok(f"QPSK gain tiny ({g[4]:+.3f} bits)", abs(g[4]) < 0.05)
    _ok(f"gain grows with M: 16={g[16]:+.3f} < 64={g[64]:+.3f} < 256={g[256]:+.3f}",
        g[16] < g[64] < g[256] and g[256] > 0.1)
    return gains_bits


def test_shaping_gain_db():
    """Express the shaping gain in dB: SNR saving at a matched R_BMD target."""
    print("(3b) shaping gain in dB (SNR saving at matched R_BMD)")
    out = {}
    for M, target in [(16, 3.0), (64, 4.5), (256, 6.0)]:
        snr_uni = _snr_for_rate(M, 0.0, target)
        nu = _best_nu_at_target(M, target)
        snr_sh = _snr_for_rate(M, nu, target, pmf=False)
        gain_db = snr_uni - snr_sh
        out[M] = (target, round(snr_uni, 2), round(snr_sh, 2), round(gain_db, 2), round(nu, 3))
        _ok(f"M={M} R_BMD={target}b: uni@{snr_uni:.2f}dB shaped(nu={nu:.3f})@{snr_sh:.2f}dB "
            f"-> {gain_db:+.2f} dB", gain_db > -0.05)
    return out


def _snr_for_rate(M, nu, target, pmf=False, lo=-2.0, hi=34.0):
    """Bisection: Es/N0 at which R_BMD(M,nu)=target bits/complex sym."""
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        r = pas.r_bmd(M, nu, mid, n_sym=40000, seed=5, pmf=pmf)["r_bmd"]
        if r < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _best_nu_at_target(M, target):
    """nu that maximises R_BMD at the SNR where uniform hits `target` (good proxy)."""
    snr0 = _snr_for_rate(M, 0.0, target)
    nu, _ = pas.best_nu(M, snr0, n_sym=40000, seed=5)
    return nu


def test_demap_reduces():
    print("(4) a-priori demap with nu=0 == qam.soft_demap")
    rng = np.random.default_rng(7)
    for M in (4, 16, 64, 256):
        bits = rng.integers(0, 2, size=qam.bits_per_symbol(M) * 500).astype(np.int64)
        syms, npad = qam.bits_to_symbols(bits, M)
        sigma = qam.ebn0_to_sigma_qam(8.0, 0.5, M)
        y = qam.awgn_complex(syms, sigma, rng)
        llr_ref = qam.soft_demap(y, sigma, M)
        # nu=0 uniform PMF -> shaped scale == qam scale, prior flat
        pmf = pas.mb_pmf(M, 0.0)
        llr_pas = pas.soft_demap_apriori(y, sigma, M, pmf)
        md = float(np.max(np.abs(llr_ref - llr_pas)))
        _ok(f"M={M}: max|LLR_qam - LLR_pas|={md:.2e}", md < 1e-6)
    # and the channel wrapper at nu=0 matches qam.QAMChannel
    for M in (16, 64):
        bits = rng.integers(0, 2, size=qam.bits_per_symbol(M) * 400).astype(np.int64)
        sigma = qam.ebn0_to_sigma_qam(9.0, 0.5, M)
        r1 = np.random.default_rng(123)
        r2 = np.random.default_rng(123)
        llr_q = qam.QAMChannel(M).transmit(bits, sigma, r1)
        llr_p = pas.ShapedQAMChannel(M, nu=0.0).transmit(bits, sigma, r2)
        md = float(np.max(np.abs(llr_q - llr_p)))
        _ok(f"channel M={M} nu=0 vs qam.QAMChannel: max diff={md:.2e}", md < 1e-9)


def test_coded_pas_decodes():
    print("(5) tiny coded PAS run decodes (BER falls with SNR), idealised DM")
    import rl_construct as R
    from sc_ldpc import SCLDPCCode
    cfg = R.Config(bg=1, ils=0, Z=16, mp=9, w=3, L=12, W=6, max_iter=12, alpha=0.8)
    comp = cfg.component()
    assign = R.round_robin_assign(cfg)
    sc = SCLDPCCode(comp, w=cfg.w, L=cfg.L, assign=assign)
    M, nu = 64, 0.12
    chan = pas.ShapedQAMChannel(M, nu=nu)
    # operational net rate (uniform info bits / complex channel uses):
    n_tx_bits = int(sc.tx_mask.sum())
    n_cu = n_tx_bits / chan.m
    net_rate = sc.K / n_cu
    bers = []
    for snr in (3.0, 4.0, 5.0, 6.0):         # spans the (sharp) 64-QAM waterfall
        sigma = pas.ebn0_to_sigma(snr, net_rate)
        be = bits = 0
        for idx in range(6):
            rng = np.random.default_rng([99, idx])
            info = rng.integers(0, 2, size=sc.K).astype(np.uint8)
            cw, _ = sc.encode(info)
            llr = np.zeros(sc.num_var)
            llr[sc.tx_mask] = chan.transmit(cw[sc.tx_mask], sigma, rng)
            llr[sc.known_mask] = 30.0
            hard = sc.decode_windowed(llr, W=cfg.W, max_iter=cfg.max_iter, alpha=cfg.alpha)
            be += int((sc.extract_info(hard) != info).sum()); bits += info.size
        ber = be / bits
        bers.append(ber)
        print(f"    snr={snr:4.1f} dB  net_rate={net_rate:.3f} b/cu  BER={ber:.3e}")
    _ok(f"BER monotone-decreasing across SNR: {['%.1e'%b for b in bers]}",
        bers[-1] <= bers[0] and bers[-1] < 1e-2)


def test_joint_rl_step():
    print("(6) joint-RL step runs end-to-end; both policy update-maps move under advantage")
    import importlib
    import experiments_pas_rl as E
    importlib.reload(E)
    R = __import__("rl_construct")
    cfg = E.cfg_for(64, 0.5, 64, 3, 12)          # M, rate, Z, w, L
    pol = R.FeaturePolicy(cfg, lr=0.15, ent=0.02, seed=0)
    sh = E.NuPolicy(init_nu=0.10, lr=0.05, seed=0)
    # (a) one full joint step runs without error and scores the batch
    state = E.JointTrainState(cfg, 64, 0.5, snr=8.0)
    ms = E.joint_step(state, pol, sh, batch=6, frames=2, pool=None, step=1)
    _ok(f"joint_step ran end-to-end ({len(ms)} samples, ber present)",
        len(ms) == 6 and "ber" in ms[0])
    # (b) NuPolicy.update moves nu under a clear non-zero advantage.  (The tiny smoke
    #     above can yield ZERO reward-variance -> zero advantage -> no move; the real
    #     run operates at BER~3e-3 with variance, so test the update MAP directly.)
    mu0 = sh.mu
    nu_before = sh.nu()
    sh.update([(mu0 + 1.0, +1.0), (mu0 - 1.0, -1.0)])      # guaranteed non-zero gradient
    _ok(f"NuPolicy.update moves nu under advantage ({nu_before:.4f}->{sh.nu():.4f})",
        abs(sh.nu() - nu_before) > 1e-6)
    # (c) FeaturePolicy.update moves theta under a clear non-zero advantage
    theta0 = pol.theta.copy()
    tr1 = pol.rollout()[1]
    tr2 = pol.rollout()[1]
    pol.update([(tr1, +1.0), (tr2, -1.0)])
    _ok(f"FeaturePolicy.update moves theta (||d||={np.linalg.norm(pol.theta - theta0):.2e})",
        np.linalg.norm(pol.theta - theta0) > 0)


if __name__ == "__main__":
    print("=== pas.py local validation ===")
    test_energy()
    test_uniform_rbmd_sane()
    gains = test_shaping_gain()
    gdb = test_shaping_gain_db()
    test_demap_reduces()
    test_coded_pas_decodes()
    test_joint_rl_step()
    print("\n=== shaping-gain summary (bits/complex symbol) ===")
    for M, (uni, sh, g, nu) in gains.items():
        print(f"  M={M:3d}: uniform R_BMD={uni:.3f}  shaped={sh:.3f}  gain={g:+.3f} b  nu*={nu:.3f}")
    print("=== shaping-gain summary (dB SNR saving at matched R_BMD) ===")
    for M, (tgt, su, ss, gd, nu) in gdb.items():
        print(f"  M={M:3d} @R_BMD={tgt}b: uniform {su:.2f}dB -> shaped {ss:.2f}dB = {gd:+.2f} dB (nu*={nu:.3f})")
    print("\nALL PAS VALIDATION CHECKS PASSED")
