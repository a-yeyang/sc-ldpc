"""Tiny dependency-free SVG plotter for BER/FER curves (semilog-y).

matplotlib is not available in this environment, so we emit a self-contained
SVG (viewable in any browser) plus a CSV of the raw data.
"""
from __future__ import annotations
import math


def _nice_log_ticks(ymin, ymax):
    lo = math.floor(math.log10(ymin))
    hi = math.ceil(math.log10(ymax))
    return [10.0 ** k for k in range(lo, hi + 1)]


def semilogy(curves, xlabel="Eb/N0 [dB]", ylabel="BER", title="",
             path="plot.svg", width=760, height=520):
    """curves: list of dicts {x:[...], y:[...], label:str, color:str}."""
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf",
              "#8c564b", "#e377c2", "#bcbd22", "#393b79", "#7f7f7f", "#aec7e8"]
    ml, mr, mt, mb = 78, 170, 48, 60
    pw, ph = width - ml - mr, height - mt - mb

    xs = [x for c in curves for x in c["x"]]
    ys = [y for c in curves for y in c["y"] if y > 0]
    if not xs or not ys:
        return
    xmin, xmax = min(xs), max(xs)
    if xmax == xmin:
        xmax = xmin + 1
    ymin = min(ys) * 0.6
    ymax = max(ys) * 1.5
    ymin = max(ymin, 1e-7)
    lymin, lymax = math.log10(ymin), math.log10(ymax)

    def X(v): return ml + (v - xmin) / (xmax - xmin) * pw
    def Y(v):
        v = max(v, ymin)
        return mt + (lymax - math.log10(v)) / (lymax - lymin) * ph

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
         f'font-family="Helvetica,Arial,sans-serif" font-size="13">']
    s.append(f'<rect width="{width}" height="{height}" fill="white"/>')
    if title:
        s.append(f'<text x="{ml+pw/2}" y="26" text-anchor="middle" '
                 f'font-size="16" font-weight="bold">{title}</text>')
    # y grid + labels
    for tick in _nice_log_ticks(ymin, ymax):
        if tick < ymin or tick > ymax:
            continue
        y = Y(tick)
        s.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" '
                 f'stroke="#e0e0e0"/>')
        exp = int(round(math.log10(tick)))
        s.append(f'<text x="{ml-8}" y="{y+4:.1f}" text-anchor="end">1e{exp}</text>')
    # x grid + labels
    xt = xmin
    step = max(0.5, round((xmax - xmin) / 8 * 2) / 2)
    while xt <= xmax + 1e-9:
        x = X(xt)
        s.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt+ph}" stroke="#eee"/>')
        s.append(f'<text x="{x:.1f}" y="{mt+ph+20:.1f}" text-anchor="middle">{xt:.1f}</text>')
        xt += step
    # axes
    s.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="#333"/>')
    s.append(f'<text x="{ml+pw/2}" y="{height-16}" text-anchor="middle">{xlabel}</text>')
    s.append(f'<text x="20" y="{mt+ph/2}" text-anchor="middle" '
             f'transform="rotate(-90 20 {mt+ph/2})">{ylabel}</text>')
    # curves
    for k, c in enumerate(curves):
        col = c.get("color", colors[k % len(colors)])
        pts = [(X(x), Y(y)) for x, y in zip(c["x"], c["y"]) if y > 0]
        if pts:
            path_d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            s.append(f'<path d="{path_d}" fill="none" stroke="{col}" stroke-width="2.2"/>')
            for x, y in pts:
                s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.3" fill="{col}"/>')
        ly = mt + 18 + k * 22
        s.append(f'<line x1="{ml+pw+14}" y1="{ly}" x2="{ml+pw+34}" y2="{ly}" '
                 f'stroke="{col}" stroke-width="2.2"/>')
        s.append(f'<text x="{ml+pw+38}" y="{ly+4}">{c["label"]}</text>')
    s.append('</svg>')
    with open(path, "w") as f:
        f.write("\n".join(s))


def write_csv(curves, path="results.csv"):
    with open(path, "w") as f:
        for c in curves:
            f.write(f"# {c['label']}\n")
            f.write("ebn0_db,ber,fer\n")
            fer = c.get("fer", [None] * len(c["x"]))
            for i, (x, y) in enumerate(zip(c["x"], c["y"])):
                f.write(f"{x},{y},{fer[i] if fer[i] is not None else ''}\n")
