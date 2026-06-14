"""Tiny dependency-free SVG plotter for BER/FER curves (semilog-y).

matplotlib is not available in this environment, so we emit a self-contained
SVG (viewable in any browser) plus a CSV of the raw data.
"""
from __future__ import annotations
import math

from outpaths import route as _route


def _nice_log_ticks(ymin, ymax):
    lo = math.floor(math.log10(ymin))
    hi = math.ceil(math.log10(ymax))
    return [10.0 ** k for k in range(lo, hi + 1)]


FONT = "DejaVu Sans, Arial, sans-serif"


def _legend_margin(curves, font_size=13, stub=38, pad=22):
    """Right margin wide enough for the longest legend label.

    The legend draws a colour stub (``stub`` px) followed by the label text.
    We estimate text width at ~0.58*font_size per character (generous for a
    sans-serif font) so even the longest label stays inside the canvas.
    """
    longest = max((len(str(c.get("label", ""))) for c in curves), default=0)
    return int(stub + longest * font_size * 0.58 + pad)


def semilogy(curves, xlabel="Eb/N0 [dB]", ylabel="BER", title="",
             path="plot.svg", width=760, height=520):
    """curves: list of dicts {x:[...], y:[...], label:str, color:str}."""
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf",
              "#8c564b", "#e377c2", "#bcbd22", "#393b79", "#7f7f7f", "#aec7e8"]
    ml, mt, mb = 78, 48, 60
    mr = _legend_margin(curves)
    pw, ph = width - ml - 170, height - mt - mb  # plot width as before (mr was 170)
    width = ml + pw + mr  # widen canvas so the longest legend label fits

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
         f'font-family="{FONT}" font-size="13">']
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
    with open(_route(path), "w") as f:
        f.write("\n".join(s))


def linear(curves, xlabel="x", ylabel="y", title="", path="plot.svg",
           width=760, height=520):
    """Linear-axis line plot (same curve dict format as `semilogy`)."""
    ml, mt, mb = 78, 48, 60
    mr = _legend_margin(curves)
    pw, ph = width - ml - 200, height - mt - mb  # plot width as before (mr was 200)
    width = ml + pw + mr  # widen canvas so the longest legend label fits
    xs = [x for c in curves for x in c["x"]]
    ys = [y for c in curves for y in c["y"]]
    if not xs or not ys:
        return
    xmin, xmax = min(xs), max(xs)
    if xmax == xmin:
        xmax = xmin + 1
    ymin, ymax = min(ys), max(ys)
    pad = (ymax - ymin) * 0.1 or 0.5
    ymin, ymax = ymin - pad, ymax + pad

    def X(v): return ml + (v - xmin) / (xmax - xmin) * pw
    def Y(v): return mt + (ymax - v) / (ymax - ymin) * ph

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
         f'font-family="{FONT}" font-size="13">',
         f'<rect width="{width}" height="{height}" fill="white"/>']
    if title:
        s.append(f'<text x="{ml+pw/2}" y="26" text-anchor="middle" '
                 f'font-size="16" font-weight="bold">{title}</text>')
    for k in range(6):                                   # y grid
        v = ymin + (ymax - ymin) * k / 5
        y = Y(v)
        s.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" stroke="#e8e8e8"/>')
        s.append(f'<text x="{ml-8}" y="{y+4:.1f}" text-anchor="end">{v:.2f}</text>')
    xt = xmin
    step = max(0.05, round((xmax - xmin) / 6, 2))
    while xt <= xmax + 1e-9:
        x = X(xt)
        s.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt+ph}" stroke="#f0f0f0"/>')
        s.append(f'<text x="{x:.1f}" y="{mt+ph+20:.1f}" text-anchor="middle">{xt:.2f}</text>')
        xt += step
    s.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="#333"/>')
    s.append(f'<text x="{ml+pw/2}" y="{height-16}" text-anchor="middle">{xlabel}</text>')
    s.append(f'<text x="20" y="{mt+ph/2}" text-anchor="middle" '
             f'transform="rotate(-90 20 {mt+ph/2})">{ylabel}</text>')
    for k, c in enumerate(curves):
        col = c.get("color", "#1f77b4")
        pts = [(X(x), Y(y)) for x, y in zip(c["x"], c["y"])]
        if pts:
            s.append(f'<path d="M' + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
                     + f'" fill="none" stroke="{col}" stroke-width="2.2"/>')
            for x, y in pts:
                s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.3" fill="{col}"/>')
        ly = mt + 18 + k * 22
        s.append(f'<line x1="{ml+pw+14}" y1="{ly}" x2="{ml+pw+34}" y2="{ly}" '
                 f'stroke="{col}" stroke-width="2.2"/>')
        s.append(f'<text x="{ml+pw+38}" y="{ly+4}">{c["label"]}</text>')
    s.append('</svg>')
    with open(_route(path), "w") as f:
        f.write("\n".join(s))


def heatmap(M, row_labels, col_labels, title="", path="heatmap.svg",
            vmin=-5.0, vmax=-0.7, cell=66, note=""):
    """Block heatmap of log10(BER). M[r][c] is a BER (0 -> best). Green=low, red=high."""
    nr, nc = len(row_labels), len(col_labels)
    ml, mt = 92, 56
    W = ml + nc * cell + 30
    H = mt + nr * cell + 64

    def color(ber):
        if ber is None:
            return "#dddddd", "-"
        v = math.log10(ber) if ber > 0 else vmin
        f = max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))   # 0=good ->1=bad
        if f < 0.5:                       # green -> yellow
            r, g, b = int(2 * f * 255), 180, 40
        else:                             # yellow -> red
            r, g, b = 220, int(180 * (1 - (f - 0.5) * 2)), 30
        txt = "0" if ber == 0 else (f"{ber:.0e}".replace("e-0", "e-"))
        return f"rgb({r},{g},{b})", txt

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'font-family="{FONT}" font-size="12">',
         f'<rect width="{W}" height="{H}" fill="white"/>']
    if title:
        s.append(f'<text x="{ml + nc*cell/2}" y="24" text-anchor="middle" '
                 f'font-size="15" font-weight="bold">{title}</text>')
    for c, cl in enumerate(col_labels):
        s.append(f'<text x="{ml + c*cell + cell/2}" y="{mt-8}" text-anchor="middle" '
                 f'font-weight="bold">{cl}</text>')
    for r, rl in enumerate(row_labels):
        s.append(f'<text x="{ml-8}" y="{mt + r*cell + cell/2 + 4}" text-anchor="end" '
                 f'font-weight="bold">{rl}</text>')
        for c in range(nc):
            x, y = ml + c*cell, mt + r*cell
            col, txt = color(M[r][c])
            s.append(f'<rect x="{x}" y="{y}" width="{cell-2}" height="{cell-2}" '
                     f'fill="{col}" stroke="#fff"/>')
            s.append(f'<text x="{x+cell/2}" y="{y+cell/2+4}" text-anchor="middle" '
                     f'fill="#000">{txt}</text>')
    if note:
        s.append(f'<text x="{ml}" y="{H-18}" font-size="11" fill="#555">{note}</text>')
    s.append('</svg>')
    with open(_route(path), "w") as f:
        f.write("\n".join(s))


def write_csv(curves, path="results.csv"):
    with open(_route(path), "w") as f:
        for c in curves:
            f.write(f"# {c['label']}\n")
            f.write("ebn0_db,ber,fer\n")
            fer = c.get("fer", [None] * len(c["x"]))
            for i, (x, y) in enumerate(zip(c["x"], c["y"])):
                f.write(f"{x},{y},{fer[i] if fer[i] is not None else ''}\n")
