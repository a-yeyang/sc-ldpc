"""Central output router: send result/figure files to the categorized tree.

The repo keeps all code at the root but stores generated artifacts under
``figures/<family>/`` and ``results/<family>/``.  Every script writes/reads
bare filenames (e.g. ``results_wmc_ALL.json``, ``exp_wmc_rl_vs_random.svg``);
``route()`` maps such a bare name to its categorized location (creating the
directory) so re-running an experiment or re-plotting from committed data lands
in — and reads from — the right folder regardless of the working directory.

``route()`` is idempotent: a name that already contains a directory component is
returned unchanged, so wrapping reads and writes with it always keeps them in
agreement.  Family is decided by filename prefix and must match the layout used
when the artifacts were first sorted (see the move in the repo README).
"""
from __future__ import annotations
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def family(name: str) -> str:
    """Categorize a bare artifact filename into one experiment family."""
    b = os.path.basename(name)
    if b.startswith(("exp_qam", "results_qam", "exp_pas", "results_pas")):
        return "qam_pas"
    if b.startswith(("exp_pam4", "results_pam4")):
        return "pam4"
    if b.startswith(("exp_polar", "results_polar")):
        return "polar"
    if b.startswith("fiber_smoke"):
        return "fiber"
    if b.startswith(("exp_construct", "results_construct", "exp_big", "results_big",
                     "exp_wl", "results_wl", "exp_wmc", "results_wmc")):
        return "construct"
    return "baseline"


def route(name: str) -> str:
    """Map a bare output filename to figures/<family>/ or results/<family>/.

    Names that already carry a directory component are returned unchanged.
    """
    if os.path.dirname(name):
        return name
    kind = "figures" if name.endswith((".svg", ".png")) else "results"
    d = os.path.join(ROOT, kind, family(name))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, os.path.basename(name))


def load(name):
    """json.load a routed result file."""
    import json
    return json.load(open(route(name)))


def save(obj, name):
    """Atomically json.dump ``obj`` to a routed result file."""
    import json
    p = route(name)
    with open(p + ".tmp", "w") as f:
        json.dump(obj, f)
    os.replace(p + ".tmp", p)
