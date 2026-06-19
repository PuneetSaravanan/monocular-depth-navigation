#!/usr/bin/env python3
"""
plot_results.py — render the evaluation results as a grouped bar chart for the
README. Reads eval/results/results.jsonl, writes docs/results_chart.png.

    python eval/plot_results.py
"""

import collections
import json
import pathlib

import matplotlib
matplotlib.use("Agg")            # headless
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

COURSE_ORDER = ["single", "slalom", "cluttered", "dense"]
CTRL_ORDER = ["depth", "classical_cv", "sonar"]
CTRL_LABEL = {"depth": "depth (proposed)", "classical_cv": "classical CV", "sonar": "sonar"}
CTRL_COLOR = {"depth": "#2a9d8f", "classical_cv": "#e9a23b", "sonar": "#b091c3"}


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    coll = collections.defaultdict(lambda: [0, 0])   # (course,ctrl)->[runs_with_coll, n]
    succ = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        k = (r["course"], r["controller"])
        coll[k][0] += r["collisions"] > 0
        coll[k][1] += 1
        succ[k][0] += r["outcome"] == "reached"
        succ[k][1] += 1
    return coll, succ


def _grouped(ax, data, courses, title, ylabel):
    import numpy as np
    x = np.arange(len(courses))
    w = 0.26
    for i, ctrl in enumerate(CTRL_ORDER):
        vals = [100 * (data[(c, ctrl)][0] / data[(c, ctrl)][1])
                if data[(c, ctrl)][1] else 0 for c in courses]
        ax.bar(x + (i - 1) * w, vals, w, label=CTRL_LABEL[ctrl], color=CTRL_COLOR[ctrl])
    ax.set_xticks(x)
    ax.set_xticklabels(courses)
    ax.set_ylim(0, 105)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)


def main():
    coll, succ = load(HERE / "results" / "results.jsonl")
    courses = [c for c in COURSE_ORDER if any(k[0] == c for k in coll)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    _grouped(axes[0], coll, courses, "Collision rate by course (lower is better)",
             "% of trials with a collision")
    _grouped(axes[1], succ, courses, "Success rate by course (higher is better)",
             "% of trials reaching goal")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Depth vs. classical baselines — 8 jittered trials per cell",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = ROOT / "docs" / "results_chart.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
