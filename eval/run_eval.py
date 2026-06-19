#!/usr/bin/env python3
"""
run_eval.py — batch evaluation harness for the depth-vs-classical comparison.

Runs each navigation controller through each course N times in headless Webots
and aggregates the per-run metrics (which the controllers append to a shared
JSON-Lines file via controllers/lib/metrics.py). Prints a summary table of
success rate, collision rate, mean completion time and distance, and writes a
markdown results table for the README.

Each trial's world is generated from eval/courses.py with SEEDED jitter
(eval/make_world.py). The seed depends only on (course, rep) — so in rep k all
three controllers face the IDENTICAL jittered layout (fair), while different
reps are genuinely different (meaningful collision rate).

Usage:
    python eval/run_eval.py                          # all courses, all ctrls, 5 reps
    python eval/run_eval.py --courses single slalom --reps 3
    python eval/run_eval.py --summarize-only         # re-print from existing results
"""

import argparse
import collections
import json
import os
import pathlib
import subprocess
import sys
import zlib

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
WORLDS = ROOT / "worlds"
sys.path.insert(0, str(HERE))

from courses import COURSES, GOAL, course_names_by_difficulty  # noqa: E402
from make_world import make_world                              # noqa: E402

WEBOTS_HOME = os.environ.get("WEBOTS_HOME", "/Applications/Webots.app/Contents")
WEBOTS = os.environ.get("WEBOTS_BIN", WEBOTS_HOME + "/MacOS/webots")
CONTROLLERS = ["depth_nav", "classical_cv_nav", "sonar_nav"]
NAME = {"depth_nav": "depth", "classical_cv_nav": "classical_cv", "sonar_nav": "sonar"}


def _seed(course, rep):
    return zlib.crc32(f"{course}-{rep}".encode())


def run_once(course, controller, rep, results_path):
    cfg = COURSES[course]
    text = make_world(course, seed=_seed(course, rep))
    text = text.replace('controller "depth_nav"', f'controller "{controller}"')
    tmp = WORLDS / "_eval_tmp.wbt"             # in worlds/ so ../protos resolves
    tmp.write_text(text)
    env = dict(os.environ,
               WEBOTS_HOME=WEBOTS_HOME,
               NAV_RESULTS=results_path,
               NAV_COURSE=course,
               NAV_RUN_ID=str(rep),
               NAV_GOAL=f"{GOAL[0]},{GOAL[1]}",
               NAV_MAX_TIME=str(cfg["max_time"]))
    proc = subprocess.Popen(
        [WEBOTS, "--batch", "--mode=fast", "--no-rendering", "--minimize",
         "--stdout", "--stderr", str(tmp)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wall_timeout = cfg["max_time"] * 4 + 120   # safety net; sim self-quits first
    try:
        proc.wait(timeout=wall_timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    finally:
        tmp.unlink(missing_ok=True)


def _aggregate(results_path):
    rows = [json.loads(l) for l in open(results_path) if l.strip()]
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["course"], r["controller"])].append(r)
    table = []
    for course in course_names_by_difficulty():
        if not any(c == course for c, _ in groups):
            continue
        diff = COURSES[course]["difficulty"]
        for ctrl in CONTROLLERS:
            runs = groups.get((course, NAME[ctrl]), [])
            if not runs:
                continue
            n = len(runs)
            succ = sum(r["outcome"] == "reached" for r in runs) / n
            coll_rate = sum(r["collisions"] > 0 for r in runs) / n
            mean_coll = sum(r["collisions"] for r in runs) / n
            ok = [r for r in runs if r["outcome"] == "reached"] or runs
            mt = sum(r["time_s"] for r in ok) / len(ok)
            md = sum(r["distance_m"] for r in ok) / len(ok)
            table.append(dict(course=course, difficulty=diff, controller=NAME[ctrl],
                              n=n, success=succ, coll_rate=coll_rate,
                              mean_coll=mean_coll, time_s=mt, dist_m=md))
    return table


def summarize(results_path, md_path=None):
    table = _aggregate(results_path)
    print(f"\n{'course':10} {'controller':13} {'n':>3} {'success':>8} "
          f"{'coll.rate':>9} {'mean_coll':>9} {'time_s':>8} {'dist_m':>7}")
    print("-" * 80)
    for r in table:
        print(f"{r['course']:10} {r['controller']:13} {r['n']:>3} "
              f"{r['success']:>7.0%} {r['coll_rate']:>8.0%} {r['mean_coll']:>9.2f} "
              f"{r['time_s']:>8.1f} {r['dist_m']:>7.1f}")
    print()
    if md_path:
        _write_markdown(table, md_path)
        print(f"wrote markdown table -> {md_path}")


def _write_markdown(table, md_path):
    lines = ["| Course (difficulty) | Controller | Trials | Success | Collision rate | Mean collisions | Mean time (s) | Mean dist (m) |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in table:
        lines.append(
            f"| {r['course']} ({r['difficulty']}) | {r['controller']} | {r['n']} | "
            f"{r['success']:.0%} | {r['coll_rate']:.0%} | {r['mean_coll']:.2f} | "
            f"{r['time_s']:.1f} | {r['dist_m']:.1f} |")
    pathlib.Path(md_path).write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--courses", nargs="+", default=course_names_by_difficulty())
    ap.add_argument("--controllers", nargs="+", default=CONTROLLERS)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=str(HERE / "results" / "results.jsonl"))
    ap.add_argument("--md", default=str(HERE / "results" / "results_table.md"))
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    # IMPORTANT: make the results path absolute. Webots runs each controller with
    # its cwd set to the controller's own directory, so a relative NAV_RESULTS
    # would be written under controllers/<name>/ instead of here.
    args.out = str(pathlib.Path(args.out).resolve())
    args.md = str(pathlib.Path(args.md).resolve())

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        open(args.out, "w").close()
        for course in args.courses:
            for ctrl in args.controllers:
                for rep in range(args.reps):
                    print(f"running {course} / {ctrl} / rep {rep} ...", flush=True)
                    run_once(course, ctrl, rep, args.out)
    summarize(args.out, args.md)


if __name__ == "__main__":
    main()
