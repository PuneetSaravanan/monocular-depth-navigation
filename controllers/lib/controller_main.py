"""
controller_main.py — shared entry point for all three navigation controllers.

Each controller file (depth_nav, classical_cv_nav, sonar_nav) is a thin wrapper:

    from controller_main import run
    from policies import DepthPolicy
    run(DepthPolicy)

Run parameters come from environment variables so the Day-5 evaluation harness
can launch many runs/courses/controllers without editing code:

    NAV_GOAL="x,y"   goal point            (default "5,0")
    NAV_COURSE=name  course label for logs (default "adhoc")
    NAV_RUN_ID=k     run index for logs    (default "0")
    NAV_RESULTS=path append per-run JSON summary here (optional)
    NAV_MAX_TIME=s   per-run time budget   (default "120")
"""

import os
import sys
import pathlib

LIB = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(LIB))

from robot_io import PioneerRobot          # noqa: E402
from avoidance import Navigator            # noqa: E402
from episode import EpisodeConfig, run_episode  # noqa: E402
from metrics import RunLogger              # noqa: E402


def run(policy_cls):
    if os.environ.get("NAV_CALIBRATE") == "1":
        return _calibrate(policy_cls)

    goal = tuple(float(v) for v in os.environ.get("NAV_GOAL", "5,0").split(","))
    course = os.environ.get("NAV_COURSE", "adhoc")
    run_id = os.environ.get("NAV_RUN_ID", "0")
    results = os.environ.get("NAV_RESULTS")
    max_time = float(os.environ.get("NAV_MAX_TIME", "120"))

    bot = PioneerRobot(enable_camera=True, enable_sonars=True,
                       enable_gps=True, enable_display=True)
    policy = policy_cls()
    nav = Navigator(goal, **getattr(policy_cls, "NAV_KW", {}))
    logger = RunLogger(policy.name, course, run_id, bot.timestep, results)

    record_path = os.environ.get("NAV_RECORD")   # set to an mp4 path to capture a demo
    print(f"[{policy.name}] start course={course} run={run_id} "
          f"goal={goal} max_time={max_time}s", flush=True)
    run_episode(bot, policy, nav, EpisodeConfig(max_time=max_time), logger,
                record_path=record_path)


def _calibrate(policy_cls):
    """Drive straight at a box on the +x axis, printing each policy's L/C/R
    blockage vs ground-truth distance (|x|). Used to set NAV_KW thresholds."""
    bot = PioneerRobot(enable_camera=True, enable_sonars=True,
                       enable_gps=True, enable_display=True)
    policy = policy_cls()
    print(f"[{policy.name}] CALIBRATE: dist_m  L  C  R", flush=True)
    last = None
    frame = 0
    while bot.step():
        frame += 1
        pos = bot.get_position()
        if pos is not None and pos[0] >= -0.9:   # stop before contact
            print(f"[{policy.name}] CALIBRATE done at x={pos[0]:.2f}", flush=True)
            break
        if frame % policy.perceive_every == 0:
            scores, _ = policy.perceive(bot)
            d = abs(pos[0]) if pos else None
            if d is not None and (last is None or abs(d - last) >= 0.25):
                print(f"[{policy.name}] CAL {d:5.2f} "
                      f"{scores[0]:.3f} {scores[1]:.3f} {scores[2]:.3f}", flush=True)
                last = d
        bot.drive(forward=0.30, turn=0.0)
