"""
sonar_nav.py — sonar-ring obstacle-avoidance baseline (classical range sensing).

Same robot, same steering/goal-seeking, same logging as depth_nav — ONLY the
perception differs: the Pioneer's 16-sonar ring (front zones) instead of a
camera. See controllers/lib/policies.py :: SonarPolicy.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "lib"))

from controller_main import run    # noqa: E402
from policies import SonarPolicy   # noqa: E402

run(SonarPolicy)
