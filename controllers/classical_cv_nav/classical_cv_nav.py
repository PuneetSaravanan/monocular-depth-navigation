"""
classical_cv_nav.py — classical monocular-CV obstacle-avoidance baseline.

Same robot, same steering/goal-seeking, same logging as depth_nav — ONLY the
perception differs: appearance-based free-space detection (floor colour model)
fused with Canny edges, instead of a learned depth model. See
controllers/lib/policies.py :: ClassicalCVPolicy.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "lib"))

from controller_main import run          # noqa: E402
from policies import ClassicalCVPolicy   # noqa: E402

run(ClassicalCVPolicy)
