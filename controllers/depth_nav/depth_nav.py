"""
depth_nav.py — monocular-depth obstacle-avoidance controller (Pioneer 3-AT).

The proposed method: MiDaS-Small depth -> left/center/right blockage -> shared
Navigator (avoidance blended with goal-seeking) -> wheels.

All the logic lives in the shared library so the three controllers
(depth / classical-CV / sonar) are provably identical except for perception:
  - perception      : controllers/lib/policies.py  (DepthPolicy)
  - steering        : controllers/lib/avoidance.py (Navigator)
  - loop + metrics  : controllers/lib/episode.py, metrics.py
See controller_main.run() for the env-var run parameters.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "lib"))

from controller_main import run   # noqa: E402
from policies import DepthPolicy  # noqa: E402

run(DepthPolicy)
