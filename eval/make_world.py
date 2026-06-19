"""
make_world.py — generate a Webots .wbt world from a course spec.

Keeps a single source of truth for the arena/robot/obstacle setup and lets the
evaluation harness apply seeded jitter per trial. Also used to (re)generate the
canonical, un-jittered course worlds in worlds/ for opening in the Webots GUI.
"""

import math
import random

from courses import COURSES, ARENA, BOX, START, GOAL

_HEADER = '''#VRML_SIM R2025a utf8
# AUTO-GENERATED from eval/courses.py by eval/make_world.py — edit the spec, not this.
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/backgrounds/protos/TexturedBackground.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/backgrounds/protos/TexturedBackgroundLight.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/floors/protos/RectangleArena.proto"
EXTERNPROTO "{proto}"

WorldInfo {{
  info [ "Course: {name} (difficulty {difficulty})." ]
  title "Course - {name}"
  basicTimeStep 32
}}
Viewpoint {{
  orientation -0.32 0.20 0.93 1.38
  position 0.0 -8.0 9.0
  follow "Pioneer 3-AT"
  followSmoothness 0.2
}}
TexturedBackground {{
}}
TexturedBackgroundLight {{
}}
RectangleArena {{
  floorSize {arena} {arena}
  floorAppearance PBRAppearance {{
    baseColor 0.82 0.80 0.74
    roughness 1
    metalness 0
  }}
  wallHeight 0.4
}}
'''

_OBSTACLE_FIRST = '''DEF OB{i} Solid {{
  translation {x:.3f} {y:.3f} 0.3
  children [
    DEF OBSTACLE_SHAPE Shape {{
      appearance PBRAppearance {{
        baseColor 0.85 0.35 0.15
        roughness 0.9
        metalness 0
      }}
      geometry Box {{ size {b} {b} {b} }}
    }}
  ]
  name "obstacle{i}"
  boundingObject USE OBSTACLE_SHAPE
}}
'''

_OBSTACLE_MORE = '''DEF OB{i} Solid {{
  translation {x:.3f} {y:.3f} 0.3
  children [ USE OBSTACLE_SHAPE ]
  name "obstacle{i}"
  boundingObject USE OBSTACLE_SHAPE
}}
'''

_GOAL = '''DEF GOAL Solid {{
  translation {gx} {gy} 0.02
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor 0 0.8 0.2 metalness 0 roughness 1 }}
      geometry Cylinder {{ height 0.04 radius 0.4 }}
    }}
  ]
  name "goal"
}}
'''

_ROBOT = '''NavRobot {{
  translation {sx:.3f} {sy:.3f} 0.15
  rotation 0 0 1 {theta:.4f}
  controller "depth_nav"
}}
'''


def make_world(name, seed=None, proto_path="../protos/NavRobot.proto"):
    """Return .wbt text for course `name`. If `seed` is given, apply small
    jitter to obstacle positions and the robot's start pose for trial variety."""
    spec = COURSES[name]
    rng = random.Random(seed) if seed is not None else None

    obstacles = []
    for (x, y) in spec["obstacles"]:
        if rng is not None:
            x += rng.gauss(0, 0.15)
            y += rng.gauss(0, 0.15)
        obstacles.append((x, y))

    sx, sy = START
    theta = 0.0
    if rng is not None:
        sy += rng.gauss(0, 0.30)       # start a bit off-centre
        theta += rng.gauss(0, 0.10)    # and slightly mis-aimed

    parts = [_HEADER.format(proto=proto_path, name=name,
                            difficulty=spec["difficulty"], arena=ARENA)]
    for i, (x, y) in enumerate(obstacles, 1):
        tmpl = _OBSTACLE_FIRST if i == 1 else _OBSTACLE_MORE
        parts.append(tmpl.format(i=i, x=x, y=y, b=BOX))
    parts.append(_GOAL.format(gx=GOAL[0], gy=GOAL[1]))
    parts.append(_ROBOT.format(sx=sx, sy=sy, theta=theta))
    return "".join(parts)


def regenerate_canonical(worlds_dir):
    """Write un-jittered worlds/course_<name>.wbt for every course (GUI/demo)."""
    import pathlib
    out = pathlib.Path(worlds_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in COURSES:
        (out / f"course_{name}.wbt").write_text(make_world(name))
    return [f"course_{name}.wbt" for name in COURSES]


if __name__ == "__main__":
    import pathlib
    here = pathlib.Path(__file__).resolve().parent
    files = regenerate_canonical(here.parent / "worlds")
    print("wrote:", ", ".join(files))
