"""
courses.py — obstacle-course layouts of increasing difficulty.

Each course is an obstacle list (box centre x, y) plus start/goal and a time
budget. Worlds are generated from these specs (see make_world.py) so that each
trial can apply small SEEDED jitter to obstacle positions and the robot's start
pose — turning an otherwise deterministic simulator into N genuinely different
trials, which is what makes a *collision rate* meaningful.

Coordinate frame: ENU, robot starts at the left (-x) facing +x toward the goal.
Arena is 16x16 (walls at +/-8) so goals/obstacles stay clear of the walls.
Boxes are 0.6 m; gaps in the base layouts are >=1.7 m so small jitter keeps
every course passable.
"""

ARENA = 16.0
BOX = 0.6
START = (-5.0, 0.0)
GOAL = (5.0, 0.0)

COURSES = {
    # name: dict(obstacles=[(x,y),...], goal, start, max_time, difficulty)
    "single": dict(
        difficulty=1, max_time=90,
        obstacles=[(0.0, 0.0)],
    ),
    "slalom": dict(
        difficulty=2, max_time=120,
        obstacles=[(-1.8, 0.8), (0.0, -0.9), (1.8, 0.8)],
    ),
    "cluttered": dict(
        difficulty=3, max_time=150,
        obstacles=[(-2.0, 1.2), (-1.0, -1.0), (0.5, 0.9),
                   (0.8, -1.4), (2.2, 0.3), (2.5, -1.2)],
    ),
    "dense": dict(
        difficulty=4, max_time=160,
        # Two staggered rows forming a zig-zag corridor (always a diagonal path).
        obstacles=[(-1.0, -1.6), (-1.0, 0.6),
                   (0.6, -0.6), (0.6, 1.6),
                   (2.2, -1.6), (2.2, 0.6)],
    ),
}


def course_names_by_difficulty():
    return sorted(COURSES, key=lambda c: COURSES[c]["difficulty"])
