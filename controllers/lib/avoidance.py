"""
avoidance.py — depth-based reactive obstacle avoidance + goal-seeking.

Perception (depth_obstacle_scores):
  Turn a MiDaS depth map into three numbers — how strongly an obstacle stands
  out from the floor in the LEFT / CENTER / RIGHT of the path.

  Key idea: MiDaS depth on a flat floor has a strong vertical "lower = nearer"
  gradient, and an arbitrary per-frame scale. We make obstacle detection
  invariant to BOTH by:
    1. normalizing each frame to 0..1 with robust percentiles, and
    2. subtracting, per image row, the row's median nearness (the floor /
       background baseline at that height).
  What's left (positive residual) is "stuff NEARER than the floor at that row"
  — i.e. an actual obstacle sticking up off the ground. An empty scene yields
  residuals ≈ 0; a box yields a strong positive blob in its zone. We only look
  at a horizontal BAND around the horizon, where obstacles about to be hit live.

Control (Navigator):
  When the path is clear, steer toward the goal (heading inferred from GPS
  motion — no compass needed). When an obstacle is detected, ADD a steer-away
  correction on top of the goal steering (a blend, not a hard switch) so the
  robot keeps being pulled back onto its route and can't spiral out into a wall.
"""

import math
import numpy as np


# --- Perception: depth map -> per-zone obstacle score ---------------------
def depth_obstacle_scores(depth, band=(0.38, 0.62), pct_lo=5, pct_hi=95,
                          zone_pct=85):
    """Return np.array([left, center, right]) obstacle scores (~0..1).

    0  => nothing nearer than the floor in that zone (clear).
    >0 => an obstacle stands out from the floor; larger = nearer/bigger.

    band     : (top, bottom) fractions of image height — the horizon ROI.
    zone_pct : percentile of the per-zone residual used as its score (85 =
               "is a sizeable near blob present?", robust to stray pixels).
    """
    h, w = depth.shape

    # (1) Frame-invariant normalization (far -> 0, near -> 1).
    lo = np.percentile(depth, pct_lo)
    hi = np.percentile(depth, pct_hi)
    if hi - lo < 1e-6:
        return np.zeros(3, dtype=np.float32)
    dn = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)

    # (2) Horizon band ROI.
    r0, r1 = int(band[0] * h), int(band[1] * h)
    roi = dn[r0:r1, :]

    # (3) Remove the floor/background baseline per row, keep "nearer-than-floor".
    baseline = np.median(roi, axis=1, keepdims=True)
    residual = np.clip(roi - baseline, 0.0, None)

    # (4) Summarize each left/center/right third by a high percentile.
    thirds = np.array_split(residual, 3, axis=1)
    return np.array([np.percentile(z, zone_pct) for z in thirds],
                    dtype=np.float32)


# --- Goal heading from GPS motion (no compass) ----------------------------
def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class GoalSeeker:
    def __init__(self, goal_xy, kp=1.3, reach_tol=0.5, move_eps=1e-3):
        self.goal = np.asarray(goal_xy, dtype=float)
        self.kp = kp
        self.reach_tol = reach_tol
        self.move_eps = move_eps

    def reached(self, pos):
        return np.linalg.norm(self.goal - np.asarray(pos)) < self.reach_tol

    def heading_turn(self, pos, prev_pos):
        """Turn command in [-1,1] toward the goal (positive = steer left)."""
        pos = np.asarray(pos, dtype=float)
        prev_pos = np.asarray(prev_pos, dtype=float)
        delta = pos - prev_pos
        if np.linalg.norm(delta) < self.move_eps:
            return 0.0
        heading = math.atan2(delta[1], delta[0])
        bearing = math.atan2(*(self.goal - pos)[::-1])
        err = _wrap(bearing - heading)
        return max(-1.0, min(1.0, self.kp * err / (math.pi / 2)))


# --- Combined navigation policy -------------------------------------------
class Navigator:
    """Blend goal-seeking with depth-based avoidance into (forward, turn).

    Thresholds `trigger` / `hard` are CALIBRATED against ground-truth distance
    (see scripts/calibrate output) rather than guessed."""

    # Defaults below are CALIBRATED from a straight-approach run (see Day-3
    # notes): floor/empty score < 0.05, obstacle crosses 0.15 at ~3.4 m, peaks
    # ~0.68 at ~2.2 m, then collapses as the box fills the view (<1.5 m).
    #
    # Directional hysteresis: once avoidance commits to a side, it KEEPS arcing
    # that way for `hold_steps` steps and maintains a minimum turn, even if the
    # obstacle momentarily drops out of view. Without this, a narrow/noisy
    # detector (sonar, classical CV) turns, loses the obstacle, gets yanked
    # straight back by goal-seeking, and oscillates into / sticks on the box.
    def __init__(self, goal_xy, cruise=0.40,
                 trigger=0.15, hard=0.45,
                 avoid_gain=2.5, center_gain=2.8, prefer_left=True,
                 goal_weight_when_avoiding=0.38, max_turn=1.0,
                 hold_steps=45, min_avoid_turn=0.30):
        self.goal = GoalSeeker(goal_xy)
        self.cruise = cruise
        self.trigger = trigger      # obstacle score that starts avoidance
        self.hard = hard            # score above which we slow hard / pivot
        self.avoid_gain = avoid_gain
        self.center_gain = center_gain
        self.prefer = 1.0 if prefer_left else -1.0
        self.goal_w_avoid = goal_weight_when_avoiding
        self.max_turn = max_turn
        self.hold_steps = hold_steps
        self.min_avoid_turn = min_avoid_turn
        # Hysteresis state:
        self._dir = 0.0             # committed avoid direction (+1 left, -1 right)
        self._hold = 0              # steps remaining in the current commitment

    def reached(self, pos):
        return self.goal.reached(pos)

    def decide(self, scores, pos, prev_pos):
        """scores=(L,C,R). Returns (forward, turn, mode)."""
        L, C, R = (float(s) for s in scores)
        goal_turn = (self.goal.heading_turn(pos, prev_pos)
                     if (pos is not None and prev_pos is not None) else 0.0)

        block = max(L, C, R)

        # (Re)commit to an avoid direction whenever an obstacle is in view.
        if block >= self.trigger:
            if self._hold <= 0:
                # Choose the more open side; (R>=L) => right blocked => go left.
                self._dir = 1.0 if R >= L else -1.0
            self._hold = self.hold_steps

        # Not avoiding and nothing committed: pure goal-seeking.
        if self._hold <= 0:
            return self.cruise, _clamp(goal_turn, self.max_turn), "clear"

        # --- Committed avoidance (this step counts down the hold) ---
        self._hold -= 1
        excess_c = max(0.0, C - self.trigger)
        # Magnitude grows with how blocked we are, but never below min_avoid_turn
        # so the robot keeps arcing clear even after the obstacle leaves view.
        mag = max(self.min_avoid_turn,
                  self.avoid_gain * abs(R - L) + self.center_gain * excess_c)
        turn = self._dir * mag + self.goal_w_avoid * goal_turn

        # Slow with center blockage but never stall (>=30% cruise) so it ARCS.
        forward = self.cruise * max(0.3, 1.0 - 1.2 * C)

        if block > self.hard:                 # very close: pivot toward opening
            forward = min(forward, 0.06)
            turn = self._dir * self.max_turn

        return max(0.0, forward), _clamp(turn, self.max_turn), "avoid"


def _clamp(v, m):
    return max(-m, min(m, v))
