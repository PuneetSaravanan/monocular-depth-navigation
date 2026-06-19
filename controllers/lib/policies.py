"""
policies.py — the three perception front-ends being compared.

Each Policy turns the robot's sensors into a left/center/right BLOCKAGE estimate
(~0 = clear path, higher = obstacle closer/bigger). They all feed the SAME
Navigator + episode loop, so the experiment isolates *perception quality*.

  DepthPolicy        : MiDaS-Small monocular depth  (the proposed method)
  ClassicalCVPolicy  : appearance/edge monocular CV  (classical camera baseline)
  SonarPolicy        : Pioneer 16-sonar ring         (classical range baseline)

Each policy also exposes NAV_KW: Navigator keyword args whose reaction
thresholds suit that policy's blockage scale, tuned so all three begin reacting
at a comparable ~2.5–3 m for a fair comparison.
"""

import numpy as np

from depth_model import MiDaSDepth, colorize_depth
from avoidance import depth_obstacle_scores


# =========================================================================
# 1) Depth (proposed)
# =========================================================================
class DepthPolicy:
    name = "depth"
    perceive_every = 2
    # Calibrated in Day 3: floor < 0.05, obstacle crosses 0.15 at ~3.4 m.
    # Reacts far, so it can afford a longer commitment (hold_steps) to clear.
    NAV_KW = dict(trigger=0.15, hard=0.45, avoid_gain=2.5, center_gain=2.8,
                  goal_weight_when_avoiding=0.38, hold_steps=45)

    def __init__(self):
        self.depth = MiDaSDepth()

    def perceive(self, bot):
        rgb = bot.get_camera_rgb()
        if rgb is None:
            return np.zeros(3, dtype=np.float32), None
        dmap = self.depth.infer(rgb)
        scores = depth_obstacle_scores(dmap)
        return scores, colorize_depth(dmap)


# =========================================================================
# 2) Classical monocular CV (camera baseline)
# =========================================================================
class ClassicalCVPolicy:
    """Appearance-based free-space detection (Ulrich & Nourbakhsh, 2000).

    The canonical pre-deep-learning monocular obstacle detector:
      1. Model the FLOOR's appearance from a trapezoid directly in front of the
         robot (assumed drivable) as a Hue-Saturation histogram. A histogram
         (not a single mean) is what lets it tolerate a textured / multi-tone
         floor — its whole reason for existing.
      2. Back-project that histogram over the image: pixels unlike the floor are
         candidate OBSTACLES.
      3. In the floor region of the image (below the horizon, where obstacles
         meet the ground), measure the non-floor fraction per left/center/right
         zone, weighting nearer (lower) rows more.

    Note: a naive Canny/edge-density score was tried first and saturated on the
    textured floor (every floor tile is an edge) — a real, documented weakness
    of edge-only methods. Edges are kept only for the visualization overlay.
    """
    name = "classical_cv"
    perceive_every = 2
    # Calibrated via straight-approach (NAV_CALIBRATE=1) with free-space
    # subtraction: clear ~0.00, obstacle center crosses 0.12 at ~2.4 m and
    # reaches ~0.43 at 1.45 m. CV reacts LATER than depth (~2.4 m vs ~3.4 m) —
    # an honest reflection of its smaller effective detection range.
    NAV_KW = dict(trigger=0.12, hard=0.35, avoid_gain=2.6, center_gain=3.0,
                  goal_weight_when_avoiding=0.35, hold_steps=45)

    ROI = (0.58, 0.90)                       # near-floor region rows
    FLOOR_SAMPLE = (0.82, 0.98, 0.35, 0.65)  # front-floor trapezoid (r0,r1,c0,c1)
    FLOOR_THRESH = 35                        # back-projection floor cutoff (0-255)
    EMA = 0.04                               # floor-histogram update rate

    def __init__(self):
        import cv2
        self.cv2 = cv2
        self._hist = None                    # EMA of the floor Hue-Sat histogram
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def perceive(self, bot):
        cv2 = self.cv2
        rgb = bot.get_camera_rgb()
        if rgb is None:
            return np.zeros(3, dtype=np.float32), None
        h, w = rgb.shape[:2]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        # --- (1) Floor Hue-Sat histogram, smoothed over time (EMA) so a close
        #         obstacle covering the sample patch can't hijack the model. ---
        fr0, fr1, fc0, fc1 = self.FLOOR_SAMPLE
        floor = hsv[int(fr0 * h):int(fr1 * h), int(fc0 * w):int(fc1 * w)]
        hist = cv2.calcHist([floor], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        if self._hist is None:
            self._hist = hist
        else:
            self._hist = (1 - self.EMA) * self._hist + self.EMA * hist

        # --- (2) Back-project: high = looks like floor ---
        backproj = cv2.calcBackProject([hsv], [0, 1], self._hist,
                                       [0, 180, 0, 256], 1)
        backproj = cv2.GaussianBlur(backproj, (5, 5), 0)
        not_floor = (backproj < self.FLOOR_THRESH).astype(np.uint8)

        # --- (2b) Morphological opening removes speckle (misclassified floor
        #          grain) but keeps solid obstacle blobs => big SNR win. ---
        not_floor = cv2.morphologyEx(not_floor, cv2.MORPH_OPEN, self._kernel)
        nf = not_floor.astype(np.float32)

        # --- (3) Non-floor fraction per zone in the floor ROI, near-weighted ---
        r0, r1 = int(self.ROI[0] * h), int(self.ROI[1] * h)
        roi = nf[r0:r1, :]
        weight = np.linspace(0.5, 1.5, roi.shape[0])[:, None]  # nearer rows weigh more
        wsum = weight.sum() * (roi.shape[1] / 3.0)
        raw = np.array([float((z * weight).sum() / wsum)
                        for z in np.array_split(roi, 3, axis=1)], dtype=np.float32)
        # Subtract the most-open zone as a per-frame free-space baseline. This
        # cancels the uniform floor-misclassification noise that otherwise
        # spikes when the robot turns (walls/floor fill all zones together),
        # leaving only LOCALIZED obstacles — same idea as the depth row-median.
        scores = np.clip(raw - raw.min(), 0.0, 1.0).astype(np.float32)

        # Visualization: red = candidate obstacle (non-floor).
        vis = (rgb * 0.45).astype(np.uint8)
        vis[..., 0] = np.clip(vis[..., 0] + 180 * not_floor, 0, 255)
        return scores, vis


# =========================================================================
# 3) Sonar ring (range baseline)
# =========================================================================
class SonarPolicy:
    """Classic range-based avoidance: convert the front sonar zones to a
    blockage score. (Same idea as a Braitenberg/potential-field sonar
    controller, expressed as an L/C/R blockage for the shared Navigator.)"""
    name = "sonar"
    perceive_every = 1
    # Calibrated: the sonar beam only picks up the 0.6 m box at ~2.0 m, then
    # reads accurately. So it reacts later than depth but precisely. trigger is
    # raised to 0.25 so weak grazing returns (~0.1-0.17) don't commit it the
    # wrong way; hold_steps is SHORT (reacts late => must not over-commit and
    # wander); `hard` high so it arcs rather than pivoting in place.
    NAV_KW = dict(trigger=0.25, hard=0.75, avoid_gain=2.5, center_gain=2.8,
                  goal_weight_when_avoiding=0.42, hold_steps=26)

    REACT_DIST = 3.0   # metres: distance at which blockage starts rising from 0

    def perceive(self, bot):
        z = bot.read_sonar_zones()   # nearest distance (m) per zone
        def blockage(d):
            return float(np.clip((self.REACT_DIST - d) / self.REACT_DIST, 0.0, 1.0))
        scores = np.array([blockage(z["left"]), blockage(z["center"]),
                           blockage(z["right"])], dtype=np.float32)
        return scores, None          # no camera visualization for sonar
