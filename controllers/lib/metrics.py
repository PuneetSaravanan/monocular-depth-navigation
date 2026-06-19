"""
metrics.py — per-run logging shared by all navigation controllers.

Records exactly the quantities the project compares across approaches:
  - outcome          : "reached" | "timeout" | "stuck"
  - time_s           : sim time from start to outcome
  - distance_m       : path length travelled (integrated from GPS)
  - collisions       : number of distinct contact events (bumper rising edges)
  - contact_time_s   : total time spent in contact (scraping along an obstacle)

One JSON object per run is appended to a results file (JSON-Lines), so Day-5
evaluation can aggregate many runs across controllers and courses.
"""

import json
import math
import pathlib


class RunLogger:
    def __init__(self, controller, course, run_id, timestep_ms,
                 results_path=None, contact_release_steps=8):
        self.controller = controller
        self.course = course
        self.run_id = run_id
        self.dt = timestep_ms / 1000.0
        self.results_path = results_path
        # A collision is counted on a rising edge; we require the bumper to be
        # released for `contact_release_steps` steps before a new contact counts
        # as a *separate* collision (debounce against jitter on one obstacle).
        self.release_needed = contact_release_steps

        self.distance = 0.0
        self.collisions = 0
        self.contact_steps = 0
        self._prev_pos = None
        self._in_contact = False
        self._released = self.release_needed
        self.outcome = None
        self.time_s = 0.0

    def update(self, pos, in_contact):
        """Call once per control step with current (x,y) and bumper state."""
        if pos is not None:
            if self._prev_pos is not None:
                self.distance += math.dist(pos, self._prev_pos)
            self._prev_pos = pos

        if in_contact:
            self.contact_steps += 1
            if not self._in_contact and self._released >= self.release_needed:
                self.collisions += 1          # new distinct collision event
            self._in_contact = True
            self._released = 0
        else:
            self._released += 1
            if self._released >= self.release_needed:
                self._in_contact = False

    def finalize(self, outcome, sim_time):
        self.outcome = outcome
        self.time_s = round(sim_time, 2)
        rec = self.summary()
        if self.results_path:
            p = pathlib.Path(self.results_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    def summary(self):
        return {
            "controller": self.controller,
            "course": self.course,
            "run_id": self.run_id,
            "outcome": self.outcome,
            "time_s": self.time_s,
            "distance_m": round(self.distance, 2),
            "collisions": self.collisions,
            "contact_time_s": round(self.contact_steps * self.dt, 2),
        }
