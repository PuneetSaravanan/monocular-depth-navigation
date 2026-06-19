"""
robot_io.py — thin, well-documented wrapper around the Webots Pioneer 3-AT.

All three controllers (depth_nav, classical_cv_nav, sonar_nav) talk to the
robot through this single class so that the *navigation policy* is the only
thing that differs between them. That keeps the comparison fair: same robot,
same sensors, same speed limits, same logging — only the steering brain changes.

Pioneer 3-AT device names (from the official Webots proto):
  - wheel motors : "front left wheel", "front right wheel",
                   "back left wheel",  "back right wheel"
  - sonar ring   : "so0" .. "so15"  (16 sonars; +x is the robot's forward axis)
  - camera/gps   : added by us in the world file's `extensionSlot`

Front sonar layout (looking down, +x forward, +y left):
        so2 so3          <- straight ahead
     so1       so4
   so0           so5     <- front shoulders
  so15             so6   <- sides
"""

import numpy as np
# Supervisor is a Robot subclass; we use it so the controller can end the run
# with simulationQuit() (the robot node sets supervisor TRUE). All normal
# device access works exactly the same as with Robot.
from controller import Supervisor  # provided by Webots at runtime


# Sonars grouped into the three forward zones we steer with. Indices chosen
# from the proto geometry: so0/so1 lean left, so2/so3 point dead ahead,
# so4/so5 lean right.
FRONT_SONARS = {
    "left":   ["so0", "so1"],
    "center": ["so2", "so3"],
    "right":  ["so4", "so5"],
}


class PioneerRobot:
    """Hardware-abstraction layer for the Pioneer 3-AT in Webots."""

    # Physical / tuning constants (documented so they're explainable).
    MAX_WHEEL_SPEED = 6.0      # rad/s — well under the motor's maxVelocity
    WHEEL_RADIUS    = 0.11     # m, Pioneer 3-AT wheel radius
    AXLE_LENGTH     = 0.40     # m, approx left-right wheel separation

    def __init__(self, enable_camera=True, enable_sonars=True, enable_gps=True,
                 enable_display=False):
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())

        # --- Wheels -------------------------------------------------------
        # The Pioneer 3-AT is 4-wheel skid-steer: we drive the two wheels on
        # each side together, so it behaves like a differential-drive robot.
        wheel_names = [
            "front left wheel", "back left wheel",      # left pair
            "front right wheel", "back right wheel",    # right pair
        ]
        self.wheels = [self.robot.getDevice(n) for n in wheel_names]
        for w in self.wheels:
            w.setPosition(float("inf"))   # velocity-control mode
            w.setVelocity(0.0)

        # --- Camera -------------------------------------------------------
        self.camera = None
        if enable_camera:
            self.camera = self.robot.getDevice("camera")
            self.camera.enable(self.timestep)

        # --- Sonars -------------------------------------------------------
        self.sonars = {}
        if enable_sonars:
            for i in range(16):
                name = f"so{i}"
                dev = self.robot.getDevice(name)
                if dev is not None:
                    dev.enable(self.timestep)
                    self.sonars[name] = dev

        # --- GPS (for distance-travelled / completion logging) ------------
        self.gps = None
        if enable_gps:
            self.gps = self.robot.getDevice("gps")
            if self.gps is not None:
                self.gps.enable(self.timestep)

        # --- Display (optional on-screen depth-map panel) -----------------
        self.display = None
        if enable_display:
            self.display = self.robot.getDevice("depth_display")

        # --- Bumper (collision sensor) ------------------------------------
        # A "bumper" TouchSensor returns 1.0 while its shell is in contact with
        # something (wall/obstacle), 0.0 otherwise. Used purely for measuring
        # collisions — never as a control input — so the metric is identical and
        # fair across all three navigation policies.
        self.bumper = self.robot.getDevice("bumper")
        if self.bumper is not None:
            self.bumper.enable(self.timestep)

    # ---------------------------------------------------------------------
    # Simulation stepping
    # ---------------------------------------------------------------------
    def step(self):
        """Advance the simulation one control step. Returns False when Webots
        asks the controller to quit."""
        return self.robot.step(self.timestep) != -1

    def time(self):
        """Simulation time in seconds."""
        return self.robot.getTime()

    def quit(self, code=0):
        """End the simulation so a batch run exits cleanly (Webots does not quit
        just because the controller returns)."""
        self.robot.simulationQuit(code)

    # ---------------------------------------------------------------------
    # Movie recording (Day-6 demo capture; needs rendering, not headless)
    # ---------------------------------------------------------------------
    def start_movie(self, path, width=900, height=600, quality=85):
        """Begin recording the Webots 3D view to an mp4 (Supervisor feature)."""
        self.robot.movieStartRecording(
            path, width, height, codec=0, quality=quality,
            acceleration=1, caption=False)

    def stop_movie(self):
        """Stop recording and block until the movie file is finalized."""
        self.robot.movieStopRecording()
        # Step the sim until Webots reports the file is written.
        while not self.robot.movieIsReady():
            if self.robot.step(self.timestep) == -1:
                break

    # ---------------------------------------------------------------------
    # Actuation
    # ---------------------------------------------------------------------
    def set_wheel_speeds(self, left, right):
        """Set left/right wheel angular velocity (rad/s), clamped to limits."""
        left = max(-self.MAX_WHEEL_SPEED, min(self.MAX_WHEEL_SPEED, left))
        right = max(-self.MAX_WHEEL_SPEED, min(self.MAX_WHEEL_SPEED, right))
        # wheels = [FL, BL, FR, BR]
        self.wheels[0].setVelocity(left)
        self.wheels[1].setVelocity(left)
        self.wheels[2].setVelocity(right)
        self.wheels[3].setVelocity(right)

    def drive(self, forward, turn):
        """Convenience: forward speed in [0,1], turn in [-1,1]
        (positive turn = steer left). Maps onto wheel speeds."""
        base = forward * self.MAX_WHEEL_SPEED
        delta = turn * self.MAX_WHEEL_SPEED
        self.set_wheel_speeds(base - delta, base + delta)

    def stop(self):
        self.set_wheel_speeds(0.0, 0.0)

    # ---------------------------------------------------------------------
    # Perception
    # ---------------------------------------------------------------------
    def get_camera_rgb(self):
        """Return the current camera frame as an (H, W, 3) uint8 RGB array.

        Webots delivers the image as BGRA bytes; we drop alpha and flip BGR->RGB.
        """
        if self.camera is None:
            return None
        w = self.camera.getWidth()
        h = self.camera.getHeight()
        raw = np.frombuffer(self.camera.getImage(), dtype=np.uint8)
        bgra = raw.reshape((h, w, 4))
        rgb = bgra[:, :, 2::-1]          # BGR -> RGB, drop alpha
        return np.ascontiguousarray(rgb)

    # Pioneer sonar lookup table is [0 -> 1024, 5 m -> 0], i.e. the RAW value is
    # inversely proportional to distance (larger = nearer). Convert to metres.
    SONAR_MAX_RANGE = 5.0
    SONAR_MAX_VALUE = 1024.0

    def _sonar_distance(self, raw):
        return self.SONAR_MAX_RANGE * (self.SONAR_MAX_VALUE - raw) / self.SONAR_MAX_VALUE

    def read_sonar_zones(self):
        """Return the NEAREST obstacle distance (metres) in each of the
        left/center/right forward zones (~5 m = nothing in range).

        Closest obstacle dominates, so per zone we take the min distance
        (equivalently the max raw value)."""
        zones = {}
        for zone, names in FRONT_SONARS.items():
            dists = [self._sonar_distance(self.sonars[n].getValue())
                     for n in names if n in self.sonars]
            zones[zone] = min(dists) if dists else self.SONAR_MAX_RANGE
        return zones

    def get_position(self):
        """(x, y) position from GPS, or None if no GPS device."""
        if self.gps is None:
            return None
        x, y, _z = self.gps.getValues()
        return (x, y)

    def in_contact(self):
        """True while the collision bumper is touching something."""
        return self.bumper is not None and self.bumper.getValue() > 0.5

    # ---------------------------------------------------------------------
    # Display
    # ---------------------------------------------------------------------
    def show_on_display(self, rgb):
        """Paint an (H, W, 3) uint8 RGB image onto the depth_display panel.

        Resizes implicitly only if dimensions match the display; otherwise the
        image is pasted at the top-left. Cheap enough to call every loop."""
        if self.display is None:
            return
        from controller import Display  # local import: only needed here
        h, w = rgb.shape[:2]
        ir = self.display.imageNew(rgb.tobytes(), Display.RGB, w, h)
        self.display.imagePaste(ir, 0, 0, False)
        self.display.imageDelete(ir)
