"""
episode.py — the single control loop shared by all three navigation policies.

This is the heart of the "fair comparison": depth, classical-CV and sonar runs
all execute THIS identical loop — same goal-seeking, same termination rules,
same logging. Only the `policy` (the perception front-end that produces a
left/center/right blockage estimate) differs between them.

A Policy just needs:
    policy.name           : str
    policy.perceive_every : int   (run perception every N steps)
    policy.perceive(bot)  -> (np.array([L, C, R]), vis_rgb_or_None)
"""

import numpy as np


class EpisodeConfig:
    def __init__(self, max_time=120.0, stuck_window_s=8.0, stuck_dist=0.12):
        self.max_time = max_time          # hard time budget per run (s)
        self.stuck_window_s = stuck_window_s
        self.stuck_dist = stuck_dist      # min progress over window or it's "stuck"


def run_episode(bot, policy, navigator, config, logger=None, record_path=None):
    """Drive one run to termination. Returns the result summary dict (or outcome).

    Termination: goal reached | timed out | stuck (jammed against something).
    If record_path is given, the Webots 3D view is recorded to that mp4 (needs
    a rendering run, i.e. not --no-rendering)."""
    prev_pos = bot.get_position()
    scores = np.zeros(3, dtype=np.float32)
    frame = 0
    outcome = "aborted"
    if record_path:
        bot.start_movie(record_path)

    # Rolling buffer for stuck detection: (time, pos) samples.
    window = max(1, int(config.stuck_window_s * 1000 / bot.timestep))
    pos_hist = []

    while bot.step():
        frame += 1
        pos = bot.get_position()
        t = bot.time()

        if logger is not None:
            logger.update(pos, bot.in_contact())

        # --- Termination checks ---
        if pos is not None and navigator.reached(pos):
            outcome = "reached"
            break
        if t > config.max_time:
            outcome = "timeout"
            break
        if pos is not None:
            pos_hist.append(pos)
            if len(pos_hist) > window:
                pos_hist.pop(0)
                moved = np.linalg.norm(np.subtract(pos_hist[-1], pos_hist[0]))
                if moved < config.stuck_dist:
                    outcome = "stuck"
                    break

        # --- Perception (at the policy's cadence) ---
        if frame % policy.perceive_every == 0:
            scores, vis = policy.perceive(bot)
            if vis is not None:
                bot.show_on_display(vis)

        # --- Shared control: blend goal-seeking + avoidance ---
        forward, turn, mode = navigator.decide(scores, pos, prev_pos)
        bot.drive(forward=forward, turn=turn)
        if pos is not None:
            prev_pos = pos

        if frame % 30 == 0:
            ps = f"({pos[0]:.2f},{pos[1]:.2f})" if pos else "n/a"
            c = logger.collisions if logger else 0
            print(f"[{policy.name}] t={t:5.1f}s pos={ps} "
                  f"L={scores[0]:.3f} C={scores[1]:.3f} R={scores[2]:.3f} "
                  f"{mode} col={c}", flush=True)

    bot.stop()
    rec = None
    if logger is not None:
        rec = logger.finalize(outcome, bot.time())
        print(f"[{policy.name}] DONE {rec}", flush=True)
    if record_path:
        bot.stop_movie()
    # Step once so the final wheel-stop command is applied, then quit Webots so
    # the batch harness can move on to the next run.
    bot.step()
    bot.quit(0)
    return rec if rec is not None else outcome
