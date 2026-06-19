"""
depth_model.py — MiDaS-Small monocular depth wrapper.

Loads MiDaS-Small once via torch.hub and runs inference on plain (H, W, 3) uint8
RGB frames coming from the Webots camera. Picks the best available device
(Apple MPS > CUDA > CPU) automatically.

IMPORTANT — what the output means:
  MiDaS predicts *relative inverse depth*: larger value = NEARER surface,
  smaller value = farther. It is NOT metric (no absolute metres). That's fine
  for this project: the avoidance policy only needs to compare zones to each
  other ("which direction is the closest obstacle?"), not measure true range.
"""

import os
import numpy as np

# Let unsupported MPS ops fall back to CPU instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402


def _pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MiDaSDepth:
    """Thin inference wrapper around MiDaS-Small."""

    def __init__(self, model_type="MiDaS_small", device=None, verbose=True):
        self.device = device or _pick_device()
        if verbose:
            print(f"[depth_model] loading {model_type} on {self.device} ...", flush=True)

        # MiDaS-Small builds its encoder from an EfficientNet-Lite backbone that
        # lives in a SEPARATE hub repo; MiDaS loads it internally without passing
        # trust_repo, so we pre-trust it here to avoid an interactive prompt
        # (there's no stdin inside a Webots controller). One-time, cached after.
        try:
            torch.hub.list("rwightman/gen-efficientnet-pytorch", trust_repo=True)
        except Exception:
            pass

        # torch.hub caches the repo + weights after the first download.
        self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.model.to(self.device).eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        # small_transform pairs with MiDaS_small; large transform for big models.
        self.transform = (
            transforms.small_transform
            if model_type == "MiDaS_small"
            else transforms.dpt_transform
        )
        if verbose:
            print("[depth_model] ready", flush=True)

    @torch.no_grad()
    def infer(self, rgb):
        """Run depth on an (H, W, 3) uint8 RGB frame.

        Returns a float32 (H, W) array of relative inverse depth (larger=nearer),
        resized back to the input frame's resolution.
        """
        h, w = rgb.shape[:2]
        batch = self.transform(rgb).to(self.device)
        pred = self.model(batch)
        # Upsample the network's low-res prediction back to camera resolution.
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=(h, w),
            mode="bicubic", align_corners=False,
        ).squeeze()
        return pred.detach().to("cpu").numpy().astype(np.float32)


def colorize_depth(depth, cmap_lut=None):
    """Map a relative-depth array to a uint8 (H, W, 3) RGB image for display.

    Near = bright/hot, far = dark/cool. Normalized per-frame (min-max) because
    MiDaS output has no fixed scale. Uses a precomputed 256-entry LUT so we don't
    depend on matplotlib inside the real-time loop.
    """
    d = depth.astype(np.float32)
    lo, hi = float(d.min()), float(d.max())
    if hi - lo < 1e-6:
        norm = np.zeros_like(d, dtype=np.uint8)
    else:
        norm = ((d - lo) / (hi - lo) * 255.0).astype(np.uint8)
    if cmap_lut is None:
        cmap_lut = _MAGMA_LUT
    return cmap_lut[norm]


def _build_magma_lut():
    """256x3 uint8 magma-ish LUT (dark purple -> orange -> pale yellow),
    built without a matplotlib dependency."""
    anchors = np.array([
        [0.0,  0,   0,   4],
        [0.25, 60,  16,  96],
        [0.5,  170, 48,  104],
        [0.75, 245, 110, 70],
        [1.0,  252, 253, 191],
    ])
    xs = np.linspace(0, 1, 256)
    lut = np.zeros((256, 3), dtype=np.uint8)
    for c in range(3):
        lut[:, c] = np.interp(xs, anchors[:, 0], anchors[:, c + 1]).astype(np.uint8)
    return lut


_MAGMA_LUT = _build_magma_lut()
