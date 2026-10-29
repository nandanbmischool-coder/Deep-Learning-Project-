"""Grad-CAM implemented with PyTorch forward hooks (Selvaraju et al., 2017)."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: str):
        modules = dict(model.named_modules())
        if target_layer not in modules:
            raise ValueError(
                f"Target layer '{target_layer}' not found. Available top-level: {sorted({n.split('.')[0] for n in modules if n})}"
            )
        self.model, self.target_layer = model, target_layer
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        self._handle = modules[target_layer].register_forward_hook(self._on_forward)

    def _on_forward(self, _module, _inp, output: torch.Tensor) -> None:
        self._acts = output
        if output.requires_grad:
            output.register_hook(lambda g: setattr(self, "_grads", g))

    def remove(self) -> None:
        self._handle.remove()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> dict:
        """Return {'cam': HxW float32 in [0,1] at feature resolution, 'probs': tensor, 'class_idx': int, 'flat': bool}."""
        self.model.eval()
        self._acts = self._grads = None
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)  # ensures the graph exists even if all params are frozen
            logits = self.model(x)
            probs = torch.softmax(logits.detach(), dim=1)[0]
            idx = int(probs.argmax()) if class_idx is None else class_idx
            self.model.zero_grad(set_to_none=True)
            logits[0, idx].backward()  # gradient of the target class score (pre-softmax)
        if self._acts is None or self._grads is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
        acts, grads = self._acts.detach()[0], self._grads.detach()[0]  # C,h,w
        weights = grads.mean(dim=(1, 2), keepdim=True)  # global-average-pooled gradients = channel importance
        cam = torch.relu((weights * acts).sum(dim=0))
        cam = torch.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy().astype(np.float32)
        peak = float(cam.max())
        flat = peak <= 1e-12  # no positive evidence for the target class -> an all-zero map, never a fabricated one
        cam = np.zeros_like(cam) if flat else cam / peak
        return {"cam": cam, "probs": probs.cpu(), "class_idx": idx, "flat": flat}


def render_overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> tuple[np.ndarray, np.ndarray]:
    """rgb: HxWx3 uint8 original image. Returns (heatmap_rgb, overlay_rgb), both at original resolution."""
    h, w = rgb.shape[:2]
    cam_full = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    heat_bgr = cv2.applyColorMap((cam_full * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    overlay = (rgb.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    return heat, overlay
