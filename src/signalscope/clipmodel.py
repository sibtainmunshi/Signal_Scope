"""Frozen CLIP image tower plus our trained linear head, as one logit-producing module.

The tower is a TorchScript export of the official CLIP ViT-L/14 image encoder, which we
did not train; the head is ours. Keeping both behind a single `forward` that returns one
logit per row lets the existing Detector scoring, batching and calibration code stay
unchanged.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .paths import root_path

DEFAULT_VISUAL = "model/checkpoints/clip_vitl14_visual/visual_fp16.ts"


def resolve_visual(payload: dict) -> Path:
    """Image-tower location: checkpoint field, then environment override, then default."""
    candidate = payload.get("visual_path") or os.getenv("SIGNALSCOPE_CLIP_VISUAL") or DEFAULT_VISUAL
    path = root_path(candidate)
    if not path.is_file():
        raise FileNotFoundError(
            f"CLIP image tower missing: {path}. Download the published tower or set "
            "SIGNALSCOPE_CLIP_VISUAL; see the README model setup section.")
    return path


def load_visual(payload: dict, device: torch.device) -> tuple[torch.jit.ScriptModule, str]:
    """Load the traced tower, verifying the declared digest when the checkpoint states one."""
    path = resolve_visual(payload)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    expected = payload.get("visual_sha256")
    if expected and expected != digest:
        raise ValueError(f"CLIP image tower digest mismatch for {path}; expected {expected}.")
    tower = torch.jit.load(path, map_location="cpu")
    # Stored in half precision to keep the download small; inference runs in float32.
    return tower.float().eval().to(device), digest


class ClipLinearModel(nn.Module):
    """Traced CLIP image tower, L2-normalized embedding, then our logistic head."""

    def __init__(self, tower: nn.Module, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.tower = tower
        self.register_buffer("weight", weight.to(torch.float32))
        self.register_buffer("bias", bias.to(torch.float32))

    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.tower(batch).float(), dim=-1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.embed(batch) @ self.weight.T + self.bias


class ClipMlpModel(nn.Module):
    """Traced CLIP image tower, L2-normalized embedding, then a small trained MLP head.

    Same shape of contract as ClipLinearModel (one logit per row), so the rest of the
    Detector, explanation and calibration code is unchanged. The head here has real
    capacity (a hidden ReLU layer) rather than a single hyperplane, which is what let a
    training mixture spanning both older and newer generators avoid the catastrophic
    global decision-boundary shift a linear head showed under the same reweighting.
    """

    def __init__(self, tower: nn.Module, state_dict: dict, hidden_units: int, dropout: float):
        super().__init__()
        self.tower = tower
        self.net = nn.Sequential(
            nn.Linear(768, hidden_units), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_units, 1))
        # Saved from a module whose only child was also named "net"; strip that prefix.
        self.net.load_state_dict({k.removeprefix("net."): v for k, v in state_dict.items()})
        self.net.eval().requires_grad_(False)

    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.tower(batch).float(), dim=-1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.net(self.embed(batch))
