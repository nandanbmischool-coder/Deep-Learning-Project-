"""Model factory for transfer learning (DenseNet121 primary; ResNet50 / MobileNetV3-Large optional)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torchvision import models

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # index 0 / 1; PNEUMONIA is the positive class
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}


class PretrainedWeightsError(RuntimeError):
    """Raised when pretrained weights are requested but cannot be loaded. We never fall back silently."""


@dataclass(frozen=True)
class ArchSpec:
    name: str
    cam_layer: str  # last convolutional feature layer used by Grad-CAM (verified in tests / at load time)
    trainable_prefixes: tuple[str, ...]  # parameters unfrozen in "last_block" fine-tuning mode (plus the new head)
    head_prefixes: tuple[str, ...]


ARCHS: dict[str, ArchSpec] = {
    # torchvision DenseNet: features.denseblock4 -> features.norm5 -> (functional in-place ReLU) -> pool -> classifier.
    # We hook denseblock4 (the last convolutional block) because the ReLU after norm5 is in-place.
    "densenet121": ArchSpec("densenet121", "features.denseblock4", ("features.denseblock4", "features.norm5"), ("classifier",)),
    "resnet50": ArchSpec("resnet50", "layer4", ("layer4",), ("fc",)),
    "mobilenet_v3_large": ArchSpec(
        "mobilenet_v3_large", "features.16", ("features.13", "features.14", "features.15", "features.16"), ("classifier",)
    ),
}


def _construct(arch: str, weights):
    return {"densenet121": models.densenet121, "resnet50": models.resnet50, "mobilenet_v3_large": models.mobilenet_v3_large}[
        arch
    ](weights=weights)


def _weights_enum(arch: str):
    return {
        "densenet121": models.DenseNet121_Weights.IMAGENET1K_V1,
        "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
        "mobilenet_v3_large": models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
    }[arch]


def _replace_head(model: nn.Module, arch: str, num_classes: int = 2) -> nn.Module:
    if arch == "densenet121":
        model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.classifier.in_features, num_classes))
    elif arch == "resnet50":
        model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.fc.in_features, num_classes))
    else:
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def build_model(arch: str, pretrained: bool, weights_path: Path | None = None) -> tuple[nn.Module, str]:
    """Return (model, pretrained_source). Raises PretrainedWeightsError instead of silently using random weights."""
    if arch not in ARCHS:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from {sorted(ARCHS)}.")
    if weights_path is not None:
        model = _construct(arch, None)
        try:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
        except Exception as exc:  # noqa: BLE001
            raise PretrainedWeightsError(
                f"Could not load ImageNet weights from '{weights_path.name}': {exc}. The file must be the original "
                f"torchvision {arch} ImageNet state_dict (1000-class head)."
            ) from exc
        source = f"ImageNet weights from local file {weights_path.name}"
    elif pretrained:
        try:
            model = _construct(arch, _weights_enum(arch))
        except Exception as exc:  # noqa: BLE001
            raise PretrainedWeightsError(
                f"Could not download pretrained {arch} weights ({exc}). Either fix your internet connection, or download "
                f"the torchvision ImageNet state_dict manually, put it in data/artifacts/weights/ and set the "
                f"'weights file' option. Training was NOT started with random weights."
            ) from exc
        source = f"torchvision ImageNet ({_weights_enum(arch).name})"
    else:
        model = _construct(arch, None)
        source = "none (random initialization, explicitly requested)"
    return _replace_head(model, arch), source


def apply_freeze(model: nn.Module, arch: str, mode: str) -> None:
    """mode: 'head_only' | 'last_block' | 'full'."""
    spec = ARCHS[arch]
    for name, p in model.named_parameters():
        if mode == "full":
            p.requires_grad = True
        elif any(name.startswith(pre) for pre in spec.head_prefixes):
            p.requires_grad = True
        elif mode == "last_block" and any(name.startswith(pre) for pre in spec.trainable_prefixes):
            p.requires_grad = True
        else:
            p.requires_grad = False


def set_train_mode(model: nn.Module) -> None:
    """train() but keep BatchNorm layers of frozen blocks in eval mode so ImageNet statistics are preserved."""
    model.train()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(model: nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    arch = ckpt["arch"]
    model, _ = build_model(arch, pretrained=False)  # weights come from the checkpoint, not ImageNet
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def pick_device(pref: str = "auto") -> torch.device:
    if pref == "cpu" or (pref == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    return torch.device("cuda")
