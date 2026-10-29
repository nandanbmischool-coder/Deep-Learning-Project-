"""Secure upload handling: content sniffing, size/dimension limits, re-encoding, heuristic X-ray plausibility checks."""

from __future__ import annotations

import io
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import get_settings
from app.core.errors import AppError

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
HEURISTIC_NOTE = (
    "These are simple image-statistics heuristics, not a validated chest X-ray detector. "
    "A passing check does not prove the image is a chest radiograph."
)


@dataclass
class DecodedImage:
    image: Image.Image  # RGB, EXIF-orientation applied
    source_format: str
    width: int
    height: int
    validation: dict


def read_limited(stream, max_bytes: int) -> bytes:
    data = stream.read(max_bytes + 1)
    if len(data) == 0:
        raise AppError(400, "EMPTY_FILE", "The uploaded file is empty.")
    if len(data) > max_bytes:
        raise AppError(413, "FILE_TOO_LARGE", f"File exceeds the {max_bytes // (1024 * 1024)} MB limit.")
    return data


def safe_display_name(name: str | None) -> str:
    """Display-only hint. Never used for filesystem paths."""
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", (name or "").replace("\\", "/").split("/")[-1])
    return base[:120]


def decode_image(data: bytes) -> DecodedImage:
    s = get_settings()
    Image.MAX_IMAGE_PIXELS = s.max_image_pixels  # decompression-bomb guard (Pillow raises above 2x this)
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format or ""
            if fmt not in ALLOWED_FORMATS:
                raise AppError(415, "UNSUPPORTED_FORMAT", "Only JPEG, PNG and WebP images are accepted.")
            w, h = probe.size
            if max(w, h) > s.max_image_side or w * h > s.max_image_pixels:
                raise AppError(
                    413, "IMAGE_TOO_LARGE", f"Image dimensions exceed {s.max_image_side}px / {s.max_image_pixels:,} pixels."
                )
            if min(w, h) < s.min_image_side:
                raise AppError(422, "IMAGE_TOO_SMALL", f"Image is smaller than {s.min_image_side}px on one side.")
            probe.verify()
        with Image.open(io.BytesIO(data)) as im:  # verify() invalidates the handle, so reopen to decode
            im = ImageOps.exif_transpose(im)
            rgb = im.convert("RGB")
            rgb.load()
    except AppError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError, SyntaxError) as exc:
        raise AppError(422, "INVALID_IMAGE", "The file is not a valid, readable image.") from exc
    return DecodedImage(rgb, fmt, rgb.width, rgb.height, plausibility_checks(rgb))


def plausibility_checks(img: Image.Image) -> dict:
    """Heuristic checks (labelled as such). Chest radiographs are near-grayscale, have wide tonal range and a
    roughly portrait/square aspect ratio."""
    a = np.asarray(img.resize((256, 256)), dtype=np.float32)
    colorfulness = float(np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean())
    gray = a.mean(axis=2)
    contrast = float(gray.std())
    p1, p99 = np.percentile(gray, [1, 99])
    aspect = img.width / img.height
    warnings: list[str] = []
    if colorfulness > 12:
        warnings.append("Image is clearly colour; chest X-rays are normally grayscale.")
    if contrast < 25:
        warnings.append("Very low contrast; the image may be blank, over/under-exposed or not a radiograph.")
    if p99 - p1 < 90:
        warnings.append("Narrow tonal range for a radiograph.")
    if not (0.6 <= aspect <= 1.8):
        warnings.append("Unusual aspect ratio for a chest X-ray.")
    return {
        "status": "warning" if warnings else "ok",
        "warnings": warnings,
        "note": HEURISTIC_NOTE,
        "checks": {
            "colorfulness": round(colorfulness, 2),
            "contrast_std": round(contrast, 2),
            "tonal_range_p1_p99": round(float(p99 - p1), 1),
            "aspect_ratio": round(aspect, 3),
        },
    }


def new_public_id() -> str:
    return uuid.uuid4().hex[:16]


def save_analysis_images(public_id: str, user_id: int, original: Image.Image, heat: np.ndarray, overlay: np.ndarray) -> dict:
    """Write server-named files under media/analyses/<user>/<id>/. Returns paths relative to the media dir."""
    s = get_settings()
    rel = Path("analyses") / str(user_id) / public_id
    d = s.media_dir / rel
    d.mkdir(parents=True, exist_ok=True)
    original.save(d / "original.png", format="PNG")
    Image.fromarray(heat).save(d / "heatmap.png", format="PNG")
    Image.fromarray(overlay).save(d / "overlay.png", format="PNG")
    thumb = original.copy()
    thumb.thumbnail((256, 256))
    thumb.save(d / "thumbnail.jpg", format="JPEG", quality=85)
    return {
        k: (rel / f).as_posix()
        for k, f in (
            ("original", "original.png"),
            ("heatmap", "heatmap.png"),
            ("overlay", "overlay.png"),
            ("thumbnail", "thumbnail.jpg"),
        )
    }


def resolve_media(rel_path: str) -> Path:
    """Resolve a stored relative path, refusing anything that escapes the media directory."""
    root = get_settings().media_dir.resolve()
    p = (root / rel_path).resolve()
    if root not in p.parents:
        raise AppError(404, "NOT_FOUND", "File not found.")
    return p
