import io

import numpy as np
import pytest
import torch
from PIL import Image

from app.core.config import get_settings
from app.core.errors import AppError
from app.db.session import SessionLocal
from app.ml.gradcam import GradCAM, render_overlay
from app.ml.model import ARCHS, build_model
from app.services.image_io import decode_image, resolve_media
from app.services.registry import register_run
from tests.conftest import auth_client, make_xray_png


def test_decode_rejects_bad_inputs():
    with pytest.raises(AppError) as e:
        decode_image(b"not an image at all")
    assert e.value.code == "INVALID_IMAGE"
    buf = io.BytesIO()
    Image.new("RGB", (100, 100)).save(buf, "BMP")
    with pytest.raises(AppError) as e:
        decode_image(buf.getvalue())
    assert e.value.code == "UNSUPPORTED_FORMAT"
    buf = io.BytesIO()
    Image.new("RGB", (16, 16)).save(buf, "PNG")
    with pytest.raises(AppError) as e:
        decode_image(buf.getvalue())
    assert e.value.code == "IMAGE_TOO_SMALL"


def test_heuristic_flags_colour_image_but_not_xray_like():
    assert decode_image(make_xray_png()).validation["status"] == "ok"
    d = decode_image(make_xray_png(color=True)).validation
    assert d["status"] == "warning" and "heuristic" in d["note"]


def test_path_traversal_blocked():
    with pytest.raises(AppError):
        resolve_media("../../etc/passwd")


@pytest.mark.parametrize("arch", sorted(ARCHS))
def test_gradcam_target_layer_exists_and_is_real(arch):
    model, source = build_model(arch, pretrained=False)
    assert "random initialization" in source  # never claims pretrained weights when it used none
    model.eval()
    cam = GradCAM(model, ARCHS[arch].cam_layer)
    x = torch.randn(1, 3, 224, 224)
    out = cam(x, 1)
    assert out["cam"].ndim == 2 and out["cam"].shape[0] == out["cam"].shape[1] >= 7
    assert out["cam"].min() >= 0.0 and out["cam"].max() <= 1.0
    assert not np.isnan(out["cam"]).any()
    cam.remove()


def test_gradcam_unknown_layer_raises():
    model, _ = build_model("resnet50", pretrained=False)
    with pytest.raises(ValueError):
        GradCAM(model, "does.not.exist")


def test_overlay_shapes():
    rgb = np.zeros((300, 200, 3), np.uint8)
    heat, ov = render_overlay(rgb, np.random.rand(7, 7).astype(np.float32))
    assert heat.shape == ov.shape == (300, 200, 3)


def test_pretrained_failure_is_explicit(monkeypatch):
    from app.ml import model as m

    def boom(*a, **k):
        raise OSError("no network")

    monkeypatch.setattr(m, "_construct", boom)
    with pytest.raises(m.PretrainedWeightsError):
        m.build_model("densenet121", pretrained=True)


def _install_random_model(version="densenet121-test"):
    """Create a (randomly initialised, untrained) checkpoint purely to exercise the API plumbing."""
    import json

    s = get_settings()
    run = s.artifacts_dir / "runs" / version
    run.mkdir(parents=True, exist_ok=True)
    model, _ = build_model("densenet121", pretrained=False)
    torch.save(
        {
            "arch": "densenet121",
            "state_dict": model.state_dict(),
            "class_to_idx": {"NORMAL": 0, "PNEUMONIA": 1},
            "image_size": 224,
        },
        run / "model.pt",
    )
    (run / "config.json").write_text(
        json.dumps(
            {
                "version": version,
                "architecture": "densenet121",
                "pretrained": False,
                "pretrained_source": "none (test)",
                "threshold": 0.5,
                "threshold_method": "test",
                "parameter_count": 1,
                "model_size_mb": 1.0,
                "class_to_idx": {},
            }
        )
    )
    with SessionLocal() as db:
        return register_run(db, run)


def test_prediction_flow_ownership_and_assets(client, new_client):
    owner = auth_client(new_client, "owner@example.com")
    r = owner.post("/api/v1/predictions", files={"file": ("x.png", make_xray_png(), "image/png")})
    assert r.status_code == 503 and r.json()["error"]["code"] == "MODEL_NOT_TRAINED"  # no fake predictions

    _install_random_model()
    r = owner.post("/api/v1/predictions", files={"file": ("../../evil.png", make_xray_png(), "image/png")})
    assert r.status_code == 201, r.text
    body = r.json()
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-4 and body["predicted_class"] in ("NORMAL", "PNEUMONIA")
    aid = body["analysis_id"]
    for kind in ("original", "heatmap", "overlay", "thumbnail"):
        assert owner.get(f"/api/v1/predictions/{aid}/assets/{kind}").status_code == 200
    assert owner.get(f"/api/v1/predictions/{aid}/report").status_code == 200

    intruder = auth_client(new_client, "intruder@example.com")
    assert intruder.get(f"/api/v1/predictions/{aid}").status_code == 404
    assert intruder.get(f"/api/v1/predictions/{aid}/assets/original").status_code == 404
    assert intruder.delete(f"/api/v1/predictions/{aid}").status_code == 404
    assert intruder.get("/api/v1/predictions").json()["total"] == 0

    assert owner.post("/api/v1/predictions", files={"file": ("a.png", b"garbage", "image/png")}).status_code == 422
    assert owner.post("/api/v1/predictions", files={"file": ("a.png", b"", "image/png")}).status_code == 400
    summary = owner.get("/api/v1/dashboard/summary").json()
    assert summary["operational"]["total_analyses"] == 1
    assert summary["model"]["evaluated"] is False and summary["model"]["test_accuracy"] is None  # nothing fabricated
    assert owner.delete(f"/api/v1/predictions/{aid}").status_code == 204
    assert owner.get(f"/api/v1/predictions/{aid}").status_code == 404
