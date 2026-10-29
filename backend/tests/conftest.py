import io
import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="pv_test_"))
os.environ.update(
    {
        "DATABASE_URL": f"sqlite:///{(_TMP / 'test.db').as_posix()}",
        "MEDIA_DIR": str(_TMP / "media"),
        "ARTIFACTS_DIR": str(_TMP / "artifacts"),
        "DATASET_DIR": str(_TMP / "dataset"),
        "SECRET_KEY": "test-secret-key-for-pytest-only-0123456789",
        "ENVIRONMENT": "development",
    }
)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def tmp_root() -> Path:
    return _TMP


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


def make_xray_png(size=(320, 384), seed=0, color=False) -> bytes:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0 : size[1], 0 : size[0]]
    base = 60 + 160 * np.exp(-(((xx - size[0] / 2) / (size[0] / 3)) ** 2 + ((yy - size[1] / 2) / (size[1] / 3)) ** 2))
    arr = np.clip(base + rng.normal(0, 18, base.shape), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr).convert("RGB")
    if color:
        img = Image.fromarray(np.stack([arr, 255 - arr, arr // 2], axis=2))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def auth_client(client_factory, email, password="Password123", name="Test User"):
    c = client_factory()
    r = c.post("/api/v1/auth/register", json={"email": email, "full_name": name, "password": password})
    assert r.status_code == 201, r.text
    c.headers["X-CSRF-Token"] = c.cookies.get("pv_csrf")
    return c


@pytest.fixture()
def new_client():
    def factory():
        return TestClient(app)

    return factory
