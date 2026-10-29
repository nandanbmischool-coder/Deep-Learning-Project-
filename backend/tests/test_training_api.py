"""Full API path: admin starts a real background training process, polls real progress, model gets registered."""

import time

from app.core.config import get_settings
from tests.conftest import auth_client
from tests.test_training_smoke import make_dataset


def test_training_job_lifecycle(client, new_client):
    admin = auth_client(new_client, "trainer-admin@example.com")
    from sqlalchemy import update

    from app.db.models import User
    from app.db.session import SessionLocal

    with SessionLocal() as db:  # the shared test DB may already have an admin, so promote this account explicitly
        db.execute(update(User).where(User.email == "trainer-admin@example.com").values(role="admin"))
        db.commit()
    assert admin.post("/api/v1/training/jobs", json={"epochs": 1, "head_epochs": 5}).status_code == 422
    assert admin.post("/api/v1/training/jobs", json={"weights_file": "../x.pth"}).status_code == 422

    body = {
        "architecture": "mobilenet_v3_large",
        "epochs": 2,
        "head_epochs": 1,
        "batch_size": 8,
        "pretrained": False,
        "max_images_per_class": 10,
        "val_fraction": 0.25,
    }
    r = admin.post("/api/v1/training/jobs", json=body)
    assert r.status_code == 400 and r.json()["error"]["code"] == "DATASET_MISSING"

    make_dataset(get_settings().dataset_dir, n=10)
    r = admin.post("/api/v1/training/jobs", json=body)
    assert r.status_code == 202, r.text
    jid = r.json()["job_id"]
    assert admin.post("/api/v1/training/jobs", json=body).status_code == 409  # one job at a time
    state = "running"
    for _ in range(240):
        j = admin.get(f"/api/v1/training/jobs/{jid}").json()
        state = j["state"]
        if state != "running":
            break
        time.sleep(2)
    assert state == "completed", j
    assert len(j["progress"]["history"]) >= 1 and j["progress"]["status"]["state"] == "completed"
    models = admin.get("/api/v1/models").json()["items"]
    assert any(m["version"] == j["version"] and m["evaluated"] for m in models)
    ev = admin.get("/api/v1/evaluations").json()["items"][0]
    full = admin.get(f"/api/v1/evaluations/{ev['id']}").json()["report"]
    assert full["n_samples"] == 20 and full["positive_class"] == "PNEUMONIA"
    mid = next(m["id"] for m in models if m["version"] == j["version"])
    assert admin.post(f"/api/v1/models/{mid}/activate").status_code == 200
    summ = admin.get("/api/v1/dashboard/summary").json()["model"]
    assert summ["evaluated"] is True and summ["test_accuracy"] is not None
    assert admin.get("/api/v1/training/jobs").json()["total"] >= 1
