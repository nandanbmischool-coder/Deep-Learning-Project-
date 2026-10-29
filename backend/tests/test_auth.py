from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import auth_client


def test_register_login_logout_and_roles(client, new_client):
    admin = auth_client(new_client, "admin@example.com")  # first account -> admin
    assert admin.get("/api/v1/auth/me").json()["role"] == "admin"
    other = auth_client(new_client, "user2@example.com")
    assert other.get("/api/v1/auth/me").json()["role"] == "user"
    assert (
        other.post(
            "/api/v1/auth/register", json={"email": "user2@example.com", "full_name": "Dup", "password": "Password123"}
        ).status_code
        == 409
    )


def test_weak_password_and_validation_errors(new_client):
    c = new_client()
    r = c.post("/api/v1/auth/register", json={"email": "weak@example.com", "full_name": "Weak", "password": "short"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_ERROR"
    r = c.post("/api/v1/auth/register", json={"email": "weak@example.com", "full_name": "Weak", "password": "onlyletterslong"})
    assert r.status_code == 422


def test_protected_routes_require_auth(client):
    anon = TestClient(app)
    for path in ("/api/v1/auth/me", "/api/v1/predictions", "/api/v1/dashboard/summary", "/api/v1/training/jobs"):
        assert anon.get(path).status_code == 401


def test_csrf_enforced_on_mutations(new_client):
    c = auth_client(new_client, "csrf@example.com")
    bad = TestClient(app)
    bad.cookies.set("pv_access", c.cookies.get("pv_access"))
    assert bad.patch("/api/v1/auth/me", json={"full_name": "Hacker"}).status_code == 403
    assert c.patch("/api/v1/auth/me", json={"full_name": "Renamed User"}).status_code == 200


def test_login_wrong_password_and_rate_limit(new_client):
    auth_client(new_client, "login@example.com")
    c = new_client()
    for _ in range(5):
        assert c.post("/api/v1/auth/login", json={"email": "login@example.com", "password": "wrongpass1"}).status_code == 401
    assert c.post("/api/v1/auth/login", json={"email": "login@example.com", "password": "Password123"}).status_code == 429


def test_training_is_admin_only(new_client):
    u = auth_client(new_client, "plain@example.com")
    assert u.post("/api/v1/training/jobs", json={}).status_code == 403
