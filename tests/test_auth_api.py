# tests/test_auth_api.py
"""Auth API: register, login, JWT."""

import uuid

from fastapi.testclient import TestClient


def test_register_and_login(client: TestClient):
    u = f"hr_alice_{uuid.uuid4().hex[:10]}"
    r = client.post(
        "/api/v1/auth/register",
        json={"username": u, "password": "secret12"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["token_type"] == "bearer"
    assert "access_token" in data
    assert data["user"]["username"] == u

    r2 = client.post(
        "/api/v1/auth/login",
        json={"username": u, "password": "secret12"},
    )
    assert r2.status_code == 200
    assert r2.json()["user"]["username"] == u


def test_register_duplicate(client: TestClient):
    u = f"hr_bob_{uuid.uuid4().hex[:10]}"
    body = {"username": u, "password": "secret12"}
    assert client.post("/api/v1/auth/register", json=body).status_code == 200
    r = client.post("/api/v1/auth/register", json=body)
    assert r.status_code == 409


def test_login_wrong_password(client: TestClient):
    u = f"hr_carol_{uuid.uuid4().hex[:10]}"
    client.post(
        "/api/v1/auth/register",
        json={"username": u, "password": "rightpassword"},
    )
    r = client.post(
        "/api/v1/auth/login",
        json={"username": u, "password": "wrong"},
    )
    assert r.status_code == 401
