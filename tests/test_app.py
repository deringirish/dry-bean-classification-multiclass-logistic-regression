import json
import pytest
from app import app


@pytest.fixture
def client():
    """Fixture providing a Flask test client with testing mode enabled."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home_page_loads(client):
    """Verify that GET / renders the main web interface."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Dry Bean Variety Classifier" in response.data
    assert b"Bean Measurements" in response.data


def test_predict_endpoint_success(client):
    """Verify that POST /predict returns valid predictions and probabilities."""
    sample_payload = {
        "Area": 114004,
        "Perimeter": 1279.356,
        "MajorAxisLength": 451.3613,
        "MinorAxisLength": 323.748,
        "AspectRation": 1.3942,
        "Eccentricity": 0.6968,
        "ConvexArea": 115298,
        "EquivDiameter": 380.9913,
        "Extent": 0.7490,
        "Solidity": 0.9888,
        "roundness": 0.8753,
        "Compactness": 0.8441,
        "ShapeFactor1": 0.0040,
        "ShapeFactor2": 0.0012,
        "ShapeFactor3": 0.7125,
        "ShapeFactor4": 0.9933,
    }

    response = client.post(
        "/predict",
        data=json.dumps(sample_payload),
        content_type="application/json",
    )

    assert response.status_code == 200
    data = response.get_json()

    assert data["status"] == "success"
    assert data["prediction"] == "BOMBAY"
    assert isinstance(data["confidence"], (int, float))
    assert 0 <= data["confidence"] <= 100
    assert "probabilities" in data
    assert len(data["probabilities"]) == 7


def test_predict_endpoint_missing_features(client):
    """Verify that POST /predict returns HTTP 400 when required features are missing."""
    incomplete_payload = {
        "Area": 33006,
        "Perimeter": 710.496,
    }

    response = client.post(
        "/predict",
        data=json.dumps(incomplete_payload),
        content_type="application/json",
    )

    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert "Missing required features" in data["message"]


def test_predict_endpoint_invalid_values(client):
    """Verify that POST /predict returns HTTP 400 when invalid/non-numeric values are submitted."""
    invalid_payload = {
        "Area": "not-a-number",
        "Perimeter": 1279.356,
        "MajorAxisLength": 451.3613,
        "MinorAxisLength": 323.748,
        "AspectRation": 1.3942,
        "Eccentricity": 0.6968,
        "ConvexArea": 115298,
        "EquivDiameter": 380.9913,
        "Extent": 0.7490,
        "Solidity": 0.9888,
        "roundness": 0.8753,
        "Compactness": 0.8441,
        "ShapeFactor1": 0.0040,
        "ShapeFactor2": 0.0012,
        "ShapeFactor3": 0.7125,
        "ShapeFactor4": 0.9933,
    }

    response = client.post(
        "/predict",
        data=json.dumps(invalid_payload),
        content_type="application/json",
    )

    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
