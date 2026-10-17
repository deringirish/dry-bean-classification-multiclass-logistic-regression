from pathlib import Path
import pytest
import joblib
import pandas as pd
import numpy as np

# Path to the serialized model pipeline
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "dry_bean_pipeline.joblib"

EXPECTED_CLASSES = [
    "BARBUNYA",
    "BOMBAY",
    "CALI",
    "DERMASON",
    "HOROZ",
    "SEKER",
    "SIRA",
]

EXPECTED_FEATURES = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]


@pytest.fixture(scope="module")
def model():
    """Fixture to load the saved scikit-learn pipeline."""
    assert MODEL_PATH.exists(), f"Model file not found at {MODEL_PATH}"
    pipeline = joblib.load(MODEL_PATH)
    assert pipeline is not None
    return pipeline


def test_model_artifact_exists():
    """Verify that the model file exists and is non-empty."""
    assert MODEL_PATH.is_file()
    assert MODEL_PATH.stat().st_size > 0


def test_model_pipeline_structure(model):
    """Verify that the pipeline contains StandardScaler and LogisticRegression steps."""
    step_names = [name for name, _ in model.steps]
    assert "standardscaler" in step_names
    assert "logisticregression" in step_names


def test_model_classes(model):
    """Verify that the model identifies all 7 target bean cultivars."""
    classes = list(model.classes_)
    assert sorted(classes) == sorted(EXPECTED_CLASSES)


def test_model_prediction_single_sample(model):
    """Verify inference on a representative sample vector."""
    sample = {
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
    df = pd.DataFrame([sample], columns=EXPECTED_FEATURES)

    # Prediction should be BOMBAY (large bean)
    pred = model.predict(df)[0]
    assert pred in EXPECTED_CLASSES
    assert pred == "BOMBAY"

    # Probabilities must sum to 1.0
    probabilities = model.predict_proba(df)[0]
    assert len(probabilities) == 7
    assert np.isclose(np.sum(probabilities), 1.0, atol=1e-4)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
