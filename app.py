import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Feature list matching the exact dataset columns & model expectation
FEATURE_NAMES = [
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

# Bean variety physical profiles and descriptions
CULTIVAR_INFO = {
    "SEKER": {
        "name": "Seker",
        "description": "Spherical, highly circular bean with high roundness (>0.92) and low aspect ratio.",
        "badge_color": "#2563eb",
    },
    "BARBUNYA": {
        "name": "Barbunya",
        "description": "Medium-to-large oval pinto bean with mottled reddish speckles.",
        "badge_color": "#d97706",
    },
    "BOMBAY": {
        "name": "Bombay",
        "description": "Exceptionally large white bean with massive area (>100,000 px) and perimeter.",
        "badge_color": "#059669",
    },
    "CALI": {
        "name": "Cali",
        "description": "Large kidney-shaped bean with medium-high elongation and distinctive shape factors.",
        "badge_color": "#7c3aed",
    },
    "DERMASON": {
        "name": "Dermason",
        "description": "Small, flat white bean; the most popular commercial dry bean in Turkey.",
        "badge_color": "#0284c7",
    },
    "HOROZ": {
        "name": "Horoz",
        "description": "Long, slender cylindrical white bean with very high aspect ratio (>1.85).",
        "badge_color": "#dc2626",
    },
    "SIRA": {
        "name": "Sira",
        "description": "Small-to-medium compact white bean with morphological similarity to Dermason.",
        "badge_color": "#4b5563",
    },
}

# Authentic sample presets from dataset for instant 1-click testing
SAMPLE_PRESETS = {
    "BOMBAY": {
        "label": "Bombay (Giant Bean)",
        "features": {
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
        },
    },
    "HOROZ": {
        "label": "Horoz (Elongated Cylindrical)",
        "features": {
            "Area": 33006,
            "Perimeter": 710.496,
            "MajorAxisLength": 283.0204,
            "MinorAxisLength": 149.6237,
            "AspectRation": 1.8915,
            "Eccentricity": 0.8488,
            "ConvexArea": 33354,
            "EquivDiameter": 204.9989,
            "Extent": 0.6355,
            "Solidity": 0.9896,
            "roundness": 0.8216,
            "Compactness": 0.7243,
            "ShapeFactor1": 0.0086,
            "ShapeFactor2": 0.0015,
            "ShapeFactor3": 0.5246,
            "ShapeFactor4": 0.9924,
        },
    },
    "SEKER": {
        "label": "Seker (Spherical Round)",
        "features": {
            "Area": 28395,
            "Perimeter": 610.291,
            "MajorAxisLength": 208.1781,
            "MinorAxisLength": 173.8887,
            "AspectRation": 1.1972,
            "Eccentricity": 0.5498,
            "ConvexArea": 28715,
            "EquivDiameter": 190.1411,
            "Extent": 0.7639,
            "Solidity": 0.9889,
            "roundness": 0.9580,
            "Compactness": 0.9134,
            "ShapeFactor1": 0.0073,
            "ShapeFactor2": 0.0031,
            "ShapeFactor3": 0.8342,
            "ShapeFactor4": 0.9987,
        },
    },
    "DERMASON": {
        "label": "Dermason (Small Flat White)",
        "features": {
            "Area": 20420,
            "Perimeter": 524.932,
            "MajorAxisLength": 183.6012,
            "MinorAxisLength": 141.8862,
            "AspectRation": 1.2940,
            "Eccentricity": 0.6347,
            "ConvexArea": 20684,
            "EquivDiameter": 161.2438,
            "Extent": 0.7902,
            "Solidity": 0.9872,
            "roundness": 0.9312,
            "Compactness": 0.8782,
            "ShapeFactor1": 0.0090,
            "ShapeFactor2": 0.0033,
            "ShapeFactor3": 0.7713,
            "ShapeFactor4": 0.9980,
        },
    },
}

# Load the trained model pipeline
MODEL_PATH = Path(__file__).resolve().parent / "models" / "dry_bean_pipeline.joblib"
model = None

def load_pipeline():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
    return model

# Ensure model loads at boot
load_pipeline()


@app.route("/")
def home():
    return render_template(
        "index.html",
        feature_names=FEATURE_NAMES,
        presets=SAMPLE_PRESETS,
        cultivar_info=CULTIVAR_INFO,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True, silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({"status": "error", "message": "Invalid payload. Expected JSON object."}), 400

        # Validate that all 16 features are present
        missing = [f for f in FEATURE_NAMES if f not in data]
        if missing:
            return jsonify({
                "status": "error",
                "message": f"Missing required features: {', '.join(missing)}"
            }), 400

        # Type conversion and basic domain validation
        parsed_features = {}
        for feat in FEATURE_NAMES:
            try:
                val = float(data[feat])
                if np.isnan(val) or np.isinf(val):
                    raise ValueError("Must be a finite number")
                if val <= 0 and feat not in ["ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4"]:
                    return jsonify({
                        "status": "error",
                        "message": f"Feature '{feat}' must be greater than 0."
                    }), 400
                parsed_features[feat] = val
            except (ValueError, TypeError):
                return jsonify({
                    "status": "error",
                    "message": f"Invalid numeric value for feature '{feat}'."
                }), 400

        # Construct DataFrame with exact column order
        input_df = pd.DataFrame([parsed_features], columns=FEATURE_NAMES)

        # Run inference
        clf = load_pipeline()
        prediction = str(clf.predict(input_df)[0])
        probabilities = clf.predict_proba(input_df)[0]
        class_names = list(clf.classes_)

        # Prepare formatted results
        prob_dict = {
            cls: round(float(prob) * 100, 2)
            for cls, prob in zip(class_names, probabilities)
        }
        # Sort probabilities descending
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

        confidence = round(float(np.max(probabilities)) * 100, 2)
        info = CULTIVAR_INFO.get(prediction, {
            "name": prediction,
            "description": "Dry bean cultivar.",
            "badge_color": "#2563eb",
        })

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "display_name": info["name"],
            "description": info["description"],
            "badge_color": info["badge_color"],
            "confidence": confidence,
            "probabilities": dict(sorted_probs),
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": f"Prediction error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
