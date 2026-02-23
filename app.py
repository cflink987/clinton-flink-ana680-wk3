from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import os

app = Flask(__name__)

# Default feature list (used only if the artifact doesn't provide it)
DEFAULT_FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("artifacts", "wine_lr.joblib"))

# --- Load model artifact ---
loaded = joblib.load(MODEL_PATH)

if isinstance(loaded, dict):
    # Your artifact format: {"model": LinearRegression(), "features": [...]}
    model = loaded["model"]
    FEATURES = loaded.get("features", DEFAULT_FEATURES)
else:
    model = loaded
    FEATURES = DEFAULT_FEATURES

# Build helper maps to accept multiple naming styles
# Canonical names are exactly what's in FEATURES
LOWER_FEATURE_MAP = {f.lower(): f for f in FEATURES}
CANONICAL_SET = set(FEATURES)

def normalize_payload_keys(payload: dict) -> dict:
    """
    Accept keys in multiple forms:
      - exact canonical name
      - spaces -> underscores (and '-' -> '_')
      - case-insensitive matching
    Returns dict with canonical feature names.
    """
    normalized = {}

    for k, v in payload.items():
        if not isinstance(k, str):
            continue

        # If it's already canonical, keep it
        if k in CANONICAL_SET:
            normalized[k] = v
            continue

        # Convert common variants: spaces/dashes -> underscores
        k2 = k.strip().replace(" ", "_").replace("-", "_")

        # If canonical features use underscores
        if k2 in CANONICAL_SET:
            normalized[k2] = v
            continue

        # Case-insensitive matching (supports pH/ph, etc.)
        k3 = k2.lower()
        if k3 in LOWER_FEATURE_MAP:
            normalized[LOWER_FEATURE_MAP[k3]] = v
            continue

    return normalized


HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Wine Quality Predictor</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; max-width: 900px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 16px; }
    label { font-size: 14px; }
    input { width: 100%; padding: 8px; font-size: 14px; }
    button { padding: 10px 14px; font-size: 15px; cursor: pointer; }
    .card { padding: 16px; border: 1px solid #ddd; border-radius: 10px; margin-top: 18px; }
    .result { font-size: 18px; }
    .muted { color: #666; font-size: 13px; }
    code { background: #f6f6f6; padding: 2px 6px; border-radius: 6px; }
  </style>
</head>
<body>
  <h2>Wine Quality Predictor</h2>
  <p class="muted">
    Enter the 11 feature values and click Predict. (API endpoint: <code>POST /predict</code>)
  </p>

  <div class="card">
    <form id="predictForm">
      <div class="grid">
        {% for f in features %}
          <div>
            <label for="{{ ids[f] }}">{{ f }}</label>
            <input type="number" step="any" id="{{ ids[f] }}" name="{{ ids[f] }}" required placeholder="e.g. 7.4"/>
          </div>
        {% endfor %}
      </div>
      <div style="margin-top: 14px;">
        <button type="submit">Predict</button>
      </div>
    </form>
  </div>

  <div class="card">
    <div class="result" id="result">Result will appear here.</div>
    <pre id="raw" class="muted"></pre>
  </div>

<script>
  // Use safe IDs (no spaces) but send canonical feature names back to the server
  const FEATURES = {{ features | tojson }};
  const ID_MAP = {{ ids | tojson }};

  const form = document.getElementById("predictForm");
  const resultEl = document.getElementById("result");
  const rawEl = document.getElementById("raw");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    resultEl.textContent = "Predicting...";
    rawEl.textContent = "";

    const payload = {};
    for (const f of FEATURES) {
      const id = ID_MAP[f];
      const val = document.getElementById(id).value;
      payload[f] = Number(val);
    }

    try {
      const r = await fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });

      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || JSON.stringify(data));

      resultEl.textContent = `Predicted quality: ${data.rounded_quality} (raw: ${data.predicted_quality})`;
      rawEl.textContent = JSON.stringify(data, null, 2);
    } catch (err) {
      resultEl.textContent = "Error: " + err.message;
    }
  });
</script>
</body>
</html>
"""


@app.get("/")
def index():
    # HTML element IDs should not contain spaces.
    # Create safe IDs but keep canonical feature names for payload.
    ids = {f: f.replace(" ", "_").replace("-", "_") for f in FEATURES}
    return render_template_string(HTML, features=FEATURES, ids=ids)


@app.get("/health")
def health():
    ok = hasattr(model, "predict")
    return jsonify({
        "status": "ok" if ok else "bad",
        "model_path": MODEL_PATH,
        "model_type": str(type(model)),
        "feature_order": FEATURES,
    })


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"detail": "Missing or invalid JSON body"}), 400

        # Option A: {"features": [..]} in correct order (FEATURES)
        if isinstance(payload, dict) and "features" in payload:
            feats = payload["features"]
            if not isinstance(feats, list) or len(feats) != len(FEATURES):
                return jsonify({"detail": f"'features' must be a list of length {len(FEATURES)}"}), 400
            x = np.array([feats], dtype=float)

        # Option B: {"fixed acidity": ..., ...} (accepts spaces or underscores)
        elif isinstance(payload, dict):
            normalized = normalize_payload_keys(payload)
            missing = [f for f in FEATURES if f not in normalized]
            if missing:
                return jsonify({
                    "detail": f"Missing features: {missing}",
                    "expected_features": FEATURES
                }), 400
            x = np.array([[float(normalized[f]) for f in FEATURES]], dtype=float)

        else:
            return jsonify({"detail": "JSON payload must be an object"}), 400

        if not hasattr(model, "predict"):
            return jsonify({"detail": f"Loaded model has no predict(): {type(model)}"}), 500

        pred = float(model.predict(x)[0])
        return jsonify({
            "predicted_quality": round(pred, 4),
            "rounded_quality": int(round(pred)),
            "feature_order": FEATURES
        })

    except Exception as e:
        # Return JSON error (prevents HTML 500 pages and keeps the UI readable)
        return jsonify({"detail": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)