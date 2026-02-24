from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import os
import json

app = Flask(__name__)

# Fallback if artifact doesn't supply feature order (yours does)
DEFAULT_FEATURES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("artifacts", "wine_lr.joblib"))

# --- Load model artifact ---
loaded = joblib.load(MODEL_PATH)
if isinstance(loaded, dict):
    model = loaded["model"]
    FEATURES = loaded.get("features", DEFAULT_FEATURES)
else:
    model = loaded
    FEATURES = DEFAULT_FEATURES

# --- Bounds config ---
BOUNDS_PATH = os.getenv("BOUNDS_PATH", os.path.join("artifacts", "feature_bounds.json"))
INPUT_POLICY = os.getenv("INPUT_POLICY", "clip").lower()   # "clip" or "reject"
BOUND_MODE = os.getenv("BOUND_MODE", "p01p99").lower()      # "p01p99" or "minmax"

FEATURE_BOUNDS = None
if os.path.exists(BOUNDS_PATH):
    try:
        with open(BOUNDS_PATH, "r") as f:
            FEATURE_BOUNDS = json.load(f)
    except Exception:
        FEATURE_BOUNDS = None

# --- Helper maps ---
CANONICAL_SET = set(FEATURES)
LOWER_FEATURE_MAP = {f.lower(): f for f in FEATURES}

def human_label(feature_name: str) -> str:
    # Display friendly labels in the UI
    if feature_name == "pH":
        return "pH"
    return feature_name.replace("_", " ")

def get_allowed_range(feature_name: str):
    """
    Returns (lo, hi) or (None, None) if bounds missing.
    Uses p01/p99 by default to avoid outliers; can switch to min/max.
    """
    if not FEATURE_BOUNDS:
        return (None, None)
    b = FEATURE_BOUNDS.get(feature_name)
    if not b:
        return (None, None)

    if BOUND_MODE == "minmax":
        return (b.get("min"), b.get("max"))
    # default p01p99
    return (b.get("p01"), b.get("p99"))

def normalize_payload_keys(payload: dict) -> dict:
    """
    Accepts keys in multiple forms:
      - canonical (e.g., fixed_acidity)
      - spaces -> underscores (e.g., fixed acidity)
      - dashes -> underscores
      - case-insensitive (e.g., ph -> pH)
    Returns dict with canonical feature names.
    """
    normalized = {}

    for k, v in payload.items():
        if not isinstance(k, str):
            continue

        # already canonical
        if k in CANONICAL_SET:
            normalized[k] = v
            continue

        # spaces/dashes -> underscores
        k2 = k.strip().replace(" ", "_").replace("-", "_")
        if k2 in CANONICAL_SET:
            normalized[k2] = v
            continue

        # case-insensitive map
        k3 = k2.lower()
        if k3 in LOWER_FEATURE_MAP:
            normalized[LOWER_FEATURE_MAP[k3]] = v
            continue

    return normalized

def enforce_bounds(normalized: dict):
    """
    Applies INPUT_POLICY to out-of-range values:
      - reject: return (None, error_response, status)
      - clip: clamps values and returns warnings
    """
    warnings = []
    if not FEATURE_BOUNDS:
        return normalized, warnings, None

    for f in FEATURES:
        if f not in normalized:
            continue

        try:
            val = float(normalized[f])
        except Exception:
            return None, None, (jsonify({"detail": f"Feature '{f}' must be numeric"}), 400)

        lo, hi = get_allowed_range(f)
        if lo is None or hi is None:
            continue

        # out of range?
        if val < lo or val > hi:
            if INPUT_POLICY == "reject":
                return None, None, (jsonify({
                    "detail": f"Feature '{f}' out of allowed range [{lo}, {hi}]",
                    "feature": f,
                    "value": val,
                    "allowed": {"lo": lo, "hi": hi},
                    "mode": BOUND_MODE,
                }), 400)
            else:
                clipped = min(max(val, lo), hi)
                if clipped != val:
                    warnings.append({"feature": f, "from": val, "to": clipped, "allowed": [lo, hi]})
                normalized[f] = clipped

    return normalized, warnings, None

HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Wine Quality Predictor</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; max-width: 980px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 16px; }
    label { font-size: 14px; display: block; margin-bottom: 4px; }
    input { width: 100%; padding: 8px; font-size: 14px; }
    button { padding: 10px 14px; font-size: 15px; cursor: pointer; }
    .card { padding: 16px; border: 1px solid #ddd; border-radius: 10px; margin-top: 18px; }
    .result { font-size: 18px; }
    .muted { color: #666; font-size: 13px; }
    .range { color: #777; font-size: 12px; margin-top: 4px; }
    code { background: #f6f6f6; padding: 2px 6px; border-radius: 6px; }
  </style>
</head>
<body>
  <h2>Wine Quality Predictor</h2>
  <p class="muted">
    Enter the 11 feature values and click Predict.
    (API endpoint: <code>POST /predict</code>)<br/>
    Input policy: <code>{{ input_policy }}</code>, bounds: <code>{{ bound_mode }}</code>
  </p>

  <div class="card">
    <form id="predictForm">
      <div class="grid">
        {% for f in features %}
          <div>
            <label for="{{ ids[f] }}">{{ labels[f] }}</label>
            <input
              type="number"
              step="any"
              id="{{ ids[f] }}"
              name="{{ ids[f] }}"
              required
              {% if bounds[f][0] is not none %} min="{{ bounds[f][0] }}" {% endif %}
              {% if bounds[f][1] is not none %} max="{{ bounds[f][1] }}" {% endif %}
              placeholder="e.g. 7.4"
            />
            {% if bounds[f][0] is not none %}
              <div class="range">Allowed: {{ bounds[f][0] }} to {{ bounds[f][1] }}</div>
            {% endif %}
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

      let msg = `Predicted quality: ${data.rounded_quality} (raw: ${data.predicted_quality})`;
      if (data.warnings && data.warnings.length > 0) {
        msg += ` — clipped ${data.warnings.length} value(s)`;
      }
      resultEl.textContent = msg;
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
    ids = {f: f.replace(" ", "_").replace("-", "_") for f in FEATURES}  # safe IDs
    labels = {f: human_label(f) for f in FEATURES}
    bounds = {f: get_allowed_range(f) for f in FEATURES}
    return render_template_string(
        HTML,
        features=FEATURES,
        ids=ids,
        labels=labels,
        bounds=bounds,
        input_policy=INPUT_POLICY,
        bound_mode=BOUND_MODE,
    )

@app.get("/health")
def health():
    return jsonify({
        "status": "ok" if hasattr(model, "predict") else "bad",
        "model_path": MODEL_PATH,
        "model_type": str(type(model)),
        "feature_order": FEATURES,
        "bounds_loaded": bool(FEATURE_BOUNDS),
        "input_policy": INPUT_POLICY,
        "bound_mode": BOUND_MODE,
    })

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"detail": "Missing or invalid JSON body"}), 400

    # Option A: ordered list
    if isinstance(payload, dict) and "features" in payload:
        feats = payload["features"]
        if not isinstance(feats, list) or len(feats) != len(FEATURES):
            return jsonify({"detail": f"'features' must be a list of length {len(FEATURES)}"}), 400
        x = np.array([feats], dtype=float)
        warnings = []

    # Option B: dict with feature names (spaces/underscores ok)
    elif isinstance(payload, dict):
        normalized = normalize_payload_keys(payload)
        missing = [f for f in FEATURES if f not in normalized]
        if missing:
            return jsonify({"detail": f"Missing features: {missing}", "expected_features": FEATURES}), 400

        normalized, warnings, err = enforce_bounds(normalized)
        if err is not None:
            return err

        x = np.array([[float(normalized[f]) for f in FEATURES]], dtype=float)

    else:
        return jsonify({"detail": "JSON payload must be an object"}), 400

    if not hasattr(model, "predict"):
        return jsonify({"detail": f"Loaded model has no predict(): {type(model)}"}), 500

    try:
        pred = float(model.predict(x)[0])
        return jsonify({
            "predicted_quality": round(pred, 4),
            "rounded_quality": int(round(pred)),
            "feature_order": FEATURES,
            "warnings": warnings
        })
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)