from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import os
import traceback

app = Flask(__name__)

FEATURES = [
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
loaded = joblib.load(MODEL_PATH)

if isinstance(loaded, dict):
    model = loaded["model"]
    # Use the feature order saved with the model (important!)
    if "features" in loaded and isinstance(loaded["features"], list):
        FEATURES = loaded["features"]
else:
    model = loaded

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
            <label for="{{f}}">{{f}}</label>
            <input type="number" step="any" id="{{f}}" name="{{f}}" required placeholder="e.g. 7.4"/>
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
  const form = document.getElementById("predictForm");
  const resultEl = document.getElementById("result");
  const rawEl = document.getElementById("raw");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    resultEl.textContent = "Predicting...";
    rawEl.textContent = "";

    const payload = {};
    for (const f of FEATURES) {
      const val = document.getElementById(f).value;
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
    return render_template_string(HTML, features=FEATURES)

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_path": MODEL_PATH})

@app.post("/predict")
def predict():
    try:
        payload = request.get_json(silent=True) or {}

        # Case (2): {"features": [...]}
        if isinstance(payload, dict) and "features" in payload:
            feats = payload["features"]
            if not isinstance(feats, list) or len(feats) != len(FEATURES):
                return jsonify({"detail": f"'features' must be a list of length {len(FEATURES)}"}), 400
            x = np.array([feats], dtype=float)

        # Case (1): {"fixed acidity": ..., ...}
        elif isinstance(payload, dict):
            missing = [f for f in FEATURES if f not in payload]
            if missing:
                return jsonify({"detail": f"Missing features: {missing}"}), 400
            x = np.array([[float(payload[f]) for f in FEATURES]], dtype=float)

        else:
            return jsonify({"detail": "Invalid JSON payload"}), 400

        pred = float(model.predict(x)[0])
        return jsonify({
            "predicted_quality": round(pred, 4),
            "rounded_quality": int(round(pred)),
            "feature_order": FEATURES
        })

    except Exception as e:
        # This will print into Heroku logs AND return the traceback to curl/browser
        app.logger.exception("Predict failed")
        return jsonify({
            "detail": str(e),
            "trace": traceback.format_exc()
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)