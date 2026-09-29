"""Odisha Agronomist: Flask backend.
Run:  set GEMINI_API_KEY=<your key>   (Windows)   /   export GEMINI_API_KEY=<your key>   (Mac/Linux)
      python app.py
"""
import os, json, time
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

BASE = os.path.dirname(os.path.abspath(__file__))
crop_bundle = joblib.load(os.path.join(BASE, "models", "crop_model.pkl"))      # classifier + label encoder
yield_stats = joblib.load(os.path.join(BASE, "models", "yield_stats.pkl"))     # typical yield ranges

# ---------- Gemini setup ----------
try:
    from google import genai
    from google.genai import types
    GEMINI_KEY = os.environ.get("GEMINI_API_KEY")          # never hardcode the key
    client = (genai.Client(api_key=GEMINI_KEY,
                           http_options=types.HttpOptions(timeout=60000))   # 60 s (ms); Odia/Bengali/Telugu replies are slower
              if GEMINI_KEY else None)
except ImportError:
    client = None

# Primary model first, then a fallback. Verify both names work for your key.
GEMINI_MODELS = [os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"), "gemini-3.5-flash", "gemini-flash-latest"]
RETRYABLE = ("503", "429", "500", "UNAVAILABLE", "RESOURCE_EXHAUSTED")   # timeouts (504) skip to the next model

LANGUAGES = ["English", "Odia", "Hindi", "Bengali", "Telugu"]
LIMITS = {"N": (0, 300), "P": (0, 300), "K": (0, 300), "temperature": (0, 50),
          "humidity": (0, 100), "ph": (3, 10), "rainfall": (0, 500)}
FEATURES = crop_bundle["features"]
app = Flask(__name__)

# ---------- Trained ML info shown in the app ----------
COMPARISON = [{"name": "Random Forest", "acc": 99.32}, {"name": "XGBoost", "acc": 98.86}]   # notebook step 4
REC_CSV = os.path.join(BASE, "data", "processed", "odisha_crop_rec_clean.csv")
rec_df = pd.read_csv(REC_CSV) if os.path.exists(REC_CSV) else None

# typical growing conditions per crop (10th / median / 90th percentile of the training data)
PROFILE = {}
if rec_df is not None:
    g = rec_df.groupby("label")[FEATURES]
    lo, mid, hi = g.quantile(.1), g.median(), g.quantile(.9)
    PROFILE = {c: {f: [round(float(lo.loc[c, f]), 1), round(float(mid.loc[c, f]), 1), round(float(hi.loc[c, f]), 1)]
                   for f in FEATURES} for c in mid.index}


def build_model_info():
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    m, le = crop_bundle["model"], crop_bundle["label_encoder"]
    info = {"name": {"rf": "Random Forest", "xgb": "XGBoost"}.get(crop_bundle.get("name"), str(crop_bundle.get("name"))),
            "features": FEATURES, "crops": list(le.classes_), "comparison": COMPARISON,
            "yield_crops": len(yield_stats["by_crop"]), "districts": len(yield_stats["districts"]),
            "mapped": crop_bundle["label_to_yield_crop"], "accuracy": None, "per_crop": [], "importance": []}
    if hasattr(m, "feature_importances_"):
        info["importance"] = [{"feature": f, "value": round(float(v) * 100, 1)}
                              for f, v in zip(FEATURES, m.feature_importances_)]
    if rec_df is not None:      # same 80/20 split and seed as the notebook, so this is the true held-out score
        _, X_te, _, y_te = train_test_split(rec_df[FEATURES], le.transform(rec_df["label"]), test_size=0.2,
                                            stratify=rec_df["label"], random_state=42)
        pred = m.predict(X_te)
        f1 = f1_score(y_te, pred, average=None)
        info.update(rows=len(rec_df), test_rows=len(y_te), accuracy=round(accuracy_score(y_te, pred) * 100, 2),
                    per_crop=[{"crop": c, "f1": round(float(v), 2)} for c, v in zip(le.classes_, f1)])
    return info


try:
    MODEL_INFO = build_model_info()
except Exception as e:
    MODEL_INFO = {"error": f"Could not load model details: {e}"}


def call_gemini(prompt):
    """Try each model up to 3 times with backoff on overload / rate-limit errors."""
    errors = []
    for model in GEMINI_MODELS:
        for attempt in range(3):
            try:
                r = client.models.generate_content(
                    model=model, contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json"))
                if not r.text:
                    raise ValueError("Empty response from Gemini")
                data = json.loads(r.text)
                if not isinstance(data, dict):
                    raise ValueError("Gemini did not return a JSON object")
                return data
            except Exception as e:
                errors.append(f"{model}: {str(e)[:200]}")
                print(f"[gemini] {model} attempt {attempt + 1} failed: {str(e)[:300]}", flush=True)
                if any(c in str(e) for c in RETRYABLE) and attempt < 2:
                    time.sleep(2 ** attempt)        # 1s, 2s
                    continue
                break                               # timeout or non-retryable: move to next model
    raise RuntimeError(" | ".join(errors))


def yield_info(label, district):
    """Typical yield range straight from the Odisha data (not a model prediction)."""
    crop = crop_bundle["label_to_yield_crop"].get(label)
    if not crop:
        return {"available": False}
    s, scope = yield_stats["by_crop_district"].get(crop, {}).get(district), "district"
    if not s or s["n"] < 5:
        s, scope = yield_stats["by_crop"].get(crop), "odisha"
    if not s:
        return {"available": False}
    conf = "high" if s["n"] >= 100 else "medium" if s["n"] >= 30 else "low"
    return {"available": True, "crop": crop, "scope": scope, "unit": yield_stats["unit"],
            "low": round(s["p25"], 2), "typical": round(s["median"], 2), "high": round(s["p75"], 2),
            "records": s["n"], "years": s["years"], "confidence": conf}


def agronomist_plan(d, crop_label, language, y):
    """Gemini acts as an Odisha agronomist; returns sowing / fertilizer / irrigation advice."""
    if client is None:
        return {"error": "Gemini is not configured. Set the GEMINI_API_KEY environment variable."}
    yline = (f"Typical yield in past Odisha records: {y['low']}-{y['high']} {y['unit']}." if y["available"]
             else "No Odisha yield records are available for this crop.")
    prompt = f"""You are an expert agronomist for Odisha, India, advising a farmer.
District: {d['district'].title()}. Recommended crop: {crop_label}.
Soil (kg/ha): N={d['N']}, P={d['P']}, K={d['K']}, pH={d['ph']}.
Weather: temperature {d['temperature']} C, humidity {d['humidity']} %, rainfall {d['rainfall']} mm.
{yline}
Give practical, specific advice for Odisha's Kharif and Rabi seasons and the district's soil type.
Reply ONLY as JSON with exactly these keys: "sowing_schedule", "fertilizer_management", "irrigation_strategy".
Each value is short plain text (3-5 sentences or short lines). Base fertilizer advice on the N, P, K and pH given.
Write all values in {language}. Add that doses should be confirmed with the local Krishi Vigyan Kendra."""
    try:
        return call_gemini(prompt)
    except Exception as e:                          # bad key, quota, network, or non-JSON reply
        print(f"[gemini] all models failed: {e}", flush=True)
        return {"error": "The AI advisor is busy right now (Google's servers are overloaded). Your crop matches and yield range above are still valid. Please try the advice again in a minute."}


@app.get("/")
def index():
    labels = [l for l in crop_bundle["label_encoder"].classes_]
    return render_template("index.html", districts=yield_stats["districts"], crops=labels, languages=LANGUAGES)


@app.post("/api/recommend")
def recommend():
    d = request.get_json(force=True, silent=True) or {}
    if d.get("district") not in yield_stats["districts"]:
        return jsonify(error="Choose a district."), 400
    try:
        vals = {k: float(d[k]) for k in FEATURES}
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Fill in all soil and weather values with numbers."), 400
    for k in FEATURES:                              # loop FEATURES so a missing LIMITS key can't crash
        lo, hi = LIMITS.get(k, (float("-inf"), float("inf")))
        if not lo <= vals[k] <= hi:
            return jsonify(error=f"{k} should be between {lo} and {hi}."), 400

    row = pd.DataFrame([[vals[k] for k in FEATURES]], columns=FEATURES)
    proba = pd.Series(crop_bundle["model"].predict_proba(row)[0], index=crop_bundle["label_encoder"].classes_)
    top = [{"crop": c, "score": round(float(p), 3)} for c, p in proba.sort_values(ascending=False).head(3).items()]

    chosen = d.get("crop") if d.get("crop") in proba.index else top[0]["crop"]
    y = yield_info(chosen, d["district"])
    lang = d.get("language") if d.get("language") in LANGUAGES else "English"
    plan = agronomist_plan({**vals, "district": d["district"]}, chosen, lang, y)
    return jsonify(top=top, chosen=chosen, chosen_score=round(float(proba[chosen]), 3), yield_info=y, plan=plan,
                   profile=PROFILE.get(chosen))


@app.get("/api/model")
def model_details():
    return jsonify(MODEL_INFO)


if __name__ == "__main__":
    # debug only when you explicitly ask for it: set FLASK_DEBUG=1
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")