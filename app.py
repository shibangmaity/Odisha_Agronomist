"""Odisha Agronomist: Flask backend.
Run:  put GROQ_API_KEY=<key> in a .env file next to app.py, then: python app.py
      (or set it in the terminal: set GROQ_API_KEY=<key> on Windows, export on Mac/Linux)
      (OPENAI_API_KEY or XAI_API_KEY also work)
      python app.py
"""
import os, json, re, time
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

BASE = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE, ".env")
DOTENV_OK = True
try:                                   # read keys from a .env file next to app.py (same as CampusIQ)
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH)
except ImportError:
    DOTENV_OK = False
crop_bundle = joblib.load(os.path.join(BASE, "models", "crop_model.pkl"))      # classifier + label encoder
yield_stats = joblib.load(os.path.join(BASE, "models", "yield_stats.pkl"))     # typical yield ranges

# ---------- AI provider setup: Groq (default), OpenAI or xAI, whichever key you set ----------
client, PROVIDER, MODELS, IMPORT_ERR = None, None, [], None
try:
    from openai import OpenAI          # Groq, xAI and OpenAI all speak the OpenAI API, so one library covers them
    if os.environ.get("GROQ_API_KEY"):
        PROVIDER = "groq"              # same key and model as CampusIQ
        client = OpenAI(api_key=os.environ["GROQ_API_KEY"].strip(), base_url="https://api.groq.com/openai/v1", timeout=60, max_retries=0)
        MODELS = [os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"), "llama-3.3-70b-versatile"]
    elif os.environ.get("OPENAI_API_KEY"):
        PROVIDER = "openai"
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip(), timeout=60, max_retries=0)
        MODELS = [os.environ.get("OPENAI_MODEL", "gpt-6-luna"), "gpt-5.6-luna", "gpt-5.4-mini"]
    elif os.environ.get("XAI_API_KEY"):
        PROVIDER = "xai"
        client = OpenAI(api_key=os.environ["XAI_API_KEY"].strip(), base_url="https://api.x.ai/v1", timeout=60, max_retries=0)
        MODELS = [os.environ.get("XAI_MODEL", "grok-4.5"), "grok-4", "grok-3"]
except ImportError as e:
    IMPORT_ERR = str(e)
print(f"[setup] .env file: {'found' if os.path.exists(ENV_PATH) else 'NOT found'} at {ENV_PATH} | python-dotenv: {'ok' if DOTENV_OK else 'MISSING'} | openai library: {'MISSING' if IMPORT_ERR else 'ok'}", flush=True)
print(f"AI provider: {PROVIDER or 'none (set GROQ_API_KEY, OPENAI_API_KEY or XAI_API_KEY)'}", flush=True)
RETRYABLE = ("429", "500", "502", "503", "overloaded", "rate_limit")   # timeouts skip to the next model

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


def call_ai(prompt):
    """Try each model up to 3 times with backoff on overload / rate-limit errors."""
    errors = []
    for model in MODELS:
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model=model, **({"response_format": {"type": "json_object"}} if PROVIDER == "openai" else {}),
                    messages=[{"role": "system", "content": "You are an expert Odisha agronomist. Reply only with a JSON object."},
                              {"role": "user", "content": prompt}])
                text = r.choices[0].message.content
                if not text:
                    raise ValueError(f"Empty response (finish_reason={r.choices[0].finish_reason})")
                cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
                try:
                    data = json.loads(cleaned)
                except ValueError:
                    raise ValueError(f"Reply was not valid JSON: {text[:200]!r}")
                if not isinstance(data, dict):
                    raise ValueError("Reply was not a JSON object")
                return data
            except Exception as e:
                msg = str(e)
                errors.append(f"{model}: {msg[:200]}")
                print(f"[ai] {model} attempt {attempt + 1} failed: {msg[:300]}", flush=True)
                if any(k in msg for k in ("insufficient_quota", "invalid_api_key", "Incorrect API key")):
                    raise RuntimeError(msg)         # billing or key problem: other models won't help
                if any(c in msg for c in RETRYABLE) and attempt < 2:
                    time.sleep(2 ** attempt)        # 1s, 2s
                    continue
                break                               # timeout or non-retryable: next model
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
    """The AI acts as an Odisha agronomist; returns sowing / fertilizer / irrigation advice."""
    if client is None:
        if IMPORT_ERR:
            return {"error": "The 'openai' package is not installed for this Python. Run: python -m pip install openai   then restart the app."}
        if not DOTENV_OK:
            return {"error": "The 'python-dotenv' package is not installed, so the .env file was not read. Run: python -m pip install python-dotenv   then restart."}
        return {"error": f"GROQ_API_KEY was not found. Put GROQ_API_KEY=your_key in this file: {ENV_PATH} (file exists: {os.path.exists(ENV_PATH)}), then restart."}
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
        return call_ai(prompt)
    except Exception as e:                          # bad key, quota, network, or non-JSON reply
        print(f"[ai] all models failed: {e}", flush=True)
        if "insufficient_quota" in str(e):
            return {"error": "The OpenAI account has no credit left. Add billing credit at platform.openai.com, then try again."}
        if "invalid_api_key" in str(e) or "Incorrect API key" in str(e):
            return {"error": "The OpenAI API key is invalid. Check OPENAI_API_KEY and restart the app."}
        return {"error": "The AI advisor is busy right now. Your crop matches and yield range above are still valid. Please try the advice again in a minute.",
                "detail": str(e)[:500]}


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