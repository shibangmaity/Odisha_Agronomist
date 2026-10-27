"""Odisha Agronomist: Flask backend.
Run:  put GROQ_API_KEY=<key> in a .env file next to app.py, then: python app.py
      (XAI_API_KEY or OPENAI_API_KEY also work; Groq is used first if several are set)
      (or set it in the terminal: set GROQ_API_KEY=<key> on Windows, export on Mac/Linux)
"""
import os, json, re, time, math, threading
from collections import OrderedDict, defaultdict, deque
import joblib
import pandas as pd
import sklearn
from flask import Flask, render_template, request, jsonify

BASE = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE, ".env")
DOTENV_OK = True
try:                                   # read keys from a .env file next to app.py
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH)
except ImportError:
    DOTENV_OK = False
crop_bundle = joblib.load(os.path.join(BASE, "models", "crop_model.pkl"))      # classifier + label encoder
yield_stats = joblib.load(os.path.join(BASE, "models", "yield_stats.pkl"))     # typical yield ranges

TRAINED_WITH = crop_bundle.get("sklearn_version")
if TRAINED_WITH and TRAINED_WITH != sklearn.__version__:
    print(f"[setup] WARNING: crop_model.pkl was trained with scikit-learn {TRAINED_WITH} but {sklearn.__version__} "
          f"is installed. Run: python -m pip install -r requirements.txt", flush=True)

# ---------- AI provider setup: Groq (default), xAI or OpenAI, whichever key you set ----------
client, PROVIDER, MODELS, IMPORT_ERR = None, None, [], None
KEY_NAMES = {"groq": "GROQ_API_KEY", "xai": "XAI_API_KEY", "openai": "OPENAI_API_KEY"}
BILLING = {"groq": "console.groq.com", "xai": "console.x.ai", "openai": "platform.openai.com"}
try:
    from openai import OpenAI          # Groq, xAI and OpenAI all speak the OpenAI API, so one library covers them
    if os.environ.get("GROQ_API_KEY"):
        PROVIDER = "groq"
        client = OpenAI(api_key=os.environ["GROQ_API_KEY"].strip(), base_url="https://api.groq.com/openai/v1", timeout=60, max_retries=0)
        MODELS = [os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"), "llama-3.3-70b-versatile"]
    elif os.environ.get("XAI_API_KEY"):
        PROVIDER = "xai"
        client = OpenAI(api_key=os.environ["XAI_API_KEY"].strip(), base_url="https://api.x.ai/v1", timeout=60, max_retries=0)
        MODELS = [os.environ.get("XAI_MODEL", "grok-4.5"), "grok-4", "grok-3"]
    elif os.environ.get("OPENAI_API_KEY"):
        PROVIDER = "openai"
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip(), timeout=60, max_retries=0)
        MODELS = [os.environ.get("OPENAI_MODEL", "gpt-6-luna"), "gpt-5.6-luna", "gpt-5.4-mini"]
except ImportError as e:
    IMPORT_ERR = str(e)
print(f"[setup] .env file: {'found' if os.path.exists(ENV_PATH) else 'NOT found'} at {ENV_PATH} | python-dotenv: {'ok' if DOTENV_OK else 'MISSING'} | openai library: {'MISSING' if IMPORT_ERR else 'ok'}", flush=True)
print(f"AI provider: {PROVIDER or 'none (set GROQ_API_KEY, XAI_API_KEY or OPENAI_API_KEY)'}", flush=True)
RETRYABLE = ("429", "500", "502", "503", "overloaded", "rate_limit")   # timeouts skip to the next model

LANGUAGES = ["English", "Odia", "Hindi", "Bengali", "Telugu"]
FEATURES = crop_bundle["features"]
LE = crop_bundle["label_encoder"]
app = Flask(__name__)

# ---------- Data the model was trained on ----------
REC_CSV = os.path.join(BASE, "data", "processed", "odisha_crop_rec_clean.csv")
rec_df = pd.read_csv(REC_CSV) if os.path.exists(REC_CSV) else None

# Inputs are only accepted inside the range the model saw in training. Outside it a Random Forest
# still answers confidently, but the answer means nothing (e.g. N=P=K=300 used to give "apple 47%").
FALLBACK_RANGES = {"N": (0, 140), "P": (5, 145), "K": (5, 205), "temperature": (8.8, 43.7),
                   "humidity": (14.3, 100), "ph": (3.5, 9.9), "rainfall": (20.2, 298.6)}
if crop_bundle.get("ranges"):
    _ranges = {f: tuple(crop_bundle["ranges"][f]) for f in FEATURES}
elif rec_df is not None:
    _ranges = {f: (float(rec_df[f].min()), float(rec_df[f].max())) for f in FEATURES}
else:
    _ranges = FALLBACK_RANGES
LIMITS = {f: (math.floor(lo * 10) / 10, math.ceil(hi * 10) / 10) for f, (lo, hi) in _ranges.items()}

# Crops that are not grown in Odisha are never recommended. Change with EXCLUDED_CROPS=apple,grapes,...
EXCLUDED_CROPS = {c.strip().lower() for c in os.environ.get("EXCLUDED_CROPS", "apple,grapes").split(",") if c.strip()}
CROPS = [c for c in LE.classes_ if c not in EXCLUDED_CROPS]
LOW_CONFIDENCE = float(os.environ.get("LOW_CONFIDENCE", "0.40"))   # below this top score, say "no clear match"

# typical growing conditions per crop (10th / median / 90th percentile of the training data)
PROFILE = {}
if rec_df is not None:
    g = rec_df.groupby("label")[FEATURES]
    lo, mid, hi = g.quantile(.1), g.median(), g.quantile(.9)
    PROFILE = {c: {f: [round(float(lo.loc[c, f]), 1), round(float(mid.loc[c, f]), 1), round(float(hi.loc[c, f]), 1)]
                   for f in FEATURES} for c in mid.index}

# typical weather per district and season, built once by build_district_climate.py (Open-Meteo, ERA5)
SEASONS = {"Kharif": "Jun-Oct", "Rabi": "Nov-Feb", "Zaid": "Mar-May"}

# Soil types of Odisha. Districts, reaction and N/P/K status follow "Soil of Orissa and Its Management",
# Orissa Review (Govt. of Odisha), Nov 2005. Only pH is filled in: the N/P/K status of a soil is shown as text,
# because the model's N/P/K behave like the nutrients a field receives (rice rows have a median of 80:47:40,
# close to a fertilizer dose), not the soil's own status. Filling "low" values made the model suggest coconut.
SOIL_SOURCE = "Soil of Orissa and Its Management, Orissa Review, Nov 2005"
_NS = ("medium", "not stated")
SOIL_TYPES = [   # order decides the default soil for a district (first type found there)
    {"name": "Deltaic alluvial", "ph": 6.3, "reaction": "acidic to neutral",
     "N": _NS, "P": _NS, "K": _NS,
     "districts": ["BALESHWAR", "BHADRAK", "JAJAPUR", "KENDRAPARA", "JAGATSINGHAPUR", "CUTTACK", "PURI", "GAJAPATI", "GANJAM"]},
    {"name": "Red", "ph": 5.3, "reaction": "strongly to moderately acidic",
     "N": ("low", "low"), "P": ("low", "low"), "K": _NS,
     "districts": ["KORAPUT", "RAYAGADA", "NABARANGPUR", "MALKANGIRI", "KENDUJHAR", "GANJAM", "KALAHANDI", "NUAPADA",
                   "BALANGIR", "DHENKANAL", "MAYURBHANJ"]},
    {"name": "Red and yellow", "ph": 5.8, "reaction": "moderately acidic (uplands) to slightly acidic (lowlands)",
     "N": ("low", "low (uplands)"), "P": ("low", "low (uplands), medium (lowlands)"), "K": ("medium", "high in lowlands, uplands not stated"),
     "districts": ["SAMBALPUR", "BARGARH", "DEOGARH", "SUNDARGARH", "JHARSUGUDA"]},   # Jharsuguda: formerly part of Sambalpur
    {"name": "Laterite", "ph": 5.2, "reaction": "pH 4.5 to 5.8",
     "N": ("low", "low"), "P": ("low", "low"), "K": ("medium", "medium"),
     "districts": ["PURI", "KHORDHA", "NAYAGARH", "CUTTACK", "DHENKANAL", "KENDUJHAR", "MAYURBHANJ", "SAMBALPUR"]},
    {"name": "Coastal saline and alluvial", "ph": 7.0, "reaction": "pH 6.0 to 8.0, saline",
     "N": ("high", "high"), "P": ("low", "low to medium"), "K": ("high", "high"),
     "districts": ["BALESHWAR", "BHADRAK", "JAGATSINGHAPUR", "KENDRAPARA", "PURI", "KHORDHA", "GANJAM"]},
    {"name": "Black", "ph": 7.8, "reaction": "neutral to alkaline",
     "N": _NS, "P": ("low", "low"), "K": ("low", "low"),
     "districts": ["PURI", "GANJAM", "MALKANGIRI", "KALAHANDI", "NUAPADA", "BALANGIR", "SONEPUR", "BOUDH", "SAMBALPUR",
                   "BARGARH", "ANUGUL"]},
    {"name": "Mixed red and black", "ph": 7.0, "reaction": "neutral",
     "N": _NS, "P": _NS, "K": _NS,
     "districts": ["SAMBALPUR", "BARGARH", "SONEPUR", "BALANGIR"]},
    {"name": "Brown forest", "ph": 5.5, "reaction": "acidic",
     "N": ("high", "medium to high"), "P": ("medium", "medium"), "K": ("medium", "medium"),
     "districts": ["KANDHAMAL", "RAYAGADA", "GANJAM", "NAYAGARH"]},
]
SOILS = [{"name": s["name"], "ph": s["ph"], "reaction": s["reaction"], "districts": s["districts"],
          "status": {f: s[f][1] for f in ("N", "P", "K")}, "source": SOIL_SOURCE} for s in SOIL_TYPES]

# crop suggestions outside Kharif are weak: the training data lacks Odisha's main winter and summer crops
SEASON_NOTES = {
    "Rabi": "Rabi suggestions are limited. The model's training data has no mustard, groundnut, potato, wheat or winter "
            "vegetables, so the matches below may not suit your field. Check them against your local KVK's Rabi advice.",
    "Zaid": "Summer (Zaid) suggestions are limited. The model's training data has no groundnut, sesame or summer vegetables, "
            "so the matches below may not suit your field. Check them against your local KVK's advice.",
}
CLIMATE_CSV = os.path.join(BASE, "data", "processed", "district_climate.csv")
CLIMATE = {}
if os.path.exists(CLIMATE_CSV):
    for r in pd.read_csv(CLIMATE_CSV).to_dict("records"):
        CLIMATE.setdefault(r["district"], {})[r["season"]] = {
            "temperature": float(r["temperature"]), "humidity": float(r["humidity"]),
            "rainfall": float(r["rainfall"]), "years": str(r["years"])}
print(f"[setup] district weather table: {len(CLIMATE)} districts" if CLIMATE else
      "[setup] district weather table not found (run build_district_climate.py); weather is entered by hand", flush=True)


def build_model_info():
    m = crop_bundle["model"]
    info = {"name": {"rf": "Random Forest", "xgb": "XGBoost"}.get(crop_bundle.get("name"), str(crop_bundle.get("name"))),
            "features": FEATURES, "crops": CROPS, "excluded": sorted(EXCLUDED_CROPS & set(LE.classes_)),
            "comparison": crop_bundle.get("comparison", [{"name": "Random Forest", "acc": 99.32}, {"name": "XGBoost", "acc": 98.86}]),
            "yield_crops": len(yield_stats["by_crop"]), "districts": len(yield_stats["districts"]),
            "mapped": crop_bundle["label_to_yield_crop"], "limits": LIMITS, "low_confidence": LOW_CONFIDENCE,
            "climate_districts": len(CLIMATE),
            "trained_with": TRAINED_WITH, "accuracy": None, "cv_accuracy": None, "per_crop": [], "importance": []}
    if hasattr(m, "feature_importances_"):
        info["importance"] = [{"feature": f, "value": round(float(v) * 100, 1)}
                              for f, v in zip(FEATURES, m.feature_importances_)]
    if crop_bundle.get("metrics"):          # saved by the notebook at training time
        info.update(crop_bundle["metrics"])
    elif rec_df is not None:                # older model file: recompute on the notebook's 80/20 split
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score
        _, X_te, _, y_te = train_test_split(rec_df[FEATURES], LE.transform(rec_df["label"]), test_size=0.2,
                                            stratify=rec_df["label"], random_state=42)
        pred = m.predict(X_te)
        f1 = f1_score(y_te, pred, average=None)
        info.update(rows=len(rec_df), test_rows=len(y_te), accuracy=round(accuracy_score(y_te, pred) * 100, 2),
                    per_crop=[{"crop": c, "f1": round(float(v), 2)} for c, v in zip(LE.classes_, f1)])
    return info


try:
    MODEL_INFO = build_model_info()
except Exception as e:
    MODEL_INFO = {"error": f"Could not load model details: {e}"}


# ---------- Protect the AI key: cache answers and limit calls per visitor ----------
RATE_LIMIT = int(os.environ.get("AI_CALLS_PER_MINUTE", "6"))   # AI calls per visitor per minute
_hits, _lock = defaultdict(deque), threading.Lock()
_plan_cache, CACHE_SIZE = OrderedDict(), 256


def over_rate_limit():
    ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or request.remote_addr or "?"
    now = time.time()
    with _lock:
        q = _hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            return True
        q.append(now)
        if len(_hits) > 5000:                       # forget idle visitors
            for k in [k for k, v in _hits.items() if not v]:
                del _hits[k]
    return False


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
        return {"error": f"No AI key found. Put GROQ_API_KEY=your_key (or XAI_API_KEY / OPENAI_API_KEY) in this file: {ENV_PATH} (file exists: {os.path.exists(ENV_PATH)}), then restart."}

    key = (d["district"], d.get("season"), crop_label, language, *(round(float(d[f]), 1) for f in FEATURES))
    with _lock:
        if key in _plan_cache:
            _plan_cache.move_to_end(key)
            return _plan_cache[key]
    if over_rate_limit():
        return {"error": "Too many advice requests in the last minute. Your crop matches and yield range above are still valid. Please wait a minute and try again."}

    yline = (f"Typical yield in past Odisha records: {y['low']}-{y['high']} {y['unit']}." if y["available"]
             else "No Odisha yield records are available for this crop.")
    season = d.get("season")
    sline = (f"Growing season: {season} ({SEASONS[season]}). Give advice for this season only."
             if season else "Cover Odisha's Kharif and Rabi seasons.")
    prompt = f"""You are an expert agronomist for Odisha, India, advising a farmer.
District: {d['district'].title()}. Recommended crop: {crop_label}.
{sline}
Soil (kg/ha): N={d['N']}, P={d['P']}, K={d['K']}, pH={d['ph']}.
Weather: temperature {d['temperature']} C, humidity {d['humidity']} %, average rainfall {d['rainfall']} mm per month in the season.
{yline}
Give practical, specific advice for Odisha and the district's soil type.
Reply ONLY as JSON with exactly these keys: "sowing_schedule", "fertilizer_management", "irrigation_strategy".
Each value is short plain text (3-5 sentences or short lines). Base fertilizer advice on the N, P, K and pH given.
If {crop_label} is not usually grown in this season in Odisha, say so first and name crops that are.
Write all values in {language}. Add that doses should be confirmed with the local Krishi Vigyan Kendra."""
    try:
        plan = call_ai(prompt)
    except Exception as e:                          # bad key, quota, network, or non-JSON reply
        print(f"[ai] all models failed: {e}", flush=True)
        if "insufficient_quota" in str(e):
            return {"error": f"The {PROVIDER} account has no credit left. Add credit at {BILLING[PROVIDER]}, then try again."}
        if "invalid_api_key" in str(e) or "Incorrect API key" in str(e):
            return {"error": f"The {PROVIDER} API key is invalid. Check {KEY_NAMES[PROVIDER]} and restart the app."}
        return {"error": "The AI advisor is busy right now. Your crop matches and yield range above are still valid. Please try the advice again in a minute.",
                "detail": str(e)[:500]}
    with _lock:
        _plan_cache[key] = plan
        if len(_plan_cache) > CACHE_SIZE:
            _plan_cache.popitem(last=False)
    return plan


@app.get("/")
def index():
    return render_template("index.html", districts=yield_stats["districts"], crops=CROPS, languages=LANGUAGES,
                           limits=LIMITS, seasons=SEASONS, climate=CLIMATE, soils=SOILS)


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
            return jsonify(error=f"{k} should be between {lo} and {hi}. That is the range the model was trained on, "
                                 f"so values outside it would give a meaningless result."), 400

    row = pd.DataFrame([[vals[k] for k in FEATURES]], columns=FEATURES)
    proba = pd.Series(crop_bundle["model"].predict_proba(row)[0], index=LE.classes_)[CROPS]
    if proba.sum() > 0:
        proba = proba / proba.sum()                 # re-share the excluded crops' votes
    top = [{"crop": c, "score": round(float(p), 3)} for c, p in proba.sort_values(ascending=False).head(3).items()]
    low_conf = top[0]["score"] < LOW_CONFIDENCE
    warning = ("No crop is a strong match for these values. The scores below are only the closest matches in a general "
               "crop dataset, so check them with your local Krishi Vigyan Kendra before deciding." if low_conf else None)

    chosen = d.get("crop") if d.get("crop") in proba.index else top[0]["crop"]
    y = yield_info(chosen, d["district"])
    lang = d.get("language") if d.get("language") in LANGUAGES else "English"
    season = d.get("season") if d.get("season") in SEASONS else None
    plan = agronomist_plan({**vals, "district": d["district"], "season": season}, chosen, lang, y)
    return jsonify(top=top, chosen=chosen, chosen_score=round(float(proba[chosen]), 3), yield_info=y, plan=plan,
                   profile=PROFILE.get(chosen), low_confidence=low_conf, warning=warning, season=season,
                   season_note=SEASON_NOTES.get(season))


@app.get("/api/model")
def model_details():
    return jsonify(MODEL_INFO)


if __name__ == "__main__":
    # debug only when you explicitly ask for it: set FLASK_DEBUG=1
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
