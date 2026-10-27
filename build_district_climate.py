"""Builds data/processed/district_climate.csv: typical weather per Odisha district and season.

Run once from the repo folder (needs internet, takes a few minutes):  python build_district_climate.py

Source: Open-Meteo historical weather API (ERA5 reanalysis), free, no key needed.
For each district headquarters and each season it stores:
  temperature  mean daily temperature (C)
  humidity     mean daily relative humidity (%)
  rainfall     average rainfall per month in that season (mm/month) -- the same scale as the
               crop dataset (rice there has a median of about 233 mm)
Raw downloads are cached in data/raw/climate_cache/ (git-ignored), so a rerun is instant.
"""
import json, os, sys, time, urllib.error, urllib.parse, urllib.request
import pandas as pd

START, END = "2019-01-01", "2023-12-31"            # 5 full years
SEASONS = {"Kharif": [6, 7, 8, 9, 10], "Rabi": [11, 12, 1, 2], "Zaid": [3, 4, 5]}
OUT = os.path.join("data", "processed", "district_climate.csv")
CACHE = os.path.join("data", "raw", "climate_cache")
URL = "https://archive-api.open-meteo.com/v1/archive"

# district headquarters (lat, lon); names match the yield data
DISTRICTS = {
    "ANUGUL": (20.84, 85.10), "BALANGIR": (20.70, 83.48), "BALESHWAR": (21.49, 86.93), "BARGARH": (21.33, 83.62),
    "BHADRAK": (21.05, 86.50), "BOUDH": (20.84, 84.32), "CUTTACK": (20.46, 85.88), "DEOGARH": (21.54, 84.73),
    "DHENKANAL": (20.66, 85.60), "GAJAPATI": (18.78, 84.09), "GANJAM": (19.31, 84.79), "JAGATSINGHAPUR": (20.26, 86.17),
    "JAJAPUR": (20.85, 86.34), "JHARSUGUDA": (21.86, 84.01), "KALAHANDI": (19.91, 83.17), "KANDHAMAL": (20.47, 84.23),
    "KENDRAPARA": (20.50, 86.42), "KENDUJHAR": (21.63, 85.58), "KHORDHA": (20.18, 85.62), "KORAPUT": (18.81, 82.71),
    "MALKANGIRI": (18.35, 81.89), "MAYURBHANJ": (21.94, 86.73), "NABARANGPUR": (19.23, 82.55), "NAYAGARH": (20.13, 85.10),
    "NUAPADA": (20.81, 82.53), "PURI": (19.81, 85.83), "RAYAGADA": (19.17, 83.42), "SAMBALPUR": (21.47, 83.97),
    "SONEPUR": (20.83, 83.92), "SUNDARGARH": (22.12, 84.04),
}


def get_json(params):
    url = URL + "?" + urllib.parse.urlencode(params)
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            if e.code == 429:                       # free-tier rate limit: wait and retry
                print("    rate limit reached, waiting 65 s ...", flush=True)
                time.sleep(65)
                continue
            raise RuntimeError(f"HTTP {e.code}: {body[:300]}")
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"    network error ({e}), retrying ...", flush=True)
            time.sleep(10)
    raise RuntimeError("gave up after several retries")


def daily_frame(lat, lon):
    base = {"latitude": lat, "longitude": lon, "start_date": START, "end_date": END, "timezone": "Asia/Kolkata"}
    try:
        d = get_json({**base, "daily": "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum"})["daily"]
        return pd.DataFrame({"date": d["time"], "temperature": d["temperature_2m_mean"],
                             "humidity": d["relative_humidity_2m_mean"], "rain": d["precipitation_sum"]})
    except RuntimeError as e:
        if "relative_humidity" not in str(e):
            raise
        # older API without a daily humidity mean: average the hourly values per day
        d = get_json({**base, "daily": "temperature_2m_mean,precipitation_sum"})["daily"]
        h = get_json({**base, "hourly": "relative_humidity_2m"})["hourly"]
        hum = pd.DataFrame({"t": h["time"], "h": h["relative_humidity_2m"]})
        hum = hum.groupby(hum["t"].str[:10])["h"].mean()
        return pd.DataFrame({"date": d["time"], "temperature": d["temperature_2m_mean"],
                             "humidity": hum.reindex(d["time"]).values, "rain": d["precipitation_sum"]})


def main():
    os.makedirs(CACHE, exist_ok=True)
    rows = []
    for i, (name, (lat, lon)) in enumerate(DISTRICTS.items(), 1):
        cache = os.path.join(CACHE, f"{name}.csv")
        if os.path.exists(cache):
            df = pd.read_csv(cache)
        else:
            print(f"[{i}/{len(DISTRICTS)}] downloading {name} ...", flush=True)
            df = daily_frame(lat, lon)
            df.to_csv(cache, index=False)
            time.sleep(3)                           # be polite to the free API
        df["date"] = pd.to_datetime(df["date"])
        df["month"], df["year"] = df["date"].dt.month, df["date"].dt.year
        for season, months in SEASONS.items():
            s = df[df["month"].isin(months)].dropna()
            monthly_rain = s.groupby(["year", "month"])["rain"].sum()     # total per calendar month
            rows.append({"district": name, "season": season,
                         "temperature": round(float(s["temperature"].mean()), 1),
                         "humidity": round(float(s["humidity"].mean()), 1),
                         "rainfall": round(float(monthly_rain.mean()), 1),
                         "years": f"{START[:4]}-{END[:4]}"})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"\nsaved {OUT}: {len(out)} rows")
    print(out.groupby("season")[["temperature", "humidity", "rainfall"]].agg(["min", "median", "max"]).round(1))


if __name__ == "__main__":
    if not os.path.exists("app.py"):
        sys.exit("Run this from the Odisha_Agronomist folder (the one with app.py).")
    main()
