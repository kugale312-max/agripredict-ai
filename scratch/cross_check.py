import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import requests

PROCESSED_DIR = Path("data/processed")
WEATHER_CACHE_PATH = PROCESSED_DIR / "weather_cache.csv"

STATE_CENTROIDS = {
    "Andaman and Nicobar Islands": (11.7401, 92.6586),
    "Andhra Pradesh": (15.9129, 79.7400),
    "Arunachal Pradesh": (28.2180, 94.7278),
    "Assam": (26.2006, 92.9376),
    "Bihar": (25.0961, 85.3131),
    "Chandigarh": (30.7333, 76.7794),
    "Chhattisgarh": (21.2787, 81.8661),
    "Dadra and Nagar Haveli": (20.1809, 73.0169),
    "Goa": (15.2993, 74.1240),
    "Gujarat": (22.2587, 71.1924),
    "Haryana": (29.0588, 76.0856),
    "Himachal Pradesh": (31.1048, 77.1734),
    "Jammu and Kashmir": (33.7782, 76.5762),
    "Jharkhand": (23.6102, 85.2799),
    "Karnataka": (15.3173, 75.7139),
    "Kerala": (10.8505, 76.2711),
    "Madhya Pradesh": (22.9734, 78.6569),
    "Maharashtra": (19.7515, 75.7139),
    "Manipur": (24.6637, 93.9063),
    "Meghalaya": (25.4670, 91.3662),
    "Mizoram": (23.1645, 92.9376),
    "Nagaland": (26.1584, 94.5624),
    "Odisha": (20.9517, 85.0985),
    "Puducherry": (11.9416, 79.8083),
    "Punjab": (31.1471, 75.3412),
    "Rajasthan": (27.0238, 74.2179),
    "Sikkim": (27.5330, 88.5122),
    "Tamil Nadu": (11.1271, 78.6569),
    "Telangana": (18.1124, 79.0193),
    "Tripura": (23.9408, 91.9882),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Uttarakhand": (30.0668, 79.0193),
    "West Bengal": (22.9868, 87.8550),
}

def fetch_nasa_daily(lat, lon, start_ymd, end_ymd, max_retries=3):
    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters=PRECTOTCORR,T2M&community=AG&longitude={lon}&latitude={lat}&"
        f"start={start_ymd}&end={end_ymd}&format=JSON"
    )
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=40)
            if resp.status_code == 200:
                data = resp.json()
                params = data.get("properties", {}).get("parameter", {})
                return {
                    "precip": params.get("PRECTOTCORR", {}),
                    "temp": params.get("T2M", {}),
                }
            elif resp.status_code in [429, 500, 502, 503, 504]:
                time.sleep(2 ** attempt)
            else:
                time.sleep(1)
        except Exception:
            time.sleep(2 ** attempt)
    return None

def fetch_om_daily(lat, lon, start_date, end_date, max_retries=3):
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&"
        f"daily=precipitation_sum,temperature_2m_mean&timezone=Asia%2FKolkata"
    )
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                daily = data.get("daily", {})
                times = daily.get("time", [])
                p_sum = daily.get("precipitation_sum", [])
                t_mean = daily.get("temperature_2m_mean", [])
                precip = {}
                temp = {}
                for d_str, p, t in zip(times, p_sum, t_mean):
                    ymd = d_str.replace("-", "")
                    if p is not None:
                        precip[ymd] = float(p)
                    if t is not None:
                        temp[ymd] = float(t)
                return {"precip": precip, "temp": temp}
            elif resp.status_code == 429:
                time.sleep(15 * attempt)
            else:
                time.sleep(2)
        except Exception:
            time.sleep(2 ** attempt)
    return None

def compute_seasonal_totals(p_dict, t_dict, crop_yr, season):
    if season == "Kharif":
        start_dt = f"{crop_yr}0601"
        end_dt = f"{crop_yr}1031"
    elif season == "Rabi":
        start_dt = f"{crop_yr}1101"
        end_dt = f"{crop_yr+1}0331"
    elif season == "Zaid":
        start_dt = f"{crop_yr}0301"
        end_dt = f"{crop_yr}0630"
    else:
        return None, None

    expected_days = pd.date_range(start_dt, end_dt).strftime("%Y%m%d").tolist()
    
    # Check if days exist and valid (> -900 for NASA)
    rain_vals = []
    temp_vals = []
    for d in expected_days:
        if d not in p_dict or p_dict[d] is None or p_dict[d] <= -900:
            return np.nan, np.nan
        if d not in t_dict or t_dict[d] is None or t_dict[d] <= -900:
            return np.nan, np.nan
        rain_vals.append(p_dict[d])
        temp_vals.append(t_dict[d])

    return round(sum(rain_vals), 2), round(sum(temp_vals) / len(temp_vals), 2)

def main():
    df = pd.read_csv(WEATHER_CACHE_PATH)
    states = sorted(df["State"].unique())

    # Map sample years
    sample_years = {}
    for s in states:
        s_df = df[df["State"] == s]
        yrs = sorted(s_df["Crop_Year"].unique())
        preferred = [y for y in [2003, 2005, 2007] if y in yrs]
        if len(preferred) == 3:
            sample_years[s] = preferred
        else:
            candidates = [y for y in [2002, 2003, 2004, 2005, 2006, 2007, 2008] if y in yrs]
            if len(candidates) >= 3:
                sample_years[s] = [candidates[0], candidates[len(candidates)//2], candidates[-1]]
            else:
                sample_years[s] = [yrs[0], yrs[len(yrs)//2], yrs[-1]]

    results = []
    print(f"Starting comparison for {len(states)} states...")

    for i, state in enumerate(states, 1):
        lat, lon = STATE_CENTROIDS[state]
        s_yrs = sample_years[state]
        min_yr = min(s_yrs)
        max_yr = max(s_yrs) + 1 # for Rabi
        
        start_ymd = f"{min_yr}0101"
        end_ymd = f"{max_yr}1231"
        start_date = f"{min_yr}-01-01"
        end_date = f"{max_yr}-12-31"

        print(f"[{i}/{len(states)}] {state} ({s_yrs}) at ({lat}, {lon})...", end=" ", flush=True)

        nasa_data = fetch_nasa_daily(lat, lon, start_ymd, end_ymd)
        time.sleep(0.5)
        om_data = fetch_om_daily(lat, lon, start_date, end_date)
        time.sleep(0.5)

        if not nasa_data or not om_data:
            print("FAILED to fetch!")
            continue

        s_df = df[df["State"] == state]
        seasons = sorted(s_df["Season"].unique())

        for yr in s_yrs:
            for season in seasons:
                # Check if this state, year, season was actually present in cache
                if len(s_df[(s_df["Crop_Year"] == yr) & (s_df["Season"] == season)]) == 0:
                    continue

                r_nasa, t_nasa = compute_seasonal_totals(nasa_data["precip"], nasa_data["temp"], yr, season)
                r_om, t_om = compute_seasonal_totals(om_data["precip"], om_data["temp"], yr, season)

                if np.isnan(r_nasa) or np.isnan(r_om):
                    pct_diff = np.nan
                    temp_diff = np.nan
                else:
                    mean_r = (r_nasa + r_om) / 2.0
                    if mean_r == 0:
                        pct_diff = 0.0
                    else:
                        pct_diff = round(abs(r_nasa - r_om) / mean_r * 100.0, 1)
                    temp_diff = round(abs(t_nasa - t_om), 2)

                results.append({
                    "State": state,
                    "Crop_Year": yr,
                    "Season": season,
                    "Rain_NASA": r_nasa,
                    "Rain_OM": r_om,
                    "Rain_Diff_Pct": pct_diff,
                    "Temp_NASA": t_nasa,
                    "Temp_OM": t_om,
                    "Temp_Diff_C": temp_diff,
                })
        print("OK")

    res_df = pd.DataFrame(results)
    res_df.to_csv("scratch/nasa_om_comparison.csv", index=False)
    print("\nSaved comparison to scratch/nasa_om_comparison.csv")

if __name__ == "__main__":
    main()
