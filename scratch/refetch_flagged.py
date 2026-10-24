import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import requests

PROCESSED_DIR = Path("data/processed")
WEATHER_CACHE_PATH = PROCESSED_DIR / "weather_cache.csv"

# Agricultural Multi-Point Coordinates for Flagged States
STATE_AGRICULTURAL_COORDS: Dict[str, List[Tuple[float, float]]] = {
    # Jammu and Kashmir: Kashmir Valley and Jammu agricultural plains (not high Himalayas / Ladakh)
    "Jammu and Kashmir": [
        (34.0837, 74.7973),  # Srinagar (Kashmir Valley - major rice, maize, apple)
        (32.7266, 74.8570),  # Jammu Plains (RS Pura / Samba belt - major wheat, basmati paddy)
        (33.7311, 75.1487),  # Anantnag (South Kashmir agricultural plain)
    ],
    # Assam: Brahmaputra agricultural valley (not Karbi Anglong / Dima Hasao hill plateau)
    "Assam": [
        (26.1445, 91.7362),  # Lower Brahmaputra Valley (Guwahati / Kamrup plains)
        (26.7509, 94.2037),  # Upper Brahmaputra Valley (Jorhat / Golaghat tea and paddy belt)
        (26.3465, 92.6840),  # Central Brahmaputra agricultural plains (Nagaon)
    ],
    # Mizoram: Cultivated river valleys and settled farming areas (not steep barren ridge tops)
    "Mizoram": [
        (23.7271, 92.7176),  # Central agricultural valleys (Aizawl district)
        (23.4757, 93.3265),  # Champhai Valley (the primary rice bowl of Mizoram)
        (24.2246, 92.6782),  # Kolasib (northern fertile agricultural belt)
    ],
    # Tripura: Fertile western plains and river valleys
    "Tripura": [
        (23.8315, 91.2868),  # West Tripura / Agartala fertile plains
        (23.5342, 91.4883),  # South Tripura / Gomati agricultural plains (Udaipur)
        (24.3793, 92.1678),  # North Tripura / Dharmanagar valley
    ],
    # Jharkhand: Major agricultural plateau and river basin belts
    "Jharkhand": [
        (23.3441, 85.3096),  # Ranchi Plateau (central agricultural zone)
        (24.2676, 87.2498),  # Santhal Parganas / Dumka (northeastern paddy belt)
        (23.9925, 85.3637),  # North Chota Nagpur / Hazaribagh agricultural area
    ],
    # Himachal Pradesh: Lower agricultural valleys (not high mountain ridges)
    "Himachal Pradesh": [
        (32.0998, 76.2691),  # Kangra Valley (major rice and wheat bowl of HP)
        (31.7087, 76.9320),  # Balh Valley / Mandi (fertile agricultural valley)
        (31.4685, 76.2708),  # Una / Foothills plains (low-altitude subtropical belt)
    ],
    # Uttarakhand: Terai agricultural belt and Doon / Ganges valley
    "Uttarakhand": [
        (28.9800, 79.4000),  # Udham Singh Nagar (Terai breadbasket of Uttarakhand)
        (30.3165, 78.0322),  # Dehradun (Doon Valley agricultural belt)
        (29.9457, 78.1642),  # Haridwar (Ganges agricultural plains)
    ],
    # Sikkim: Inhabited agricultural valleys and terrace farming belts (not alpine peaks)
    "Sikkim": [
        (27.3389, 88.6065),  # East Sikkim (Gangtok / Ranipool farming belt)
        (27.1667, 88.3500),  # South Sikkim (Namchi agricultural zone)
        (27.2833, 88.2500),  # West Sikkim (Geyzing terrace farming belt)
    ],
    # Arunachal Pradesh: Sub-Himalayan agricultural valleys and foothill plains
    "Arunachal Pradesh": [
        (28.0667, 95.3333),  # Pasighat / East Siang (fertile foothills plain)
        (27.9167, 96.1667),  # Tezu / Lohit Valley agricultural plain
        (27.1400, 93.7500),  # Papum Pare / Doimukh plains
    ],
    # Nagaland: Cultivated valleys and plains
    "Nagaland": [
        (25.9094, 93.7266),  # Dimapur Plains (major agricultural and rice bowl hub)
        (25.6751, 94.1086),  # Kohima agricultural valley
        (26.3256, 94.5152),  # Mokokchung agricultural belt
    ],
    # Goa: Coastal agricultural lowlands (not Western Ghats mountain ridges)
    "Goa": [
        (15.4989, 73.8278),  # North Goa (Tiswadi / Bardez coastal plains)
        (15.2832, 73.9862),  # South Goa (Salcete / Quepem coastal plains)
    ],
}

def fetch_single_point_daily(state: str, lat: float, lon: float, start_year: int, end_year: int, max_retries: int = 3):
    start_date_str = f"{start_year}0101"
    end_date_str = f"{end_year}1231"
    nasa_url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters=PRECTOTCORR,T2M&community=AG&longitude={lon}&latitude={lat}&"
        f"start={start_date_str}&end={end_date_str}&format=JSON"
    )
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(nasa_url, timeout=40)
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

    # Fallback to Open-Meteo
    om_start = f"{start_year}-01-01"
    om_end = f"{end_year}-12-31"
    om_url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date={om_start}&end_date={om_end}&"
        f"daily=precipitation_sum,temperature_2m_mean&timezone=Asia%2FKolkata"
    )
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(om_url, timeout=30)
            if resp.status_code == 200:
                daily = resp.json().get("daily", {})
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

    raise RuntimeError(f"Could not fetch data for {state} ({lat}, {lon})")

def fetch_state_daily(state: str, start_year: int, end_year: int):
    points = STATE_AGRICULTURAL_COORDS[state]
    print(f"  Fetching {len(points)} agricultural points for '{state}'...")
    pt_data = []
    for lat, lon in points:
        time.sleep(0.5)
        d = fetch_single_point_daily(state, lat, lon, start_year, end_year)
        pt_data.append(d)

    all_days = set()
    for d in pt_data:
        all_days.update(d["precip"].keys())
        all_days.update(d["temp"].keys())

    avg_precip = {}
    avg_temp = {}
    for day in sorted(all_days):
        p_vals = [d["precip"][day] for d in pt_data if day in d["precip"] and d["precip"][day] is not None and d["precip"][day] >= 0 and d["precip"][day] > -900]
        t_vals = [d["temp"][day] for d in pt_data if day in d["temp"] and d["temp"][day] is not None and d["temp"][day] > -900]
        if len(p_vals) == len(points):
            avg_precip[day] = float(np.mean(p_vals))
        else:
            avg_precip[day] = np.nan
        if len(t_vals) == len(points):
            avg_temp[day] = float(np.mean(t_vals))
        else:
            avg_temp[day] = np.nan

    return {"precip": avg_precip, "temp": avg_temp}

def compute_seasonal_stats(p_dict, t_dict, crop_yr, season):
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
        return np.nan, np.nan

    expected_days = pd.date_range(start_dt, end_dt).strftime("%Y%m%d").tolist()
    rain_vals = []
    temp_vals = []
    for d in expected_days:
        p_val = p_dict.get(d)
        t_val = t_dict.get(d)
        if p_val is None or pd.isna(p_val) or p_val < 0:
            return np.nan, np.nan
        if t_val is None or pd.isna(t_val) or t_val <= -900:
            return np.nan, np.nan
        rain_vals.append(p_val)
        temp_vals.append(t_val)

    return round(float(np.sum(rain_vals)), 2), round(float(np.mean(temp_vals)), 2)

def main():
    cache_df = pd.read_csv(WEATHER_CACHE_PATH)
    flagged_states = list(STATE_AGRICULTURAL_COORDS.keys())
    print(f"Re-fetching {len(flagged_states)} flagged states with multi-point agricultural coordinates...")

    # Filter cache to remove flagged states
    preserved_cache = cache_df[~cache_df["State"].isin(flagged_states)].copy()
    print(f"Preserving {len(preserved_cache)} rows for non-flagged states.")

    new_rows = []
    for idx, state in enumerate(flagged_states, 1):
        state_rows = cache_df[cache_df["State"] == state]
        min_yr = int(state_rows["Crop_Year"].min())
        max_yr = int(state_rows["Crop_Year"].max()) + 1
        print(f"\n[{idx}/{len(flagged_states)}] Processing {state} ({min_yr}-{max_yr}, {len(state_rows)} seasons)...")

        t0 = time.time()
        daily = fetch_state_daily(state, min_yr, max_yr)
        p_dict = daily["precip"]
        t_dict = daily["temp"]

        for _, row in state_rows.iterrows():
            crop_yr = int(row["Crop_Year"])
            season = str(row["Season"])
            r_sum, t_mean = compute_seasonal_stats(p_dict, t_dict, crop_yr, season)
            new_rows.append({
                "State": state,
                "Crop_Year": crop_yr,
                "Season": season,
                "Rainfall_mm": r_sum,
                "Temperature_C": t_mean,
            })
        print(f"  Done in {time.time()-t0:.1f}s.")

    new_df = pd.DataFrame(new_rows)
    updated_cache = pd.concat([preserved_cache, new_df], ignore_index=True)

    # Step 5: Drop records whose weather is still missing or exactly 0 mm for a whole season
    invalid_mask = (
        updated_cache["Rainfall_mm"].isna()
        | updated_cache["Temperature_C"].isna()
        | (updated_cache["Rainfall_mm"] == 0.0)
    )
    dropped_cache = updated_cache[invalid_mask]
    if len(dropped_cache) > 0:
        print(f"\nDropping {len(dropped_cache)} records with missing or 0.0 mm seasonal rainfall from weather cache:")
        for _, r in dropped_cache.iterrows():
            print(f"  - Dropped: {r['State']} {r['Crop_Year']} {r['Season']} (Rain={r['Rainfall_mm']}, Temp={r['Temperature_C']})")
        updated_cache = updated_cache[~invalid_mask].copy()

    updated_cache = updated_cache.sort_values(["State", "Crop_Year", "Season"]).reset_index(drop=True)
    updated_cache.to_csv(WEATHER_CACHE_PATH, index=False)
    print(f"\nSuccessfully updated {WEATHER_CACHE_PATH} ({len(updated_cache)} total records).")

if __name__ == "__main__":
    main()
