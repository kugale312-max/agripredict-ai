"""
backend/prepare_data.py
=======================
Data Preparation Pipeline for AgriPredict AI (Crop Yield Prediction).

This script performs end-to-end data processing on the raw Indian agricultural dataset:
1. Loads 'data/raw/crop_production.csv', strips whitespace, and normalizes casing.
2. Cleans records: drops missing values for Production and Area, filters out Area <= 0,
   and eliminates duplicate entries (with row drop statistics).
3. Computes Yield = Production / Area (tonnes per hectare).
4. Drops crops whose production is measured in nuts or bales (Coconut, Cotton, Jute, Mesta,
   Sannhamp, etc.) with explicit reasons, and retains major crops with at least 500 records.
5. Removes outliers: drops non-positive yields (Yield <= 0) and values exceeding the
   99th percentile within each individual crop.
6. Maps seasons to match the frontend form: Kharif -> Kharif, Rabi -> Rabi, Summer -> Zaid.
   Drops Whole Year, Autumn, and Winter seasons (with drop reports).
7. Downloads seasonal weather metrics (total rainfall in mm and mean temperature in °C)
   from NASA POWER daily API (with Open-Meteo fallback) using state centroids.
   Seasonal windows:
     - Kharif: Jun 1 to Oct 31 of Crop_Year
     - Rabi: Nov 1 of Crop_Year to Mar 31 of Crop_Year + 1
     - Zaid: Mar 1 to Jun 30 of Crop_Year
   Caches all responses in 'data/processed/weather_cache.csv' to ensure each data point
   is fetched exactly once. Uses retries and short polite delays. Halts on failure.
8. Merges dominant soil types from 'backend/soil_map.py' (Alluvial, Black, Red, Sandy, Loamy, Clay).
9. Saves the processed dataset to 'data/processed/crop_yield_clean.csv' with columns:
   ['State', 'Crop_Year', 'Crop', 'Season', 'Rainfall_mm', 'Temperature_C', 'Soil_Type', 'Area_ha', 'Yield_t_per_ha'].
10. Prints a comprehensive data quality and statistical summary.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import requests

# Fix random state for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Paths setup
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "crop_production.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
WEATHER_CACHE_PATH = PROCESSED_DIR / "weather_cache.csv"
CLEAN_DATA_PATH = PROCESSED_DIR / "crop_yield_clean.csv"

# Import dominant soil map
try:
    from backend.soil_map import STATE_SOIL_MAP, get_soil_type
except ImportError:
    from soil_map import STATE_SOIL_MAP, get_soil_type

# -----------------------------------------------------------------------------
# Configuration: State Centroids (Latitude, Longitude) for Weather Data
# -----------------------------------------------------------------------------
STATE_CENTROIDS: Dict[str, Tuple[float, float]] = {
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

# -----------------------------------------------------------------------------
# Configuration: Agricultural Multi-Point Coordinates for Flagged States
# (2 to 3 points averaged across main agricultural plains/valleys rather than
# high-altitude or non-agricultural terrain)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Configuration: Non-Tonne Crops Exclusion Rationale
# -----------------------------------------------------------------------------
NON_TONNE_CROPS_REASONS: Dict[str, str] = {
    "Coconut": "Production measured in thousands of nuts / nuts, not tonnes.",
    "Cotton(lint)": "Production measured in bales of 170 kg each, not tonnes.",
    "Kapas": "Raw unginned cotton measured in bales of 170 kg, not tonnes.",
    "Jute": "Production measured in bales of 180 kg each, not tonnes.",
    "Mesta": "Production measured in bales of 180 kg each, not tonnes.",
    "Jute & mesta": "Combined fiber crop measured in bales of 180 kg, not tonnes.",
    "Sannhamp": "Sunhemp bast fiber measured in bales of 180 kg, not tonnes.",
    "other fibres": "Miscellaneous fiber crops measured in bales, not tonnes.",
}


# =============================================================================
# Step 1: Load and Normalize Casing / Whitespace
# =============================================================================
def step_1_load_data(raw_path: Path) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("STEP 1: Loading raw dataset and normalizing strings")
    print("=" * 70)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at '{raw_path}'. Please ensure data/raw/crop_production.csv exists."
        )

    df = pd.read_csv(raw_path)
    initial_rows, initial_cols = df.shape
    print(f"Loaded '{raw_path.name}': {initial_rows:,} rows and {initial_cols} columns.")

    # Strip whitespace from all string/object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # Normalize Season casing (e.g. 'Kharif', 'Rabi', 'Summer')
    df["Season"] = df["Season"].str.title()
    # Normalize State_Name (standard title case, preserving 'and')
    df["State_Name"] = df["State_Name"].replace({
        "Andaman And Nicobar Islands": "Andaman and Nicobar Islands",
        "Dadra And Nagar Haveli": "Dadra and Nagar Haveli",
    })

    print("Normalized whitespace and casing across all text columns.")
    return df


# =============================================================================
# Step 2: Clean Missing Values, Invalid Area, and Duplicates
# =============================================================================
def step_2_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("STEP 2: Cleaning missing values, non-positive area, and duplicates")
    print("=" * 70)
    initial_count = len(df)

    # 1. Missing Production or Area
    missing_mask = df["Production"].isna() | df["Area"].isna() | (df["Production"] == "")
    # In case Production has non-numeric representations
    df["Production"] = pd.to_numeric(df["Production"], errors="coerce")
    df["Area"] = pd.to_numeric(df["Area"], errors="coerce")
    missing_numeric = df["Production"].isna() | df["Area"].isna()
    dropped_missing = missing_numeric.sum()
    df = df[~missing_numeric].copy()
    print(f"1. Missing Production or Area: dropped {dropped_missing:,} rows.")

    # 2. Area <= 0
    non_positive_area = df["Area"] <= 0
    dropped_area = non_positive_area.sum()
    df = df[~non_positive_area].copy()
    print(f"2. Invalid Area (Area <= 0): dropped {dropped_area:,} rows.")

    # 3. Duplicate rows
    dropped_dupes = df.duplicated().sum()
    df = df.drop_duplicates().copy()
    print(f"3. Exact Duplicates: dropped {dropped_dupes:,} rows.")

    total_cleaned = initial_count - len(df)
    print(f"Total rows removed in Step 2: {total_cleaned:,} (Remaining: {len(df):,})")
    return df


# =============================================================================
# Step 3: Compute Yield (tonnes per hectare)
# =============================================================================
def step_3_compute_yield(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("STEP 3: Computing Crop Yield (Production / Area in tonnes/ha)")
    print("=" * 70)
    df["Yield"] = df["Production"] / df["Area"]
    print(f"Yield computed for {len(df):,} records.")
    return df


# =============================================================================
# Step 4: Exclude Non-Tonne Crops and Filter Major Crops (>= 500 rows)
# =============================================================================
def step_4_filter_crops(df: pd.DataFrame, min_rows: int = 500) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("STEP 4: Excluding non-tonne unit crops & keeping major crops (>= 500 rows)")
    print("=" * 70)

    # 1. Non-tonne crops exclusion
    print("Excluding crops whose production is NOT measured in tonnes:")
    non_tonne_dropped_total = 0
    for crop_name, reason in NON_TONNE_CROPS_REASONS.items():
        crop_match = df["Crop"] == crop_name
        count = crop_match.sum()
        if count > 0:
            print(f"  - '{crop_name}': dropped {count:,} rows. [Reason: {reason}]")
            non_tonne_dropped_total += count
            df = df[~crop_match].copy()

    print(f"Total non-tonne crop rows dropped: {non_tonne_dropped_total:,} rows.")

    # 2. Major crops with at least 500 rows
    crop_counts_before = df["Crop"].value_counts()
    major_crops = crop_counts_before[crop_counts_before >= min_rows].index.tolist()
    minor_crops = crop_counts_before[crop_counts_before < min_rows]

    dropped_minor_rows = minor_crops.sum()
    print(f"\nFiltering for major crops with >= {min_rows} rows:")
    print(f"Dropped {len(minor_crops)} minor crops totaling {dropped_minor_rows:,} rows.")

    df = df[df["Crop"].isin(major_crops)].copy()
    print(f"\nFinal major crops list ({len(major_crops)} crops retained):")
    final_counts = df["Crop"].value_counts()
    for crop_name, count in final_counts.items():
        print(f"  - {crop_name:<26} : {count:,} rows")

    print(f"Remaining rows after Step 4: {len(df):,}")
    return df


# =============================================================================
# Step 5: Remove Yield Outliers
# =============================================================================
def step_5_remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("STEP 5: Removing Yield Outliers (Yield <= 0 and > 99th percentile within crop)")
    print("=" * 70)
    initial_count = len(df)

    # 1. Yield <= 0
    zero_yield = df["Yield"] <= 0
    dropped_zero = zero_yield.sum()
    df = df[~zero_yield].copy()
    print(f"1. Non-positive yield (Yield <= 0): dropped {dropped_zero:,} rows.")

    # 2. Yield > 99th percentile within each crop
    p99_by_crop = df.groupby("Crop")["Yield"].transform(lambda s: s.quantile(0.99))
    above_p99 = df["Yield"] > p99_by_crop
    dropped_p99 = above_p99.sum()
    df = df[~above_p99].copy()
    print(f"2. Above 99th percentile within each crop: dropped {dropped_p99:,} rows.")

    total_outliers = initial_count - len(df)
    print(f"Total outlier rows removed: {total_outliers:,} (Remaining: {len(df):,})")
    return df


# =============================================================================
# Step 6: Map Seasons to Frontend Form and Drop Irrelevant Seasons
# =============================================================================
def step_6_map_seasons(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("STEP 6: Mapping seasons to frontend form (Kharif, Rabi, Zaid)")
    print("=" * 70)
    initial_count = len(df)

    # Target season mapping
    season_mapping = {
        "Kharif": "Kharif",
        "Rabi": "Rabi",
        "Summer": "Zaid",
    }
    dropped_seasons = ["Whole Year", "Autumn", "Winter"]

    print("Season breakdown before filtering:")
    for season, count in df["Season"].value_counts().items():
        print(f"  - {season:<12}: {count:,} rows")

    # Count rows to be dropped
    dropped_mask = df["Season"].isin(dropped_seasons)
    dropped_count = dropped_mask.sum()
    print(f"\nDropping non-target seasons {dropped_seasons}:")
    for s in dropped_seasons:
        cnt = (df["Season"] == s).sum()
        print(f"  - '{s}': dropped {cnt:,} rows.")
    print(f"Total non-target season rows dropped: {dropped_count:,} rows.")

    # Retain target seasons and map Summer -> Zaid
    df = df[df["Season"].isin(season_mapping.keys())].copy()
    df["Season"] = df["Season"].map(season_mapping)

    print("\nSeason distribution after mapping to form:")
    for season, count in df["Season"].value_counts().items():
        print(f"  - {season:<10}: {count:,} rows")

    print(f"Remaining rows after Step 6: {len(df):,}")
    return df


# =============================================================================
# Step 7: Weather Merge via NASA POWER API (with Cache and Retries)
# =============================================================================
def _fetch_single_point_daily_weather(
    state: str,
    lat: float,
    lon: float,
    start_year: int,
    end_year: int,
    max_retries: int = 3,
) -> Dict[str, Dict[str, float]]:
    """
    Fetch daily precipitation (PRECTOTCORR) and temperature (T2M) for a specific
    coordinate from NASA POWER daily API. Falls back to Open-Meteo Archive API if necessary.
    
    Returns:
        Dict with keys:
          - 'precip': { 'YYYYMMDD': mm }
          - 'temp': { 'YYYYMMDD': °C }
    """
    start_date_str = f"{start_year}0101"
    end_date_str = f"{end_year}1231"

    # Primary: NASA POWER Daily API
    nasa_url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters=PRECTOTCORR,T2M&community=AG&longitude={lon}&latitude={lat}&"
        f"start={start_date_str}&end={end_date_str}&format=JSON"
    )

    last_error: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(nasa_url, timeout=40)
            if resp.status_code == 200:
                data = resp.json()
                if "properties" in data and "parameter" in data["properties"]:
                    params = data["properties"]["parameter"]
                    return {
                        "precip": params.get("PRECTOTCORR", {}),
                        "temp": params.get("T2M", {}),
                    }
            elif resp.status_code in [429, 500, 502, 503, 504]:
                wait_sec = 2 ** attempt
                print(f"    [NASA POWER {state} ({lat}, {lon})] HTTP {resp.status_code}. Retrying in {wait_sec}s (attempt {attempt}/{max_retries})...")
                time.sleep(wait_sec)
                last_error = f"NASA HTTP {resp.status_code}: {resp.text[:100]}"
            else:
                last_error = f"NASA HTTP {resp.status_code}: {resp.text[:150]}"
                break
        except requests.RequestException as e:
            wait_sec = 2 ** attempt
            print(f"    [NASA POWER {state} ({lat}, {lon})] Network error: {e}. Retrying in {wait_sec}s (attempt {attempt}/{max_retries})...")
            time.sleep(wait_sec)
            last_error = str(e)

    # Secondary Fallback: Open-Meteo Archive API
    print(f"    [Fallback] Trying Open-Meteo Archive API for {state} ({lat}, {lon})...")
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
                data = resp.json()
                daily = data.get("daily", {})
                times = daily.get("time", [])
                p_sum = daily.get("precipitation_sum", [])
                t_mean = daily.get("temperature_2m_mean", [])
                precip_dict = {}
                temp_dict = {}
                for d_str, p, t in zip(times, p_sum, t_mean):
                    ymd = d_str.replace("-", "")
                    if p is not None:
                        precip_dict[ymd] = float(p)
                    if t is not None:
                        temp_dict[ymd] = float(t)
                return {"precip": precip_dict, "temp": temp_dict}
            elif resp.status_code == 429:
                wait_sec = 60  # Minutely quota exceeded
                print(f"    [Open-Meteo {state}] 429 Rate limit. Waiting {wait_sec}s...")
                time.sleep(wait_sec)
                last_error = "Open-Meteo 429 Rate Limit"
            else:
                last_error = f"Open-Meteo HTTP {resp.status_code}: {resp.text[:100]}"
                break
        except requests.RequestException as e:
            time.sleep(2 ** attempt)
            last_error = str(e)

    # If both APIs failed, STOP and report failure without generating fake data
    raise RuntimeError(
        f"CRITICAL ERROR: Failed to retrieve weather data for State='{state}' (lat={lat}, lon={lon}) "
        f"from both NASA POWER and Open-Meteo.\n"
        f"Last reported error: {last_error}\n"
        f"Stopping execution as instructed: no fake weather values will be generated."
    )


def _fetch_state_daily_weather(
    state: str,
    start_year: int,
    end_year: int,
) -> Dict[str, Dict[str, float]]:
    """
    Fetch daily weather for a state.
    If the state is in STATE_AGRICULTURAL_COORDS, averages the daily time-series
    across the 2-3 agricultural points.
    Otherwise, uses the state centroid coordinate.
    """
    if state in STATE_AGRICULTURAL_COORDS:
        coords = STATE_AGRICULTURAL_COORDS[state]
        print(f"    Using {len(coords)} agricultural points for '{state}'...")
        pt_data = []
        for lat, lon in coords:
            time.sleep(0.5)
            d = _fetch_single_point_daily_weather(state, lat, lon, start_year, end_year)
            pt_data.append(d)

        all_days = set()
        for d in pt_data:
            all_days.update(d["precip"].keys())
            all_days.update(d["temp"].keys())

        avg_precip = {}
        avg_temp = {}
        for day in sorted(all_days):
            p_vals = [
                d["precip"][day]
                for d in pt_data
                if day in d["precip"] and d["precip"][day] is not None and d["precip"][day] >= 0 and d["precip"][day] > -900
            ]
            t_vals = [
                d["temp"][day]
                for d in pt_data
                if day in d["temp"] and d["temp"][day] is not None and d["temp"][day] > -900
            ]
            if len(p_vals) == len(coords):
                avg_precip[day] = float(np.mean(p_vals))
            else:
                avg_precip[day] = np.nan
            if len(t_vals) == len(coords):
                avg_temp[day] = float(np.mean(t_vals))
            else:
                avg_temp[day] = np.nan

        return {"precip": avg_precip, "temp": avg_temp}
    else:
        if state not in STATE_CENTROIDS:
            raise KeyError(f"State '{state}' not found in STATE_CENTROIDS dictionary.")
        lat, lon = STATE_CENTROIDS[state]
        return _fetch_single_point_daily_weather(state, lat, lon, start_year, end_year)


def step_7_weather_merge(df: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("STEP 7: Merging seasonal weather data (NASA POWER / Open-Meteo Archive)")
    print("=" * 70)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing cache if present
    cached_df = pd.DataFrame(columns=["State", "Crop_Year", "Season", "Rainfall_mm", "Temperature_C"])
    if cache_path.exists():
        try:
            cached_df = pd.read_csv(cache_path)
            cached_df["Crop_Year"] = cached_df["Crop_Year"].astype(int)
            cached_df["State"] = cached_df["State"].astype(str)
            cached_df["Season"] = cached_df["Season"].astype(str)
            print(f"Loaded existing weather cache: {len(cached_df):,} cached seasonal records found.")
        except Exception as e:
            print(f"Warning: Could not read existing weather cache ({e}). Starting fresh.")
            cached_df = pd.DataFrame(columns=["State", "Crop_Year", "Season", "Rainfall_mm", "Temperature_C"])

    # Determine unique (State, Crop_Year, Season) needed
    needed_combos = (
        df[["State_Name", "Crop_Year", "Season"]]
        .drop_duplicates()
        .rename(columns={"State_Name": "State"})
    )
    print(f"Total unique (State, Crop_Year, Season) combinations needed: {len(needed_combos):,}")

    # Identify missing entries
    if not cached_df.empty:
        merged_check = needed_combos.merge(
            cached_df[["State", "Crop_Year", "Season", "Rainfall_mm"]],
            on=["State", "Crop_Year", "Season"],
            how="left",
        )
        missing_combos = merged_check[merged_check["Rainfall_mm"].isna()][["State", "Crop_Year", "Season"]]
    else:
        missing_combos = needed_combos.copy()

    missing_states = sorted(missing_combos["State"].unique())
    print(f"Combinations already in cache: {len(needed_combos) - len(missing_combos):,}")
    print(f"Combinations to fetch from API: {len(missing_combos):,} (across {len(missing_states)} states)")

    new_cache_rows: List[Dict] = []

    if missing_states:
        for idx, state in enumerate(missing_states, 1):
            state_needed = missing_combos[missing_combos["State"] == state]
            min_yr = int(state_needed["Crop_Year"].min())
            max_yr = int(state_needed["Crop_Year"].max()) + 1  # +1 for Rabi season into next calendar year

            print(f"[{idx}/{len(missing_states)}] Fetching weather for '{state}' ({min_yr}-{max_yr})...")
            t0 = time.time()
            daily_weather = _fetch_state_daily_weather(state, min_yr, max_yr)
            elapsed = time.time() - t0

            p_dict = daily_weather["precip"]
            t_dict = daily_weather["temp"]

            # Compute seasonal windows for each required (Crop_Year, Season)
            for _, row in state_needed.iterrows():
                crop_yr = int(row["Crop_Year"])
                season = str(row["Season"])

                if season == "Kharif":
                    start_ymd = f"{crop_yr}0601"
                    end_ymd = f"{crop_yr}1031"
                elif season == "Rabi":
                    start_ymd = f"{crop_yr}1101"
                    end_ymd = f"{crop_yr + 1}0331"
                elif season == "Zaid":
                    start_ymd = f"{crop_yr}0301"
                    end_ymd = f"{crop_yr}0630"
                else:
                    continue

                # Generate full date range to verify no days are omitted
                expected_days = pd.date_range(start_ymd, end_ymd).strftime("%Y%m%d").tolist()

                # Sum precipitation and average temperature.
                # Missing values (-999, None, or < 0 for precip) MUST STAY MISSING (np.nan),
                # never clipped or silently filled with 0.
                rain_vals = []
                temp_vals = []
                has_missing = False

                for d in expected_days:
                    p_val = p_dict.get(d)
                    t_val = t_dict.get(d)

                    if p_val is None or pd.isna(p_val) or p_val < 0 or p_val <= -900:
                        has_missing = True
                        break
                    if t_val is None or pd.isna(t_val) or t_val <= -900:
                        has_missing = True
                        break

                    rain_vals.append(p_val)
                    temp_vals.append(t_val)

                if has_missing or not rain_vals or not temp_vals:
                    rain_sum = np.nan
                    temp_mean = np.nan
                else:
                    rain_sum = round(float(np.sum(rain_vals)), 2)
                    temp_mean = round(float(np.mean(temp_vals)), 2)

                new_cache_rows.append({
                    "State": state,
                    "Crop_Year": crop_yr,
                    "Season": season,
                    "Rainfall_mm": rain_sum,
                    "Temperature_C": temp_mean,
                })

            print(f"    -> Done in {elapsed:.1f}s ({len(state_needed)} seasonal records calculated).")
            time.sleep(1.0)

            if new_cache_rows:
                updated_cache = pd.concat([cached_df, pd.DataFrame(new_cache_rows)], ignore_index=True)
                updated_cache = updated_cache.drop_duplicates(subset=["State", "Crop_Year", "Season"])
                updated_cache.to_csv(cache_path, index=False)

    # Final combined cache
    if new_cache_rows:
        cached_df = pd.concat([cached_df, pd.DataFrame(new_cache_rows)], ignore_index=True)
        cached_df = cached_df.drop_duplicates(subset=["State", "Crop_Year", "Season"])

    # Clean cache of missing or 0.0 mm rainfall records
    cache_invalid = (
        cached_df["Rainfall_mm"].isna()
        | cached_df["Temperature_C"].isna()
        | (cached_df["Rainfall_mm"] == 0.0)
    )
    if cache_invalid.sum() > 0:
        print(f"Cleaning {cache_invalid.sum()} records from weather cache where rainfall is missing or exactly 0.0 mm...")
        cached_df = cached_df[~cache_invalid].copy()

    cached_df.to_csv(cache_path, index=False)
    print(f"Weather cache successfully verified at '{cache_path.name}' ({len(cached_df):,} total records).")

    # Merge weather into main DataFrame
    df = df.rename(columns={"State_Name": "State"})
    df["Crop_Year"] = df["Crop_Year"].astype(int)
    initial_rows = len(df)
    df = df.merge(cached_df, on=["State", "Crop_Year", "Season"], how="left")

    # Step 5: Drop records whose weather is still missing or exactly 0.0 mm for a whole season
    invalid_mask = (
        df["Rainfall_mm"].isna()
        | df["Temperature_C"].isna()
        | (df["Rainfall_mm"] == 0.0)
    )
    dropped_weather_rows = invalid_mask.sum()
    if dropped_weather_rows > 0:
        print(f"\nStep 5: Dropping {dropped_weather_rows:,} crop records whose seasonal weather was missing or exactly 0.0 mm:")
        dropped_combos = (
            df[invalid_mask][["State", "Crop_Year", "Season"]]
            .drop_duplicates()
            .to_dict(orient="records")
        )
        for combo in dropped_combos:
            print(f"  - Dropped whole-season weather: {combo['State']} {combo['Crop_Year']} {combo['Season']}")
        df = df[~invalid_mask].copy()

    print(f"Successfully merged weather variables into {len(df):,} records (retained {len(df):,}/{initial_rows:,}).")
    return df


# =============================================================================
# Step 8: Dominant Soil Type Mapping
# =============================================================================
def step_8_soil_map(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("STEP 8: Mapping dominant soil type from backend/soil_map.py")
    print("=" * 70)
    # Check all states in df exist in STATE_SOIL_MAP
    unmapped_states = set(df["State"].unique()) - set(STATE_SOIL_MAP.keys())
    if unmapped_states:
        raise KeyError(f"The following states are missing from STATE_SOIL_MAP: {unmapped_states}")

    df["Soil_Type"] = df["State"].map(STATE_SOIL_MAP)

    print("Soil Type breakdown in the cleaned dataset:")
    for soil, count in df["Soil_Type"].value_counts().items():
        print(f"  - {soil:<12}: {count:,} rows ({count / len(df) * 100:.1f}%)")

    return df


# =============================================================================
# Step 9: Save Cleaned Dataset
# =============================================================================
def step_9_save_clean_data(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print(f"STEP 9: Saving processed dataset to '{output_path.name}'")
    print("=" * 70)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Rename and order columns exactly as requested:
    # State, Crop_Year, Crop, Season, Rainfall_mm, Temperature_C, Soil_Type, Area_ha, Yield_t_per_ha
    df = df.rename(columns={
        "Area": "Area_ha",
        "Yield": "Yield_t_per_ha",
    })

    final_columns = [
        "State",
        "Crop_Year",
        "Crop",
        "Season",
        "Rainfall_mm",
        "Temperature_C",
        "Soil_Type",
        "Area_ha",
        "Yield_t_per_ha",
    ]

    final_df = df[final_columns].copy()
    final_df.to_csv(output_path, index=False)
    print(f"Successfully saved clean dataset to '{output_path}' ({len(final_df):,} rows, {len(final_columns)} columns).")
    return final_df


# =============================================================================
# Step 10: Print Comprehensive Summary
# =============================================================================
def step_10_print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("STEP 10: DATA PREPARATION & QUALITY SUMMARY")
    print("=" * 70)

    # 1. Total row count and columns
    print(f"1. Total Dataset Records : {len(df):,}")
    print(f"   Total Feature Columns : {df.shape[1]} -> {list(df.columns)}")

    # 2. Missing values per column
    print("\n2. Missing Values per Column:")
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        print(f"   - {col:<16}: {count} missing values")

    # 3. Row count per season
    print("\n3. Row Count per Season:")
    for season, count in df["Season"].value_counts().items():
        print(f"   - {season:<10}: {count:,} records ({count / len(df) * 100:.1f}%)")

    # 4. Row count per crop
    print(f"\n4. Row Count per Crop (Total {df['Crop'].nunique()} Major Crops):")
    for crop, count in df["Crop"].value_counts().items():
        print(f"   - {crop:<26}: {count:,} records ({count / len(df) * 100:.2f}%)")

    # 5. Yield Statistics
    print("\n5. Yield Statistics (tonnes / hectare):")
    desc = df["Yield_t_per_ha"].describe()
    print(f"   - Count : {desc['count']:,.0f}")
    print(f"   - Mean  : {desc['mean']:.4f} t/ha")
    print(f"   - Std   : {desc['std']:.4f} t/ha")
    print(f"   - Min   : {desc['min']:.4f} t/ha")
    print(f"   - 25%   : {desc['25%']:.4f} t/ha")
    print(f"   - 50%   : {desc['50%']:.4f} t/ha (Median)")
    print(f"   - 75%   : {desc['75%']:.4f} t/ha")
    print(f"   - Max   : {desc['max']:.4f} t/ha")

    # 6. Environmental metrics summary
    print("\n6. Environmental Metrics Overview:")
    print(f"   - Rainfall (mm)     : Min={df['Rainfall_mm'].min():.1f}, Mean={df['Rainfall_mm'].mean():.1f}, Max={df['Rainfall_mm'].max():.1f}")
    print(f"   - Temperature (C)   : Min={df['Temperature_C'].min():.1f}, Mean={df['Temperature_C'].mean():.1f}, Max={df['Temperature_C'].max():.1f}")

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")


def print_per_state_seasonal_table(df: pd.DataFrame) -> None:
    """
    Print mean seasonal rainfall and mean temperature per state for each season,
    counting each State-Year-Season once (not once per crop row).
    """
    print("\n" + "=" * 70)
    print("PER-STATE MEAN SEASONAL WEATHER TABLE (Unique State-Year-Season)")
    print("=" * 70)
    weather_unique = df[["State", "Crop_Year", "Season", "Rainfall_mm", "Temperature_C"]].drop_duplicates()
    seasonal_stats = weather_unique.groupby(["State", "Season"])[["Rainfall_mm", "Temperature_C"]].agg(["mean", "count"]).round(2)
    pd.set_option("display.max_rows", 150)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 1000)
    print(seasonal_stats)


# =============================================================================
# Main Pipeline Runner
# =============================================================================
def main():
    start_total = time.time()
    print(f"Starting AgriPredict AI Data Preparation Pipeline [Random Seed = {RANDOM_STATE}]")
    print(f"Working Directory: {PROJECT_ROOT}")

    # Steps 1 to 10
    df = step_1_load_data(RAW_DATA_PATH)
    df = step_2_clean_data(df)
    df = step_3_compute_yield(df)
    df = step_4_filter_crops(df, min_rows=500)
    df = step_5_remove_outliers(df)
    df = step_6_map_seasons(df)
    df = step_7_weather_merge(df, WEATHER_CACHE_PATH)
    df = step_8_soil_map(df)
    final_df = step_9_save_clean_data(df, CLEAN_DATA_PATH)
    step_10_print_summary(final_df)
    print_per_state_seasonal_table(final_df)

    total_duration = time.time() - start_total
    print(f"\nTotal Pipeline Runtime: {total_duration:.1f} seconds.")


if __name__ == "__main__":
    main()

