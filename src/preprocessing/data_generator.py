"""
Phase 2 – Data Generation / Collection
Generates a synthetic but statistically realistic Indian real estate dataset
with 20,000+ records covering all required features.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import sys

# ── Allow running as a standalone script ─────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    RAW_DATA_PATH, RANDOM_STATE
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Seed ─────────────────────────────────────────────────────────────────────
np.random.seed(RANDOM_STATE)
N = 20_000

# ── Reference Data ────────────────────────────────────────────────────────────
CITIES = {
    "Mumbai":    {"base": 18_000, "growth": 0.085, "density": 21_000},
    "Delhi":     {"base": 12_000, "growth": 0.078, "density": 11_000},
    "Bangalore": {"base": 10_000, "growth": 0.092, "density":  4_400},
    "Hyderabad": {"base":  8_500, "growth": 0.088, "density":  3_100},
    "Chennai":   {"base":  8_000, "growth": 0.075, "density":  7_100},
    "Pune":      {"base":  7_500, "growth": 0.082, "density":  5_800},
    "Kolkata":   {"base":  6_000, "growth": 0.065, "density": 24_000},
    "Ahmedabad": {"base":  5_500, "growth": 0.072, "density":  6_300},
    "Jaipur":    {"base":  5_000, "growth": 0.068, "density":  6_700},
    "Noida":     {"base":  7_000, "growth": 0.080, "density":  5_500},
}

LOCALITIES = {
    "Mumbai":    ["Bandra", "Andheri", "Worli", "Powai", "Juhu", "Borivali", "Thane", "Navi Mumbai"],
    "Delhi":     ["Dwarka", "Rohini", "Saket", "Vasant Kunj", "Lajpat Nagar", "Pitampura"],
    "Bangalore": ["Koramangala", "Whitefield", "Indiranagar", "HSR Layout", "Electronic City"],
    "Hyderabad": ["Gachibowli", "HITEC City", "Banjara Hills", "Jubilee Hills", "Madhapur"],
    "Chennai":   ["Anna Nagar", "Velachery", "OMR", "Porur", "Adyar", "Nungambakkam"],
    "Pune":      ["Hinjewadi", "Kothrud", "Wakad", "Viman Nagar", "Baner", "Hadapsar"],
    "Kolkata":   ["Salt Lake", "New Town", "Behala", "Howrah", "Rajarhat"],
    "Ahmedabad": ["SG Highway", "Bopal", "Prahlad Nagar", "Satellite", "Maninagar"],
    "Jaipur":    ["Malviya Nagar", "C-Scheme", "Vaishali Nagar", "Mansarovar", "Jagatpura"],
    "Noida":     ["Sector 18", "Sector 62", "Sector 137", "Greater Noida", "Sector 50"],
}

PROPERTY_TYPES = ["Apartment", "Villa", "Independent House", "Plot", "Penthouse"]
PT_WEIGHTS     = [0.55,        0.15,    0.15,                0.08,   0.07]

FURNISHED = ["Fully Furnished", "Semi Furnished", "Unfurnished"]
FURNISHED_WEIGHTS = [0.35, 0.40, 0.25]

AREAS = ["North", "South", "East", "West", "Central"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def _city_series(n: int) -> np.ndarray:
    cities = list(CITIES.keys())
    weights = np.array([1 / len(cities)] * len(cities))
    # slightly skew toward major metros
    weights[0] += 0.10   # Mumbai
    weights[1] += 0.08   # Delhi
    weights[2] += 0.07   # Bangalore
    weights /= weights.sum()
    return np.random.choice(cities, size=n, p=weights)


def _sqft_for_bhk(bhk: np.ndarray) -> np.ndarray:
    base = 400 + bhk * 250
    noise = np.random.normal(0, 80, len(bhk))
    return np.clip(base + noise, 200, 10_000).astype(int)


def _price(city_arr, sqft_arr, prop_type_arr, furnished_arr,
           floor_arr, age_arr, metro_dist_arr,
           crime_arr, infra_arr):
    prices = []
    for i in range(len(city_arr)):
        city = city_arr[i]
        base_per_sqft = CITIES[city]["base"]

        # Property type premium
        pt_factor = {
            "Apartment": 1.0, "Villa": 1.6,
            "Independent House": 1.3, "Plot": 0.7, "Penthouse": 2.0
        }[prop_type_arr[i]]

        # Furnished premium
        f_factor = {
            "Fully Furnished": 1.12,
            "Semi Furnished": 1.06,
            "Unfurnished": 1.0
        }[furnished_arr[i]]

        # Floor premium (higher floors command premium)
        floor_factor = 1 + (floor_arr[i] / 100)

        # Age discount
        age_factor = max(0.70, 1.0 - age_arr[i] * 0.015)

        # Metro proximity premium (closer = costlier)
        metro_factor = 1 + max(0, (5 - metro_dist_arr[i]) / 20)

        # Crime discount
        crime_factor = 1 - (crime_arr[i] / 500)

        # Infra premium
        infra_factor = 1 + (infra_arr[i] / 200)

        per_sqft = (base_per_sqft * pt_factor * f_factor * floor_factor
                    * age_factor * metro_factor * crime_factor * infra_factor)

        total = per_sqft * sqft_arr[i]
        # Add ±8% noise
        total *= np.random.uniform(0.92, 1.08)
        prices.append(round(total, -3))   # round to nearest thousand

    return np.array(prices, dtype=float)


# ── Generator ─────────────────────────────────────────────────────────────────
def generate_dataset(n: int = N) -> pd.DataFrame:
    log.info("Generating %d synthetic real estate records …", n)

    # Core
    cities     = _city_series(n)
    prop_types = np.random.choice(PROPERTY_TYPES, n, p=PT_WEIGHTS)
    bhk        = np.random.choice([1, 2, 3, 4, 5], n, p=[0.08, 0.30, 0.38, 0.18, 0.06])
    bathrooms  = np.clip(bhk + np.random.choice([-1, 0, 1], n, p=[0.15, 0.70, 0.15]), 1, 6)
    balcony    = np.random.choice([0, 1, 2, 3], n, p=[0.10, 0.40, 0.35, 0.15])
    area_sqft  = _sqft_for_bhk(bhk)

    # Location
    localities = np.array([
        np.random.choice(LOCALITIES[c]) for c in cities
    ])
    areas      = np.random.choice(AREAS, n)
    pincodes   = np.random.randint(100_000, 999_999, n)

    # Building details
    total_floors = np.random.choice(range(1, 41), n,
                                    p=np.array([1]*40) / 40)
    floor_number = np.array([
        np.random.randint(0, tf + 1) for tf in total_floors
    ])
    prop_age   = np.clip(np.random.exponential(7, n).astype(int), 0, 50)
    furnished  = np.random.choice(FURNISHED, n, p=FURNISHED_WEIGHTS)
    parking    = np.random.choice(["Yes", "No"], n, p=[0.65, 0.35])

    # Distances (km)
    hospital_dist  = np.round(np.random.exponential(3, n), 1)
    school_dist    = np.round(np.random.exponential(2, n), 1)
    metro_dist     = np.round(np.random.exponential(4, n), 1)
    mall_dist      = np.round(np.random.exponential(5, n), 1)
    airport_dist   = np.round(np.abs(np.random.normal(25, 12, n)), 1)

    # Socioeconomic
    pop_density    = np.array([CITIES[c]["density"] +
                                np.random.randint(-2000, 2000) for c in cities])
    crime_index    = np.clip(np.random.normal(45, 18, n), 0, 100).round(1)
    infra_score    = np.clip(np.random.normal(55, 15, n), 0, 100).round(1)

    # Target
    price = _price(cities, area_sqft, prop_types, furnished,
                   floor_number, prop_age, metro_dist,
                   crime_index, infra_score)

    df = pd.DataFrame({
        "property_id":                np.arange(1, n + 1),
        "property_type":              prop_types,
        "bhk":                        bhk,
        "bathrooms":                  bathrooms,
        "balcony":                    balcony,
        "floor_number":               floor_number,
        "total_floors":               total_floors,
        "area_sqft":                  area_sqft,
        "property_age":               prop_age,
        "furnished_status":           furnished,
        "parking_availability":       parking,
        "city":                       cities,
        "area":                       areas,
        "locality":                   localities,
        "pincode":                    pincodes,
        "hospital_distance_km":       np.clip(hospital_dist, 0.1, 30),
        "school_distance_km":         np.clip(school_dist, 0.1, 20),
        "metro_distance_km":          np.clip(metro_dist, 0.1, 40),
        "mall_distance_km":           np.clip(mall_dist, 0.1, 30),
        "airport_distance_km":        np.clip(airport_dist, 1, 80),
        "population_density":         np.clip(pop_density, 1000, 50_000),
        "crime_index":                crime_index,
        "infrastructure_growth_score": infra_score,
        "price":                      price,
    })

    # Inject ~3 % missing values in selected columns for realism
    for col in ["balcony", "hospital_distance_km", "crime_index",
                "infrastructure_growth_score", "metro_distance_km"]:
        mask = np.random.rand(n) < 0.03
        df.loc[mask, col] = np.nan

    # Inject a small number of duplicates (~0.5 %)
    dupe_idx = np.random.choice(df.index, size=int(n * 0.005), replace=False)
    df = pd.concat([df, df.loc[dupe_idx]], ignore_index=True)

    log.info("Dataset shape: %s", df.shape)
    return df


def save_dataset(df: pd.DataFrame, path: Path = RAW_DATA_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Dataset saved → %s", path)


if __name__ == "__main__":
    df = generate_dataset()
    save_dataset(df)
    log.info("Sample:\n%s", df.head(3).to_string())
