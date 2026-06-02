"""
distance_matrix.py — Per-City Distance Matrix Pipeline
========================================================

Usage:
    python3 distance_matrix.py               # process ALL cities in the database
    python3 distance_matrix.py "Mumbai"      # process ONE specific city

For each city in places_database.json this script builds (or incrementally
updates) a NxN driving-distance matrix using the Google Distance Matrix API,
where N = number of places in that city.

Output (one pair of files per city):
    data/matrices/<city_slug>_matrix.npy   — NxN numpy float64 array (metres)
    data/matrices/<city_slug>_meta.json    — index ↔ place metadata

Matrix values:
    0.0   → same place (diagonal)
    > 0   → driving distance in metres
   -1.0   → no driving route found

Smart incremental updates (per city):
    • City places unchanged → skips that city entirely
    • New places added      → copies existing pairs, API only for new combinations
    • Places removed        → shrinks matrix, no API calls needed

API batching:
    a 10 origins × 10 destinations per request (Distance Matrix API limit).
    A 50-place city = ceil(50/10)² = 25 chunks = 25 API requests.
"""

import os
import sys
import json
import time
import math
import requests
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

# ─── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR    = Path(__file__).resolve().parent / "data"
DB_PATH     = DATA_DIR / "places_database.json"
MATRICES_DIR = DATA_DIR / "matrices"

# ─── Constants ────────────────────────────────────────────────────────────────

DISTANCE_MATRIX_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"
CHUNK_SIZE    = 10    # Standard tier: max 100 elements per request (10×10)
                      # Upgrade to 25 only if on a premium/paid API plan
REQUEST_DELAY = 0.5   # Seconds between API requests
NO_ROUTE      = -1.0  # No driving route found


# ─── File path helpers ────────────────────────────────────────────────────────

def matrix_path(city_slug: str) -> Path:
    return MATRICES_DIR / f"{city_slug}_matrix.npy"

def meta_path(city_slug: str) -> Path:
    return MATRICES_DIR / f"{city_slug}_meta.json"


# ─── DB helpers ───────────────────────────────────────────────────────────────

def load_db() -> dict[str, list[dict]]:
    """
    Reads places_database.json and returns a dict of
    { city_slug → list of place dicts }.
    """
    if not DB_PATH.exists():
        print(f"❌  Database not found at {DB_PATH}")
        print("    Run:  python3 main.py \"<city name>\"  first.")
        sys.exit(1)

    with open(DB_PATH, "r", encoding="utf-8") as f:
        db = json.load(f)

    cities: dict[str, list[dict]] = {}
    for city in db.get("cities", []):
        slug = city["city_slug"]
        places = []
        for place in city.get("places", []):
            places.append({
                "place_id":  place["place_id"],
                "name":      place["name"],
                "city":      city["city_name"],
                "city_slug": slug,
                "lat":       place["latitude"],
                "lng":       place["longitude"],
                "category":  place.get("category", ""),
            })
        cities[slug] = places

    return cities


# ─── Per-city matrix I/O ──────────────────────────────────────────────────────

def load_city_matrix(city_slug: str) -> tuple[list | None, np.ndarray | None]:
    """
    Loads the existing matrix and meta for a city.
    Returns (place_ids_list, matrix) or (None, None) if not yet computed.
    """
    mp = matrix_path(city_slug)
    mm = meta_path(city_slug)

    if not mp.exists() or not mm.exists():
        return None, None

    with open(mm, "r", encoding="utf-8") as f:
        meta = json.load(f)

    matrix = np.load(str(mp))
    return meta["place_ids"], matrix


def save_city_matrix(city_slug: str, matrix: np.ndarray, places: list[dict]) -> None:
    """Saves a city's matrix (.npy) and metadata (.json)."""
    MATRICES_DIR.mkdir(parents=True, exist_ok=True)

    np.save(str(matrix_path(city_slug)), matrix)

    meta = {
        "city_slug":    city_slug,
        "city_name":    places[0]["city"] if places else city_slug,
        "computed_at":  datetime.now(timezone.utc).isoformat(),
        "total_places": len(places),
        "mode":         "driving",
        "unit":         "meters",
        "no_route_value": NO_ROUTE,
        "place_ids":    [p["place_id"]  for p in places],
        "place_names":  [p["name"]      for p in places],
        "place_coords": [[p["lat"], p["lng"]] for p in places],
        "categories":   [p["category"]  for p in places],
    }

    with open(meta_path(city_slug), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    size_kb = matrix.nbytes / 1024
    print(f"    💾 {city_slug}_matrix.npy  ({size_kb:.1f} KB,  {len(places)}×{len(places)})")
    print(f"    💾 {city_slug}_meta.json")


# ─── API helpers ──────────────────────────────────────────────────────────────

def _call_api(origins: list[dict], destinations: list[dict], api_key: str) -> list[list[float]]:
    """
    One Distance Matrix API request for a batch of origins and destinations.
    Returns a 2D list [origin_idx][dest_idx] → distance in metres or NO_ROUTE.
    """
    def coords(p):
        return f"{p['lat']},{p['lng']}"

    params = {
        "origins":      "|".join(coords(p) for p in origins),
        "destinations": "|".join(coords(p) for p in destinations),
        "mode":         "driving",
        "key":          api_key,
    }

    try:
        resp = requests.get(DISTANCE_MATRIX_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"\n    ⚠️  API error: {e}")
        return [[NO_ROUTE] * len(destinations) for _ in origins]

    if data.get("status") != "OK":
        err = data.get("error_message", "")
        print(f"\n    ⚠️  API status: {data.get('status')}  {err}")
        return [[NO_ROUTE] * len(destinations) for _ in origins]

    results = []
    for row in data.get("rows", []):
        row_vals = []
        for elem in row.get("elements", []):
            if elem.get("status") == "OK":
                row_vals.append(float(elem["distance"]["value"]))
            else:
                row_vals.append(NO_ROUTE)
        results.append(row_vals)

    return results


def _fill_block(
    origins: list[dict],
    destinations: list[dict],
    matrix: np.ndarray,
    id_to_idx: dict[str, int],
    api_key: str,
) -> int:
    """
    Fills matrix entries for all (origin, dest) pairs via chunked API calls.
    Skips i==j pairs (diagonal stays 0). Returns number of API requests made.
    """
    n_oc = math.ceil(len(origins)      / CHUNK_SIZE)
    n_dc = math.ceil(len(destinations) / CHUNK_SIZE)
    total_chunks = n_oc * n_dc
    req = 0

    for oi in range(0, len(origins), CHUNK_SIZE):
        obatch = origins[oi : oi + CHUNK_SIZE]
        for di in range(0, len(destinations), CHUNK_SIZE):
            dbatch = destinations[di : di + CHUNK_SIZE]
            req += 1

            elems = len(obatch) * len(dbatch)
            print(
                f"    chunk ({req}/{total_chunks})  "
                f"{len(obatch)}×{len(dbatch)}={elems} elems... ",
                end="", flush=True,
            )

            results = _call_api(obatch, dbatch, api_key)

            for li, origin in enumerate(obatch):
                for lj, dest in enumerate(dbatch):
                    if origin["place_id"] == dest["place_id"]:
                        continue  # diagonal stays 0
                    ni = id_to_idx[origin["place_id"]]
                    nj = id_to_idx[dest["place_id"]]
                    if li < len(results) and lj < len(results[li]):
                        matrix[ni][nj] = results[li][lj]

            print("✓")
            if req < total_chunks:
                time.sleep(REQUEST_DELAY)

    return req


# ─── Per-city pipeline ────────────────────────────────────────────────────────

def process_city(city_slug: str, places: list[dict], api_key: str) -> None:
    """
    Builds or incrementally updates the distance matrix for one city.

    Args:
        city_slug: e.g. "mumbai"
        places:    list of place dicts for this city
        api_key:   Google API key
    """
    N             = len(places)
    city_name     = places[0]["city"] if places else city_slug
    current_ids   = [p["place_id"] for p in places]
    current_id_set = set(current_ids)
    id_to_idx     = {pid: i for i, pid in enumerate(current_ids)}

    print(f"\n🏙️  {city_name}  ({N} places, {N*N - N} pairs to compute)")

    # ── Load existing ────────────────────────────────────────────────────────
    existing_ids, existing_matrix = load_city_matrix(city_slug)
    total_requests = 0

    if existing_ids is not None:
        existing_id_set = set(existing_ids)
        added_ids   = current_id_set - existing_id_set
        removed_ids = existing_id_set - current_id_set

        # Nothing changed — skip entirely
        if not added_ids and not removed_ids:
            print(f"    ✅ Already up to date — skipping.")
            return

        # Report what changed
        if added_ids:   print(f"    ➕ {len(added_ids)} new place(s)")
        if removed_ids: print(f"    ➖ {len(removed_ids)} removed place(s)")

        # Create new matrix (all -1, diagonal 0)
        matrix = np.full((N, N), NO_ROUTE, dtype=np.float64)
        np.fill_diagonal(matrix, 0.0)

        # Copy existing pairs from old matrix
        old_id_to_idx = {pid: i for i, pid in enumerate(existing_ids)}
        copied = 0
        for pid_i in existing_ids:
            if pid_i not in current_id_set:
                continue
            for pid_j in existing_ids:
                if pid_j not in current_id_set:
                    continue
                matrix[id_to_idx[pid_i]][id_to_idx[pid_j]] = (
                    existing_matrix[old_id_to_idx[pid_i]][old_id_to_idx[pid_j]]
                )
                copied += 1

        print(f"    ♻️  Copied {copied} existing pairs")

        if not added_ids:
            # Only removals — save trimmed matrix, no API calls
            save_city_matrix(city_slug, matrix, places)
            print(f"    ✅ Done (removals only, no API calls).")
            return

        # API calls only for new place combinations
        new_places = [p for p in places if p["place_id"] in added_ids]
        old_places = [p for p in places if p["place_id"] not in added_ids]

        print(f"    🌐 Computing new pairs via API:")

        # new → all (new rows fully)
        print(f"    [new → all]  {len(new_places)}×{N}:")
        total_requests += _fill_block(new_places, places, matrix, id_to_idx, api_key)

        # old → new (new columns for existing rows)
        if old_places:
            print(f"    [old → new]  {len(old_places)}×{len(new_places)}:")
            total_requests += _fill_block(old_places, new_places, matrix, id_to_idx, api_key)

    else:
        # ── Full computation (first run for this city) ───────────────────────
        matrix = np.full((N, N), NO_ROUTE, dtype=np.float64)
        np.fill_diagonal(matrix, 0.0)

        chunks_needed = math.ceil(N / CHUNK_SIZE) ** 2
        print(f"    🌐 Full compute — {chunks_needed} API chunk(s):")
        total_requests += _fill_block(places, places, matrix, id_to_idx, api_key)

    # ── Stats & save ─────────────────────────────────────────────────────────
    reachable = int(np.sum(matrix > 0))
    no_route  = int(np.sum(matrix == NO_ROUTE)) - 0  # diagonal is 0, not -1

    save_city_matrix(city_slug, matrix, places)

    avg_km = np.mean(matrix[matrix > 0]) / 1000 if reachable else 0
    print(f"    📊 {reachable} reachable pairs  |  {no_route} no-route  |  avg {avg_km:.1f} km  |  {total_requests} API calls")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "═" * 62)
    print("  📐 Distance Matrix Pipeline  (per-city)")
    print("═" * 62)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "YOUR_GOOGLE_API_KEY_HERE":
        print("❌  GOOGLE_API_KEY not set in .env")
        sys.exit(1)

    print("\n📂 Loading database...")
    all_cities = load_db()

    if not all_cities:
        print("❌  No cities in database. Run python3 main.py \"<city>\" first.")
        sys.exit(1)

    # Optional: filter to one city if given as CLI arg
    target_city = " ".join(sys.argv[1:]).strip().lower() if len(sys.argv) > 1 else None

    cities_to_process = {}
    for slug, places in all_cities.items():
        city_name = places[0]["city"].lower() if places else slug
        if target_city and target_city not in (slug, city_name):
            continue
        cities_to_process[slug] = places

    if not cities_to_process:
        print(f"❌  City '{target_city}' not found in database.")
        print(f"   Available: {', '.join(all_cities.keys())}")
        sys.exit(1)

    print(f"   Processing {len(cities_to_process)} city/cities:")
    for slug, places in cities_to_process.items():
        print(f"   • {places[0]['city']}: {len(places)} places")

    start = time.time()

    for slug, places in cities_to_process.items():
        process_city(slug, places, api_key)

    elapsed = time.time() - start
    print("\n" + "═" * 62)
    print(f"  ✅ Done in {elapsed:.1f}s")
    print(f"  Matrices saved in:  {MATRICES_DIR}/")
    for slug in cities_to_process:
        print(f"    • {slug}_matrix.npy  +  {slug}_meta.json")
    print("═" * 62 + "\n")


if __name__ == "__main__":
    main()
