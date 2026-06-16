"""
pipeline/hotel_resolver.py
==========================

Pre-processing step that converts a raw hotel address string into a
**hotel proxy place** — the closest place already present in the city's
distance matrix.

Why do we need this?
  The distance matrix only contains distances between places in our database.
  The user's actual hotel is (almost certainly) not one of those places, so
  we cannot compute exact hotel-to-place distances.

Solution:
  1. Geocode the hotel address with OpenStreetMap Nominatim (free, no key).
  2. Compute the straight-line (Haversine) distance from the hotel's
     lat/lng to every place in the matrix metadata.
  3. The nearest place becomes the "hotel proxy":
       - It is added to the selected_place_ids list (if not already there)
       - It is inserted at index-0 of must_visit_ids with the highest priority
       - Downstream agents receive it as the effective base of operations
  4. If no hotel address is provided (or geocoding fails), fall back to the
     place nearest to the city's geometric centre (average of all place coords).

Nominatim API:
  - Free, no authentication required.
  - Rate limit: 1 request/second — enforced here with a 1-second sleep.
  - User-Agent header is required (we use a project-specific string).

Usage:
    from pipeline.hotel_resolver import resolve_hotel_proxy

    result = resolve_hotel_proxy(
        city_slug     = "mumbai",
        hotel_address = "Taj Mahal Palace Hotel, Apollo Bunder, Colaba, Mumbai",
    )
    # result.proxy_place_id   → place_id of the nearest matrix entry
    # result.proxy_place_name → its human-readable name
    # result.hotel_lat, .hotel_lng → geocoded hotel coordinates
    # result.distance_km      → straight-line distance hotel→proxy (km)
    # result.fallback_used    → True if city-centre fallback was applied

Standalone test:
    source venv/bin/activate
    python3 -m pipeline.hotel_resolver
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

ROOT         = Path(__file__).resolve().parent.parent
MATRICES_DIR = ROOT / "data" / "matrices"

# Nominatim settings
NOMINATIM_URL    = "https://nominatim.openstreetmap.org/search"
NOMINATIM_UA     = "ItineraryPlanner/1.0 (travel-planner-app)"
NOMINATIM_DELAY  = 1.1   # seconds — respect 1 req/s limit


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class HotelProxyResult:
    """All output from the hotel-resolution step."""
    # The proxy place (nearest matrix entry to the hotel)
    proxy_place_id:   str
    proxy_place_name: str
    proxy_place_idx:  int           # row/col index in the full matrix

    # Geocoded hotel coordinates (or city-centre coords on fallback)
    hotel_lat: float
    hotel_lng: float

    # Straight-line distance from hotel to proxy (km)
    distance_km: float

    # True when we used city-centre fallback (no address given / geocoding failed)
    fallback_used: bool
    fallback_reason: str            # human-readable explanation

    # The "canonical" hotel description to pass to LLM prompts
    hotel_label: str                # e.g. "Taj Mahal Palace Hotel (proxy: Gateway of India area)"


# ─── Haversine distance ────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Straight-line (great-circle) distance in kilometres between two WGS84 points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlng / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ─── Nominatim geocoding ──────────────────────────────────────────────────────

def _geocode_nominatim(address: str) -> tuple[float, float] | None:
    """
    Geocode `address` using OpenStreetMap Nominatim.

    Tries up to two queries:
      1. The full cleaned address as-is.
      2. If that fails, drops the first comma-separated token (often the hotel
         name) and retries with just the street / area part — Nominatim
         handles street addresses much better than named buildings.

    Returns (lat, lng) as floats, or None if both queries fail.
    Waits 1.1 s after each request to respect the 1 req/s rate limit.
    """
    def _query(q: str) -> tuple[float, float] | None:
        params  = {"q": q, "format": "json", "limit": 1, "addressdetails": 0}
        headers = {"User-Agent": NOMINATIM_UA}
        try:
            resp = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            results = resp.json()
        except Exception as exc:
            print(f"   ⚠️  Nominatim request failed: {exc}")
            return None
        finally:
            time.sleep(NOMINATIM_DELAY)
        if not results:
            return None
        try:
            return float(results[0]["lat"]), float(results[0]["lon"])
        except (KeyError, ValueError, IndexError):
            return None

    # Attempt 1 — full address
    result = _query(address)
    if result:
        return result

    # Attempt 2 — drop first token (hotel name) if address has ≥ 2 comma parts
    parts = [p.strip() for p in address.split(",") if p.strip()]
    if len(parts) >= 2:
        shorter = ", ".join(parts[1:])
        print(f"   ↪ Retrying with shortened address: \"{shorter}\"")
        result = _query(shorter)
        if result:
            return result

    return None


def _preprocess_address(raw_address: str, city_name: str) -> str:
    """
    Light cleanup of the user's raw hotel address before geocoding.

    - Strips leading/trailing whitespace
    - Appends the city name if it isn't already in the address
      (increases Nominatim accuracy significantly)
    - Collapses repeated commas / spaces
    """
    addr = raw_address.strip()

    # Remove redundant punctuation
    while "  " in addr:
        addr = addr.replace("  ", " ")
    while ",," in addr:
        addr = addr.replace(",,", ",")

    # Append city name if missing (case-insensitive check)
    if city_name.lower() not in addr.lower():
        addr = f"{addr}, {city_name}"

    return addr


# ─── Matrix meta loader ────────────────────────────────────────────────────────

def _load_matrix_meta(city_slug: str) -> dict:
    """Load the matrix metadata JSON for a city."""
    path = MATRICES_DIR / f"{city_slug}_meta.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Matrix metadata not found: {path}\n"
            f"Run:  python3 distance_matrix.py \"{city_slug}\"  first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─── Nearest-place finder ──────────────────────────────────────────────────────

def _find_nearest_place(
    lat: float,
    lng: float,
    meta: dict,
) -> tuple[int, str, str, float]:
    """
    Find the place in `meta` whose coordinates are closest to (lat, lng).

    Returns:
        (idx, place_id, place_name, distance_km)
    """
    best_idx  = 0
    best_dist = float("inf")

    place_ids    = meta["place_ids"]
    place_names  = meta["place_names"]
    place_coords = meta["place_coords"]   # list of [lat, lng]

    for i, (coords, pid) in enumerate(zip(place_coords, place_ids)):
        d = _haversine_km(lat, lng, coords[0], coords[1])
        if d < best_dist:
            best_dist = d
            best_idx  = i

    return (
        best_idx,
        place_ids[best_idx],
        place_names[best_idx],
        best_dist,
    )


def _city_centre(meta: dict) -> tuple[float, float]:
    """
    Compute the geometric centre of all places in the city
    (simple average of all place latitudes and longitudes).
    """
    coords = meta["place_coords"]
    if not coords:
        return 0.0, 0.0
    avg_lat = sum(c[0] for c in coords) / len(coords)
    avg_lng = sum(c[1] for c in coords) / len(coords)
    return avg_lat, avg_lng


# ─── Public API ───────────────────────────────────────────────────────────────

def resolve_hotel_proxy(
    city_slug: str,
    hotel_address: Optional[str] = None,
    hotel_name: Optional[str] = None,
    city_name: Optional[str] = None,
    verbose: bool = True,
) -> HotelProxyResult:
    """
    Resolve a user-supplied hotel address to the nearest place in the distance
    matrix, or fall back to the place nearest the city's geometric centre.

    Args:
        city_slug     : e.g. "mumbai" — used to load the right matrix meta
        hotel_address : raw address string from the user form (may be empty / None)
        hotel_name    : hotel display name (e.g. "Taj Mahal Palace Hotel") — used
                        in the label only; does not affect geocoding
        city_name     : city display name for Nominatim hint (defaults to city_slug.title())
        verbose       : print progress lines (True when called from pipeline scripts)

    Returns:
        HotelProxyResult with all resolution details.
    """
    meta      = _load_matrix_meta(city_slug)
    city_disp = city_name or meta.get("city_name", city_slug.title())

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    # ── Step 1: Geocode the hotel address ─────────────────────────────────────
    hotel_lat:      Optional[float] = None
    hotel_lng:      Optional[float] = None
    fallback_used:  bool            = False
    fallback_reason: str            = ""

    if hotel_address and hotel_address.strip():
        clean_addr = _preprocess_address(hotel_address.strip(), city_disp)
        log(f"🏨 Geocoding hotel address via Nominatim…")
        log(f"   Query: \"{clean_addr}\"")
        coords = _geocode_nominatim(clean_addr)

        if coords:
            hotel_lat, hotel_lng = coords
            log(f"   ✅ Geocoded → ({hotel_lat:.5f}, {hotel_lng:.5f})")
        else:
            fallback_used   = True
            fallback_reason = (
                f"Nominatim could not geocode \"{hotel_address[:60]}\". "
                f"Using city-centre proxy instead."
            )
            log(f"   ⚠️  {fallback_reason}")
    else:
        fallback_used   = True
        fallback_reason = "No hotel address provided. Using city-centre proxy."
        log(f"🏨 {fallback_reason}")

    # ── Step 2: Apply city-centre fallback if needed ──────────────────────────
    if fallback_used or hotel_lat is None:
        hotel_lat, hotel_lng = _city_centre(meta)
        log(f"   City centre: ({hotel_lat:.5f}, {hotel_lng:.5f})")

    # ── Step 3: Find the nearest matrix place ─────────────────────────────────
    log(f"📍 Finding nearest place in distance matrix…")
    idx, proxy_id, proxy_name, dist_km = _find_nearest_place(hotel_lat, hotel_lng, meta)
    log(f"   ✅ Proxy place → \"{proxy_name}\"  ({dist_km:.2f} km away)")

    # ── Step 4: Build hotel label for LLM prompts ─────────────────────────────
    base_label = hotel_name or hotel_address or "City centre"
    hotel_label = (
        f"{base_label}"
        + (f" [area proxy: {proxy_name}]" if not fallback_used else f" [city-centre proxy: {proxy_name}]")
    )

    return HotelProxyResult(
        proxy_place_id   = proxy_id,
        proxy_place_name = proxy_name,
        proxy_place_idx  = idx,
        hotel_lat        = hotel_lat,
        hotel_lng        = hotel_lng,
        distance_km      = dist_km,
        fallback_used    = fallback_used,
        fallback_reason  = fallback_reason,
        hotel_label      = hotel_label,
    )


def inject_hotel_proxy(
    proxy: HotelProxyResult,
    selected_place_ids: list[str],
    must_visit_ids: list[str],
) -> tuple[list[str], list[str]]:
    """
    Inject the hotel proxy into the place selection lists.

    - Ensures proxy_place_id is in selected_place_ids
    - Inserts proxy_place_id at index-0 of must_visit_ids (highest priority)
      so downstream agents always treat it as the anchor point

    Returns updated (selected_place_ids, must_visit_ids) — does NOT mutate inputs.
    """
    selected = list(selected_place_ids)
    must     = list(must_visit_ids)

    pid = proxy.proxy_place_id

    # Add to selected if missing
    if pid not in selected:
        selected = [pid] + selected

    # Place at front of must-visit (highest priority)
    if pid in must:
        must.remove(pid)
    must = [pid] + must

    return selected, must


# ─── Standalone smoke-test ────────────────────────────────────────────────────

def _run_test() -> None:
    print("\n" + "═" * 64)
    print("  🧪  Hotel Resolver — smoke test (Mumbai)")
    print("═" * 64 + "\n")

    # Test 1: Real hotel address
    print("── Test 1: Real hotel address ──")
    result1 = resolve_hotel_proxy(
        city_slug     = "mumbai",
        hotel_address = "Taj Mahal Palace Hotel, Apollo Bunder, Colaba, Mumbai",
        hotel_name    = "Taj Mahal Palace Hotel",
    )
    print(f"   Proxy place : {result1.proxy_place_name}")
    print(f"   Proxy ID    : {result1.proxy_place_id}")
    print(f"   Hotel coords: ({result1.hotel_lat:.5f}, {result1.hotel_lng:.5f})")
    print(f"   Distance    : {result1.distance_km:.2f} km")
    print(f"   Fallback    : {result1.fallback_used}")
    print(f"   Label       : {result1.hotel_label}")

    # Test 2: No address → city-centre fallback
    print("\n── Test 2: No hotel address (city-centre fallback) ──")
    result2 = resolve_hotel_proxy(
        city_slug     = "mumbai",
        hotel_address = "",
    )
    print(f"   Proxy place : {result2.proxy_place_name}")
    print(f"   Distance    : {result2.distance_km:.2f} km")
    print(f"   Fallback    : {result2.fallback_used}")
    print(f"   Reason      : {result2.fallback_reason}")

    # Test 3: inject_hotel_proxy
    print("\n── Test 3: inject_hotel_proxy ──")
    dummy_selected = ["place_A", "place_B", "place_C"]
    dummy_must     = ["place_B"]
    new_sel, new_must = inject_hotel_proxy(result1, dummy_selected, dummy_must)
    print(f"   Before selected : {dummy_selected}")
    print(f"   After  selected : {new_sel}")
    print(f"   Before must     : {dummy_must}")
    print(f"   After  must     : {new_must}")

    print("\n" + "═" * 64 + "\n")


if __name__ == "__main__":
    _run_test()
