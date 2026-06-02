"""
main.py — Tourist Location Data Pipeline
=========================================

Usage:
    python3 main.py "Mumbai"
    python3 main.py "Paris, France"
    python3 main.py "Manali, Himachal Pradesh"

Behaviour:
  ① City NOT in database → run full pipeline (geocode → search → details → normalize → LLM → save)
  ② City EXISTS, all places have overall_note → print "already complete", exit
  ③ City EXISTS, some places have overall_note=null → partial update:
       Re-fetch Place Details only for those places → re-normalize → LLM → merge → save
     This handles the case where Groq's rate limit was hit mid-run.

LLM enrichment via Groq / llama-3.3-70b-versatile:
  - Compacts description to ≤200 words
  - Summarizes top reviews into a review_summary paragraph
  - Generates overall_note (what to do / experience at this place)
  If rate limit is hit, stops gracefully — saved places keep their LLM data,
  null-filled places can be completed on the next run.
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from slugify import slugify

# Load .env from project root
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

from pipeline.geocode import geocode_city
from pipeline.text_search import fetch_all_categories
from pipeline.place_details import fetch_all_place_details
from pipeline.normalize import normalize_places, curate_places
from pipeline.llm_enricher import enrich_all_places
from pipeline.database import load_database, save_database, upsert_city, find_city


# ─── Helpers ────────────────────────────────────────────────────────────────

def _print_header(city_name: str, mode: str) -> None:
    print("\n" + "═" * 62)
    print(f"  🗺️  Tourist Location Pipeline  [{mode}]")
    print(f"  City: {city_name}")
    print("═" * 62)


def _print_footer(city_name: str, n_places: int, total_cities: int, elapsed: float) -> None:
    print("\n" + "═" * 62)
    print(f"  ✅ Pipeline complete in {elapsed:.1f}s")
    print(f"  📦 {n_places} places saved for '{city_name}'")
    print(f"  🌍 Database now contains {total_cities} city/cities")
    print("═" * 62 + "\n")


def _print_breakdown(places: list) -> None:
    breakdown: dict[str, int] = {}
    for p in places:
        breakdown[p["category"]] = breakdown.get(p["category"], 0) + 1
    print("\n📊 Category breakdown:")
    for cat, count in sorted(breakdown.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


# ─── Full pipeline (new city) ────────────────────────────────────────────────

def _run_full(city_name: str, db: dict) -> None:
    """Runs the complete pipeline for a city not yet in the database."""
    _print_header(city_name, "FULL PIPELINE")
    start = time.time()

    # Step 1: Geocode
    print("\n📍 Step 1: Geocoding city...")
    geo = geocode_city(city_name)
    lat, lng, formatted_address = geo["lat"], geo["lng"], geo["formatted_address"]
    print(f"  ✓ {formatted_address} → ({lat}, {lng})")

    # Step 2: Category searches
    print("\n🔍 Step 2: Running category searches...")
    raw_search_results = fetch_all_categories(city_name)
    if not raw_search_results:
        print("❌ No places found. Check your API key and city name.")
        sys.exit(1)

    # Step 3: Place details
    print("\n📋 Step 3: Fetching full place details...")
    raw_details = fetch_all_place_details(raw_search_results)

    # Step 4: Normalize
    print("\n🔧 Step 4: Normalizing data...")
    places = normalize_places(raw_details)
    if not places:
        print("❌ No valid places after normalization.")
        sys.exit(1)
    _print_breakdown(places)

    # Step 4b: Curate to top 50
    places = curate_places(places)

    # Step 5: LLM enrichment
    places = enrich_all_places(places)

    # Step 6: Save
    print("\n💾 Step 6: Saving to database...")
    city_slug = slugify(city_name)
    country = formatted_address.split(",")[-1].strip()

    city_entry = {
        "city_name":         city_name,
        "city_slug":         city_slug,
        "country":           country,
        "formatted_address": formatted_address,
        "coordinates":       {"lat": lat, "lng": lng},
        "fetched_at":        datetime.now(timezone.utc).isoformat(),
        "total_places":      len(places),
        "places":            places,
    }
    db = upsert_city(db, city_entry)
    save_database(db)

    _print_footer(city_name, len(places), db["total_cities"], time.time() - start)


# ─── Partial update (fill null LLM fields) ──────────────────────────────────

def _run_partial(city_name: str, existing: dict, db: dict) -> None:
    """
    Fills in LLM data for places that have overall_note=null.
    Re-fetches Place Details for only those places, re-normalizes,
    runs LLM, then merges back into the existing city entry.
    """
    incomplete = [p for p in existing["places"] if p.get("overall_note") is None]
    n_incomplete = len(incomplete)
    n_total = len(existing["places"])

    _print_header(city_name, f"PARTIAL UPDATE — {n_incomplete}/{n_total} places need LLM data")
    start = time.time()

    # Build a minimal place-list format that fetch_all_place_details expects
    # It needs dicts with 'id' and optionally 'displayName'
    stub_list = [
        {"id": p["place_id"], "displayName": {"text": p["name"]}}
        for p in incomplete
    ]

    print(f"\n📋 Step 1: Re-fetching Place Details for {n_incomplete} incomplete places...")
    raw_details = fetch_all_place_details(stub_list)

    print(f"\n🔧 Step 2: Re-normalizing...")
    fresh_places = normalize_places(raw_details)

    if not fresh_places:
        print("❌ No places returned from API. Check your API key.")
        sys.exit(1)

    print(f"\n🤖 Step 3: Running LLM enrichment on {len(fresh_places)} places...")
    enriched = enrich_all_places(fresh_places)

    # Merge: build a lookup of newly enriched places by place_id
    enriched_by_id = {p["place_id"]: p for p in enriched}

    # Replace incomplete places in the existing list with freshly enriched versions.
    # Places that already had overall_note are left completely untouched.
    merged_places = []
    for p in existing["places"]:
        if p.get("overall_note") is None and p["place_id"] in enriched_by_id:
            merged_places.append(enriched_by_id[p["place_id"]])
        else:
            merged_places.append(p)

    # Update city entry metadata
    existing["places"]     = merged_places
    existing["total_places"] = len(merged_places)
    existing["fetched_at"] = datetime.now(timezone.utc).isoformat()

    still_null = sum(1 for p in merged_places if p.get("overall_note") is None)

    print(f"\n💾 Step 4: Saving updated database...")
    db = upsert_city(db, existing)
    save_database(db)

    elapsed = time.time() - start
    print("\n" + "═" * 62)
    print(f"  ✅ Partial update complete in {elapsed:.1f}s")
    print(f"  📦 {n_incomplete - still_null} places newly enriched")
    if still_null:
        print(f"  ⚠️  {still_null} places still have null LLM data (rate limit hit again)")
        print(f"     Re-run to continue filling them in.")
    else:
        print(f"  🎉 All {len(merged_places)} places for '{city_name}' are now fully enriched!")
    print("═" * 62 + "\n")


# ─── Entry point ─────────────────────────────────────────────────────────────

def run(city_name: str) -> None:
    db = load_database()
    city_slug = slugify(city_name)
    existing = find_city(db, city_slug)

    if existing is None:
        # City not in database — run full pipeline
        _run_full(city_name, db)
        return

    # City exists — check how many places are missing LLM data
    incomplete = [p for p in existing["places"] if p.get("overall_note") is None]

    if not incomplete:
        n = len(existing["places"])
        print(f"\n✅ '{city_name}' is already complete in the database ({n} places, all enriched).")
        print("   To force a full refresh, delete the city entry from places_database.json and re-run.\n")
        return

    # Some places need LLM data — run partial update
    _run_partial(city_name, existing, db)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "\nUsage:  python3 main.py \"<city name>\"\n\n"
            "Examples:\n"
            '  python3 main.py "Mumbai"\n'
            '  python3 main.py "Paris, France"\n'
            '  python3 main.py "Manali, Himachal Pradesh"\n'
        )
        sys.exit(1)

    city = " ".join(sys.argv[1:])
    run(city)
