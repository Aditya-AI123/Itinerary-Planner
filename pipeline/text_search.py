"""
text_search.py
Performs multi-category Text Search using the Google Places API (New).
Returns a deduplicated list of { id, displayName, types } dicts.

Docs: https://developers.google.com/maps/documentation/places/web-service/text-search
"""

import os
import time
import requests

PLACES_TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"

# Max results per single category search (Google allows up to 20)
# 3 per category × 20 categories = up to 60 before dedup → typically lands 35–55 unique
MAX_PER_CATEGORY = 3

# 20 category-specific searches covering a wide variety of place types
SEARCH_CATEGORIES = [
    # Landmarks & Tourism
    {"query": "famous tourist attractions",             "type": "tourist_attraction"},
    {"query": "iconic landmarks",                       "type": None},

    # History & Culture
    {"query": "historical monuments and heritage sites","type": None},
    {"query": "museums",                                "type": "museum"},
    {"query": "art galleries",                          "type": "art_gallery"},
    {"query": "temples churches mosques religious sites","type": None},

    # Nature & Outdoors
    {"query": "beaches",                                "type": "beach"},
    {"query": "parks and gardens",                      "type": "park"},
    {"query": "national parks wildlife sanctuaries",    "type": None},
    {"query": "scenic viewpoints and nature spots",     "type": None},

    # Adventure & Thrill
    {"query": "adventure sports and activities",        "type": None},
    {"query": "amusement and theme parks",              "type": "amusement_park"},
    {"query": "hiking trekking trails",                 "type": None},

    # Food & Drink
    {"query": "best restaurants local cuisine",         "type": "restaurant"},
    {"query": "cafes and coffee shops",                 "type": "cafe"},
    {"query": "street food markets",                    "type": None},

    # Nightlife & Entertainment
    {"query": "rooftop bars and nightlife",             "type": "bar"},
    {"query": "entertainment venues",                   "type": None},

    # Shopping
    {"query": "local markets and bazaars",              "type": "market"},
    {"query": "shopping malls",                         "type": "shopping_mall"},
]


def _search_category(city_name: str, category: dict, api_key: str) -> list:
    """
    Runs one Text Search for the given category in the given city.

    Returns a list of raw place dicts (id, displayName, types, location, rating).
    """
    text_query = f"{category['query']} in {city_name}"

    body = {
        "textQuery": text_query,
        "maxResultCount": MAX_PER_CATEGORY,
        "languageCode": "en",
    }
    if category.get("type"):
        body["includedType"] = category["type"]

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.id,places.displayName,places.types,places.location,places.rating",
    }

    try:
        response = requests.post(
            PLACES_TEXT_SEARCH_URL,
            json=body,
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()
        return response.json().get("places", [])
    except requests.exceptions.HTTPError as e:
        msg = ""
        try:
            msg = e.response.json().get("error", {}).get("message", "")
        except Exception:
            pass
        print(f"  ⚠️  '{category['query']}' failed (HTTP {e.response.status_code}): {msg}")
        return []
    except Exception as e:
        print(f"  ⚠️  '{category['query']}' error: {e}")
        return []


def fetch_all_categories(city_name: str) -> list:
    """
    Runs all category searches for a city.
    Returns a deduplicated list of place dicts (by place id).

    Args:
        city_name: e.g. "Mumbai"

    Returns:
        list of { id, displayName, types, ... } dicts
    """
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key or api_key == "YOUR_GOOGLE_API_KEY_HERE":
        raise ValueError(
            "❌  No valid API key found. Please update GOOGLE_API_KEY in your .env file."
        )

    print(f"\n🔍 Running {len(SEARCH_CATEGORIES)} category searches for '{city_name}'...")

    seen_ids: set = set()
    all_places: list = []

    for category in SEARCH_CATEGORIES:
        print(f"  → {category['query']}... ", end="", flush=True)

        results = _search_category(city_name, category, api_key)
        added = 0

        for place in results:
            if place.get("id") and place["id"] not in seen_ids:
                seen_ids.add(place["id"])
                all_places.append(place)
                added += 1

        print(f"{len(results)} found, {added} new (total: {len(all_places)})")

        # Small delay between requests to be respectful to the API
        time.sleep(0.2)

    print(f"\n✅ Total unique places found: {len(all_places)}")
    return all_places
