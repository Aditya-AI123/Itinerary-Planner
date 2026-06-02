"""
place_details.py
Fetches full place details for each place ID using the Google Places API (New).

Photo references are stored as raw resource_name strings only.
URL construction is left to the frontend via a separate API call.

Docs: https://developers.google.com/maps/documentation/places/web-service/place-details
"""

import os
import time
import requests

PLACE_DETAILS_BASE_URL = "https://places.googleapis.com/v1/places"

# Fields requested from the Place Details endpoint.
# Only fetch essential fields — Google bills per field group requested.
# Omitting: website, phone, amenities, opening hours, accessibility, parking.
# Photos: only resource names stored (no URL construction).
REQUESTED_FIELDS = ",".join([
    "id",
    "displayName",
    "editorialSummary",
    "types",
    "rating",
    "userRatingCount",
    "location",
    "formattedAddress",
    "photos",
    "reviews",
])


def fetch_place_details(place_id: str, api_key: str) -> dict | None:
    """
    Fetches full details for a single place.

    Args:
        place_id: Google Place ID string (e.g. "ChIJ...")
        api_key: Google API key

    Returns:
        Raw place detail dict, or None on failure
    """
    url = f"{PLACE_DETAILS_BASE_URL}/{place_id}"
    headers = {
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": REQUESTED_FIELDS,
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        msg = ""
        try:
            msg = e.response.json().get("error", {}).get("message", "")
        except Exception:
            pass
        print(f"  ⚠️  Details failed for {place_id} (HTTP {e.response.status_code}): {msg}")
        return None
    except Exception as e:
        print(f"  ⚠️  Details error for {place_id}: {e}")
        return None


def fetch_all_place_details(places: list) -> list:
    """
    Fetches details for all places in the list with rate limiting.

    Args:
        places: list of { id, displayName, ... } dicts from text search

    Returns:
        list of raw detail dicts (failed fetches are excluded)
    """
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key or api_key == "YOUR_GOOGLE_API_KEY_HERE":
        raise ValueError(
            "❌  No valid API key. Please update GOOGLE_API_KEY in your .env file."
        )

    print(f"\n📋 Fetching details for {len(places)} places...")

    results = []
    total = len(places)

    for i, place in enumerate(places):
        place_id = place.get("id", "")
        display_name = (
            place.get("displayName", {}).get("text", place_id)
            if isinstance(place.get("displayName"), dict)
            else place_id
        )
        print(f"  [{i+1}/{total}] {display_name}... ", end="", flush=True)

        details = fetch_place_details(place_id, api_key)

        if details:
            results.append(details)
            print("✓")
        else:
            print("✗ skipped")

        # Rate-limit delay between requests
        if i < total - 1:
            time.sleep(0.15)

    print(f"\n✅ Successfully fetched details for {len(results)}/{total} places")
    return results
