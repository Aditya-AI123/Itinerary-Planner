"""
geocode.py
Resolves a city/place name to (lat, lng) using the Google Geocoding API.
"""

import os
import requests


GEOCODING_API_URL = "https://maps.googleapis.com/maps/api/geocode/json"


def geocode_city(city_name: str) -> dict:
    """
    Geocodes a city name to coordinates.

    Args:
        city_name: e.g. "Mumbai", "Paris", "New York"

    Returns:
        dict with keys: lat, lng, formatted_address

    Raises:
        ValueError: if the API key is missing or the geocoding fails
    """
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key or api_key == "YOUR_GOOGLE_API_KEY_HERE":
        raise ValueError(
            "❌  No valid API key found. "
            "Please update GOOGLE_API_KEY in your .env file."
        )

    response = requests.get(
        GEOCODING_API_URL,
        params={"address": city_name, "key": api_key},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    if data.get("status") != "OK" or not data.get("results"):
        raise ValueError(
            f"❌  Geocoding failed for '{city_name}'. "
            f"Status: {data.get('status')}. {data.get('error_message', '')}"
        )

    result = data["results"][0]
    location = result["geometry"]["location"]

    return {
        "lat": location["lat"],
        "lng": location["lng"],
        "formatted_address": result["formatted_address"],
    }
