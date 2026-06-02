"""
database.py
Manages the single master places_database.json file.

Database structure:
{
  "last_updated": "2024-...",
  "total_cities": 2,
  "cities": [
    {
      "city_name": "Mumbai",
      "city_slug": "mumbai",
      "country": "India",
      "formatted_address": "Mumbai, Maharashtra, India",
      "coordinates": { "lat": 19.076, "lng": 72.877 },
      "fetched_at": "2024-...",
      "total_places": 38,
      "places": [ ...place objects... ]
    },
    ...
  ]
}

When a city that already exists is fetched again,
its entry is REPLACED (refreshed), not duplicated.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "places_database.json"

EMPTY_DB = {
    "last_updated": None,
    "total_cities": 0,
    "cities": [],
}


def load_database() -> dict:
    """
    Loads the database from disk. Returns the empty structure
    if the file doesn't exist yet.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists():
        try:
            with open(DB_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"⚠️  Could not read existing database, starting fresh: {e}")

    return dict(EMPTY_DB)


def save_database(db: dict) -> None:
    """
    Writes the database dict to disk as formatted JSON.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Database saved → {DB_PATH}")


def upsert_city(db: dict, city_entry: dict) -> dict:
    """
    Inserts or replaces a city entry in the database.

    - If the city_slug already exists → its entry is replaced (data refreshed).
    - If it's new → appended to the cities list.

    Args:
        db: The loaded database dict
        city_entry: The city object to upsert

    Returns:
        Updated database dict
    """
    cities = db.get("cities", [])
    slug = city_entry["city_slug"]

    existing_idx = next(
        (i for i, c in enumerate(cities) if c.get("city_slug") == slug),
        None,
    )

    if existing_idx is not None:
        print(f"\n🔄 City '{city_entry['city_name']}' already in database — refreshing data.")
        cities[existing_idx] = city_entry
    else:
        print(f"\n➕ Adding new city '{city_entry['city_name']}' to database.")
        cities.append(city_entry)

    db["cities"] = cities
    db["total_cities"] = len(cities)
    db["last_updated"] = datetime.now(timezone.utc).isoformat()

    return db


def find_city(db: dict, city_slug: str) -> dict | None:
    """
    Looks up a city in the database by its slug.

    Args:
        db: The loaded database dict
        city_slug: e.g. "mumbai", "paris-france"

    Returns:
        City entry dict or None if not found
    """
    return next(
        (c for c in db.get("cities", []) if c.get("city_slug") == city_slug),
        None,
    )
