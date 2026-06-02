"""
normalize.py
Transforms raw Google Places API detail objects into clean, structured
place records ready for LLM enrichment and the database.

Fields stored here are the raw pre-LLM values. The llm_enricher step
will:
  - Compact description to ≤200 words
  - Summarize reviews into a review_summary paragraph
  - Generate overall_note using google_types + moods as context
  - Strip google_types, moods, raw_reviews from the final record

Photo URLs are NOT constructed here — only raw photo resource_name
strings are stored (max 5). The frontend will request actual image URLs
via a separate API call: GET /v1/{resource_name}/media?maxWidthPx=800&key=...
"""

from pipeline.category_map import resolve_category

MAX_PLACES = 50  # Hard cap for any city — best for itinerary use


def normalize_place(raw: dict) -> dict:
    """
    Normalizes a single raw place details response into a clean record.

    google_types and moods are kept as _llm_context fields — they are
    used by the LLM enricher to generate overall_note, then stripped
    from the final saved record.

    Args:
        raw: Raw dict from the Place Details endpoint

    Returns:
        Clean place dict (pre-LLM enrichment)
    """
    google_types = raw.get("types", [])

    # Category and mood resolution
    category_info = resolve_category(google_types)
    category = category_info["label"]
    moods = category_info["moods"]

    # Location
    location = raw.get("location", {})
    latitude = location.get("latitude")
    longitude = location.get("longitude")

    # Display name
    display_name_obj = raw.get("displayName", {})
    name = (
        display_name_obj.get("text", "Unknown Place")
        if isinstance(display_name_obj, dict)
        else "Unknown Place"
    )

    # Raw description — LLM enricher will compact to ≤200 words
    editorial = raw.get("editorialSummary", {})
    description = editorial.get("text") if isinstance(editorial, dict) else None

    # Raw review texts — LLM enricher will summarize into one paragraph
    raw_reviews = []
    for r in raw.get("reviews", [])[:5]:
        text_obj = r.get("text", {})
        text = text_obj.get("text") if isinstance(text_obj, dict) else None
        if text:
            raw_reviews.append({
                "author": r.get("authorAttribution", {}).get("displayName", "Anonymous"),
                "rating": r.get("rating"),
                "text":   text,
            })

    # Photo references — raw only, max 5, no URL construction
    photo_references = []
    for p in raw.get("photos", [])[:5]:
        resource_name = p.get("name")
        if resource_name:
            photo_references.append({
                "resource_name": resource_name,  # e.g. "places/ChIJ.../photos/AXCi..."
                "width_px":      p.get("widthPx"),
                "height_px":     p.get("heightPx"),
            })

    return {
        "place_id":          raw.get("id"),
        "name":              name,
        "description":       description,     # raw; LLM will compact to ≤200 words
        "category":          category,
        "rating":            raw.get("rating"),
        "total_ratings":     raw.get("userRatingCount"),
        "address":           raw.get("formattedAddress"),
        "latitude":          latitude,
        "longitude":         longitude,
        "photo_references":  photo_references,
        # ── LLM context fields (stripped after enrichment) ──────────
        "_google_types":     google_types,    # used by LLM for overall_note
        "_moods":            moods,           # used by LLM for overall_note
        "_raw_reviews":      raw_reviews,     # used by LLM for review_summary
    }


def normalize_places(raw_places: list) -> list:
    """
    Normalizes a list of raw place detail objects.
    Filters out entries missing critical fields (lat/lng or name).

    Args:
        raw_places: list of raw API response dicts

    Returns:
        list of clean place dicts (ready for LLM enrichment)
    """
    normalized = []
    skipped = 0

    for raw in raw_places:
        place = normalize_place(raw)
        if place["name"] == "Unknown Place" or place["latitude"] is None or place["longitude"] is None:
            skipped += 1
            continue
        normalized.append(place)

    print(f"\n🔧 Normalized {len(normalized)} places ({skipped} filtered out for missing fields)")
    return normalized


def curate_places(places: list, max_places: int = MAX_PLACES) -> list:
    """
    Intelligently caps the place list at max_places (default 50) while
    preserving category diversity — essential for quality itinerary building.

    Strategy:
      1. If already ≤50 → return as-is
      2. Guarantee up to 2 best-rated places per category (diversity first)
      3. Fill remaining slots up to 50 with highest-rated places overall
      4. Final sort by rating desc for consistent ordering

    Args:
        places: Normalized place list
        max_places: Hard cap (default 50)

    Returns:
        Curated list of up to max_places places
    """
    if len(places) <= max_places:
        return places

    print(f"\n✂️  Curating {len(places)} places → top {max_places} (diversity-aware)...")

    # Group by category, sort each group by rating descending
    by_category: dict[str, list] = {}
    for p in places:
        cat = p["category"]
        by_category.setdefault(cat, []).append(p)

    for cat in by_category:
        by_category[cat].sort(key=lambda x: x.get("rating") or 0, reverse=True)

    # Phase 1: take top 2 per category to guarantee diversity
    selected_ids: set = set()
    selected: list = []

    for cat_places in by_category.values():
        for p in cat_places[:2]:
            if p["place_id"] not in selected_ids:
                selected_ids.add(p["place_id"])
                selected.append(p)

    # Phase 2: fill remaining slots with highest-rated unselected places
    if len(selected) < max_places:
        remaining = [p for p in places if p["place_id"] not in selected_ids]
        remaining.sort(key=lambda x: x.get("rating") or 0, reverse=True)
        fill_count = max_places - len(selected)
        selected.extend(remaining[:fill_count])

    # Phase 3: if still over cap (edge case), trim by rating
    if len(selected) > max_places:
        selected.sort(key=lambda x: x.get("rating") or 0, reverse=True)
        selected = selected[:max_places]

    # Final sort: highest rated first
    selected.sort(key=lambda x: x.get("rating") or 0, reverse=True)

    print(f"  ✓ Kept {len(selected)} places across {len(by_category)} categories")
    return selected
