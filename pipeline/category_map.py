"""
category_map.py
Maps Google Places API types to:
  - friendly category labels
  - mood tags (used for filtering/recommendation)
"""

CATEGORY_MAP = {
    # Landmarks & Tourism
    "tourist_attraction":       {"label": "Tourist Attraction",       "moods": ["cultural", "sightseeing"]},
    "point_of_interest":        {"label": "Point of Interest",         "moods": ["sightseeing"]},
    "landmark":                 {"label": "Landmark",                  "moods": ["cultural", "sightseeing"]},

    # History & Culture
    "museum":                   {"label": "Museum",                    "moods": ["cultural", "educational"]},
    "art_gallery":              {"label": "Art Gallery",               "moods": ["cultural", "relaxing"]},
    "church":                   {"label": "Religious Site",            "moods": ["cultural", "spiritual"]},
    "mosque":                   {"label": "Religious Site",            "moods": ["cultural", "spiritual"]},
    "hindu_temple":             {"label": "Religious Site",            "moods": ["cultural", "spiritual"]},
    "place_of_worship":         {"label": "Religious Site",            "moods": ["cultural", "spiritual"]},
    "historical_landmark":      {"label": "Historic & Cultural",       "moods": ["cultural", "educational"]},

    # Nature & Outdoors
    "park":                     {"label": "Park & Gardens",            "moods": ["relaxing", "nature"]},
    "natural_feature":          {"label": "Natural Feature",           "moods": ["nature", "adventure"]},
    "beach":                    {"label": "Beach",                     "moods": ["relaxing", "nature", "adventure"]},
    "campground":               {"label": "Camping & Outdoors",        "moods": ["adventure", "nature"]},
    "national_park":            {"label": "National Park",             "moods": ["nature", "adventure"]},
    "wildlife_park":            {"label": "Wildlife & Safari",         "moods": ["nature", "adventure", "educational"]},
    "zoo":                      {"label": "Zoo & Wildlife",            "moods": ["nature", "family", "educational"]},
    "aquarium":                 {"label": "Aquarium",                  "moods": ["nature", "family", "educational"]},
    "botanical_garden":         {"label": "Botanical Garden",          "moods": ["relaxing", "nature"]},

    # Adventure & Thrill
    "amusement_park":           {"label": "Adventure & Thrill",        "moods": ["adventure", "thrill", "family"]},
    "adventure_sports_center":  {"label": "Adventure Sports",          "moods": ["thrill", "adventure"]},
    "hiking_area":              {"label": "Hiking & Trekking",         "moods": ["adventure", "nature"]},
    "sports_complex":           {"label": "Sports & Activities",       "moods": ["adventure", "thrill"]},
    "stadium":                  {"label": "Stadium",                   "moods": ["thrill", "cultural"]},

    # Food & Drink
    "restaurant":               {"label": "Restaurant",                "moods": ["foodie", "relaxing"]},
    "cafe":                     {"label": "Café",                      "moods": ["foodie", "relaxing"]},
    "bar":                      {"label": "Bar & Nightlife",           "moods": ["nightlife", "social"]},
    "night_club":               {"label": "Nightclub",                 "moods": ["nightlife", "thrill"]},
    "bakery":                   {"label": "Bakery & Desserts",         "moods": ["foodie", "relaxing"]},
    "meal_delivery":            {"label": "Restaurant",                "moods": ["foodie"]},
    "meal_takeaway":            {"label": "Street Food & Takeaway",    "moods": ["foodie"]},

    # Shopping & Leisure
    "shopping_mall":            {"label": "Shopping",                  "moods": ["shopping", "social"]},
    "market":                   {"label": "Local Market",              "moods": ["shopping", "cultural"]},
    "clothing_store":           {"label": "Shopping",                  "moods": ["shopping"]},

    # Wellness & Relaxation
    "spa":                      {"label": "Spa & Wellness",            "moods": ["relaxing", "wellness"]},
    "gym":                      {"label": "Fitness & Wellness",        "moods": ["wellness"]},

    # Entertainment
    "movie_theater":            {"label": "Entertainment",             "moods": ["social", "relaxing"]},
    "theater":                  {"label": "Theatre & Performing Arts", "moods": ["cultural", "social"]},
    "casino":                   {"label": "Casino & Entertainment",    "moods": ["thrill", "nightlife"]},

    # Accommodation
    "lodging":                  {"label": "Accommodation",             "moods": []},
    "hotel":                    {"label": "Hotel",                     "moods": []},
}

FALLBACK = {"label": "Point of Interest", "moods": ["sightseeing"]}


def resolve_category(google_types: list[str]) -> dict:
    """
    Given a list of Google place types, return the most descriptive
    { label, moods } entry from the map.
    Priority is given to the first type that matches.
    """
    for t in google_types:
        if t in CATEGORY_MAP:
            return CATEGORY_MAP[t]
    return FALLBACK
