# 🗺️ Itinerary Planner — Tourist Location Data Pipeline

A Python pipeline that fetches **30–40+ tourist places** for any city using the **Google Places API (New)**, covering a wide variety of categories — landmarks, beaches, cafés, adventure spots, historical sites, restaurants, and more.

---

## 🚀 Quick Start

### 1. Activate the virtual environment
```bash
source venv/bin/activate
```

### 2. Add your Google API Key
Edit the `.env` file and replace the placeholder:
```
GOOGLE_API_KEY=your_actual_api_key_here
```

### 3. Run the pipeline
```bash
python main.py "Mumbai"
python main.py "Paris, France"
python main.py "Manali, Himachal Pradesh"
```

Results are saved to `data/places_database.json`.
- **New cities** are appended to the database.
- **Re-running the same city** refreshes its data (no duplicates).

---

## 🔑 Google Cloud Setup

Enable the following APIs on your Google Cloud project:

| API | Purpose |
|-----|---------|
| **Places API (New)** | Text Search + Place Details |
| **Geocoding API** | Convert city name → lat/lng |

**Steps:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create or select a project
3. Enable **Places API (New)** and **Geocoding API**
4. Go to **Credentials** → Create an **API Key**
5. Paste the key into `.env` as `GOOGLE_API_KEY=...`

Both APIs can share the same key.

---

## 📁 Project Structure

```
Itinerary Planner/
├── .env                          # API keys (never commit!)
├── .gitignore
├── requirements.txt
├── main.py                       # ← Run this
│
├── pipeline/
│   ├── __init__.py
│   ├── geocode.py                # Step 1: City name → lat/lng
│   ├── text_search.py            # Step 2: 20 category searches
│   ├── place_details.py          # Step 3: Full place details
│   ├── normalize.py              # Step 4: Clean & enrich data
│   ├── category_map.py           # Google type → friendly label + mood tags
│   └── database.py               # Load/save places_database.json
│
├── data/
│   └── places_database.json      # Master database (auto-created on first run)
│
└── venv/                         # Python virtual environment
```

---

## 🗂️ Database Structure (`data/places_database.json`)

```json
{
  "last_updated": "2024-06-01T10:30:00Z",
  "total_cities": 2,
  "cities": [
    {
      "city_name": "Mumbai",
      "city_slug": "mumbai",
      "country": "India",
      "formatted_address": "Mumbai, Maharashtra, India",
      "coordinates": { "lat": 19.076, "lng": 72.877 },
      "fetched_at": "2024-06-01T10:30:00Z",
      "total_places": 38,
      "places": [ ...place objects... ]
    }
  ]
}
```

---

## 📦 Place Object Schema

```json
{
  "place_id": "ChIJwe1EZjDG5zsRmKl1",
  "name": "Gateway of India",
  "description": "Iconic arch monument on the waterfront...",
  "category": "Historic & Cultural",
  "moods": ["cultural", "sightseeing"],
  "google_types": ["tourist_attraction", "point_of_interest"],
  "rating": 4.6,
  "total_ratings": 84521,
  "price_level": 0,
  "address": "Apollo Bandar, Colaba, Mumbai, Maharashtra 400001",
  "latitude": 18.9220,
  "longitude": 72.8347,

  "photo_references": [
    {
      "resource_name": "places/ChIJ.../photos/AXCi...",
      "width_px": 4032,
      "height_px": 3024,
      "author_attributions": [{ "display_name": "Ravi S.", "uri": "..." }]
    }
  ],

  "reviews": [
    {
      "author": "Ravi Sharma",
      "rating": 5,
      "text": "Magnificent monument, must visit!",
      "published_at": "2024-03-10T...",
      "relative_time": "3 months ago"
    }
  ],

  "opening_hours": {
    "open_now": true,
    "weekday_text": ["Monday: Open 24 hours", "..."]
  },

  "website": "https://...",
  "phone_national": "022 2202 6060",
  "amenities": {
    "good_for_children": true,
    "good_for_groups": true,
    "outdoor_seating": true
  },
  "accessibility": {
    "wheelchair_accessible_entrance": true
  },
  "parking": {
    "paid_parking_lot": true
  }
}
```

### 📸 Photo References

Photo URLs are **not** pre-built — only raw `resource_name` strings are stored.
When the frontend needs an image, construct the URL as:

```
GET https://places.googleapis.com/v1/{resource_name}/media
    ?maxWidthPx=800
    &key=YOUR_API_KEY
```

Example:
```
https://places.googleapis.com/v1/places/ChIJ.../photos/AXCi.../media?maxWidthPx=800&key=...
```

---

## 🏷️ Categories Searched (20 total)

| Category | Search Query |
|----------|-------------|
| Tourist Attractions | Famous tourist attractions |
| Landmarks | Iconic landmarks |
| Historic Sites | Historical monuments and heritage sites |
| Museums | Museums |
| Art Galleries | Art galleries |
| Religious Sites | Temples, churches, mosques |
| Beaches | Beaches |
| Parks & Gardens | Parks and gardens |
| National Parks | Wildlife sanctuaries |
| Scenic Viewpoints | Nature spots, viewpoints |
| Adventure Sports | Adventure activities |
| Amusement Parks | Theme parks |
| Hiking | Trekking trails |
| Restaurants | Best restaurants, local cuisine |
| Cafés | Coffee shops |
| Street Food | Local markets, food streets |
| Nightlife | Rooftop bars |
| Entertainment | Entertainment venues |
| Local Markets | Bazaars |
| Shopping Malls | Shopping malls |

---

## 🏷️ Mood Tags

Each place is tagged with moods for filtering & recommendation:

`cultural` · `sightseeing` · `relaxing` · `adventure` · `thrill` · `nature` · `foodie` · `nightlife` · `shopping` · `family` · `educational` · `spiritual` · `wellness` · `social`

---

## 💡 Tips

- The pipeline is rate-limited (150–200ms between requests) to stay within Google's quota.
- Re-running the same city **refreshes** its data, it will not create a duplicate.
- You can query the database manually by loading the JSON and filtering by `city_slug`.
