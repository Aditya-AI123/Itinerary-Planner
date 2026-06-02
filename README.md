# AI Itinerary Planner

### Self Project | Jun '25 – Aug '

## 1. Problem Statement

Planning a trip to Mumbai or Goa is overwhelming. Generic travel sites dump hundreds of places with no
personalisation — travellers waste hours reading reviews, checking weather, estimating distances and manually
building day-by-day schedules. The goal of this project was to build an end-to-end, AI-powered travel planner
that learns what the user actually wants (through an intuitive swipe interface), then automatically generates a
realistic, weather-aware, distance-optimised itinerary.

## 2. System Overview

```
Phase Description Key Technology
```
```
Phase 1 Database Creation LLM + Google Places API
```
```
Phase 2 User Onboarding Custom Form + Swipe UI
```
```
Phase 3 Itinerary Generation Agentic LLM Pipeline
```
```
Phase 4 Post-Generation Edit & Modify Output
```
## 3. Phase 1 — LLM-Powered Attraction Database

Before any user interaction, a one-time database of top attractions across Mumbai and Goa was built
programmatically. The goal was to have rich, structured data for ~70 places covering beaches, historical sites,
food spots, adventure activities and more.

### How the database was built:

- Text Search calls — ~5 queries per city (e.g. 'tourist attractions in Mumbai', 'beaches in Goa') via Google
    Places Text Search API, yielding ~60-100 candidate Place IDs.
- Place Details calls — one call per unique Place ID to fetch name, coordinates, rating, reviews, photos,
    address and opening hours.
- LLM formatting — an LLM parsed and structured the raw API JSON into a clean schema: id, name, lat/long,
    description, activities[], rating, best_time, reviews[].
- Total API cost — ~200-350 calls, well within Google's $200/month free credit (effectively $0).

### DB Schema (simplified):

```
{ id, name, lat, long, city, address, rating, total_reviews, types[], primary_type,
reviews[], photos[], best_time_to_visit }
```
Obstacle: Risk of LLM hallucinating place details.

Fix: Used agentic LLM with tool-calling (Google Places API as a tool) — the model fetched real, live data and
only reformatted the API response rather than synthesising from memory. 20% of entries were manually
spot-checked for accuracy.

## 4. Phase 2 — User Onboarding & Swipe Preference Engine


Before the swipe screen, users fill in trip details via a structured form. These inputs, combined with swipe
signals, form the complete preference payload sent to the backend.

### Form inputs collected:

- Arrival & departure dates
- Number of adults
- Trip nature — relax / adventure / thrill / fun (multi-select)
- Flight booked? Hotel booked? (with accommodation location if yes)
- If unbooked — the system suggests optimal travel windows and stay options

### Swipe mechanism (Bumble-style cards):

Attractions from the database are shown one at a time as rich cards — containing the place image, name,
description, activities, best visit time, weather, rating and reviews. The user signals preference through three
actions:

```
Action Meaning Effect on Itinerary
```
```
Swipe Left Not interested Place excluded entirely
```
```
Swipe Right Would like to visit Added to candidate pool
```
```
Star / Super-like Must visit Marked as high-priority; always included
```
## 5. Phase 3 — Agentic AI Itinerary Generation

The core of the project. Once user preferences are collected, a multi-stage LLM pipeline processes them along
with real-time weather and pre-computed distances to produce a detailed, realistic itinerary.

### 5.1 Weather Integration (Open-Meteo API)

Why Open-Meteo? Free, no API key required, global coverage with hourly forecasts up to 16 days — perfect
for Mumbai/Goa.

The backend queries temperature, precipitation probability and weather codes for every day of the trip. This raw
JSON is then fed to a dedicated Weather Interpreter LLM that produces a concise human-readable summary
(e.g. 'Day 2: 65% rain in PM — move beach to morning, swap afternoon to museum'). This reduces ~2,
tokens of raw JSON to ~100 tokens of actionable insight.

### Weather decision rules passed to LLM:

- Rain probability >50%: swap outdoor beach activities for aquariums / indoor markets.
- Temperature >32°C: schedule outdoor activities 7–11 AM; indoor/AC venues for midday.
- Clear & cool: prioritise beaches, forts and open-air sites.

### 5.2 Distance Optimisation (Google Maps / Precomputed Matrix)

Obstacle: Calling the Distance Matrix API live for every itinerary — 15 places = 225 elements ≈ $1.12 per user
— would make the project uneconomical at scale.

Fix — Precomputation: A one-time script runs the Distance Matrix API across all 70 attractions + ~10 popular
hotel areas (80 locations = 6,400 elements ≈ $0.03 total). This matrix is saved as a NumPy array. At runtime,
user-selected places are simply sliced from the matrix — zero additional API cost per itinerary.

Fallback: For hotels outside the precomputed set, a Haversine approximation estimates driving time with ~85%
accuracy (sufficient for clustering purposes).


A dedicated Distance Interpreter LLM receives the sliced matrix and outputs day-by-day geographic clusters
— grouping nearby attractions on the same day to minimise travel, ensuring the user returns to accommodation
by 9 PM, and flagging special logistics (e.g. Elephanta Caves requires a 45-min ferry).

### 5.3 The 3-Stage LLM Pipeline

```
Stage Input Output Token Saving
```
```
Weather Interpreter Raw Open-Meteo JSON
(~2,000 tokens)
```
```
Concise day-by-day
weather guidance
```
```
~90%
```
```
Distance Interpreter Sliced duration matrix
+ place names
```
```
Day clusters +
travel notes
```
```
~85%
```
```
Final Itinerary LLM Both summaries +
user prefs
```
```
Full JSON itinerary
(editable)
```
```
—
```
### 5.4 Parallelism — Reducing Wait Time

Obstacle: Weather and Distance LLM calls run sequentially = 9+ seconds total latency.

Fix: The Weather Interpreter and Distance Interpreter are independent of each other. Using Python's
asyncio.gather(), both fire simultaneously. Result: 9 s → ~4 s total (67% faster). The final itinerary LLM then
runs once both summaries are ready.

Additional optimisations: weather results are cached by city + date range (Mumbai summer is consistent
year-on-year), and the final LLM response is streamed token-by-token so users see Day 1 appear before Day 5
is generated.

## 6. Phase 4 — Post-Generation Editing

After the itinerary is generated, users are not locked in. The output is presented as an editable structured view.
Users can swap activities, adjust timings, or request a modified version by describing the change in natural
language (e.g. 'Move Elephanta to Day 3 and add a lunch stop near Gateway'). The system re-runs only the
affected parts of the pipeline.

## 7. Key Obstacles & Fixes — Summary

```
Obstacle Fix
```
```
LLM hallucinating place data Switched to tool-augmented agentic LLM that calls real APIs; LLM only reformats verified API responses.
```
```
Distance Matrix API too expensive at scale
($1+/itinerary)
```
```
One-time precomputation of full 80×80 matrix ($0.03 total); runtime uses NumPy slice — $0 per user.
```
```
Token overload / LLM accuracy drop
with raw JSON data
```
```
Specialist Interpreter LLMs compress weather and distance data into concise summaries before the final call. Cost cut ~70%.
```
```
High latency from sequential LLM calls
(9+ seconds)
```
```
asyncio.gather() parallelises independent calls; streaming UI shows content as it generates. Latency → ~4 s.
```
```
Photo costs from Google Places API Store only photoReference at build time (free). Frontend constructs the photo URL on demand, lazy-loading images only for cards the user actually views.
```
## 8. Tech Stack


```
Layer Technology / Service
```
```
LLM GPT-4o / Claude (tool-augmented, agentic)
```
```
Attraction Discovery Google Places Text Search API
```
```
Place Enrichment Google Places Details API
```
```
Weather Open-Meteo API (free, no key required)
```
```
Distance / Travel Time Google Distance Matrix API (precomputed once)
```
```
Async Orchestration Python asyncio (asyncio.gather for parallel LLM calls)
```
```
Database JSON flat-file / SQLite (static attractions DB)
```
```
Distance Approximation Haversine formula (fallback for unknown hotels)
```
```
Frontend (Swipe UI) React (Bumble-style card swipe interaction)
```
```
Backend FastAPI (async endpoints, streaming responses)
```

```
AI Itinerary Planner | Self Project | Jun '25 – Aug '
```

