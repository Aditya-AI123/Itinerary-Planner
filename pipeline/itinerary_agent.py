"""
pipeline/itinerary_agent.py
============================

The Head Planner LLM — a Gemini 2.5 Flash agent that acts as the senior
court master of itinerary creation.

It receives every piece of prepared intelligence:
  • Trip brief    (geographic clusters + distance constraints) from trip_planner_agent
  • Weather brief (day-by-day conditions + scheduling notes)   from weather_agent
  • Full user preferences from the form (dates, pace, budget, trip style …)
  • All selected places (liked ✓ + must-visit ♥) with their full metadata

And produces a complete, structured day-by-day itinerary as a JSON object
that is ready to be consumed by the frontend renderer.

Design decisions:
  ─ The LLM outputs a LEAN JSON (scheduling decisions + notes only).
    Photo references, coordinates, and verbose place metadata are NOT
    passed into the prompt (to keep token count low), and are NOT in the
    raw LLM output. Instead, a post-processing step (_enrich_itinerary)
    injects full place data from the database into every activity slot.
  ─ Gemini JSON mode (response_mime_type="application/json") is used to
    guarantee well-formed JSON output every time.
  ─ Temperature is set slightly higher (0.5) to allow creative scheduling
    while still being consistent.

Output JSON schema (see _ITINERARY_SCHEMA docstring for full structure):
  {
    "trip_summary":  { city, dates, hotel, travellers, budget, weather_summary },
    "days":          [ { day_number, date, theme, weather_note, slots: [...] } ],
    "overflow_places": [ { place_id, place_name, reason } ],
    "packing_tips":  [ "string" ],
    "general_tips":  [ "string" ]
  }

Usage:
    from pipeline.itinerary_agent import build_itinerary
    itinerary = build_itinerary(
        city_slug          = "mumbai",
        city_name          = "Mumbai",
        start_date         = "2025-07-15",
        end_date           = "2025-07-17",
        selected_place_ids = ["ChIJ...", ...],
        must_visit_ids     = ["ChIJ..."],
        trip_brief         = trip_planner_agent.build_trip_brief(...),
        weather_brief      = weather_agent.build_weather_brief(...),
        hotel_name         = "Taj Mahal Palace Hotel",
        hotel_address      = "Apollo Bunder, Colaba, Mumbai 400001",
        num_adults         = 2,
        num_children       = 0,
        trip_types         = ["Cultural", "Foodie"],
        budget_level       = "mid-range",
        travel_pace        = 3,
    )
    # itinerary is a fully enriched dict ready for the frontend

Standalone smoke-test:
    source venv/bin/activate
    python3 -m pipeline.itinerary_agent
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from pipeline.hotel_resolver import resolve_hotel_proxy, inject_hotel_proxy

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH  = DATA_DIR / "places_database.json"

load_dotenv(dotenv_path=ROOT / ".env")

# ─── Model config ─────────────────────────────────────────────────────────────

MODEL_NAME   = "gemini-2.5-flash"
TEMPERATURE  = 0.5    # Creative scheduling, still consistent
MAX_TOKENS   = 65536  # Gemini 2.5 Flash max — needed for large multi-day JSON itineraries

# ─── Categories that are food/leisure (do not count toward activity cap) ──────

FOOD_CATEGORIES = {
    "Restaurant", "Café", "Street Food", "Food", "Bakery",
    "Bar", "Night Market", "Local Market",
}

# ─── Slot type values the LLM may use ─────────────────────────────────────────

SLOT_TYPES = (
    "activity",         # proper sightseeing / attraction
    "meal",             # breakfast / lunch / dinner
    "leisure",          # free time / shopping / beach walk (no specific place)
    "transit",          # notable travel leg worth calling out
    "hotel_return",     # end-of-day return to hotel
)

# ─── JSON schema docstring (embedded in the prompt) ───────────────────────────

_ITINERARY_SCHEMA = """\
{
  "trip_summary": {
    "city":                    "string",
    "start_date":              "YYYY-MM-DD",
    "end_date":                "YYYY-MM-DD",
    "trip_days":               <integer>,
    "hotel": {
      "name":    "string",
      "address": "string"
    },
    "travellers": {
      "adults":   <integer>,
      "children": <integer>
    },
    "trip_style":              ["string"],
    "budget_level":            "budget|mid-range|luxury|ultra-luxury",
    "overall_weather_summary": "one-sentence trip-level weather summary"
  },

  "days": [
    {
      "day_number":           <integer>,
      "date":                 "YYYY-MM-DD",
      "day_label":            "Monday, 15 July 2025",
      "theme":                "Short descriptive theme, e.g. South Mumbai Heritage & Waterfront",
      "weather_note":         "Actionable one-liner: conditions + any warnings for this day",
      "estimated_driving_km": <number>,
      "activity_count":       <integer — proper sightseeing stops only, not meals>,

      "slots": [
        {
          "slot_id":               "d<day>_s<seq>",
          "time_slot":             "HH:MM–HH:MM",
          "type":                  "activity|meal|leisure|transit|hotel_return",
          "place_id":              "Google place_id string, or null for meals/leisure",
          "place_name":            "Name of place or meal suggestion",
          "duration_minutes":      <integer>,
          "notes":                 "What to do / see / eat here. Practical, specific.",
          "tips":                  ["Actionable tip 1", "Actionable tip 2"],
          "is_must_visit":         true|false,
          "weather_consideration": "Specific note if weather affects this slot, else null"
        }
      ]
    }
  ],

  "overflow_places": [
    {
      "place_id":   "string",
      "place_name": "string",
      "reason":     "Why it was not included (too isolated / trip too short / weather risk)"
    }
  ],

  "packing_tips": ["string"],
  "general_tips": ["string"]
}"""


# ─── Trip preferences container ─────────────────────────────────────────────

@dataclass
class TripPreferences:
    """All user-supplied preferences from the frontend form."""
    city_slug:    str
    city_name:    str
    start_date:   str
    end_date:     str
    num_adults:   int              = 1
    num_children: int              = 0
    trip_types:   list[str]        = field(default_factory=list)
    budget_level: str              = "mid-range"
    travel_pace:  int              = 3
    hotel_name:   str              = ""
    hotel_address: str             = ""


# ─── Database loader ──────────────────────────────────────────────────────────

def _load_place_details(city_slug: str, place_ids: list[str]) -> dict[str, dict]:
    """
    Load full place records from places_database.json for the given IDs.
    Returns {place_id: place_dict}.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    with open(DB_PATH, encoding="utf-8") as f:
        db = json.load(f)

    id_set  = set(place_ids)
    details: dict[str, dict] = {}

    for city in db.get("cities", []):
        if city["city_slug"] != city_slug:
            continue
        for place in city.get("places", []):
            if place["place_id"] in id_set:
                details[place["place_id"]] = place

    return details


# ─── Prompt builders ──────────────────────────────────────────────────────────

def _build_place_catalog(
    place_details: dict[str, dict],
    must_visit_ids: set[str],
) -> str:
    """
    Build a concise, structured text catalog of all selected places.
    Passed to the LLM so it knows what it has to schedule.
    """
    lines: list[str] = []

    # Sort: must-visit first, then by rating descending
    def sort_key(pid: str) -> tuple:
        p = place_details.get(pid, {})
        return (0 if pid in must_visit_ids else 1, -(p.get("rating") or 0))

    for pid in sorted(place_details.keys(), key=sort_key):
        p       = place_details[pid]
        name    = p.get("name", "Unknown")
        cat     = p.get("category", "N/A")
        rating  = p.get("rating")
        moods   = p.get("moods", [])
        note    = p.get("overall_note") or ""
        addr    = p.get("address", "")
        is_food = cat in FOOD_CATEGORIES

        priority  = "♥ MUST-VISIT" if pid in must_visit_ids else "✓ liked"
        type_tag  = "[food/drink – not an activity]" if is_food else "[activity]"
        mood_str  = f"  Moods: {', '.join(moods)}" if moods else ""
        rating_str = f"⭐{rating}" if rating else ""

        lines.append(
            f"  ID: {pid}\n"
            f"  Name: {name}  |  {cat}  |  {rating_str}  |  {priority}  {type_tag}\n"
            f"  Address: {addr}\n"
            f"  What to do: {note}"
            f"{mood_str}"
        )
        lines.append("")  # blank line between places

    return "\n".join(lines)


def _build_prompt(
    prefs: TripPreferences,
    trip_days: int,
    selected_count: int,
    must_visit_count: int,
    place_catalog: str,
    trip_brief: str,
    weather_brief: str,
) -> str:

    pace_map = {1: "very relaxed (1–2 activities/day)", 2: "relaxed (2 activities/day)",
                3: "balanced (3 activities/day)", 4: "busy (3 full activities/day)",
                5: "fully packed (max activities every day)"}
    pace_desc    = pace_map.get(prefs.travel_pace, "balanced")
    num_adults   = prefs.num_adults
    num_children = prefs.num_children
    trip_types   = prefs.trip_types
    budget_level = prefs.budget_level
    hotel_name   = prefs.hotel_name
    hotel_address = prefs.hotel_address

    has_children = num_children > 0
    child_note = (
        f"\n⚠️  Group includes {num_children} child(ren) — avoid late-night slots, "
        "prefer family-friendly timings, include rest breaks."
    ) if has_children else ""

    return f"""You are the senior Head Planner LLM — the master itinerary architect.
You have received pre-processed intelligence from two specialist agents and must
now produce a final, complete, day-by-day travel itinerary as a JSON object.

Your itinerary must be practical, realistic, well-paced, and tailored to the
user's preferences. Every decision must be grounded in the trip brief and weather
brief you have received. Do NOT hallucinate place names, addresses, or distances.
Only schedule places from the SELECTED PLACES catalog below. For meal slots you
may suggest general area recommendations or named restaurants from the catalog.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRIP CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
City              : {prefs.city_name}
Dates             : {prefs.start_date} → {prefs.end_date}  ({trip_days} day(s))
Hotel / base      : {hotel_name or "city centre (not specified)"}
                    {hotel_address}
Travellers        : {num_adults} adult(s), {num_children} child(ren){child_note}
Trip style        : {", ".join(trip_types) if trip_types else "general"}
Budget level      : {budget_level}
Travel pace       : {pace_desc}
Selected places   : {selected_count} total  ({must_visit_count} must-visit ♥)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GEOGRAPHIC & DISTANCE BRIEF  (from Trip Planner Agent)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{trip_brief}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WEATHER BRIEF  (from Weather Agent)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{weather_brief}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SELECTED PLACES CATALOG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{place_catalog}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HARD SCHEDULING RULES  (non-negotiable)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1.  HOTEL ANCHOR: Every day starts from the hotel (earliest slot ≥ 07:00) and
    ends with a return to the hotel (latest slot ≤ 22:30 for adults-only;
    ≤ 21:00 if children are present). Add a "hotel_return" slot at day end.

2.  ACTIVITY CAP: Max {3 if prefs.travel_pace <= 3 else 4} proper sightseeing/activity
    stops per day (places tagged [activity] in the catalog). Food, drinks, and
    leisure do not count toward this cap.

3.  DRIVING LIMIT: Max 100 km total driving per day (as stated in the trip brief).
    Respect the cluster allocations the Trip Planner Agent recommended.

4.  MUST-VISIT PRIORITY: All ♥ MUST-VISIT places MUST appear in the itinerary
    unless the trip brief explicitly flags them as physically impossible. They
    take absolute scheduling priority over liked-only places.

5.  WEATHER RESPECT: Follow all scheduling notes from the Weather Brief.
    — Rain/storm days → prefer indoor activities; flag outdoor slots.
    — Extreme heat → schedule outdoor activities before 11:00 or after 17:00.
    — Golden-hour slots → use sunrise/sunset windows from the weather brief.

6.  MEAL SLOTS: Include breakfast (≈07:00–08:00), lunch (≈12:30–13:30), and
    dinner (≈19:30–20:30) every day. For food/drink places from the catalog,
    assign them to the nearest appropriate meal slot. For other meals, suggest
    area/cuisine based on which cluster the user is in that day.

7.  PACE RESPECT: Match the travel pace the user selected.
    — Very relaxed / relaxed → add leisure / rest slots, shorter days.
    — Busy / fully packed → fill all available time; minimise gaps.

8.  OVERFLOW: Any place from the catalog that cannot fit into the {trip_days}-day
    schedule must appear in "overflow_places" with a clear reason.

9.  REALISM: Allow realistic travel/transit time between places (15–30 min for
    nearby places, 45–60 min for cross-city). Add transit slots for journeys
    > 30 km.

10. TIPS: Every activity slot must include 2–3 practical, specific tips
    (best time to visit, entry fees, what to order, what to wear, etc.).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY a valid JSON object matching this exact schema. No markdown fences,
no explanation text, no commentary outside the JSON.

{_ITINERARY_SCHEMA}

CRITICAL JSON RULES:
  • "place_id" must be the exact ID string from the catalog, or null for
    meal/leisure/transit/hotel_return slots.
  • Every slot MUST have a valid "type" from: {", ".join(SLOT_TYPES)}.
  • "slot_id" must be unique: "d1_s1", "d1_s2", "d2_s1" etc.
  • "time_slot" must use 24-hour HH:MM–HH:MM format.
  • "is_must_visit" is true only for ♥ must-visit catalog entries.
  • Slots within each day must be in chronological order.
  • "activity_count" = number of slots with type="activity" in that day.
  • Do not invent place_ids — only use IDs from the catalog above.
"""


# ─── Post-processing: inject full place data ──────────────────────────────────

def _enrich_itinerary(
    raw_itinerary: dict,
    place_details: dict[str, dict],
) -> dict:
    """
    Inject full place metadata (address, coordinates, photo_references,
    rating, category, description) into every slot that has a place_id.

    This keeps the LLM prompt lean (no verbose photo/coordinate data)
    while giving the frontend everything it needs to render each card.
    """
    for day in raw_itinerary.get("days", []):
        for slot in day.get("slots", []):
            pid = slot.get("place_id")
            if not pid:
                continue
            p = place_details.get(pid)
            if not p:
                continue
            slot["place_details"] = {
                "name":              p.get("name"),
                "category":          p.get("category"),
                "rating":            p.get("rating"),
                "total_ratings":     p.get("total_ratings"),
                "address":           p.get("address"),
                "latitude":          p.get("latitude"),
                "longitude":         p.get("longitude"),
                "description":       p.get("description"),
                "review_summary":    p.get("review_summary"),
                "overall_note":      p.get("overall_note"),
                "moods":             p.get("moods", []),
                "photo_references":  p.get("photo_references", []),
            }
    return raw_itinerary


# ─── JSON extraction helper ───────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """
    Parse the LLM response as JSON. Handles optional markdown code fences
    that some models add even when instructed not to.
    """
    # Strip markdown fences if present
    stripped = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    stripped = re.sub(r"\s*```$", "", stripped.strip(), flags=re.MULTILINE)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Gemini returned invalid JSON.\n"
            f"Parse error: {exc}\n"
            f"Raw output (first 500 chars):\n{text[:500]}"
        ) from exc


# ─── Public API ───────────────────────────────────────────────────────────────

def build_itinerary(
    prefs: TripPreferences,
    selected_place_ids: list[str],
    must_visit_ids: Optional[list[str]] = None,
    trip_brief: str = "",
    weather_brief: str = "",
) -> dict:
    """
    Core function — orchestrates the Head Planner LLM call and returns a
    fully enriched itinerary dict ready for the frontend.

    Flow:
      1. Load full place metadata from the database
      2. Build a structured place catalog for the prompt
      3. Build the master prompt with all context (trip brief + weather brief
         + preferences + catalog + schema)
      4. Call Gemini 2.5 Flash in JSON mode
      5. Parse and validate the JSON
      6. Enrich every slot with full place data (photos, coordinates, etc.)
      7. Return the enriched dict

    Args:
        prefs              : TripPreferences dataclass with all form data
        selected_place_ids : All place IDs the user liked or hearted
        must_visit_ids     : Subset that were hearted ♥ (highest priority)
        trip_brief         : Output from trip_planner_agent.build_trip_brief()
        weather_brief      : Output from weather_agent.build_weather_brief()

    Returns:
        Fully enriched itinerary dict.

    Raises:
        FileNotFoundError : if places_database.json does not exist
        RuntimeError      : if GEMINI_API_KEY is not set or JSON parsing fails
    """
    must_visit_ids = set(must_visit_ids or [])

    start     = date.fromisoformat(prefs.start_date)
    end       = date.fromisoformat(prefs.end_date)
    trip_days = (end - start).days + 1

    # ── 0. Hotel proxy resolution ───────────────────────────────────────────
    # Geocode the hotel address and find the nearest database place as a proxy.
    # The proxy anchors the itinerary: it gets highest-priority must-visit status
    # so the LLM treats it as the daily departure / return point.
    print("\n" + "─" * 50)
    proxy = resolve_hotel_proxy(
        city_slug     = prefs.city_slug,
        hotel_address = prefs.hotel_address,
        hotel_name    = prefs.hotel_name,
        city_name     = prefs.city_name,
        verbose       = True,
    )

    # Inject proxy into selection lists
    sel_list  = list(selected_place_ids)
    must_list = list(must_visit_ids)
    sel_list, must_list = inject_hotel_proxy(proxy, sel_list, must_list)
    must_visit_ids     = set(must_list)
    selected_place_ids = sel_list

    # Enrich the prefs label so the LLM prompt carries proxy context
    import dataclasses as _dc
    prefs = _dc.replace(prefs, hotel_address=proxy.hotel_label)

    print(f"   Hotel proxy → '{proxy.proxy_place_name}'  ({proxy.distance_km:.2f} km)")
    print("─" * 50 + "\n")

    # ── 1. Load place details ─────────────────────────────────────────────────
    print(f"📚 Loading place details for {len(selected_place_ids)} selected places…")
    place_details = _load_place_details(prefs.city_slug, selected_place_ids)
    if not place_details:
        raise RuntimeError(
            "No place details found in the database for the selected IDs. "
            f"Run the pipeline for '{prefs.city_slug}' first."
        )
    print(f"   {len(place_details)} place records loaded.")

    # ── 2. Build place catalog ────────────────────────────────────────────────
    place_catalog = _build_place_catalog(place_details, must_visit_ids)

    # ── 3. Build master prompt ────────────────────────────────────────────────
    print("📝 Building master prompt…")
    prompt = _build_prompt(
        prefs            = prefs,
        trip_days        = trip_days,
        selected_count   = len(place_details),
        must_visit_count = len(must_visit_ids),
        place_catalog    = place_catalog,
        trip_brief       = trip_brief if trip_brief else "(No trip brief provided — use general judgment.)",
        weather_brief    = weather_brief if weather_brief else "(No weather brief provided — assume normal conditions.)",
    )

    # ── 4. Call LLM (Gemini or Llama, driven by PIPELINE_LLM toggle) — with truncation recovery ─
    from pipeline.model_config import get_llm_client, get_model_name, call_llm, provider_label, _active_provider
    _model_name = get_model_name()
    _client     = get_llm_client()

    def _call_llm_with_truncation_check(prompt_text: str, label: str = "") -> tuple[str, bool]:
        """Call the active LLM and return (raw_text, was_truncated)."""
        tag = f" [{label}]" if label else ""
        print(f"🤖 Calling {provider_label()} (JSON mode){tag}…")
        truncated = False

        if _active_provider() == "llama":
            # Groq — no native finish_reason for MAX_TOKENS in the same way
            text = call_llm(
                client      = _client,
                model_name  = _model_name,
                prompt      = prompt_text,
                temperature = TEMPERATURE,
                max_tokens  = MAX_TOKENS,
                json_mode   = True,
            )
        else:
            # Gemini — can inspect finish_reason
            from google import genai as _genai
            from google.genai import types as _genai_types
            resp = _client.models.generate_content(
                model    = _model_name,
                contents = prompt_text,
                config   = _genai_types.GenerateContentConfig(
                    temperature        = TEMPERATURE,
                    max_output_tokens  = MAX_TOKENS,
                    response_mime_type = "application/json",
                ),
            )
            text = resp.text.strip() if resp.text else ""
            try:
                candidate = resp.candidates[0] if resp.candidates else None
                if candidate:
                    reason = str(getattr(candidate, 'finish_reason', '') or '')
                    if 'MAX_TOKENS' in reason.upper() or reason == '2':
                        truncated = True
                        print(f"   ⚠️  finish_reason=MAX_TOKENS — output was truncated! ({len(text)} chars)")
            except Exception:
                pass

        print(f"   Raw response: {len(text)} chars | truncated={truncated}")
        return text, truncated

    raw_text, truncated = _call_llm_with_truncation_check(prompt, label="attempt-1")

    # ── 5. Parse JSON — retry once with concise mode if truncated / invalid ───
    print("🔍 Parsing itinerary JSON…")
    raw_itinerary = None

    for parse_attempt in range(2):
        try:
            raw_itinerary = _extract_json(raw_text)
            break   # success
        except RuntimeError as parse_err:
            if parse_attempt == 0:
                print(f"   ⚠️  Parse attempt 1 failed ({parse_err!s:.120}…)")
                print("   🔄  Retrying with CONCISE mode (fewer tips per slot)…")
                concise_note = (
                    "\n\n⚠️  CONCISE MODE: The previous response was too long and got truncated. "
                    "This time, include ONLY 1 short tip per slot (max 80 characters). "
                    "Keep all slot names, times, notes, and place_ids — only reduce the tips array. "
                    "Return the same complete JSON schema but with minimal tips."
                )
                raw_text, truncated = _call_llm_with_truncation_check(prompt + concise_note, label="attempt-2-concise")
            else:
                raise RuntimeError(
                    f"LLM returned invalid JSON (both attempts failed).\n"
                    f"Parse error: {parse_err}\n"
                    f"Raw output (first 800 chars):\n{raw_text[:800]}"
                ) from parse_err

    if raw_itinerary is None:
        raise RuntimeError("Itinerary parsing produced no result after 2 attempts.")

    # Basic validation
    if "days" not in raw_itinerary:
        raise RuntimeError(
            "LLM response is missing the required 'days' key.\n"
            f"Keys found: {list(raw_itinerary.keys())}"
        )
    days_found = len(raw_itinerary["days"])
    print(f"   {days_found} day(s) parsed.")

    # ── 6. Enrich with full place data ────────────────────────────────────────
    print("💎 Enriching slots with place details…")
    itinerary = _enrich_itinerary(raw_itinerary, place_details)

    # ── 7. Attach metadata ────────────────────────────────────────────────────
    itinerary["_meta"] = {
        "generated_at":       date.today().isoformat(),
        "model":              _model_name,
        "city_slug":          prefs.city_slug,
        "total_selected":     len(place_details),
        "must_visit_count":   len(must_visit_ids),
        "had_trip_brief":     bool(trip_brief),
        "had_weather_brief":  bool(weather_brief),
    }

    total_slots = sum(len(d.get("slots", [])) for d in itinerary["days"])
    print(f"✅ Itinerary ready — {days_found} day(s), {total_slots} slot(s) total.")
    return itinerary


# ─── Standalone smoke-test ────────────────────────────────────────────────────

def _run_test() -> None:
    """
    Full end-to-end smoke test using real Mumbai data.
    Chains all three agents: trip_planner → weather → itinerary.
    """
    from pipeline.trip_planner_agent import build_trip_brief
    from pipeline.weather_agent import build_weather_brief

    print("\n" + "═" * 70)
    print("  🧪  Head Planner (Itinerary Agent) — smoke test (Mumbai, 3 days)")
    print("═" * 70 + "\n")

    # ── Load real Mumbai place IDs from the matrix meta ───────────────────────
    import json as _json
    meta_path = ROOT / "data" / "matrices" / "mumbai_meta.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = _json.load(f)

    all_ids   = meta["place_ids"]
    all_names = meta["place_names"]

    # Simulate user selecting 10 places (3 hearted, 7 liked)
    step     = max(1, len(all_ids) // 10)
    selected = all_ids[::step][:10]
    must     = selected[:3]

    name_map = dict(zip(all_ids, all_names))
    print("Selected places:")
    for pid in selected:
        tag = " ♥" if pid in must else " ✓"
        print(f"  {tag} {name_map.get(pid, pid)}")

    # ── Trip dates: 3 days from now (forecast mode) ───────────────────────────
    today      = date.today()
    start_str  = (today + timedelta(days=4)).isoformat()
    end_str    = (today + timedelta(days=6)).isoformat()

    # ── Step 1: Trip planner brief ────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("Step 1: Building trip brief (trip_planner_agent)…")
    trip_brief = build_trip_brief(
        city_slug          = "mumbai",
        selected_place_ids = selected,
        must_visit_ids     = must,
        trip_days          = 3,
        num_adults         = 2,
        num_children       = 0,
        trip_types         = ["Cultural", "Foodie"],
        budget_level       = "mid-range",
        travel_pace        = 3,
        hotel_address      = "Taj Mahal Palace Hotel, Apollo Bunder, Colaba, Mumbai 400001",
    )
    print("Trip brief generated ✓")

    # ── Step 2: Weather brief ─────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("Step 2: Building weather brief (weather_agent)…")
    weather_brief = build_weather_brief(
        city_name    = "Mumbai",
        lat          = 19.0760,
        lng          = 72.8777,
        start_date   = start_str,
        end_date     = end_str,
        trip_types   = ["Cultural", "Foodie"],
        num_adults   = 2,
        num_children = 0,
        timezone     = "Asia/Kolkata",
    )
    print("Weather brief generated ✓")

    # ── Step 3: Head LLM — build itinerary ────────────────────────────────────
    print(f"\n{'─'*70}")
    print("Step 3: Building itinerary (itinerary_agent)…")
    prefs = TripPreferences(
        city_slug     = "mumbai",
        city_name     = "Mumbai",
        start_date    = start_str,
        end_date      = end_str,
        num_adults    = 2,
        num_children  = 0,
        trip_types    = ["Cultural", "Foodie"],
        budget_level  = "mid-range",
        travel_pace   = 3,
        hotel_name    = "Taj Mahal Palace Hotel",
        hotel_address = "Apollo Bunder, Colaba, Mumbai 400001",
    )
    itinerary = build_itinerary(
        prefs              = prefs,
        selected_place_ids = selected,
        must_visit_ids     = must,
        trip_brief         = trip_brief,
        weather_brief      = weather_brief,
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("ITINERARY SUMMARY")
    print(f"{'═'*70}")

    summary = itinerary.get("trip_summary", {})
    print(f"  City      : {summary.get('city')}")
    print(f"  Dates     : {summary.get('start_date')} → {summary.get('end_date')}")
    print(f"  Weather   : {summary.get('overall_weather_summary')}")

    for day in itinerary.get("days", []):
        print(f"\n  {day['day_label']} — {day['theme']}")
        print(f"  Weather: {day['weather_note']}")
        for slot in day.get("slots", []):
            icon = {"activity": "🏛", "meal": "🍽", "leisure": "🌊",
                    "transit": "🚗", "hotel_return": "🏨"}.get(slot["type"], "•")
            mv   = " ♥" if slot.get("is_must_visit") else ""
            print(f"    {icon} {slot['time_slot']}  {slot['place_name']}{mv}")

    overflow = itinerary.get("overflow_places", [])
    if overflow:
        print(f"\n  ⚠️  Overflow ({len(overflow)} places not scheduled):")
        for p in overflow:
            print(f"     • {p['place_name']}: {p['reason']}")

    print(f"\n  Packing tips: {', '.join(itinerary.get('packing_tips', []))}")
    print(f"  General tips: {', '.join(itinerary.get('general_tips', []))}")

    # Save the full JSON output
    out_path = ROOT / "data" / "itinerary_test_output.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(itinerary, f, ensure_ascii=False, indent=2)
    print(f"\n  Full JSON saved → {out_path}")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    _run_test()