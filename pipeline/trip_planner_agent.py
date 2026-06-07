"""
pipeline/trip_planner_agent.py
==============================

A Gemini 2.5 Flash agent that acts as a **smart trip planning analyst**.

Given:
  • The user's selected places (liked + must-visit) for a city
  • The driving-distance sub-matrix for only those places (sliced from the
    full city matrix)

It produces a structured **plain-English planning brief** that the downstream
Head LLM can use directly when building the actual day-by-day itinerary,
without ever seeing the raw distance matrix.

Rules enforced by this agent:
  ─ User starts and ends every day at their hotel (round-trip constraint)
  ─ Max 100 km total driving per day
  ─ Max 3 proper sightseeing / activity stops per day
    (restaurants, cafés, street food, markets → do NOT count toward the 3)
  ─ Must-visit places get priority scheduling
  ─ Cluster nearby places onto the same day to minimise backtracking
  ─ Flag any places that are very isolated (>40 km from all others) so the
    Head LLM can decide whether to skip or schedule a dedicated half-day

Output is a plain-text brief (no JSON, no tables) — prose + structured
bullet points — designed to be pasted directly into a Head-LLM prompt.

Usage (standalone test):
    source venv/bin/activate
    python3 -m pipeline.trip_planner_agent

    Or import and call:
        from pipeline.trip_planner_agent import build_trip_brief
        brief = build_trip_brief(city_slug, selected_place_ids, must_visit_ids)
        print(brief)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv

# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT / "data"
MATRICES_DIR = DATA_DIR / "matrices"
DB_PATH      = DATA_DIR / "places_database.json"

load_dotenv(dotenv_path=ROOT / ".env")

# ─── Model config ─────────────────────────────────────────────────────────────

MODEL_NAME   = "gemini-2.5-flash-preview-05-20"
TEMPERATURE  = 0.3        # Low temperature → consistent, logical clustering
MAX_TOKENS   = 8192

# ─── Planning rules (also injected into the prompt) ───────────────────────────

MAX_KM_PER_DAY           = 100
MAX_ACTIVITIES_PER_DAY   = 3     # proper sightseeing only
ISOLATION_THRESHOLD_KM   = 40   # flag if all neighbours > this distance
NO_ROUTE                 = -1.0  # sentinel in the distance matrix

# Food categories that do NOT count toward the activity cap
FOOD_CATEGORIES = {
    "Restaurant", "Café", "Street Food", "Food", "Bakery",
    "Bar", "Night Market", "Local Market", "Shopping Mall",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_meta(city_slug: str) -> dict:
    """Load the matrix metadata JSON for a city."""
    path = MATRICES_DIR / f"{city_slug}_meta.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Matrix metadata not found: {path}\n"
            f"Run:  python3 distance_matrix.py \"{city_slug}\"  first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_matrix(city_slug: str) -> np.ndarray:
    """Load the full NxN distance matrix (.npy) for a city."""
    path = MATRICES_DIR / f"{city_slug}_matrix.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Distance matrix not found: {path}\n"
            f"Run:  python3 distance_matrix.py \"{city_slug}\"  first."
        )
    return np.load(str(path))


def _load_place_details(city_slug: str, place_ids: list[str]) -> dict[str, dict]:
    """
    Fetch full place records from places_database.json for the given IDs.
    Returns {place_id: place_dict}.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    with open(DB_PATH, encoding="utf-8") as f:
        db = json.load(f)

    id_set = set(place_ids)
    details: dict[str, dict] = {}

    for city in db.get("cities", []):
        if city["city_slug"] != city_slug:
            continue
        for place in city.get("places", []):
            if place["place_id"] in id_set:
                details[place["place_id"]] = place

    return details


def _slice_matrix(
    full_matrix: np.ndarray,
    all_ids: list[str],
    selected_ids: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    Extract the sub-matrix for the selected place IDs only.

    Returns:
        sub_matrix : (K x K) ndarray where K = len(selected_ids)
        ordered_ids: the IDs in the row/column order of sub_matrix
    """
    id_to_idx = {pid: i for i, pid in enumerate(all_ids)}

    # Only keep IDs that exist in the full matrix
    valid_ids = [pid for pid in selected_ids if pid in id_to_idx]
    indices   = [id_to_idx[pid] for pid in valid_ids]

    sub = full_matrix[np.ix_(indices, indices)]
    return sub, valid_ids


def _build_distance_table(
    sub_matrix: np.ndarray,
    ordered_ids: list[str],
    place_details: dict[str, dict],
    must_visit_ids: Optional[set[str]] = None,
) -> str:
    """
    Convert the sub-matrix into a compact, human-readable table string
    for the LLM prompt.  Uses short numeric labels (P1, P2 …) so the
    table stays narrow even for 30+ places.

    Format:
        P1=Gateway of India (Historic & Cultural)
        P2=Juhu Beach (Beach)
        ...

        Distance table (km, -1 = no road route):
              P1    P2    P3  ...
        P1   0.0   12.3   8.5
        P2  12.3    0.0   5.1
        ...
    """
    n = len(ordered_ids)
    labels = [f"P{i+1}" for i in range(n)]

    # Legend
    must_visit_ids = must_visit_ids or set()
    legend_lines = []
    for label, pid in zip(labels, ordered_ids):
        p        = place_details.get(pid, {})
        name     = p.get("name", pid[:12])
        cat      = p.get("category", "")
        rating   = p.get("rating")
        r_str    = f" ⭐{rating}" if rating else ""
        mv_tag   = " ★MUST-VISIT" if pid in must_visit_ids else ""
        legend_lines.append(f"  {label} = {name} [{cat}]{r_str}{mv_tag}")
    legend = "\n".join(legend_lines)

    # Column header
    col_w = 6
    header = " " * 6 + "".join(f"{lbl:>{col_w}}" for lbl in labels)

    # Rows
    row_lines = []
    for i, lbl in enumerate(labels):
        cells = []
        for j in range(n):
            val = sub_matrix[i, j]
            if val == NO_ROUTE:
                cells.append(f"{'N/A':>{col_w}}")
            else:
                km = val / 1000.0
                cells.append(f"{km:>{col_w}.1f}")
        row_lines.append(f"{lbl:<6}" + "".join(cells))

    table = "\n".join([header] + row_lines)
    return f"Place legend:\n{legend}\n\nDriving distances (km):\n{table}"


def _is_food(place: dict) -> bool:
    """Returns True if this place is food/market and should not count as an activity."""
    cat = place.get("category", "")
    moods = set(place.get("moods", []))
    return cat in FOOD_CATEGORIES or moods & {"foodie", "nightlife", "shopping"}


def _build_prompt(
    city_name: str,
    trip_days: int,
    num_adults: int,
    num_children: int,
    trip_types: list[str],
    budget_level: str,
    travel_pace: int,
    hotel_address: str,
    must_visit_ids: set[str],
    ordered_ids: list[str],
    place_details: dict[str, dict],
    distance_table: str,
) -> str:
    """Construct the full prompt for the Gemini agent."""

    # Annotate each place
    place_annotations = []
    for pid in ordered_ids:
        p        = place_details.get(pid, {})
        name     = p.get("name", "Unknown")
        cat      = p.get("category", "N/A")
        rating   = p.get("rating", "N/A")
        is_food  = _is_food(p)
        priority = "MUST-VISIT ★" if pid in must_visit_ids else "liked"
        food_tag = " [food/market – does not count as activity]" if is_food else " [activity]"
        note     = p.get("overall_note") or (p.get("description") or "")[:120]
        place_annotations.append(
            f"  • {name} | {cat} | ⭐{rating} | {priority}{food_tag}\n"
            f"    Note: {note}"
        )

    annotations_str = "\n".join(place_annotations)

    pace_words = {1: "very relaxed", 2: "relaxed", 3: "balanced", 4: "busy", 5: "fully packed"}
    pace_desc  = pace_words.get(travel_pace, "balanced")

    return f"""You are a smart trip planning analyst. Your job is to analyse a set of tourist places
and their driving distances, then produce a concise **planning brief** for a Head Planner LLM
that will build the actual day-by-day itinerary.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRIP CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
City            : {city_name}
Trip duration   : {trip_days} day(s)
Travellers      : {num_adults} adult(s), {num_children} child(ren)
Trip style      : {", ".join(trip_types) if trip_types else "general"}
Budget          : {budget_level}
Travel pace     : {pace_desc} (scale 1–5, user selected {travel_pace})
Hotel / base    : {hotel_address if hotel_address else "city centre (exact address not provided)"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SELECTED PLACES ({len(ordered_ids)} total)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{annotations_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISTANCE DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{distance_table}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PLANNING RULES (non-negotiable)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. The user STARTS and ENDS every day at their hotel. The distance matrix
   only covers place-to-place distances; exact hotel-to-place distances are
   NOT in the data. Estimate hotel-to-cluster driving based on the hotel
   location provided above and the cluster's general area — clearly mark
   any such estimates as approximate in your brief.
2. Maximum total driving per day: {MAX_KM_PER_DAY} km (one-way legs summed).
3. Maximum proper sightseeing/activity stops per day: {MAX_ACTIVITIES_PER_DAY}.
   Restaurants, cafés, street food, local markets — these do NOT count
   toward the activity cap and can be added freely between activities.
4. MUST-VISIT places (marked ★) have the highest scheduling priority.
   They should appear before liked-only places and should not be dropped
   unless physically impossible given distance constraints.
5. Group nearby places together on the same day to minimise backtracking.
   Prefer clusters where the total inter-place driving is under 30 km.
6. If a place is isolated (all distances to other selected places > {ISOLATION_THRESHOLD_KM} km),
   flag it explicitly so the Head Planner can decide: schedule a dedicated
   half-day trip, combine with hotel transit, or drop it.
7. If there are more place-days needed than trip days available, note the
   overflow and recommend which places to deprioritise (always keep ★ places).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR OUTPUT (planning brief for the Head LLM)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Write a structured plain-English brief with the following sections:

1. **OVERVIEW**
   One short paragraph: total places, trip feasibility (can everything fit in
   {trip_days} days?), general geographic spread of the selected places.

2. **GEOGRAPHIC CLUSTERS**
   List the natural clusters you identified. For each cluster:
   - Give it a descriptive name (e.g. "South Mumbai Heritage Belt")
   - List the place labels (P1, P2 …) and their full names
   - State the approximate intra-cluster driving span (km)
   - Suggest which day(s) this cluster suits

3. **DAY-BY-DAY RECOMMENDED ALLOCATION**
   For each day, suggest:
   - Which cluster / places to visit
   - Estimated total driving (km) for the day
   - Number of activity stops (not counting food)
   - Any must-visit places included in that day

4. **ISOLATED PLACES** (if any)
   List places that are too far from all clusters and flag them.

5. **OVERFLOW / DROPPED PLACES** (if applicable)
   If the trip is too short to cover everything, list what should be
   deprioritised and why.

6. **KEY CONSTRAINTS FOR THE HEAD PLANNER**
   A bullet-point summary of the most important routing facts the Head LLM
   must respect when building the final schedule (distances, hard limits,
   special notes).

Write clearly and concisely. The Head LLM will use this brief directly —
do not include the raw distance table in your output.
"""


# ─── Public API ───────────────────────────────────────────────────────────────

def build_trip_brief(
    city_slug: str,
    selected_place_ids: list[str],
    must_visit_ids: Optional[list[str]] = None,
    trip_days: int = 3,
    num_adults: int = 1,
    num_children: int = 0,
    trip_types: Optional[list[str]] = None,
    budget_level: str = "mid-range",
    travel_pace: int = 3,
    hotel_address: str = "",
) -> str:
    """
    Core function — builds the planning brief by:
      1. Loading the distance matrix for the city
      2. Slicing it to only the selected places
      3. Calling Gemini 2.5 Flash with the distance table + constraints
      4. Returning the model's plain-text planning brief

    Args:
        city_slug          : e.g. "mumbai"
        selected_place_ids : all place IDs the user liked OR marked must-visit
        must_visit_ids     : subset of selected_place_ids marked as must-visit (♥)
        trip_days          : number of days for the trip
        num_adults         : number of adult travellers
        num_children       : number of child travellers
        trip_types         : list of trip style strings, e.g. ["Cultural", "Foodie"]
        budget_level       : "budget" | "mid-range" | "luxury" | "ultra-luxury"
        travel_pace        : 1 (very relaxed) – 5 (fully packed)
        hotel_address      : hotel name + address (or empty string)

    Returns:
        Planning brief as a plain-text string.

    Raises:
        FileNotFoundError : if matrix files don't exist for the city
        RuntimeError      : if GEMINI_API_KEY is not set
    """
    must_visit_ids = set(must_visit_ids or [])
    trip_types     = trip_types or []

    # ── 1. Load matrix + metadata ────────────────────────────────────────────
    print(f"📐 Loading distance matrix for '{city_slug}'…")
    meta       = _load_meta(city_slug)
    full_matrix = _load_matrix(city_slug)
    all_ids    = meta["place_ids"]
    city_name  = meta.get("city_name", city_slug.title())

    # ── 2. Slice to selected places only ────────────────────────────────────
    sub_matrix, ordered_ids = _slice_matrix(full_matrix, all_ids, selected_place_ids)
    K = len(ordered_ids)
    print(f"   {K} selected places → {K}×{K} sub-matrix extracted")

    if K == 0:
        return "⚠️  No valid place IDs found in the matrix. Cannot build trip brief."

    # ── 3. Load full place details (name, category, moods, etc.) ────────────
    print("📚 Loading place details from database…")
    place_details = _load_place_details(city_slug, ordered_ids)

    # ── 4. Build distance table string ──────────────────────────────────────
    distance_table = _build_distance_table(sub_matrix, ordered_ids, place_details, must_visit_ids)

    # ── 5. Build prompt ──────────────────────────────────────────────────────
    prompt = _build_prompt(
        city_name      = city_name,
        trip_days      = trip_days,
        num_adults     = num_adults,
        num_children   = num_children,
        trip_types     = trip_types,
        budget_level   = budget_level,
        travel_pace    = travel_pace,
        hotel_address  = hotel_address,
        must_visit_ids = must_visit_ids,
        ordered_ids    = ordered_ids,
        place_details  = place_details,
        distance_table = distance_table,
    )

    # ── 6. Call Gemini 2.5 Flash ─────────────────────────────────────────────
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set in .env. "
            "Get a key at https://aistudio.google.com/app/apikey"
        )

    print(f"🤖 Calling {MODEL_NAME}…")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model  = MODEL_NAME,
        contents = prompt,
        config = genai_types.GenerateContentConfig(
            temperature       = TEMPERATURE,
            max_output_tokens = MAX_TOKENS,
        ),
    )
    brief = response.text.strip()

    print(f"✅ Planning brief generated ({len(brief)} chars)")
    return brief


# ─── Standalone test ──────────────────────────────────────────────────────────

def _run_test() -> None:
    """
    Quick smoke-test using real Mumbai data.
    Simulates a user who liked 10 places and hearted 3 of them.
    """
    print("\n" + "═" * 64)
    print("  🧪  Trip Planner Agent — smoke test (Mumbai)")
    print("═" * 64 + "\n")

    # Load a handful of real Mumbai place IDs from the matrix meta
    meta = _load_meta("mumbai")
    all_ids  = meta["place_ids"]
    all_names = meta["place_names"]

    # Pick 12 places spread across the list to simulate variety
    step = max(1, len(all_ids) // 12)
    selected = all_ids[::step][:12]
    must     = selected[:3]   # first 3 are "must-visit"

    print(f"Selected {len(selected)} places, {len(must)} must-visit:\n")
    name_map = dict(zip(all_ids, all_names))
    for pid in selected:
        tag = " ★" if pid in must else ""
        print(f"  {'♥' if pid in must else '✓'} {name_map.get(pid, pid)}{tag}")

    print()
    brief = build_trip_brief(
        city_slug          = "mumbai",
        selected_place_ids = selected,
        must_visit_ids     = must,
        trip_days          = 3,
        num_adults         = 2,
        num_children       = 0,
        trip_types         = ["Cultural", "Foodie", "Adventure"],
        budget_level       = "mid-range",
        travel_pace        = 3,
        hotel_address      = "Taj Mahal Palace Hotel, Apollo Bunder, Colaba, Mumbai 400001",
    )

    print("\n" + "─" * 64)
    print("PLANNING BRIEF")
    print("─" * 64)
    print(brief)
    print("─" * 64 + "\n")


if __name__ == "__main__":
    _run_test()