"""
pipeline/weather_agent.py
=========================

A Gemini 2.5 Flash agent that acts as a **weather briefing analyst**.

Given:
  • Trip destination (lat, lng, city name)
  • Trip start and end dates

It:
  1. Decides data source automatically:
       - Trip starts within 12 days  → Open-Meteo Forecast API (live data, incl. rain probability)
       - Trip starts after 12 days   → Open-Meteo Historical API (same calendar dates, averaged
                                        over the past 2 years — no probability, but real observed data)
  2. Fetches daily weather aggregates (max/min temp, precipitation, wind, UV, WMO codes, etc.)
  3. Converts the raw response into a human-readable per-day digest
  4. Calls Gemini 2.5 Flash to interpret the digest and produce a plain-English
     weather brief for the Head Planner LLM.

Business rules enforced by the Gemini agent:
  ─ Rain probability / precip thresholds → umbrella / indoor-activities / storm warning
  ─ Temperature extremes (heat advisory, cold advisory)
  ─ Wind speed / gust warnings
  ─ UV index sunscreen recommendations (forecast only)
  ─ WMO weather code warnings (thunderstorm, snow, fog, freezing rain …)
  ─ Trip-level "wet trip" or "ideal weather" summary
  ─ Sunrise/sunset windows for golden-hour activity scheduling

Output is a plain-text brief designed to be inserted directly into the Head-LLM prompt.

Usage:
    from pipeline.weather_agent import build_weather_brief
    brief = build_weather_brief(
        city_name    = "Mumbai",
        lat          = 19.076,
        lng          = 72.877,
        start_date   = "2025-07-15",
        end_date     = "2025-07-17",
        trip_types   = ["Beach", "Cultural"],
        timezone     = "Asia/Kolkata",
    )
    print(brief)

Standalone smoke-test:
    source venv/bin/activate
    python3 -m pipeline.weather_agent
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT / ".env")

# ─── Model config (driven by PIPELINE_LLM env var set in main.py) ───────────────
# Import lazily inside functions so env var is read at call-time, not import-time.

TEMPERATURE = 0.2        # Low → consistent, rule-following output
MAX_TOKENS  = 4096

# ─── API endpoints ────────────────────────────────────────────────────────────

FORECAST_API_URL       = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_API_URL        = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_HORIZON_DAYS  = 12   # max reliable forecast window for Open-Meteo

# ─── Variables to request ─────────────────────────────────────────────────────

# Forecast API supports precipitation_probability_max and uv_index_max
FORECAST_DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    "precipitation_probability_max",   # forecast only
    "weather_code",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "sunrise",
    "sunset",
    "sunshine_duration",
    "uv_index_max",                    # forecast only
]

# Historical/archive API (ERA5) does not have probability or UV index
HISTORICAL_DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    "weather_code",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "sunrise",
    "sunset",
    "sunshine_duration",
]

# ─── WMO weather code look-up ─────────────────────────────────────────────────

WMO_DESCRIPTIONS: dict[int, str] = {
    0:  "Clear sky",
    1:  "Mainly clear",
    2:  "Partly cloudy",
    3:  "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Heavy freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

_DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ─── Data fetching ─────────────────────────────────────────────────────────────

def _fetch_forecast_daily(
    lat: float,
    lng: float,
    start_date: str,
    end_date: str,
    timezone: str,
) -> dict:
    """Call Open-Meteo Forecast API and return the raw `daily` block."""
    params = {
        "latitude":   lat,
        "longitude":  lng,
        "start_date": start_date,
        "end_date":   end_date,
        "daily":      ",".join(FORECAST_DAILY_VARS),
        "timezone":   timezone or "auto",
    }
    resp = requests.get(FORECAST_API_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"Open-Meteo forecast error: {data.get('reason', data)}")
    return data["daily"]


def _shifted_date(d: date, year_offset: int) -> date:
    """Shift a date back by `year_offset` years, clamping Feb-29 → Feb-28."""
    try:
        return d.replace(year=d.year - year_offset)
    except ValueError:
        # Feb 29 in a leap year shifted to a non-leap year
        return d.replace(year=d.year - year_offset, day=28)


def _fetch_historical_daily(
    lat: float,
    lng: float,
    start: date,
    end: date,
    year_offset: int,
    timezone: str,
) -> dict:
    """Fetch historical data for the same calendar window shifted back by `year_offset` years."""
    hist_start = _shifted_date(start, year_offset)
    hist_end   = _shifted_date(end,   year_offset)
    params = {
        "latitude":   lat,
        "longitude":  lng,
        "start_date": hist_start.isoformat(),
        "end_date":   hist_end.isoformat(),
        "daily":      ",".join(HISTORICAL_DAILY_VARS),
        "timezone":   timezone or "auto",
    }
    resp = requests.get(ARCHIVE_API_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"Open-Meteo archive error: {data.get('reason', data)}")
    return data["daily"]


def _average_two_years(year1: dict, year2: dict) -> dict:
    """
    Element-wise average of two historical daily dicts.
    Non-numeric fields use year1 as the source.
    For `weather_code` the *worse* (higher) code is kept as a conservative estimate.
    """
    n = min(len(year1.get("time", [])), len(year2.get("time", [])))

    numeric_vars = [
        "temperature_2m_max", "temperature_2m_min",
        "apparent_temperature_max", "apparent_temperature_min",
        "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours",
        "wind_speed_10m_max", "wind_gusts_10m_max",
        "sunshine_duration",
    ]

    merged: dict = {"time": year1["time"][:n]}

    for var in numeric_vars:
        v1 = year1.get(var, [None] * n)
        v2 = year2.get(var, [None] * n)
        avg_vals = []
        for a, b in zip(v1[:n], v2[:n]):
            if a is not None and b is not None:
                avg_vals.append(round((a + b) / 2.0, 1))
            else:
                avg_vals.append(a if a is not None else b)
        merged[var] = avg_vals

    # Use the more severe weather code of the two years (conservative)
    wc1 = year1.get("weather_code", [0] * n)
    wc2 = year2.get("weather_code", [0] * n)
    merged["weather_code"] = [
        max(int(a or 0), int(b or 0)) for a, b in zip(wc1[:n], wc2[:n])
    ]

    # Sunrise/sunset shift ~1 min/year — year1 values are close enough
    merged["sunrise"] = year1.get("sunrise", [None] * n)[:n]
    merged["sunset"]  = year1.get("sunset",  [None] * n)[:n]

    return merged


# ─── Digest builder helpers ───────────────────────────────────────────────────

def _hhmm(s: object) -> str:
    """Extract HH:MM from an ISO datetime string like '2025-07-15T06:12'."""
    if s and "T" in str(s):
        return str(s).split("T")[1][:5]
    return str(s) if s else "N/A"


def _dval(daily: dict, key: str, idx: int) -> object:
    """Safely index a key in the daily dict; returns None if missing."""
    vals = daily.get(key, [])
    return vals[idx] if idx < len(vals) else None


def _fmt_temp(daily: dict, i: int) -> tuple[str, str]:
    t_max  = _dval(daily, "temperature_2m_max", i)
    t_min  = _dval(daily, "temperature_2m_min", i)
    ft_max = _dval(daily, "apparent_temperature_max", i)
    ft_min = _dval(daily, "apparent_temperature_min", i)
    return (
        f"{t_min}–{t_max}°C" if t_max is not None else "N/A",
        f"{ft_min}–{ft_max}°C" if ft_min is not None else "N/A",
    )


def _fmt_precip(daily: dict, i: int) -> str:
    precip = _dval(daily, "precipitation_sum", i)
    snow   = _dval(daily, "snowfall_sum", i)
    p_hrs  = _dval(daily, "precipitation_hours", i)
    rain_p = _dval(daily, "precipitation_probability_max", i)
    base   = f"{precip:.1f} mm" if precip is not None else "N/A"
    hrs    = f" over {p_hrs:.0f} h" if p_hrs else ""
    sno    = f" | Snow: {snow:.1f} cm" if snow and snow > 0 else ""
    prob   = f" | Rain prob: {rain_p}%" if rain_p is not None else ""
    return f"{base}{hrs}{sno}{prob}"


def _fmt_wind(daily: dict, i: int) -> str:
    w_max  = _dval(daily, "wind_speed_10m_max", i)
    w_gust = _dval(daily, "wind_gusts_10m_max", i)
    base   = f"{w_max:.0f} km/h max" if w_max is not None else "N/A"
    gust   = f", gusts {w_gust:.0f} km/h" if w_gust is not None else ""
    return base + gust


def _fmt_sun(daily: dict, i: int) -> str:
    sr      = _hhmm(_dval(daily, "sunrise", i))
    ss      = _hhmm(_dval(daily, "sunset", i))
    sun_sec = _dval(daily, "sunshine_duration", i)
    sun_hrs = f"{sun_sec / 3600:.1f} h" if sun_sec is not None else "N/A"
    return f"Sunrise: {sr}  |  Sunset: {ss}  |  Sunshine: {sun_hrs}"


def _build_day_block(daily: dict, idx: int, trip_start: date, is_historical: bool) -> str:
    """Format the text block for a single trip day."""
    d     = trip_start + timedelta(days=idx)
    label = f"Day {idx+1} ({_DAY_NAMES[d.weekday()]} {d.strftime('%d %b')})"
    if is_historical:
        label += "  [historical avg]"

    wmo      = _dval(daily, "weather_code", idx)
    wmo_int  = int(wmo) if wmo is not None else None
    wmo_desc = WMO_DESCRIPTIONS.get(wmo_int, f"Code {wmo_int}") if wmo_int is not None else "Unknown"

    temp_str, feels_str = _fmt_temp(daily, idx)
    uv      = _dval(daily, "uv_index_max", idx)
    uv_line = f"\n  • UV Index      : {uv} (max)" if uv is not None else ""

    return (
        f"{label}\n"
        f"  • Condition     : {wmo_desc} (WMO {wmo_int})\n"
        f"  • Temperature   : {temp_str}  |  Feels like: {feels_str}\n"
        f"  • Precipitation : {_fmt_precip(daily, idx)}\n"
        f"  • Wind          : {_fmt_wind(daily, idx)}"
        f"{uv_line}\n"
        f"  • {_fmt_sun(daily, idx)}"
    )


# ─── Digest builder ───────────────────────────────────────────────────────────

def _build_weather_digest(
    daily: dict,
    trip_start: date,
    is_historical: bool,
) -> str:
    """
    Convert the raw daily dict into a clean, structured per-day text block
    ready to be inserted into the LLM prompt.
    """
    n = len(daily.get("time", []))

    source_note = (
        "⚠️  Data source: Historical averages (ERA5 reanalysis, same calendar dates averaged "
        "over the past 2 years).\n"
        "    Precipitation probability is NOT available — rain risk is inferred from "
        "precipitation_sum and WMO weather codes.\n"
    ) if is_historical else (
        "ℹ️  Data source: Open-Meteo live forecast (precipitation probability and UV index included).\n"
    )

    day_blocks = [
        _build_day_block(daily, i, trip_start, is_historical)
        for i in range(n)
    ]
    return source_note + "\n" + "\n\n".join(day_blocks)


# ─── Prompt builder ───────────────────────────────────────────────────────────

def _build_prompt(
    city_name: str,
    trip_days: int,
    start_date: str,
    end_date: str,
    trip_types: list[str],
    num_adults: int,
    num_children: int,
    is_historical: bool,
    weather_digest: str,
) -> str:

    data_caveat = (
        "NOTE: This is HISTORICAL weather data (same calendar dates, averaged over the "
        "past 2 years from ERA5 reanalysis). Rain probability percentages are NOT available "
        "— infer rain risk from precipitation totals and WMO weather codes. Present findings "
        "as 'historically typical for this time of year', not as a live forecast."
    ) if is_historical else (
        "NOTE: This is a LIVE FORECAST from Open-Meteo. Rain probability percentages and "
        "UV index are available and reliable for the stated dates."
    )

    return f"""You are a travel weather analyst. Your job is to interpret raw daily weather data
for a trip and produce a concise, actionable **weather brief** for a Head Planner LLM
that will build the actual day-by-day itinerary. The Head LLM will NOT see the raw
weather data — it will only read your brief.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRIP CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
City         : {city_name}
Dates        : {start_date} → {end_date}  ({trip_days} day(s))
Travellers   : {num_adults} adult(s), {num_children} child(ren)
Trip style   : {", ".join(trip_types) if trip_types else "general"}

{data_caveat}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WEATHER DATA (per day)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{weather_digest}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTERPRETATION RULES  (apply all that are relevant)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RAIN / PRECIPITATION:
  • Rain prob ≥ 70%  OR precipitation_sum > 8 mm  → "Heavy rain — recommend indoor activities;
                                                      carry umbrella and rain jacket."
  • Rain prob 40–69% OR precipitation_sum 4–8 mm  → "Significant rain likely — plan indoor
                                                      backups; carry umbrella."
  • Rain prob 20–39% OR precipitation_sum 1–4 mm  → "Light rain possible — carry an umbrella."
  • Rain prob < 20%  OR precipitation_sum < 1 mm  → "Dry day — no rain precautions needed."
  • WMO 95 / 96 / 99  (Thunderstorm)             → "⚠️ THUNDERSTORM WARNING — avoid open spaces,
                                                      boat tours, and hilltop viewpoints. Strongly
                                                      prefer indoor activities."
  • WMO 65 / 67 / 82  (Heavy/violent rain)        → Flag as severe; recommend indoor backup plan.
  • WMO 71–77 / 85–86 (Snow)                      → "❄️ SNOWFALL — outdoor travel may be disrupted."
  • WMO 45 / 48       (Fog)                        → "🌫️ FOG — early-morning sightseeing may have
                                                      reduced visibility; allow extra travel time."

TEMPERATURE (use apparent/feels-like for recommendations):
  • Feels-like max > 38°C  → "⚠️ EXTREME HEAT — avoid outdoor activity 11 am–4 pm; carry water,
                               seek shade; high exertion is risky for children."
  • Feels-like max 32–38°C → "Hot day — apply high-SPF sunscreen, carry water; schedule outdoor
                               activities in the morning or after 5 pm."
  • Feels-like max 24–32°C → "Comfortable/warm — excellent for outdoor activity."
  • Feels-like min < 5°C   → "Cold morning — warm layers recommended."
  • Feels-like min < 0°C   → "⚠️ FREEZING — heavy winter clothing essential."
  • Daily swing (max − min > 15°C) → "Large temperature swing — dress in layers."

WIND:
  • Wind gusts > 80 km/h  → "⚠️ DANGEROUS WINDS — avoid exposed outdoor activities; check
                              local advisories before boat tours or cliff walks."
  • Wind gusts 50–80 km/h → "Strong winds — skip boat trips, cliff walks, open hilltop sites."
  • Wind gusts 30–50 km/h → "Moderately windy — breezy but manageable for most activities."

UV INDEX  (forecast days only):
  • UV ≥ 11     → "⚠️ EXTREME UV — SPF 50+ mandatory; avoid midday sun."
  • UV 8–10     → "Very high UV — SPF 50+, hat and sunglasses essential."
  • UV 6–7      → "High UV — apply SPF 30+ sunscreen."
  • UV 3–5      → "Moderate UV — SPF 15 recommended."
  • UV < 3      → "Low UV — minimal sunscreen needed."

TRIP-LEVEL PATTERNS:
  • ≥ 50% of days have significant rain (prob ≥ 40% or precip > 4 mm)
                → Label the trip as ⛈️ "WET TRIP" and recommend the travellers bring
                  waterproof gear and prioritise covered/indoor attractions.
  • All days comfortable (24–32°C feels-like, rain prob < 20%, wind gusts < 30 km/h)
                → Label as ☀️ "Ideal weather — great conditions for outdoor sightseeing."
  • Any day with a weather warning (thunderstorm / extreme heat / dangerous wind)
                → Mark that day clearly so the Head Planner knows to schedule safe activities.

SUNRISE / SUNSET:
  • Always mention these so the Head Planner can assign golden-hour visits (forts,
    viewpoints, beaches) and know the safe outdoor window for each day.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR OUTPUT  (weather brief for the Head LLM)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Write a structured plain-English brief with these four sections:

1. **TRIP-LEVEL WEATHER OVERVIEW**
   2–3 sentences: overall conditions, dominant pattern, and whether the weather
   is favourable for the stated trip style. Apply the ⛈️ / ☀️ labels if applicable.

2. **DAY-BY-DAY WEATHER SUMMARY**
   For each day:
   - Date label   (e.g. "Day 1 — Mon 15 Jul")
   - One-line condition  (e.g. "Hot and partly cloudy, 28–35°C feels like 30–40°C")
   - Applicable warnings or recommendations  (rain / heat / wind / UV — from the rules above)
   - Best outdoor time window for that day  (e.g. "Outdoor activities best before 11 am or after 5 pm")

3. **PACKING & PREPARATION CHECKLIST**
   Concise bullet list of what the travellers should bring based on the full trip
   (umbrella, rain jacket, SPF 50 sunscreen, warm layers, etc.).

4. **SCHEDULING NOTES FOR THE HEAD PLANNER**
   Bullet points the Head LLM must respect when assigning activities to days:
   - Which days are best for outdoor sightseeing / beaches / hikes
   - Which days should prefer indoor attractions (museums, galleries, malls)
   - Any severe-weather days that need a contingency plan
   - Sunrise/sunset windows for time-sensitive golden-hour activities

Be concise and directive. Do NOT copy back the raw numbers —
the Head LLM needs actionable guidance, not a data table.
"""


# ─── Public API ───────────────────────────────────────────────────────────────

def build_weather_brief(
    city_name: str,
    lat: float,
    lng: float,
    start_date: str,
    end_date: str,
    trip_types: Optional[list[str]] = None,
    num_adults: int = 1,
    num_children: int = 0,
    timezone: str = "auto",
) -> str:
    """
    Core function — generates a weather planning brief for the Head Planner LLM.

    Decision logic:
        • Trip starts within 12 days  → Forecast API  (live data, rain probability, UV)
        • Trip starts after  12 days  → Historical API (ERA5, same calendar dates,
                                          averaged over past 2 years)

    Args:
        city_name    : Display name used in the brief (e.g. "Mumbai")
        lat, lng     : WGS84 coordinates of the destination city
        start_date   : Trip start as "YYYY-MM-DD"
        end_date     : Trip end   as "YYYY-MM-DD" (inclusive)
        trip_types   : User's selected trip styles e.g. ["Beach", "Cultural"]
        num_adults   : Number of adult travellers
        num_children : Number of child travellers
        timezone     : IANA timezone string or "auto" (resolved by Open-Meteo)

    Returns:
        Weather brief as a plain-text string.

    Raises:
        RuntimeError  : if GEMINI_API_KEY is not set or API calls fail
        requests.HTTPError : on Open-Meteo HTTP errors
    """
    trip_types = trip_types or []

    import time as _t
    _fn_start = _t.time()
    from datetime import datetime as _dt
    def _ts(): return _dt.now().strftime("%H:%M:%S.%f")[:-3]

    today           = date.today()
    start           = date.fromisoformat(start_date)
    end             = date.fromisoformat(end_date)
    trip_days       = (end - start).days + 1
    days_until_trip = (start - today).days
    is_historical   = not (0 <= days_until_trip <= FORECAST_HORIZON_DAYS)

    print(f"\n{'='*64}")
    print(f"[WEATHER] ► START  {_ts()}")
    print(f"[WEATHER]   city={city_name} | lat={lat} | lng={lng} | tz={timezone}")
    print(f"[WEATHER]   dates={start_date} → {end_date} ({trip_days} days) | days_until={days_until_trip}")
    print(f"[WEATHER]   mode={'HISTORICAL (ERA5)' if is_historical else 'LIVE FORECAST'}")
    print(f"[WEATHER]   types={trip_types} | adults={num_adults} | children={num_children}")
    print(f"{'='*64}")

    # ── 1. Fetch weather data ─────────────────────────────────────────────────
    _t1 = _t.time()
    if not is_historical:
        print(f"\n[WEATHER] Step 1: Live forecast API  |  START {_ts()}")
        print(f"[WEATHER]   Trip starts in {days_until_trip} day(s) — fetching Open-Meteo forecast…")
        daily = _fetch_forecast_daily(lat, lng, start_date, end_date, timezone)
        print(f"[WEATHER] Step 1 ✅  |  END {_ts()}  |  elapsed={_t.time()-_t1:.2f}s  |  {len(daily)} field(s)")
    else:
        mode = "past date" if days_until_trip < 0 else f"far future ({days_until_trip}d out)"
        print(f"\n[WEATHER] Step 1: Historical ERA5 archive ({mode})  |  START {_ts()}")

        _ty1 = _t.time()
        print(f"[WEATHER]   Fetching year −1…")
        year1 = _fetch_historical_daily(lat, lng, start, end, year_offset=1, timezone=timezone)
        print(f"[WEATHER]   Year −1 done  |  elapsed={_t.time()-_ty1:.2f}s")

        _ty2 = _t.time()
        print(f"[WEATHER]   Fetching year −2…")
        try:
            year2 = _fetch_historical_daily(lat, lng, start, end, year_offset=2, timezone=timezone)
            print(f"[WEATHER]   Year −2 done  |  elapsed={_t.time()-_ty2:.2f}s")
        except Exception as exc:
            print(f"[WEATHER]   ⚠️  Year −2 fetch failed ({exc}) — using year −1 only")
            year2 = year1

        daily = _average_two_years(year1, year2)
        print(f"[WEATHER] Step 1 ✅  |  END {_ts()}  |  total elapsed={_t.time()-_t1:.2f}s")

    # ── 2. Build human-readable weather digest ────────────────────────────────
    _t2 = _t.time()
    print(f"\n[WEATHER] Step 2: Build weather digest  |  START {_ts()}")
    weather_digest = _build_weather_digest(daily, start, is_historical)
    print(f"[WEATHER] Step 2 ✅  |  END {_ts()}  |  elapsed={_t.time()-_t2:.2f}s  |  digest chars={len(weather_digest)}")
    print(f"[WEATHER]   Weather digest preview:\n{weather_digest[:800]}{'...' if len(weather_digest)>800 else ''}")

    # ── 3. Build prompt ───────────────────────────────────────────────────────
    _t3 = _t.time()
    print(f"\n[WEATHER] Step 3: Build LLM prompt  |  START {_ts()}")
    prompt = _build_prompt(
        city_name      = city_name,
        trip_days      = trip_days,
        start_date     = start_date,
        end_date       = end_date,
        trip_types     = trip_types,
        num_adults     = num_adults,
        num_children   = num_children,
        is_historical  = is_historical,
        weather_digest = weather_digest,
    )
    print(f"[WEATHER] Step 3 ✅  |  END {_ts()}  |  elapsed={_t.time()-_t3:.2f}s  |  prompt chars={len(prompt)}")

    # ── 4. Call LLM ───────────────────────────────────────────────────────────
    from pipeline.model_config import get_llm_client, get_model_name, call_llm, provider_label
    model_name = get_model_name()
    client     = get_llm_client()
    _t4 = _t.time()
    print(f"\n[WEATHER] Step 4: LLM call  |  model={provider_label()}  |  START {_ts()}")

    brief = call_llm(
        client      = client,
        model_name  = model_name,
        prompt      = prompt,
        temperature = TEMPERATURE,
        max_tokens  = MAX_TOKENS,
        json_mode   = False,
    )

    _t4_elapsed = _t.time() - _t4
    _fn_elapsed = _t.time() - _fn_start
    print(f"[WEATHER] Step 4 ✅  |  END {_ts()}  |  LLM elapsed={_t4_elapsed:.2f}s")
    print(f"[WEATHER] ◄ TOTAL elapsed={_fn_elapsed:.2f}s  |  brief chars={len(brief)}")
    print(f"{'='*64}\n")
    return brief


# ─── Standalone smoke-test ────────────────────────────────────────────────────

def _run_test() -> None:
    """
    Quick smoke-test using real Mumbai coordinates.
    Runs two scenarios: forecast mode (3 days from now) and historical mode (60 days out).
    """
    print("\n" + "═" * 64)
    print("  🧪  Weather Agent — smoke test (Mumbai, 19.076°N 72.877°E)")
    print("═" * 64 + "\n")

    today = date.today()

    # ── Test 1: Forecast mode ────────────────────────────────────────────────
    fs  = (today + timedelta(days=3)).isoformat()
    fe  = (today + timedelta(days=5)).isoformat()
    print(f"── Test 1: Forecast mode  ({fs} → {fe}) ──\n")

    brief1 = build_weather_brief(
        city_name    = "Mumbai",
        lat          = 19.0760,
        lng          = 72.8777,
        start_date   = fs,
        end_date     = fe,
        trip_types   = ["Cultural", "Foodie"],
        num_adults   = 2,
        num_children = 0,
        timezone     = "Asia/Kolkata",
    )
    print("\n" + "─" * 64)
    print("WEATHER BRIEF (Forecast)")
    print("─" * 64)
    print(brief1)

    # ── Test 2: Historical mode ───────────────────────────────────────────────
    hs  = (today + timedelta(days=60)).isoformat()
    he  = (today + timedelta(days=62)).isoformat()
    print(f"\n── Test 2: Historical mode  ({hs} → {he}) ──\n")

    brief2 = build_weather_brief(
        city_name    = "Mumbai",
        lat          = 19.0760,
        lng          = 72.8777,
        start_date   = hs,
        end_date     = he,
        trip_types   = ["Beach", "Adventure"],
        num_adults   = 2,
        num_children = 1,
        timezone     = "Asia/Kolkata",
    )
    print("\n" + "─" * 64)
    print("WEATHER BRIEF (Historical)")
    print("─" * 64)
    print(brief2)
    print("─" * 64 + "\n")


if __name__ == "__main__":
    _run_test()
