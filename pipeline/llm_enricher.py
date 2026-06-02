"""
llm_enricher.py
Uses Groq (llama-3.3-70b-versatile) to enrich normalized place records.

Per place, a single LLM call generates three things:
  1. description    → Compacted to ≤200 words, engaging travel-writing style
  2. review_summary → A 2–3 sentence paragraph summarizing top reviews
  3. overall_note   → What you can do / experience at this place
                       (generated using category, google_types, moods as context)

After enrichment, the temp context fields (_google_types, _moods, _raw_reviews)
are stripped from the final saved record.

Rate limit handling:
  If Groq returns a 429 (RateLimitError), processing stops immediately.
  Already-enriched places are kept; remaining get null LLM fields.
  Re-run the pipeline to fill in the remaining places (smart resume will
  pick up only the null ones automatically).
"""

import os
import time
import json
import re
from groq import Groq, RateLimitError

MODEL_NAME = "llama-3.3-70b-versatile"

# Delay between Groq API calls (free tier: 30 req/min → 2s gap is comfortable)
GROQ_DELAY_S = 2.0

SYSTEM_PROMPT = (
    "You are a travel content editor. You write concise, engaging descriptions "
    "of tourist places and summarize visitor reviews. Always respond with valid "
    "JSON only — no markdown fences, no extra text, no explanation."
)


def _build_prompt(place: dict) -> str:
    """Builds a single prompt that generates description, review_summary, and overall_note."""
    name        = place.get("name", "this place")
    category    = place.get("category", "")
    types       = ", ".join(place.get("_google_types", [])[:6])
    moods       = ", ".join(place.get("_moods", []))
    description = place.get("description") or ""
    raw_reviews = place.get("_raw_reviews", [])

    reviews_text = ""
    if raw_reviews:
        reviews_text = "\n".join(
            f'- [{r["rating"]}★] {r["author"]}: "{r["text"]}"'
            for r in raw_reviews
            if r.get("text")
        )

    return f"""Given the details of a tourist place, return a JSON object with exactly three keys:

1. "description"
   A concise, engaging travel-style paragraph about the place in at most 200 words.
   Use the editorial text as your base. If none is provided, write from the name and category.
   Do NOT mention ratings, prices, or hours. Focus on what makes this place special.

2. "review_summary"
   A 2–3 sentence paragraph capturing the overall visitor sentiment from the reviews.
   Highlight mood, standout positives, and any notable mentions.
   Set to null if no reviews are provided.

3. "overall_note"
   A single sentence (max 25 words) describing what a visitor can DO or EXPERIENCE at this place.
   Think: activities, atmosphere, what kind of traveller this suits.
   Use the place type context (types, moods, category) to make this specific and useful.
   Examples:
     "Perfect for history buffs — explore ancient cave temples and colonial-era monuments."
     "Ideal for food lovers looking to sample Mumbai street food and local snacks."
     "Great for families — enjoy water rides, theme park attractions, and live shows."

Place: {name}
Category: {category}
Google place types: {types if types else "N/A"}
Vibe tags: {moods if moods else "N/A"}

Editorial description:
{description if description else "(none provided)"}

Recent visitor reviews:
{reviews_text if reviews_text else "(none provided)"}

Return ONLY a JSON object: {{"description": "...", "review_summary": "...", "overall_note": "..."}}"""


def _parse_response(text: str) -> dict | None:
    """Extracts the JSON object from the model response."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "description" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    return None


def _clean_context_fields(place: dict) -> dict:
    """Strips temporary LLM context fields that should not be saved to the DB."""
    for key in ("_google_types", "_moods", "_raw_reviews"):
        place.pop(key, None)
    return place


def _enrich_one(place: dict, client: Groq) -> dict:
    """
    Enriches a single place via one Groq API call.

    Adds:  description (compacted), review_summary, overall_note
    Strips: _google_types, _moods, _raw_reviews

    Raises:
        RateLimitError: propagated up so the caller can stop the loop cleanly
    """
    has_content = bool(place.get("description") or place.get("_raw_reviews"))

    if not has_content:
        place["review_summary"] = None
        place["overall_note"]   = None
        return _clean_context_fields(place)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_prompt(place)},
            ],
            temperature=0.4,
            max_tokens=700,
        )

        raw_text = response.choices[0].message.content or ""
        parsed = _parse_response(raw_text)

        if parsed:
            place["description"]    = parsed.get("description") or place.get("description")
            place["review_summary"] = parsed.get("review_summary")
            place["overall_note"]   = parsed.get("overall_note")
        else:
            print(f"\n    ⚠️  Could not parse LLM response for '{place['name']}'")
            place["review_summary"] = None
            place["overall_note"]   = None

    except RateLimitError:
        # Propagate rate limit up — caller handles the graceful stop
        raise

    except Exception as e:
        print(f"\n    ⚠️  Groq error for '{place['name']}': {e}")
        place["review_summary"] = None
        place["overall_note"]   = None

    return _clean_context_fields(place)


def enrich_all_places(places: list) -> list:
    """
    Runs LLM enrichment on all places using Groq + llama-3.3-70b-versatile.

    Stops gracefully if Groq's rate limit is hit — already-enriched places
    are kept; remaining places get null LLM fields so the smart-resume
    logic in main.py can fill them in on the next run.

    Args:
        places: List of normalized place dicts (from normalize.py)

    Returns:
        List of place dicts — enriched ones have full LLM data,
        rate-limited ones have null review_summary / overall_note
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key or api_key == "YOUR_GROQ_API_KEY_HERE":
        print(
            "\n⚠️  GROQ_API_KEY not set — skipping LLM enrichment.\n"
            "   Set GROQ_API_KEY in .env and re-run to fill in LLM data.\n"
            "   Get a free key at https://console.groq.com/keys"
        )
        for place in places:
            place["review_summary"] = None
            place["overall_note"]   = None
            _clean_context_fields(place)
        return places

    client = Groq(api_key=api_key)
    total = len(places)
    print(f"\n🤖 LLM enrichment via Groq / {MODEL_NAME} ({total} places)...")
    print(f"   Generating: description · review_summary · overall_note")

    enriched = []
    rate_limited = False

    for i, place in enumerate(places):
        print(f"  [{i+1}/{total}] {place['name']}... ", end="", flush=True)

        try:
            place = _enrich_one(place, client)
            enriched.append(place)
            print("✓")
        except RateLimitError:
            print("⛔ rate limit")
            print(
                f"\n  ⛔ Groq daily rate limit reached after {i} place(s).\n"
                f"  {total - i} place(s) will be saved with null LLM fields.\n"
                f"  Re-run this city tomorrow or with a higher-tier key to fill them in."
            )
            # Clean up and null-fill remaining places without calling the API
            place["review_summary"] = None
            place["overall_note"]   = None
            _clean_context_fields(place)
            enriched.append(place)

            for remaining in places[i + 1:]:
                remaining["review_summary"] = None
                remaining["overall_note"]   = None
                _clean_context_fields(remaining)
                enriched.append(remaining)

            rate_limited = True
            break

        if i < total - 1:
            time.sleep(GROQ_DELAY_S)

    done = sum(1 for p in enriched if p.get("overall_note"))
    if not rate_limited:
        print(f"\n✅ LLM enrichment complete — {done}/{total} places fully enriched")

    return enriched
