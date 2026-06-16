"""
HTTP server for the Itinerary Planner frontend.
- Serves all static files from the project root
- Exposes GET /api/config with the Google API key (for photo fetching)
- Exposes POST /api/generate-itinerary  → SSE stream of pipeline events

Run from project root:  python3 serve.py
Then open:  http://localhost:8000/frontend/index.html
"""

import http.server
import socketserver
import json
import re
import os
import sys
import time
import threading
import traceback
from datetime import date, timedelta
from pathlib import Path

PORT = 8000
ROOT = Path(__file__).resolve().parent

# Make sure imports from pipeline/ work
sys.path.insert(0, str(ROOT))

os.chdir(str(ROOT))


# ─── Env helpers ──────────────────────────────────────────────────────────────

def read_env_key(key_name):
    try:
        with open('.env') as f:
            for line in f:
                m = re.match(rf'{key_name}\s*=\s*(.+)', line.strip())
                if m:
                    return m.group(1).strip()
    except Exception:
        pass
    return ''


def load_all_env():
    """Load .env variables into os.environ so pipeline modules can use them."""
    try:
        with open('.env') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                m = re.match(r'([A-Z_][A-Z0-9_]*)\s*=\s*(.+)', line)
                if m:
                    os.environ.setdefault(m.group(1), m.group(2).strip())
    except Exception:
        pass


load_all_env()

# ───────────────────────────────────────────────────────────────────────────────
# 🔄 LLM BACKEND TOGGLE — mirrors the setting in main.py
#    Change this one line to switch all three pipeline agents (trip_planner,
#    weather, itinerary) between Gemini 2.5 Flash and Llama 3.3 70B via Groq.
#
#    USE_LLAMA = False  → Google Gemini 2.5 Flash  (default, requires GEMINI_API_KEY)
#    USE_LLAMA = True   → Llama 3.3 70B via Groq   (requires GROQ_API_KEY)
#
#    The LLM enricher (llm_enricher.py) always uses Llama/Groq regardless.
# ───────────────────────────────────────────────────────────────────────────────
USE_LLAMA = True

os.environ["PIPELINE_LLM"] = "llama" if USE_LLAMA else "gemini"
print(f"[CONFIG] PIPELINE_LLM = {'llama (Groq llama-3.3-70b-versatile)' if USE_LLAMA else 'gemini (gemini-2.5-flash)'}")


# ─── SSE helpers ──────────────────────────────────────────────────────────────

def sse_event(data: dict) -> bytes:
    """Encode a dict as an SSE data line."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode('utf-8')


# ─── Retry helper for transient Gemini errors (503 / 429) ────────────────────

MAX_RETRIES   = 3
RETRY_BACKOFF = [5, 15, 30]   # seconds to wait before each retry attempt

def _call_with_retry(stage_name: str, fn: callable, send: callable, **kwargs):
    """
    Call `fn(**kwargs)` with up to MAX_RETRIES retries on 503/429 errors.
    Sends a 'status' SSE event before each retry so the user knows what's
    happening. Raises the last exception if all retries are exhausted.
    """
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            if attempt > 0:
                wait = RETRY_BACKOFF[min(attempt - 1, len(RETRY_BACKOFF) - 1)]
                retry_msg = (f"⏳ {stage_name} is temporarily overloaded "
                             f"(attempt {attempt}/{MAX_RETRIES}) — retrying in {wait}s…")
                print(f"[PIPELINE] {retry_msg}")
                send({'stage': 'status', 'msg': retry_msg})
                time.sleep(wait)

            print(f"[PIPELINE] Calling {stage_name} (attempt {attempt + 1}/{MAX_RETRIES + 1})…")
            result = fn(**kwargs)
            print(f"[PIPELINE] {stage_name} succeeded on attempt {attempt + 1}.")
            return result

        except Exception as exc:
            last_exc = exc
            err_str  = str(exc)
            # Only retry on transient server-side errors
            if any(code in err_str for code in ('503', '429', 'UNAVAILABLE', 'RESOURCE_EXHAUSTED')):
                print(f"[PIPELINE] {stage_name} transient error (attempt {attempt + 1}): {exc}")
                if attempt < MAX_RETRIES:
                    continue   # retry
            # Non-retriable error — raise immediately
            raise

    # All retries exhausted
    raise last_exc


# ─── Pipeline runner (runs in the SSE handler's thread) ─────────────────────

PIPELINE_SEP = "═" * 70
STAGE_SEP    = "─" * 60

def _ts() -> str:
    """Current timestamp string for logs."""
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]   # HH:MM:SS.mmm

def run_pipeline(request_body: dict, send: callable):
    """
    Full pipeline orchestration.
    `send(event_dict)` writes an SSE event to the client.

    Events emitted:
      {stage: 'status', msg: '...'}           – loader stage message
      {stage: 'day',    day: {...}}            – one rendered day object
      {stage: 'meta',   ...}                  – overflow / tips / summary after days
      {stage: 'done'}                          – all done
      {stage: 'error',  msg: '...'}            – fatal error (named stage)
    """
    import time as _time
    pipeline_start = _time.time()

    from pipeline.model_config import provider_label as _provider_label
    _llm_label = _provider_label()   # e.g. "Google Gemini 2.5 Flash (gemini-2.5-flash)"

    print(f"\n{PIPELINE_SEP}")
    print(f"[PIPELINE] ►►► START  {_ts()}")
    print(f"[PIPELINE] LLM backend : {_llm_label}")
    print(f"[PIPELINE] Request keys: {list(request_body.keys())}")
    print(PIPELINE_SEP)

    # ── Extract request data ──────────────────────────────────────────────────
    city_slug     = request_body.get('citySlug', '')
    city_name     = request_body.get('cityName', city_slug.title())
    selected_ids  = request_body.get('selectedPlaceIds', [])
    must_visit_ids = request_body.get('mustVisitIds', [])
    prefs_raw     = request_body.get('preferences', {})

    start_date    = prefs_raw.get('startDate', '')
    end_date      = prefs_raw.get('endDate', '')
    num_adults    = int(prefs_raw.get('numAdults', 1))
    num_children  = int(prefs_raw.get('numChildren', 0))
    trip_types    = prefs_raw.get('tripTypes', [])
    budget_level  = prefs_raw.get('budgetLevel', 'mid-range')
    travel_pace   = int(prefs_raw.get('travelPace', 3))
    hotel_name    = prefs_raw.get('hotelName', '')
    hotel_address = prefs_raw.get('accommodationAddress', '')

    if not start_date:
        start_date = (date.today() + timedelta(days=7)).isoformat()
    if not end_date:
        end_date = (date.today() + timedelta(days=9)).isoformat()

    trip_days = (date.fromisoformat(end_date) - date.fromisoformat(start_date)).days + 1

    # ── Stage 0: Import pipeline modules ─────────────────────────────────────
    _s0_start = _time.time()
    send({'stage': 'status', 'msg': '⚙️  Loading pipeline modules…'})
    print(f"\n{STAGE_SEP}")
    print(f"[STAGE 0] Import modules  |  START {_ts()}")
    print(STAGE_SEP)
    try:
        from pipeline.trip_planner_agent import build_trip_brief
        from pipeline.weather_agent import build_weather_brief
        from pipeline.itinerary_agent import build_itinerary, TripPreferences
        _s0_elapsed = _time.time() - _s0_start
        print(f"[STAGE 0] ✅ Modules loaded  |  END {_ts()}  |  elapsed={_s0_elapsed:.2f}s")
    except Exception as exc:
        msg = f"Import error (pipeline modules): {exc}"
        print(f"[STAGE 0] ❌ {msg}")
        send({'stage': 'error', 'msg': msg})
        return

    # ── Stage 1: Load city coordinates ───────────────────────────────────────
    _s1_start = _time.time()
    send({'stage': 'status', 'msg': '🗺️  Loading city data…'})
    print(f"\n{STAGE_SEP}")
    print(f"[STAGE 1] City coordinates  |  START {_ts()}")
    print(STAGE_SEP)

    import json as _json
    db_path = ROOT / 'data' / 'places_database.json'
    lat, lng, timezone_str = 19.0760, 72.8777, 'Asia/Kolkata'  # safe Mumbai defaults

    if db_path.exists():
        try:
            with open(db_path, encoding='utf-8') as f:
                db = _json.load(f)
            for city in db.get('cities', []):
                if city.get('city_slug') == city_slug:
                    coords = city.get('coordinates', {})
                    lat = coords.get('lat', lat)
                    lng = coords.get('lng', lng)
                    print(f"[STAGE 1]   Source: places_database.json → lat={lat}, lng={lng}")
                    break
            else:
                meta_path = ROOT / 'data' / 'matrices' / f'{city_slug}_meta.json'
                if meta_path.exists():
                    with open(meta_path, encoding='utf-8') as f:
                        meta = _json.load(f)
                    coords_list = meta.get('place_coords', [])
                    if coords_list:
                        lat = sum(c[0] for c in coords_list) / len(coords_list)
                        lng = sum(c[1] for c in coords_list) / len(coords_list)
                        print(f"[STAGE 1]   Source: matrix centroid → lat={lat:.4f}, lng={lng:.4f}")
        except Exception as exc:
            print(f"[STAGE 1]   ⚠️  Could not load coords ({exc}) — using defaults")
    else:
        print("[STAGE 1]   ⚠️  places_database.json not found — using Mumbai defaults")

    _s1_elapsed = _time.time() - _s1_start
    print(f"[STAGE 1] ✅ Done  |  END {_ts()}  |  elapsed={_s1_elapsed:.2f}s  |  lat={lat}, lng={lng}, tz={timezone_str}")

    # ── Stage 2: Trip planning brief ─────────────────────────────────────────
    _s2_start = _time.time()
    send({'stage': 'status', 'msg': '📍 Resolving hotel & calculating distances…'})
    print(f"\n{STAGE_SEP}")
    print(f"[STAGE 2] Trip Planner Agent  |  {_llm_label}  |  START {_ts()}")
    print(f"[STAGE 2]   city={city_slug}  |  days={trip_days}  |  selected={len(selected_ids)}  |  must-visit={len(must_visit_ids)}")
    print(f"[STAGE 2]   hotel='{hotel_address or hotel_name}'  |  types={trip_types}  |  pace={travel_pace}  |  budget={budget_level}")
    print(STAGE_SEP)
    try:
        trip_brief = _call_with_retry(
            stage_name = f'Trip Planner Agent ({_llm_label})',
            fn         = build_trip_brief,
            send       = send,
            city_slug          = city_slug,
            selected_place_ids = list(selected_ids),
            must_visit_ids     = list(must_visit_ids),
            trip_days          = trip_days,
            num_adults         = num_adults,
            num_children       = num_children,
            trip_types         = trip_types,
            budget_level       = budget_level,
            travel_pace        = travel_pace,
            hotel_address      = hotel_address or hotel_name,
        )
        _s2_elapsed = _time.time() - _s2_start
        print(f"[STAGE 2] ✅ Trip brief done  |  END {_ts()}  |  elapsed={_s2_elapsed:.2f}s  |  {len(trip_brief)} chars")
        print(f"\n{'v'*30}  TRIP BRIEF OUTPUT  {'v'*30}")
        print(trip_brief)
        print(f"{'─'*30}  END TRIP BRIEF  {'─'*30}\n")
    except Exception as exc:
        _s2_elapsed = _time.time() - _s2_start
        msg = f"Trip Planner Agent ({_llm_label}) failed after {MAX_RETRIES} retries: {exc}"
        print(f"[STAGE 2] ❌ FAILED  |  END {_ts()}  |  elapsed={_s2_elapsed:.2f}s")
        print(f"[STAGE 2] Error: {msg}")
        traceback.print_exc()
        send({'stage': 'error', 'msg': msg})
        return

    # ── Stage 3: Weather brief ────────────────────────────────────────────────
    _s3_start = _time.time()
    send({'stage': 'status', 'msg': '🌤️  Checking weather forecast…'})
    print(f"\n{STAGE_SEP}")
    print(f"[STAGE 3] Weather Agent  |  {_llm_label}  |  START {_ts()}")
    print(f"[STAGE 3]   city={city_name}  |  lat={lat}  |  lng={lng}  |  tz={timezone_str}")
    print(f"[STAGE 3]   dates={start_date} → {end_date}  |  types={trip_types}")
    print(STAGE_SEP)
    try:
        weather_brief = _call_with_retry(
            stage_name = f'Weather Agent ({_llm_label})',
            fn         = build_weather_brief,
            send       = send,
            city_name    = city_name,
            lat          = lat,
            lng          = lng,
            start_date   = start_date,
            end_date     = end_date,
            trip_types   = trip_types,
            num_adults   = num_adults,
            num_children = num_children,
            timezone     = timezone_str,
        )
        _s3_elapsed = _time.time() - _s3_start
        print(f"[STAGE 3] ✅ Weather brief done  |  END {_ts()}  |  elapsed={_s3_elapsed:.2f}s  |  {len(weather_brief)} chars")
        print(f"\n{'v'*30}  WEATHER BRIEF OUTPUT  {'v'*30}")
        print(weather_brief)
        print(f"{'─'*30}  END WEATHER BRIEF  {'─'*30}\n")
    except Exception as exc:
        _s3_elapsed = _time.time() - _s3_start
        msg = f"Weather Agent ({_llm_label}) failed after {MAX_RETRIES} retries: {exc}"
        print(f"[STAGE 3] ❌ FAILED  |  END {_ts()}  |  elapsed={_s3_elapsed:.2f}s")
        print(f"[STAGE 3] Error: {msg}")
        traceback.print_exc()
        send({'stage': 'error', 'msg': msg})
        return

    # ── Stage 4: Head LLM — itinerary ────────────────────────────────────────
    _s4_start = _time.time()
    send({'stage': 'status', 'msg': '🤖 Generating your personalised itinerary… (this may take ~30s)'})
    print(f"\n{STAGE_SEP}")
    print(f"[STAGE 4] Itinerary Head LLM  |  {_llm_label}  |  START {_ts()}")
    print(f"[STAGE 4]   selected={len(selected_ids)}  |  must-visit={len(must_visit_ids)}")
    print(f"[STAGE 4]   trip_brief_chars={len(trip_brief)}  |  weather_brief_chars={len(weather_brief)}")
    print(f"[STAGE 4]   NOTE: Head LLM full JSON is NOT logged here (see day/slot counts after parse)")
    print(STAGE_SEP)
    try:
        prefs = TripPreferences(
            city_slug     = city_slug,
            city_name     = city_name,
            start_date    = start_date,
            end_date      = end_date,
            num_adults    = num_adults,
            num_children  = num_children,
            trip_types    = trip_types,
            budget_level  = budget_level,
            travel_pace   = travel_pace,
            hotel_name    = hotel_name,
            hotel_address = hotel_address,
        )
        itinerary = _call_with_retry(
            stage_name = f'Itinerary Agent / Head LLM ({_llm_label})',
            fn         = build_itinerary,
            send       = send,
            prefs              = prefs,
            selected_place_ids = list(selected_ids),
            must_visit_ids     = list(must_visit_ids),
            trip_brief         = trip_brief,
            weather_brief      = weather_brief,
        )
        _s4_elapsed = _time.time() - _s4_start
        _days     = itinerary.get('days', [])
        _slots    = sum(len(d.get('slots', [])) for d in _days)
        _overflow = itinerary.get('overflow_places', [])
        print(f"[STAGE 4] ✅ Itinerary done  |  END {_ts()}  |  elapsed={_s4_elapsed:.2f}s")
        print(f"[STAGE 4]   days={len(_days)}  |  total_slots={_slots}  |  overflow={len(_overflow)}")
        for _i, _d in enumerate(_days):
            _day_slots = _d.get('slots', [])
            print(f"[STAGE 4]   Day {_i+1}: '{_d.get('day_label','')}' — {len(_day_slots)} slots")
        if _overflow:
            print(f"[STAGE 4]   Overflow: {[o.get('place_name','?') for o in _overflow]}")
    except Exception as exc:
        _s4_elapsed = _time.time() - _s4_start
        msg = f"Itinerary Agent / Head LLM ({_llm_label}) failed after {MAX_RETRIES} retries: {exc}"
        print(f"[STAGE 4] ❌ FAILED  |  END {_ts()}  |  elapsed={_s4_elapsed:.2f}s")
        print(f"[STAGE 4] Error: {msg}")
        traceback.print_exc()
        send({'stage': 'error', 'msg': msg})
        return

    # ── Stage 5: Stream days to client ───────────────────────────────────────
    _s5_start = _time.time()
    print(f"\n{STAGE_SEP}")
    print(f"[STAGE 5] Streaming days to client  |  START {_ts()}")
    print(STAGE_SEP)
    days = itinerary.get('days', [])
    for i, day in enumerate(days):
        print(f"[STAGE 5]   → Sending day {i+1}/{len(days)}: '{day.get('day_label', '')}'")
        send({'stage': 'day', 'day': day})
    _s5_elapsed = _time.time() - _s5_start
    print(f"[STAGE 5] ✅ All days streamed  |  END {_ts()}  |  elapsed={_s5_elapsed:.2f}s")

    # ── Stage 6: Meta (overflow, tips, summary) ───────────────────────────────
    print(f"[STAGE 6] Sending meta event (overflow, packing_tips, general_tips, summary)…")
    send({
        'stage':           'meta',
        'trip_summary':    itinerary.get('trip_summary', {}),
        'overflow_places': itinerary.get('overflow_places', []),
        'packing_tips':    itinerary.get('packing_tips', []),
        'general_tips':    itinerary.get('general_tips', []),
        '_meta':           itinerary.get('_meta', {}),
    })

    # ── Done ──────────────────────────────────────────────────────────────────
    _total = _time.time() - pipeline_start
    print(f"\n{PIPELINE_SEP}")
    print(f"[PIPELINE] ◄◄◄ DONE  |  END {_ts()}  |  TOTAL={_total:.2f}s")
    print(f"[PIPELINE] Stage timings:  S2(trip)={_s2_elapsed:.2f}s  |  S3(weather)={_s3_elapsed:.2f}s  |  S4(itinerary)={_s4_elapsed:.2f}s")
    print(f"[PIPELINE] LLM backend: {_llm_label}")
    print(PIPELINE_SEP + "\n")
    send({'stage': 'done'})


# ─── HTTP Handler ──────────────────────────────────────────────────────────────

class Handler(http.server.SimpleHTTPRequestHandler):

    def do_GET(self):
        if self.path == '/api/config':
            body = json.dumps({
                'googleApiKey': read_env_key('GOOGLE_API_KEY'),
            }).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()

    def do_OPTIONS(self):
        """Handle CORS preflight for POST requests."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        if self.path == '/api/generate-itinerary':
            print(f"\n[SERVER] POST /api/generate-itinerary from {self.client_address}")

            # ── Read request body ─────────────────────────────────────────────
            try:
                content_len  = int(self.headers.get('Content-Length', 0))
                raw_body     = self.rfile.read(content_len)
                request_body = json.loads(raw_body)
                print(f"[SERVER] Parsed: citySlug={request_body.get('citySlug')}, "
                      f"selected={len(request_body.get('selectedPlaceIds', []))}, "
                      f"mustVisit={len(request_body.get('mustVisitIds', []))}")
            except Exception as exc:
                print(f"[SERVER] Bad request body: {exc}")
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': f'Bad request: {exc}'}).encode())
                return

            # ── Set SSE headers ───────────────────────────────────────────────
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            print("[SERVER] SSE headers sent — starting pipeline…")

            lock = threading.Lock()

            def send(event_dict):
                try:
                    with lock:
                        self.wfile.write(sse_event(event_dict))
                        self.wfile.flush()
                        print(f"[SSE →] stage={event_dict.get('stage')!r:16} "
                              f"keys={list(event_dict.keys())}")
                except BrokenPipeError:
                    print("[SERVER] Client disconnected (BrokenPipe).")

            run_pipeline(request_body, send)

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        msg = fmt % args
        if any(x in str(args) for x in ['/api/', 'POST', '404', '500']):
            print(f"[HTTP] {msg}")


# ─── Main ─────────────────────────────────────────────────────────────────────

socketserver.TCPServer.allow_reuse_address = True

print(f'🚀 Server running at http://localhost:{PORT}/frontend/index.html')
print('Press Ctrl+C to stop.\n')

with socketserver.TCPServer(('', PORT), Handler) as httpd:
    httpd.serve_forever()
