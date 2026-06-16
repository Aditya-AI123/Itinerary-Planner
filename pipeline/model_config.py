"""
pipeline/model_config.py
========================

Centralised model configuration for the three pipeline LLM agents:
  • trip_planner_agent  (build_trip_brief)
  • weather_agent       (build_weather_brief)
  • itinerary_agent     (build_itinerary)

The LLM backend is controlled by the PIPELINE_LLM environment variable,
which is set by main.py before any agent is imported.

  PIPELINE_LLM=gemini   → Google Gemini 2.5 Flash  (default)
  PIPELINE_LLM=llama    → Groq  llama-3.3-70b-versatile

Usage inside each agent:
    from pipeline.model_config import get_llm_client, AGENT_MODEL, AGENT_PROVIDER

The enricher (llm_enricher.py) always uses Llama/Groq and does NOT use this module.
"""

import os
from pathlib import Path

# ─── Model identifiers ────────────────────────────────────────────────────────

GEMINI_MODEL = "gemini-2.5-flash"
LLAMA_MODEL  = "llama-3.3-70b-versatile"

# ─── Token limits per provider ────────────────────────────────────────────────
# Gemini 2.5 Flash supports up to 65,536 output tokens.
# Groq llama-3.3-70b-versatile hard-caps at 32,768 output tokens — sending more
# causes a 400 error: "'max_tokens' must be <= 32768".

GEMINI_MAX_TOKENS = 65536   # Gemini 2.5 Flash limit
LLAMA_MAX_TOKENS  = 32768   # Groq llama-3.3-70b-versatile hard cap

# ─── Provider detection ───────────────────────────────────────────────────────

def _active_provider() -> str:
    """Return 'gemini' or 'llama' based on the PIPELINE_LLM env var."""
    val = os.getenv("PIPELINE_LLM", "gemini").strip().lower()
    if val in ("llama", "groq"):
        return "llama"
    return "gemini"


def get_model_name() -> str:
    """Return the model name string for the active provider."""
    return LLAMA_MODEL if _active_provider() == "llama" else GEMINI_MODEL


def get_max_tokens(agent_default: int | None = None) -> int:
    """
    Return the correct max_tokens for the active provider.

    Groq hard-caps llama-3.3-70b-versatile at 32,768.
    Gemini 2.5 Flash supports up to 65,536.

    Pass agent_default to use a smaller value (e.g. for trip_planner which
    doesn't need the full budget). If None, the provider max is used.
    """
    provider_max = LLAMA_MAX_TOKENS if _active_provider() == "llama" else GEMINI_MAX_TOKENS
    if agent_default is None:
        return provider_max
    return min(agent_default, provider_max)


def get_llm_client():
    """
    Return an initialised LLM client for the active provider.

    For Gemini  → google.genai.Client (uses GEMINI_API_KEY)
    For Llama   → groq.Groq           (uses GROQ_API_KEY)
    """
    provider = _active_provider()

    if provider == "llama":
        try:
            from groq import Groq
        except ImportError as exc:
            raise ImportError(
                "The 'groq' package is required for Llama mode.\n"
                "Install it with:  pip install groq"
            ) from exc

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set in .env — required for PIPELINE_LLM=llama.\n"
                "Get a key at https://console.groq.com/keys"
            )
        from groq import Groq
        return Groq(api_key=api_key)

    else:
        # Gemini
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "The 'google-genai' package is required for Gemini mode.\n"
                "Install it with:  pip install google-genai"
            ) from exc

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) not set in .env.\n"
                "Get a key at https://aistudio.google.com/app/apikey"
            )
        from google import genai as _genai
        return _genai.Client(api_key=api_key)


def call_llm(client, model_name: str, prompt: str,
             temperature: float = 0.3, max_tokens: int = 8192,
             json_mode: bool = False) -> str:
    """
    Unified LLM call that works for both Gemini and Groq/Llama.

    Args:
        client      : Client from get_llm_client()
        model_name  : Model string from get_model_name()
        prompt      : The full prompt text
        temperature : Sampling temperature
        max_tokens  : Max output tokens
        json_mode   : If True, request JSON output (Gemini only for now;
                      Groq enforces via system prompt)

    Returns:
        Plain text response string (stripped).
    """
    provider = _active_provider()

    if provider == "llama":
        from groq import Groq
        assert isinstance(client, Groq)

        messages = [{"role": "user", "content": prompt}]
        if json_mode:
            messages.insert(0, {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Always respond with valid JSON only. "
                    "Do not include markdown code fences or any text outside the JSON object."
                ),
            })

        resp = client.chat.completions.create(
            model       = model_name,
            messages    = messages,
            temperature = temperature,
            max_tokens  = max_tokens,
        )
        return resp.choices[0].message.content.strip()

    else:
        # Gemini
        from google import genai as _genai
        from google.genai import types as genai_types

        cfg_kwargs: dict = {
            "temperature":       temperature,
            "max_output_tokens": max_tokens,
        }
        if json_mode:
            cfg_kwargs["response_mime_type"] = "application/json"

        resp = client.models.generate_content(
            model    = model_name,
            contents = prompt,
            config   = genai_types.GenerateContentConfig(**cfg_kwargs),
        )
        return resp.text.strip() if resp.text else ""


def get_finish_reason(response_obj) -> str:
    """
    Safely extract finish_reason string from a Gemini response.
    Returns '' for Groq responses (not applicable).
    """
    try:
        candidate = response_obj.candidates[0] if response_obj.candidates else None
        if candidate:
            return str(getattr(candidate, "finish_reason", "") or "")
    except Exception:
        pass
    return ""


def provider_label() -> str:
    """Human-readable label for logging. e.g. 'Gemini 2.5 Flash' or 'Llama 3.3 70B (Groq)'"""
    p = _active_provider()
    if p == "llama":
        return f"Llama 3.3 70B via Groq ({LLAMA_MODEL})"
    return f"Google Gemini 2.5 Flash ({GEMINI_MODEL})"
