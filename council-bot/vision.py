import io
import os
import json
import asyncio
from typing import Dict, Optional
from PIL import Image

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

_client = None
if GEMINI_API_KEY:
    try:
        from google import genai
        _client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        pass


VISION_PROMPT = """You are an expert municipal infrastructure inspector for Singapore Town Councils.
Analyze this photo submitted by an HDB resident.
Return ONLY a valid JSON object with these keys:
{
  "fault": "Short description of detected problem (e.g. Cracked concrete footpath, Broken corridor light fixture, Bulky furniture dumped at lift lobby)",
  "severity": "Low" | "Medium" | "High" | "Critical",
  "category": "roads" | "lighting" | "waste" | "water" | "noise" | "parks" | "other",
  "crew": "Recommended dispatch crew (e.g. Electrical Contractor, Conservancy Cleaning Crew, Road Paving Team, Pest Control, Lift Maintenance)",
  "summary": "1 brief sentence summarizing the hazard for staff dispatch."
}
"""


def analyze_fault_photo(image_bytes: bytes) -> Dict[str, str]:
    """Analyzes an uploaded photo to diagnose the municipal fault."""
    fallback = {
        "fault": "Resident photo attached",
        "severity": "Medium",
        "category": "other",
        "crew": "Estate Maintenance Team",
        "summary": "Photo attached by resident for officer review.",
    }

    if not image_bytes or not _client:
        return fallback

    try:
        image = Image.open(io.BytesIO(image_bytes))
        resp = _client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image, VISION_PROMPT],
            config={"response_mime_type": "application/json", "temperature": 0.1},
        )
        data = json.loads(resp.text)
        return {
            "fault": data.get("fault", fallback["fault"]),
            "severity": data.get("severity", fallback["severity"]),
            "category": data.get("category", fallback["category"]),
            "crew": data.get("crew", fallback["crew"]),
            "summary": data.get("summary", fallback["summary"]),
        }
    except Exception:
        return fallback
