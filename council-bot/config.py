import os
import re
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
STAFF_CHAT_ID = int(os.getenv("STAFF_CHAT_ID", "0"))
COUNCIL_NAME = os.getenv("COUNCIL_NAME", "EstateKaki (Singapore Town Councils)")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

ANSWER_MODEL = os.getenv("ANSWER_MODEL", "claude-sonnet-5-5")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "claude-haiku-5-5")
# Multilingual, small, runs on CPU. Swap for bge-m3 if you need better quality.
EMBED_MODEL = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
DB_PATH = os.getenv("DB_PATH", "council.db")
DOCS_DIR = os.getenv("DOCS_DIR", "data/docs")

TOP_K = 4
# Cosine distance (0 = identical). Chunks further away than this are treated as "not found".
# Tune this using eval.py, it depends on your documents and embedding model.
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.6"))
RATE_LIMIT_PER_MIN = 8

EMERGENCY_TEXT = os.getenv(
    "EMERGENCY_TEXT",
    "🚨 **EMERGENCY ASSISTANCE REQUIRED**\n\n"
    "This sounds like a life-threatening situation or crisis. I am an automated Town Council assistant and **cannot** dispatch emergency rescue teams.\n\n"
    "Please contact emergency services immediately:\n"
    "• **Police:** 999\n"
    "• **Ambulance / Fire (SCDF):** 995\n"
    "• **24-Hour Crisis Support (Samaritans of Singapore):** 1767 (or WhatsApp 9151 1767)\n\n"
    "Please call right away. Help is available.",
)
# Keyword check that runs BEFORE any LLM call
EMERGENCY_RE = re.compile(
    r"\b(fire|on fire|ambulance|bleeding|unconscious|heart attack|gas leak|"
    r"drowning|collapsed|life[- ]threatening|someone is dying|jump(ing)? (off|down)|"
    r"suicide|kill (myself|himself|herself)|end(ing)? my life|fall from height)\b",
    re.IGNORECASE,
)
