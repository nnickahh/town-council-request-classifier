import os
import re
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
STAFF_CHAT_ID = int(os.getenv("STAFF_CHAT_ID", "0"))
COUNCIL_NAME = os.getenv("COUNCIL_NAME", "EstateKaki (Singapore Town Councils)")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# LLM & Embedding configuration
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "claude-sonnet-4-20250514")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "claude-haiku-4-5-20251001")
EMBED_MODEL = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

# Storage & Knowledge Base paths
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
RAG_INDEX_PATH = os.getenv("RAG_INDEX_PATH", "data/semantic_rag_index.json")
DB_PATH = os.getenv("DB_PATH", "council.db")
DOCS_DIR = os.getenv("DOCS_DIR", "data/docs")

# Semantic Hybrid RAG retrieval thresholds
TOP_K = int(os.getenv("TOP_K", "4"))
# Minimum combined hybrid score (0-1) required to answer; below this, escalate to human staff.
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.28"))
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", str(round(1.0 - MIN_RELEVANCE_SCORE, 2))))
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "8"))

# Optional Trello Work-Order Dispatch configuration
TRELLO_BOARD_ID = os.getenv("TRELLO_BOARD_ID", "")
TRELLO_API_KEY = os.getenv("TRELLO_API_KEY", "")
TRELLO_TOKEN = os.getenv("TRELLO_TOKEN", "")
TRELLO_TODO_LIST = os.getenv("TRELLO_TODO_LIST", "To Do")
TRELLO_URGENT_LIST = os.getenv("TRELLO_URGENT_LIST", "Urgent")
TRELLO_IN_PROGRESS_LIST = os.getenv("TRELLO_IN_PROGRESS_LIST", "In Progress")
TRELLO_DONE_LIST = os.getenv("TRELLO_DONE_LIST", "Done")

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
