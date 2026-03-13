"""Agent configuration constants.

Single source for all tunables. Environment variables override defaults.
"""

import os


# ── Backend API ──────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))

# ── OpenAI ───────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "2048"))

# ── Polling ──────────────────────────────────────────────────────
TASK_POLL_INTERVAL = int(os.getenv("TASK_POLL_INTERVAL", "5"))     # seconds
TASK_POLL_MAX_ATTEMPTS = int(os.getenv("TASK_POLL_MAX_ATTEMPTS", "60"))

# ── Defaults ─────────────────────────────────────────────────────
DEFAULT_CONFIDENCE_LEVEL = 95.0
DEFAULT_POWER_OF_TEST = 80.0
DEFAULT_CONTROL_ALLOCATION = 20.0
DEFAULT_TREATMENT_ALLOCATION = 80.0
DEFAULT_MDE_PERCENT = 5.0

# ── Specialist retries ───────────────────────────────────────────
MAX_SPECIALIST_RETRIES = 3

# ── Conversation ─────────────────────────────────────────────────
MAX_CONVERSATION_TURNS = 50