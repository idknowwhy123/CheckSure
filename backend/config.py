"""Single tunables file for CheckSure v3 backend."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

INDEX_HTML_PATH = PROJECT_ROOT / "index.html"

OLLAMA_HOST = "http://localhost:11434"
# gemma3:4b — official Ollama model, good Thai + no tool_calls template bug on 0.30.x
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:4b")
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "8192"))
LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "1500"))

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
SEARCH_MAX_RESULTS = 5
SEARCH_HITS_PER_QUERY = 2
SEARCH_DEPTH = "advanced"
SEARCH_MIN_SCORE = float(os.getenv("SEARCH_MIN_SCORE", "0.3"))
SEARCH_OPEN_WEB_FALLBACK = os.getenv("SEARCH_OPEN_WEB_FALLBACK", "true").lower() == "true"
SEARCH_DEBUNK_SUFFIXES = ["จริงไหม", "ข่าวปลอม"]
SEARCH_SUFFIX_ON_BROAD_ONLY = True
SEARCH_QUERY_MAX_LEN = int(os.getenv("SEARCH_QUERY_MAX_LEN", "120"))
DEBUG_SEARCH_ENABLED = os.getenv("DEBUG_SEARCH_ENABLED", "true").lower() == "true"
FACTCHECK_DOMAINS = [
    "tna.mcot.net",
    "antifakenewscenter.com",
    "cofact.org",
    "ddc.moph.go.th",
    "thaipbs.or.th",
    "afp.com",
    "thairath.co.th",
    "matichon.co.th",
    "bbc.com",
    "prachachat.net",
    "pptvhd36.com",
    "mgronline.com",
    "factcheckthailand.afp.com",
    "sec.or.th",
    "mitihoon.com",
    "tnnthailand.com",
]

STANCE_ENABLED = os.getenv("STANCE_ENABLED", "true").lower() == "true"
STANCE_MAX_SOURCES = int(os.getenv("STANCE_MAX_SOURCES", "3"))
STANCE_CONTENT_MAX_CHARS = int(os.getenv("STANCE_CONTENT_MAX_CHARS", "2000"))
AUTHORITATIVE_DOMAINS = [
    "antifakenewscenter.com",
    "tna.mcot.net",
    "afp.com",
    "factcheckthailand.afp.com",
    "cofact.org",
]

SOURCES_UI = 5

REPLY_SUGGESTIONS_ENABLED = (
    os.getenv("REPLY_SUGGESTIONS_ENABLED", "false").lower() == "true"
)

OCR_MAX_BYTES = 5 * 1024 * 1024
OCR_ALLOWED_TYPES = frozenset({"image/jpeg", "image/png"})
OCR_LANGS = ["th", "en"]
OCR_ORDER_ENABLED = os.getenv("OCR_ORDER_ENABLED", "true").lower() == "true"
OCR_MAX_BOXES = int(os.getenv("OCR_MAX_BOXES", "40"))
OCR_ORDER_TIMEOUT_SEC = int(os.getenv("OCR_ORDER_TIMEOUT_SEC", "30"))
OCR_INVERT_LUMINANCE_THRESHOLD = int(
    os.getenv("OCR_INVERT_LUMINANCE_THRESHOLD", "110")
)
OCR_THUMBNAIL_MAX_SIDE = int(os.getenv("OCR_THUMBNAIL_MAX_SIDE", "512"))

POLITICAL_KEYWORDS = [
    "พรรค",
    "เลือกตั้ง",
    "นายกรัฐมนตรี",
    "รัฐบาล",
    "ฝ่ายค้าน",
    "ส.ส.",
    "ส.ว.",
    "เพื่อไทย",
    "ก้าวไกล",
    "ภูมิใจไทย",
    "ประชาธิปัตย์",
    "พลังประชารัฐ",
    "รวมไทยสร้างชาติ",
    "เพื่อไทยพรรค",
    "ทักษิณ",
    "ยิ่งลักษณ์",
    "ประยุทธ์",
    "เศรษฐา",
    "การเมือง",
    "หาเสียง",
    "นโยบายพรรค",
    "แคนดิเดต",
    "ผู้สมัคร",
    "คะแนนเสียง",
    "รัฐประหาร",
    "รัฐธรรมนูญ",
    "ศาลรัฐธรรมนูญ",
    "กกต.",
    "ล้มเจ้า",
    "ชุมนุม",
]

VERDICT_ENUM = ("fake", "suspicious", "unverified", "credible")
CONFIDENCE_ENUM = ("high", "medium", "low")
CATEGORY_ENUM = ("health", "scam", "official", "other")
HIGHLIGHT_TYPE_ENUM = ("scam", "caution", "trust")

SOURCE_ID_ENUM = (
    "antifake",
    "sureandshare",
    "fda",
    "doctor",
    "aoc",
    "sec",
    "bot",
    "gov",
)
DEFAULT_SOURCE_IDS = ["antifake", "sureandshare"]

DOMAIN_LABELS = {
    "tna.mcot.net": "ชัวร์ก่อนแชร์",
    "antifakenewscenter.com": "ศูนย์ต่อต้านข่าวปลอม",
    "cofact.org": "Cofact",
    "ddc.moph.go.th": "กรมควบคุมโรค",
    "thaipbs.or.th": "Thai PBS Verify",
    "afp.com": "AFP Fact Check",
    "thairath.co.th": "ไทยรัฐ",
    "matichon.co.th": "มติชน",
    "bbc.com": "BBC Thai",
    "prachachat.net": "ประชาชาติธุรกิจ",
    "pptvhd36.com": "PPTV",
    "mgronline.com": "ผู้จัดการ",
    "factcheckthailand.afp.com": "AFP Fact Check",
    "sec.or.th": "สำนักงาน ก.ล.ต.",
    "mitihoon.com": "มิติหูฟัง",
    "tnnthailand.com": "TNN",
}
