# เช็กชัวร์ (CheckSure) v3

Thai LINE message fact-checker — **local Gemma3 reasoning** + **web search grounding** via Tavily.

Paste a forwarded message **or upload a LINE screenshot** → the backend searches Thai fact-check sites → returns a color-coded verdict with real source URLs. When nothing is found, it falls back to tone-based analysis (clearly labelled).

Photo flow: **อัปโหลดรูป** → EasyOCR reads each text box → **gemma3:4b sequences boxes by layout** → you review/edit in a modal → auto-check runs on the confirmed text.

## First-time setup

1. **Install prerequisites**
   - [Ollama](https://ollama.com/) (GPU recommended)
   - Python 3.11+
   - [Tavily API key](https://tavily.com/) (free tier: 1,000 req/month)

2. **Pull the LLM model**
   ```powershell
   .\setup-model.ps1
   ```
   Default: `gemma3:4b` (~3.3 GB). Verify with `pwsh ./check_llm.ps1`.

3. **Python dependencies**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\pip install -r requirements.txt
   ```
   First install pulls **EasyOCR + PyTorch** (~1–2 GB) and downloads Thai/English OCR models on first server start. Run `.\start.ps1` once before demo day so models are cached.

4. **Configure Tavily**
   ```powershell
   copy .env.example .env
   # Edit .env and set TAVILY_API_KEY=tvly-...
   ```

## Daily use

```powershell
.\start.ps1
```

Open **http://localhost:8000** — mode chip should show `AI ในเครื่อง + ค้นเว็บ`.

**Note:** Server startup loads EasyOCR models (adds ~30–60s on first run). `/health` reports `ocr_ready: true` when upload is available.

## 3-step demo

1. Run `.\start.ps1` — chip shows `AI ในเครื่อง + ค้นเว็บ`
2. Click the fake-health example chip → verdict + real `sources` citations appear
3. Copy a polite or firm reply from the result card and paste into your chat

### Photo upload demo

1. Tap **อัปโหลดรูป** and pick a LINE screenshot (JPEG/PNG, max 5 MB)
2. Review/edit extracted text in the modal → **ใช้ข้อความนี้**
3. Verdict card appears automatically (same as paste flow)

## Architecture

```
index.html  →  GET /health, POST /check, POST /ocr
backend/    →  websearch.py (Tavily) + llm.py (Gemma3) + ocr.py (EasyOCR) + ocr_order.py + prompt.py + pipeline.py
```

Tunables live in `backend/config.py` (models, domains, search limits).

### OCR reading order

Two-stage local OCR (no extra model):

1. **EasyOCR** — detects and recognizes each text box (`detail=1`)
2. **gemma3:4b** (same model as verdict) — orders box ids using normalized `cx`/`cy` coordinates

If LLM ordering fails or is disabled, falls back to geometric sort (top→bottom, left→right).

| Env var | Default | Purpose |
|---|---|---|
| `OCR_ORDER_ENABLED` | `true` | Set `false` for geometric-only (faster, less accurate on messy layouts) |
| `OCR_MAX_BOXES` | `40` | Above this count, skip LLM and use geometric sort |
| `OCR_ORDER_TIMEOUT_SEC` | `30` | LLM ordering timeout before geometric fallback |

Typical `/ocr` latency adds ~2–5s for LLM ordering on multi-box LINE screenshots.

## Troubleshooting

**503 on `/check` with `selectattr: unknown test 'tool_calls'`**

The stock Typhoon model crashes on Ollama 0.30.x (`tool_calls` template bug). Use the default `gemma3:4b` (`.\setup-model.ps1`) or run `pwsh ./check_llm.ps1` to confirm the LLM path works. See `MODEL-COMPAT-FIX.md`.

**Port 8000 already in use**

Stop the other server (Ctrl+C) or run `taskkill /PID <pid> /F` after `netstat -ano | findstr ":8000"`.

## Privacy note

The **LLM runs locally**. **OCR runs locally** — images are processed in memory only (never saved to disk); only the text you confirm is sent to `/check`. Only **extracted search queries** (not the full message) are sent to Tavily. See `docs/CheckSure-WebSearch-First-Plan.md` for full details.

## Project layout

```
docs/           Plans and proposal
tools/          rootan-harvester (future KB growth, not used at runtime)
backend/        FastAPI app
index.html      Frontend UI
start.ps1       One-command launch
```
