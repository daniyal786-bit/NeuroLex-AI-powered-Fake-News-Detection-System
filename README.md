# NeuroLex AI

NeuroLex is a full-stack **fake news detection** application. It combines a multi-model transformer ensemble (BERT, RoBERTa, DeBERTa), URL and image analysis, optional external fact-check APIs, and an AI chat assistant for explanations and media literacy.

## Features

- **Text analysis** — 5-stage pipeline with ensemble voting and confidence tiers
- **URL analysis** — Domain credibility, content extraction, pattern detection
- **Image analysis** — OCR (English, Urdu, Arabic) plus text classification
- **AI chatbot** — Groq-powered assistant (`/llm/chat`) with rule-based fallback (`/chat`)
- **Credibility checks** — Source and domain heuristics
- **Multi-language UI** — English, Urdu, Pashto, Hindi (translation via deep-translator)

## Requirements

- Python 3.11+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (for image analysis; default path on Windows: `C:\Program Files\Tesseract-OCR\tesseract.exe`)
- Optional: [Groq API key](https://console.groq.com) for full AI chat (without it, basic canned responses still work)

## Quick start

### 1. Clone and install

```bash
cd NeuroLex_2
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Configure environment

Copy the example env file and add your keys:

```bash
copy .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | For AI chat | Enables Llama-powered chat via Groq |
| `GOOGLE_FACTCHECK_API_KEY` | Optional | Google Fact Check Tools API |
| `NEWS_API_KEY` | Optional | NewsAPI domain verification |
| `TEXTRAZOR_API_KEY` | Optional | NLP entity extraction |

### 3. Run the server

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000** in your browser.

First startup downloads Hugging Face models (~1–2 GB) and may take 1–2 minutes on CPU.

## API overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI |
| GET | `/healthz` | Health check |
| GET | `/readyz` | Readiness (models + static files) |
| GET | `/docs` | Swagger UI |
| POST | `/predict` | Text fake-news prediction |
| POST | `/analyze_url` | URL analysis |
| POST | `/analyze_image` | Image upload + OCR + analysis |
| POST | `/llm/chat` | AI chat (Groq or fallback) |
| POST | `/llm/explain_prediction` | Natural-language explanation |
| POST | `/chat` | Rule-based chat + optional inline analysis |
| POST | `/credibility` | Source credibility |

## Chatbot behavior

1. **With `GROQ_API_KEY`** — Uses `llama-3.3-70b-versatile` (configurable via `GROQ_MODEL`) for conversational answers and context-aware history.
2. **Without Groq** — Falls back to helpful canned responses on `/llm/chat` and the `/chat` router for greetings, tips, and short-text analysis when the ensemble is loaded.

The web UI calls `/llm/chat` first, then `/chat` if needed.

## Project structure

```
NeuroLex_2/
├── main.py              # FastAPI app entry
├── routers/             # text, url, image, llm, chat, credibility, auth, history
├── services/            # ensemble_loader, model_loader, external APIs
├── core/                # config, database
├── static/              # index.html, app.js, neurolex.css
├── requirements.txt
└── .env.example
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Chat only gives generic replies | Set `GROQ_API_KEY` in `.env` and install `groq`: `pip install groq` |
| Page has no styling / chat does nothing | Ensure assets load from `/static/` (fixed in current `index.html`) |
| `Groq library not installed` in logs | Run `pip install groq` and restart the server |
| Image OCR fails | Install Tesseract and verify path in `main.py` |
| Models slow to load | Normal on first run; models are cached by Hugging Face |

## Development

```bash
# Health check
curl http://127.0.0.1:8000/healthz

# Chat (fallback mode)
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"message\":\"hello\"}"
```

## License

Academic / FYP project — see repository owner for usage terms.
