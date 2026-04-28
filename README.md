# RAG-based Document Q&A System

An AI-powered document question-answering system built with a **FastAPI** backend and **Angular** frontend. Upload a PDF or TXT file and ask questions about its content — the system retrieves the most relevant sections and generates accurate answers using a large language model.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local) |
| Vector Store | FAISS |
| LLM | Groq API (`llama-3.3-70b-versatile`) |
| RAG Framework | LangChain |
| Frontend | Angular (Phase 5 — in progress) |

---

## Project Structure

```
rag-document-qa/
├── backend/
│   ├── main.py              # FastAPI entry point
│   ├── rag_pipeline.py      # LangChain RAG logic
│   ├── file_handler.py      # PDF + TXT loading and chunking
│   ├── requirements.txt     # Python dependencies
│   └── .env                 # API keys (never commit)
├── frontend/                # Angular app (Phase 5)
├── .gitignore
└── README.md
```

---

## How It Works

```
User uploads PDF/TXT
        ↓
File is split into chunks (500 chars each)
        ↓
Each chunk is converted to a vector embedding (HuggingFace)
        ↓
Embeddings stored in FAISS vector database
        ↓
User asks a question
        ↓
Question is embedded → top 4 relevant chunks retrieved
        ↓
Chunks + question sent to Groq LLM
        ↓
Answer returned to user
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com)
- Git

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/rag-document-qa.git
cd rag-document-qa
```

**2. Create and activate a virtual environment**

```bash
python -m venv rag-env

# Linux/Mac
source rag-env/bin/activate

# Windows
rag-env\Scripts\activate
```

**3. Install dependencies**

```bash
cd backend
pip install -r requirements.txt
```

**4. Set up environment variables**

Create a `.env` file inside the `backend/` folder:

```
GROQ_API_KEY=your_groq_api_key_here
```

**5. Download the embedding model (one-time setup)**

Run this script once to download and save the model locally:

```bash
python -c "
import os, ssl
os.environ['CURL_CA_BUNDLE'] = ''
ssl._create_default_https_context = ssl._create_unverified_context
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('./local_model')
print('Model saved to ./local_model')
"
```

**6. Run the server**

```bash
uvicorn main:app --reload
```

API is now running at `http://localhost:8000`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/upload` | Upload a PDF or TXT file |
| `POST` | `/ask` | Ask a question about the uploaded document |
| `GET` | `/debug/chunks` | View sample chunks (development only) |

### Example — Upload a file

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@your_document.pdf"
```

### Example — Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

### Example Response

```json
{
  "answer": "This document is about..."
}
```

---

## Interactive API Docs

FastAPI auto-generates a Swagger UI at:

```
http://localhost:8000/docs
```

Use it to test `/upload` and `/ask` directly from your browser without any frontend.

---

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `GROQ_API_KEY` | Your Groq API key from console.groq.com | Yes |

---

## What NOT to Commit

The `.gitignore` excludes the following large or sensitive files:

- `.env` — contains your API key
- `local_model/` — ~90MB embedding model (regenerate with step 5 above)
- `faiss_index/` — vector database (regenerated on each upload)
- `rag-env/` — virtual environment (recreate with `pip install -r requirements.txt`)

---

## Roadmap

- [x] Phase 1 — Python environment setup
- [x] Phase 2 — RAG pipeline (LangChain + FAISS + HuggingFace embeddings)
- [x] Phase 3 — FastAPI backend with `/upload` and `/ask` endpoints
- [ ] Phase 4 — Docker containerization
- [ ] Phase 5 — Angular frontend (upload UI + chat interface)

---

## Author

**Your Name**
- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [your-linkedin](https://linkedin.com/in/your-linkedin)

---

## License

This project is licensed under the MIT License.