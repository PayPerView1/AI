# 🤖 PayPerView AI Service (Python FastAPI)

Production-ready AI & RAG Chatbot Service built with **FastAPI**, **LangChain**, **Google Gemini LLM**, **SQLModel/SQLite**, and **ChromaDB**.

Supports JWT Bearer authentication, multi-tenant thread scoping, dynamic system prompt context injection for `CLIPPER` and `BRAND` roles, 12 function calling tools, and grounded RAG search.

---

## 📁 Project Structure

```text
ai-service/
│
├── app/
│   ├── main.py              # FastAPI app setup, CORS, health endpoint, lifespan
│   ├── core/
│   │   ├── config.py        # Environment settings (Gemini Key, JWT Secret, Ports)
│   │   └── security.py      # JWT Bearer token decoder & authentication dependency
│   ├── db/
│   │   ├── database.py      # Portable database connection engine & session dependency
│   │   └── models.py        # ChatThread & ChatMessage SQLModel schemas
│   ├── rag/
│   │   ├── tools.py         # 12 Function calling tools for CLIPPER & BRAND
│   │   ├── prompts.py       # Role-based dynamic system prompt generator
│   │   ├── retriever.py     # Similarity search logic for ChromaDB
│   │   ├── embeddings.py    # Gemini embedding factory
│   │   └── vector_store.py  # ChromaDB collection store manager
│   ├── routes/
│   │   └── chat.py          # REST endpoints (/api/v1/ai/threads, /messages)
│   └── services/
│       └── chat_service.py  # Thread ownership check, RAG search, LLM orchestration
│
├── documents/               # Platform Markdown documentation files
├── chroma_db/               # Persistent Chroma vector database
├── create_test_token.py     # CLI helper script to generate test JWT tokens
├── ingest.py                # Build-time and on-demand document ingestion script
├── requirements.txt         # Production Python dependencies
├── Dockerfile               # Production multi-stage Docker container build
├── .dockerignore            # Docker context ignore rules
├── .gitignore               # Git repository ignore rules
├── render.yaml              # Render Blueprint deployment configuration
└── README.md                # Service documentation
```

---

## ⚙️ Environment Variables

The service reads settings from system environment variables (or local `.env` during development):

| Variable Name | Required | Default Value | Description |
| :--- | :---: | :--- | :--- |
| `GEMINI_API_KEY` | **Yes** | `""` | Google Gemini API key used for LLM and embeddings. |
| `JWT_SECRET` | **Yes** | `supersecretkey_...` | Secret key used to sign and verify JWT Bearer tokens. |
| `JWT_ALGORITHM` | No | `HS256` | JWT signing algorithm. |
| `CHROMA_PERSIST_DIRECTORY` | No | `./chroma_db` | Path to persistent ChromaDB storage (relative to service root). |
| `CHROMA_COLLECTION` | No | `brands_clippers_kb` | Name of the ChromaDB vector collection. |
| `AI_SERVICE_PORT` | No | `8001` | Default local FastAPI port. |
| `PORT` | No | `8001` | Dynamic port set by cloud providers (Render/Railway). |
| `DATABASE_URL` | No | `sqlite:///.../chat_history.db` | Relational database connection URL. |

---

## 🧠 ChromaDB & RAG Ingestion Flow

The Retrieval-Augmented Generation (RAG) pipeline operates as follows:

```text
documents/ (*.md)
   │
   ▼ (python ingest.py)
Text Splitter (800 char chunks, 100 overlap)
   │
   ▼ (Gemini Embeddings: models/gemini-embedding-001)
chroma_db/ (Vector Store)
   │
   ▼ (User Query similarity search)
RAG Retriever -> Dynamic System Prompt -> Gemini LLM -> AI API Response
```

* During **Docker Build**, `RUN python ingest.py` parses all documents in `documents/` and pre-builds `chroma_db/` inside the image.
* At **Runtime**, `get_vector_store()` loads `chroma_db/` from the container file system portably.

---

## 💻 Local Development Setup

### 1. Create Virtual Environment & Install Dependencies
```bash
cd ai-service
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Local `.env`
Copy `.env.example` to `.env` and fill in your `GEMINI_API_KEY`:
```bash
cp .env.example .env
```

### 3. Run Knowledge Ingestion
```bash
python ingest.py
```

### 4. Generate Test Bearer Tokens
```bash
python create_test_token.py
```

### 5. Start the FastAPI Development Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

* API Health Check: `GET http://localhost:8001/health`
* Interactive API Documentation (Swagger UI): `http://localhost:8001/docs`

---

## 🐳 Docker Setup

### 1. Build Docker Image
```bash
docker build -t ai-service .
```

### 2. Run Docker Container
```bash
docker run -d \
  -p 8001:8001 \
  --env-file .env \
  --name payperview-ai-container \
  ai-service
```

---

## ☁️ Cloud Deployment (Render & Railway)

### Deploying to Render
1. Push your repository to **GitHub** / **GitLab**.
2. Go to [Render Dashboard](https://dashboard.render.com/) -> **New +** -> **Web Service**.
3. Select **Docker** as the Runtime environment and root directory as `ai-service/`.
4. Add the following **Environment Variables** in Render:
   - `GEMINI_API_KEY`: *(Your Google Gemini API Key)*
   - `JWT_SECRET`: *(Your JWT Secret Key)*
5. Render will automatically assign `$PORT` and build the container using `Dockerfile`.

### Deploying to Railway
1. Create a new project in [Railway.app](https://railway.app/).
2. Connect your GitHub repository and set Root Directory to `ai-service/`.
3. Set Environment Variables: `GEMINI_API_KEY`, `JWT_SECRET`.
4. Railway will automatically build the Dockerfile and expose the web service.
