from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.routes.chat import router as chat_router
from app.routes.review import router as review_router
from app.core.config import get_settings
from app.db.database import init_db

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize DB tables
    init_db()
    print("[ai-service] Database tables initialized successfully.")
    yield
    # Shutdown
    print("[ai-service] Shutting down.")


app = FastAPI(
    title="PayPerView Platform - AI Chatbot Service",
    description="Python FastAPI AI Chatbot microservice supporting JWT auth, dynamic prompt context injection for CLIPPER & BRAND roles, function tools calling, and vector RAG search.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware allowing access from Frontend and Express backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(chat_router)
app.include_router(review_router)


@app.get("/health")
async def health_check():
    """Service health check endpoint (GET /health)."""
    return {
        "status": "ok",
        "service": "PayPerView AI Chatbot Service",
        "version": "1.0.0",
    }
