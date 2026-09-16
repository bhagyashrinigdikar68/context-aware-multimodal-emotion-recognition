from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings

app = FastAPI(
    title="Context-Aware Multimodal Emotion Recognition in Conversations",
    description=(
        "Research-oriented API for multimodal (text+audio+visual), "
        "context-aware conversational emotion recognition. "
        "Predictions are probabilistic and must not be used for high-stakes "
        "decisions about individuals -- see README for full ethical notes."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
def root():
    return {
        "message": "Context-Aware Multimodal Emotion Recognition API",
        "docs": "/docs",
        "device": settings.DEVICE,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
