from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from api.routers.face import router as face_router
from api.routers.text import router as text_router

load_dotenv()

FRONTEND_URL = os.getenv("FRONTEND_URL")

app = FastAPI(
    title="Cognitive model - API",
    description="API for emotion detection in text and face",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

app.include_router(face_router)
app.include_router(text_router)