from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

app = FastAPI(title="Chess AI Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://chesst1024.vercel.app",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
