import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import health, stocks, ipo

app = FastAPI(
    title="Sensei AI API",
    description="Institutional-grade AI trading intelligence for 500 Indian stocks",
    version="2.0.0",
)

# Allow localhost in dev + any *.vercel.app domain in production.
# Set ALLOWED_ORIGINS env var on Railway to add your custom domain.
_extra = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    *_extra,
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(stocks.router, prefix="/api")
app.include_router(ipo.router, prefix="/api")


@app.get("/")
async def root():
    return {"message": "Sensei AI API v2.0", "docs": "/docs"}
