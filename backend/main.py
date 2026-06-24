from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import health, stocks, ipo

app = FastAPI(
    title="Sensei AI API",
    description="Institutional-grade AI trading intelligence for 500 Indian stocks",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
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
