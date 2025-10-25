# api_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

# Import your main AI functions (must exist)
try:
    from arsemble_ai import get_ai_response
except Exception as e:
    raise RuntimeError(
        "Make sure arsemble_ai.py exists and defines get_ai_response") from e

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ARsemble-API")

app = FastAPI(title="ARsemble AI API", version="1.0")

# CORS for frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {"message": "ARsemble AI API running!"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/ask")
async def ask_ai(payload: QueryRequest):
    """Primary backend endpoint. Expects JSON: { "query": "..." }"""
    q = (payload.query or "").strip()
    logger.info("API ask recieved query: %s", q[:200])
    if not q:
        return {"error": "query is required"}
    try:
        result = get_ai_response(q)
        # Ensure JSON-serializable
        return result
    except Exception as e:
        logger.exception("Error in get_ai_response:")
        return {"error": "internal error", "detail": str(e)}

# If you want to run this file directly:
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("ARSSEMBLE_API_PORT", "8080"))
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=True)
