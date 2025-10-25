# api_server.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import uvicorn
from typing import Any, Dict

# Import your main AI functions (must be in same project)
# arsemble_ai.get_ai_response should return a dict or text as your handler expects
from arsemble_ai import get_ai_response

# --- Logging ---
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ARsemble-API")

# --- FastAPI App ---
app = FastAPI(title="ARsemble AI API", version="1.0")

# Allow cross-origin (for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request model ---
class QueryRequest(BaseModel):
    query: str


def _call_ai_and_safe_return(query: str) -> Dict[str, Any]:
    """
    Wrapper to call your AI function and normalize the return to a JSON-serializable dict.
    """
    try:
        result = get_ai_response(query)
        # If the AI returns a plain string, wrap it
        if isinstance(result, str):
            return {"source": "local", "text": result}
        if isinstance(result, dict):
            return result
        # Unknown type -> convert to str
        return {"source": "local", "text": str(result)}
    except Exception as e:
        logger.exception("Error calling get_ai_response:")
        return {"error": "internal_error", "detail": str(e)}


# --- Root (for quick browser test) ---
@app.get("/")
def root():
    return {"message": "ARsemble AI API running!"}


# --- Main AI endpoint (canonical) ---
@app.post("/api/ask")
async def ask_ai(req: QueryRequest):
    """
    Primary endpoint: expects JSON {"query": "..."}.
    """
    q = (req.query or "").strip()
    logger.info("ask_ai received query: %s", q[:200])
    resp = _call_ai_and_safe_return(q)
    return resp


# --- Backwards-compatible endpoint (some clients call /api/recommend) ---
@app.post("/api/recommend")
async def recommend_ai(request: Request):
    """
    Backwards-compatible route. Accepts either:
      - JSON body {"query": "..."}
      - or plain form / other shapes where we attempt to pull 'query' key.
    """
    try:
        body = await request.json()
    except Exception:
        # If body isn't JSON, try to read text
        try:
            body_text = await request.body()
            body = {"query": body_text.decode("utf-8", errors="ignore")}
        except Exception:
            body = {}

    # Normalize different payload shapes
    query = ""
    if isinstance(body, dict):
        # common keys
        query = body.get("query") or body.get("q") or body.get("text") or ""
    elif isinstance(body, str):
        query = body
    query = (query or "").strip()

    if not query:
        # Try form data / fallback
        form = await request.form()
        query = (form.get("query") or form.get("q") or "").strip()

    logger.info("recommend endpoint received query: %s",
                query[:200] if query else "<empty>")

    if not query:
        return {"error": "missing_query", "message": "Provide a JSON body like {\"query\":\"PSU for i5-13400 + RTX 3060\"}"}

    resp = _call_ai_and_safe_return(query)
    return resp


# --- Health Check ---
@app.get("/health")
def health():
    return {"status": "ok"}


# --- Optional convenience: single GET to test recommend quickly (not required) ---
@app.get("/api/demo")
def demo():
    sample = "PSU needed for i5-13400 + RTX 3060 + B760 motherboard"
    logger.info("demo: running sample query")
    return _call_ai_and_safe_return(sample)


# --- Run manually (uvicorn) ---
if __name__ == "__main__":
    # Run directly: uvicorn will serve the FastAPI app
    # You can also run via: uvicorn api_server:app --reload --host 0.0.0.0 --port 8080
    logger.info("Starting ARsemble API on 0.0.0.0:8080")
    uvicorn.run("api_server:app", host="0.0.0.0", port=8080, log_level="info")
