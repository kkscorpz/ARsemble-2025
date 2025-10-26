# api_server.py
# FastAPI wrapper for ARsemble AI (keeps existing /api/ask and adds streaming /api/ask_stream)

from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import logging
import os
import asyncio
import json

# Import your main AI functions (must exist)
try:
    import arsemble_ai as arsemble_ai_mod
    from arsemble_ai import get_ai_response
except Exception as e:
    raise RuntimeError(
        "Make sure arsemble_ai.py exists and defines get_ai_response") from e

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ARsemble-API")

app = FastAPI(title="ARsemble AI API", version="1.0")

# CORS for frontend dev (adjust origins if needed)
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
    logger.info("API ask received query: %s", q[:200])
    if not q:
        return {"error": "query is required"}
    try:
        result = get_ai_response(q)
        # Ensure JSON-serializable
        return result
    except Exception as e:
        logger.exception("Error in get_ai_response:")
        return {"error": "internal error", "detail": str(e)}


# Streaming ask endpoint: streams JSON-lines while model produces tokens
@app.post("/api/ask_stream")
async def ask_ai_stream(request: Request):
    """
    POST JSON {"query":"..."} to stream newline-delimited JSON objects as chunks become available.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json body"}, status_code=400)

    q = (body.get("query") or "").strip()
    if not q:
        return JSONResponse({"error": "query is required"}, status_code=400)

    async def stream_gen():
        # Preferred: if arsemble_ai_mod provides an async generator stream_response(query)
        if hasattr(arsemble_ai_mod, "stream_response"):
            try:
                async for item in arsemble_ai_mod.stream_response(q):
                    # item may be bytes/str/dict
                    if isinstance(item, bytes):
                        yield item + b"\n"
                    elif isinstance(item, str):
                        yield (item + "\n").encode("utf-8")
                    else:
                        yield (json.dumps(item) + "\n").encode("utf-8")
                return
            except Exception as e:
                yield (json.dumps({"error": "stream_failed", "msg": str(e)}) + "\n").encode("utf-8")
                return

        # Fallback: run blocking get_ai_response in thread and stream slices
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, lambda: arsemble_ai_mod.get_ai_response(q))
        except Exception as e:
            yield (json.dumps({"error": "generation_failed", "msg": str(e)}) + "\n").encode("utf-8")
            return

        if result is None:
            yield (json.dumps({"error": "no_result"}) + "\n").encode("utf-8")
            return

        if isinstance(result, dict):
            yield (json.dumps(result) + "\n").encode("utf-8")
            return

        text = str(result)
        chunk_size = 256
        for i in range(0, len(text), chunk_size):
            part = text[i: i + chunk_size]
            yield (json.dumps({"chunk": i // chunk_size, "text": part}) + "\n").encode("utf-8")
        return

    return StreamingResponse(stream_gen(), media_type="application/json")


# If you want to run this file directly:
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("ARSSEMBLE_API_PORT", "8080"))
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=True)
