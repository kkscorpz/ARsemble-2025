# server.py — Unified ARsemble AI server (UI + API)
# Replaces the previous versions and avoids 405 by accepting GET/POST where appropriate.
# Adds streaming JSON-lines endpoint /api/ai/stream and gzip compression.

from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from waitress import serve
import os
import logging
import traceback

# Import AI functions (must exist)
# Keep your function imports for backward-compatible endpoints:
from arsemble_ai import get_ai_response, get_component_details, list_shops

# Also import the module for optional stream_response detection
import arsemble_ai as arsemble_ai_mod

# Additional helpers for streaming + compression
from flask import Response, stream_with_context
from flask_compress import Compress
import json
import threading

HOST = os.getenv("ARSSEMBLE_HOST", "0.0.0.0")
PORT = int(os.getenv("ARSSEMBLE_PORT", "10000"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ARsemble")

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
# enable gzip compression for dynamic responses
Compress(app)


# Serve index / static files
@app.route("/", methods=["GET"])
def index():
    try:
        return send_from_directory(app.static_folder, "index.html")
    except Exception:
        abort(404)


@app.route("/<path:filename>", methods=["GET"])
def static_files(filename):
    try:
        return send_from_directory(app.static_folder, filename)
    except Exception:
        abort(404)


# Unified AI endpoint (POST only for queries that change state or supply body)
@app.route("/api/recommend", methods=["POST", "GET"])
def recommend_api():
    if request.method == "GET":
        return jsonify({
            "info": "POST JSON {'query':'.'} to this endpoint",
            "example": {"query": "Recommend me a ₱30000 PC build"}
        })
    try:
        body = request.get_json(force=True) or {}
        query = (body.get("query") or "").strip()
        if not query:
            return jsonify({"error": "Missing 'query' in request"}), 400

        logger.info("AI query: %s", query)
        result = get_ai_response(query)

        # If local-psu or local-lookup or other structured responses exist, keep them
        if isinstance(result, dict):
            return jsonify(result)
        else:
            return jsonify({"source": "local", "text": str(result)})
    except Exception as e:
        logger.exception("Error in /api/recommend")
        return jsonify({
            "error": "server error in /api/recommend",
            "detail": str(e),
            "traceback": traceback.format_exc()
        }), 500


# Streaming endpoint: streams JSON-lines as chunks become available
@app.route("/api/ai/stream", methods=["POST", "GET"])
def ai_stream():
    """
    POST JSON {"prompt":"..."} to stream incremental results.
    GET will return a short usage message.
    """
    if request.method == "GET":
        return jsonify({
            "info": "POST JSON {'prompt':'...'} to this endpoint to receive streaming JSON-lines"
        })

    try:
        body = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid json body"}), 400

    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "missing 'prompt' in body"}), 400

    # stream generator wrapper
    def gen():
        """
        Yields bytes. Uses arsemble_ai_mod.stream_response(prompt) if available (preferred).
        Otherwise uses get_ai_response(prompt) and yields small chunks (fallback).
        Each yielded item is a newline-terminated JSON text (JSON-lines).
        """
        try:
            # Preferred streaming interface if your ai module exposes it:
            if hasattr(arsemble_ai_mod, "stream_response"):
                for item in arsemble_ai_mod.stream_response(prompt):
                    # item may be bytes, str, or dict
                    if isinstance(item, bytes):
                        yield item + b"\n"
                    elif isinstance(item, str):
                        yield (item + "\n").encode("utf-8")
                    else:
                        yield (json.dumps(item) + "\n").encode("utf-8")
                return

            # Fallback: call your existing synchronous get_ai_response and stream in chunks
            if hasattr(arsemble_ai_mod, "get_ai_response"):
                # run blocking generation in a background thread and capture the full result
                result_container = {"result": None, "exc": None}

                def _runner():
                    try:
                        result_container["result"] = arsemble_ai_mod.get_ai_response(
                            prompt)
                    except Exception as e:
                        result_container["exc"] = e

                t = threading.Thread(target=_runner)
                t.start()
                t.join()
                if result_container["exc"]:
                    raise result_container["exc"]
                result = result_container["result"]
            else:
                result = None

            if result is None:
                yield (json.dumps({"error": "no_result"}) + "\n").encode("utf-8")
                return

            # If result is dict => send as single JSON object
            if isinstance(result, dict):
                yield (json.dumps(result) + "\n").encode("utf-8")
                return

            # Otherwise treat as text and stream in slices
            text = str(result)
            chunk_size = 256
            for i in range(0, len(text), chunk_size):
                part = text[i: i + chunk_size]
                yield (json.dumps({"chunk": i // chunk_size, "text": part}) + "\n").encode("utf-8")
        except Exception as e:
            try:
                yield (json.dumps({"error": "server_error", "msg": str(e)}) + "\n").encode("utf-8")
            except Exception:
                pass

    # stream_with_context ensures request context available while streaming
    return Response(stream_with_context(gen()), mimetype="application/json")


# Lookup endpoint: support POST (JSON) and GET (?chip_id=...)
@app.route("/api/lookup", methods=["POST", "GET"])
def lookup_api():
    try:
        if request.method == "GET":
            chip_id = request.args.get(
                "chip_id") or request.args.get("id") or ""
        else:
            body = request.get_json(force=True) or {}
            chip_id = (body.get("chip_id") or body.get("id") or "").strip()

        if not chip_id:
            return jsonify({"found": False, "error": "chip_id required (POST JSON or GET ?chip_id=)"}), 400

        details = get_component_details(chip_id)
        # Expect details to be a dict with at least 'found' key
        return jsonify(details)
    except Exception as e:
        logger.exception("server error in /api/lookup")
        return jsonify({"error": "server error in /api/lookup", "detail": str(e)}), 500


# Shops list (GET)
@app.route("/api/shops", methods=["GET"])
def get_shops():
    try:
        shops = list_shops(only_public=True)
        return jsonify({"shops": shops})
    except Exception as e:
        logger.exception("server error in /api/shops")
        return jsonify({"error": "server error in /api/shops", "detail": str(e)}), 500


# Health
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "ARsemble AI running!"})


if __name__ == "__main__":
    logger.info(f"Starting ARsemble unified server on http://{HOST}:{PORT}")
    serve(app, host=HOST, port=PORT)
