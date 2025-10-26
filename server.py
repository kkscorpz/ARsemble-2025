# server.py — Unified ARsemble AI server (UI + API)
# Replaces previous versions. Adds streaming JSON-lines endpoint /api/ai/stream and gzip compression.
# Static routes moved to the bottom so API routes are matched first (avoids 404/405 on /api/*).

from flask import Flask, request, jsonify, send_from_directory, abort, Response, stream_with_context
from flask_cors import CORS
from waitress import serve
import os
import logging
import traceback
import json
import threading
from flask_compress import Compress

# Import AI functions (must exist)
from arsemble_ai import get_ai_response, get_component_details, list_shops
import arsemble_ai as arsemble_ai_mod

# Environment: prefer standard PORT env var (Render), fallback to ARSEMBLE_PORT then 10000
HOST = os.getenv("ARSSEMBLE_HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", os.getenv("ARSSEMBLE_PORT", "10000")))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ARsemble")

app = Flask(__name__, static_folder="static", static_url_path="")
# allow all origins for API routes (change to specific origin in production)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
# enable gzip compression for dynamic responses
Compress(app)


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

    def gen():
        try:
            # If arsemble_ai_mod provides a stream_response generator, prefer it
            if hasattr(arsemble_ai_mod, "stream_response"):
                for item in arsemble_ai_mod.stream_response(prompt):
                    if isinstance(item, bytes):
                        yield item + b"\n"
                    elif isinstance(item, str):
                        yield (item + "\n").encode("utf-8")
                    else:
                        yield (json.dumps(item) + "\n").encode("utf-8")
                return

            # Fallback: blocking get_ai_response, run in thread and stream slices
            if hasattr(arsemble_ai_mod, "get_ai_response"):
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

            if isinstance(result, dict):
                yield (json.dumps(result) + "\n").encode("utf-8")
                return

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


# Serve index / static files (moved to bottom so API routes are matched first)
@app.route("/", methods=["GET"])
def index():
    try:
        return send_from_directory(app.static_folder, "index.html")
    except Exception:
        abort(404)


@app.route("/<path:filename>", methods=["GET"])
def static_files(filename):
    try:
        full = os.path.join(app.static_folder, filename)
        if os.path.exists(full):
            return send_from_directory(app.static_folder, filename)
        abort(404)
    except Exception:
        abort(404)


if __name__ == "__main__":
    logger.info(f"Starting ARsemble unified server on http://{HOST}:{PORT}")
    serve(app, host=HOST, port=PORT)
