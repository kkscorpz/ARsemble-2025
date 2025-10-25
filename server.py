# server.py — Unified ARsemble AI server (UI + API)
# Replaces the previous versions and avoids 405 by accepting GET/POST where appropriate.

from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from waitress import serve
import os
import logging
import traceback

# Import AI functions (must exist)
from arsemble_ai import get_ai_response, get_component_details, list_shops

HOST = os.getenv("ARSSEMBLE_HOST", "0.0.0.0")
PORT = int(os.getenv("ARSSEMBLE_PORT", "10000"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ARsemble")

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)


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
            "info": "POST JSON {'query':'...'} to this endpoint",
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
