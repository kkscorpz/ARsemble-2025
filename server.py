# server.py
# Unified ARsemble AI Server (Frontend + API)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from waitress import serve
import os
import logging
import traceback

# --- Import the main AI handler ---
from arsemble_ai import get_ai_response

# --- Config ---
HOST = os.getenv("ARSSEMBLE_HOST", "0.0.0.0")
PORT = int(os.getenv("ARSSEMBLE_PORT", "10000"))

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger("ARsemble")

# --- Flask App ---
app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# --- Serve frontend ---


@app.route("/")
def index():
    """Serve the main UI (index.html)."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    """Serve static frontend files."""
    return send_from_directory(app.static_folder, filename)

# --- Unified AI API ---


@app.route("/api/recommend", methods=["POST"])
def recommend_api():
    """Handles all AI queries: build, PSU, bottleneck, etc."""
    try:
        body = request.get_json(force=True) or {}
        query = (body.get("query") or "").strip()

        if not query:
            return jsonify({"error": "Missing 'query' in request"}), 400

        logger.info(f"Received AI query: {query}")

        # Call AI core
        result = get_ai_response(query)

        # Normalize output
        if isinstance(result, dict):
            return jsonify(result)
        else:
            return jsonify({"source": "local", "text": str(result)})

    except Exception as e:
        logger.exception("Error in /api/recommend")
        return jsonify({
            "error": "Server error",
            "detail": str(e),
            "traceback": traceback.format_exc()
        }), 500

# --- Health check ---


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "ARsemble AI running!"})


# --- Run with Waitress ---
if __name__ == "__main__":
    logger.info(f"Starting ARsemble unified API on http://{HOST}:{PORT}")
    serve(app, host=HOST, port=PORT)
