# server.py
# Unified ARsemble server — serves static UI and proxies API to FastAPI backend

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from waitress import serve
import requests
import os
import logging

# --- Config ---
API_BACKEND = os.getenv(
    "ARSSEMBLE_API", "http://127.0.0.1:8080")  # FastAPI backend
HOST = os.getenv("ARSSEMBLE_HOST", "0.0.0.0")
PORT = int(os.getenv("ARSSEMBLE_PORT", "10000"))

# --- Logging ---
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ARsemble-Server")

# --- Flask App ---
app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)


@app.before_request
def log_request():
    logger.info(f"Request: {request.method} {request.path}")

# --- Serve index.html ---


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

# --- Proxy endpoint for /api/recommend ---


@app.route("/api/recommend", methods=["POST"])
def proxy_recommend():
    try:
        body = request.get_json(force=True)
        resp = requests.post(f"{API_BACKEND}/api/ask", json=body)
        return jsonify(resp.json())
    except Exception as e:
        logger.exception("Proxy /api/recommend failed:")
        return jsonify({"error": str(e)}), 500

# --- Simple ping ---


@app.route("/api/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok", "message": "ARsemble Flask proxy running"})


# --- Run with Waitress ---
if __name__ == "__main__":
    logger.info(f"Starting unified ARsemble server on {HOST}:{PORT}")
    serve(app, host=HOST, port=PORT)
