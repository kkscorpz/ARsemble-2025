# server.py
from flask import Flask, request, jsonify, send_from_directory
import os
import traceback

# Import your AI functions from ARsemble_ai.py
try:
    from arsemble_ai import get_ai_response, get_component_details
except Exception:
    try:
        from arsemble_ai import get_ai_response, get_component_details
    except Exception as e:
        raise RuntimeError(
            "Can't import get_ai_response/get_component_details. "
            "Make sure ARsemble_ai.py exists and defines those functions."
        ) from e

app = Flask(__name__, static_folder="static")

# Serve index.html at root


@app.route("/", methods=["GET"])
def index_route():
    return send_from_directory(app.static_folder, "index.html")

# Serve other static assets if needed (safe: only inside static folder)


@app.route("/<path:filename>")
def static_files_route(filename):
    return send_from_directory(app.static_folder, filename)

# API: recommend (unique function name)


@app.route("/api/recommend", methods=["POST"])
def recommend_api():
    try:
        body = request.get_json(force=True) or {}
        query = body.get("query", "")
        resp = get_ai_response(query)

        if resp.get("source") == "local-recommendation":
            return jsonify(resp)

        if resp.get("source") == "gemini-fallback":
            return jsonify({"source": "gemini-fallback", "text": resp.get("text", "")})

        return jsonify(resp)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "server error in /api/recommend", "detail": str(e)}), 500

# API: lookup (unique function name)


@app.route("/api/lookup", methods=["POST"])
def lookup_api():
    try:
        body = request.get_json(force=True) or {}
        chip_id = (body.get("chip_id") or body.get("id") or "").strip()
        if not chip_id:
            return jsonify({"found": False, "error": "chip_id required"}), 400
        details = get_component_details(chip_id)
        return jsonify(details)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "server error in /api/lookup", "detail": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(
        f"Starting server at http://127.0.0.1:{port} — serving static/index.html and /api/*")
    app.run(host="127.0.0.1", port=port, debug=True)
