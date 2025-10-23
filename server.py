# server.py
from flask import Flask, request, jsonify, send_from_directory, abort
import os
import traceback
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AI-Fallback")

# Import your AI functions (defensive import with helpful error)
try:
    # file/name you used earlier was "arsemble_ai.py"
    from arsemble_ai import get_ai_response, get_component_details
except Exception as e:
    logger.exception("Failed to import arsemble_ai functions.")
    raise RuntimeError(
        "Can't import get_ai_response/get_component_details. "
        "Make sure arsemble_ai.py exists and defines those functions."
    ) from e

# Create Flask app (static folder = ./static)
app = Flask(__name__, static_folder="static", static_url_path="")

# --- Routes ---------------------------------------------------------------

# Basic health check


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

# Serve index.html at root


@app.route("/", methods=["GET"])
def index_route():
    try:
        return send_from_directory(app.static_folder, "index.html")
    except Exception:
        abort(404)

# Serve other static assets (only inside static folder)


@app.route("/<path:filename>")
def static_files_route(filename):
    try:
        return send_from_directory(app.static_folder, filename)
    except Exception:
        abort(404)

# API: recommend


@app.route("/api/recommend", methods=["POST"])
def recommend_api():
    try:
        body = request.get_json(force=True) or {}
        query = body.get("query", "") or ""
        resp = get_ai_response(query)

        # If the response is a dict coming from your local lookup (component details),
        # format it into clean plain text so the client sees readable specs.
        if isinstance(resp, dict):
            src = resp.get("source", "")

            # Local component details found -> return nice text under "text"
            if src == "local-lookup" and resp.get("found"):
                comp = resp.get("component", {})
                name = comp.get("name", "Unknown Component")
                ctype = (comp.get("type") or "").upper()
                price = comp.get("price", "N/A")
                specs = comp.get("specs", {}) or {}

                lines = [
                    f"🎯 {name} — {price}",
                    f"Type: {ctype}",
                    "Specifications:"
                ]
                # maintain a friendly ordering for common keys
                common_order = ["vram", "clock", "power", "tdp", "wattage",
                                "slot", "interface", "capacity", "compatibility"]
                added = set()
                for k in common_order:
                    if k in specs:
                        lines.append(f"  • {k.capitalize()}: {specs[k]}")
                        added.add(k)
                # add remaining specs
                for k, v in specs.items():
                    if k in added:
                        continue
                    lines.append(f"  • {k.capitalize()}: {v}")

                text_output = "\n".join(lines)
                return jsonify({"source": "local-lookup", "text": text_output})

            # Local lookup returned no exact match -> let client handle suggestions
            if src == "local-lookup" and not resp.get("found"):
                # If suggestions exist, include them in plain text
                suggestions = resp.get("suggestions", []) or []
                if suggestions:
                    lines = ["Could not find an exact match. Close matches:"]
                    for s in suggestions[:8]:
                        lines.append(
                            f"  • {s.get('name')} ({s.get('type')}) — score {s.get('score')}")
                    return jsonify({"source": "local-lookup", "text": "\n".join(lines)})
                # else fall back to returning the original dict (so client can inspect)
                return jsonify(resp)

            # Local recommendations (builds) or other structured outputs: return JSON unchanged
            if src in ("local-recommendation", "local-psu", "local-compatibility"):
                return jsonify(resp)

            # Gemini fallback or other dicts -> if it has 'text' key, return that wrapped
            if resp.get("text"):
                return jsonify(resp)

            # default: return the dict as-is
            return jsonify(resp)

        # If the response is plain text, wrap it into a JSON object
        return jsonify({"source": "local", "text": str(resp)})

    except Exception as e:
        logger.exception("server error in /api/recommend")
        return jsonify({"error": "server error in /api/recommend", "detail": str(e)}), 500

# API: lookup


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
        logger.exception("server error in /api/lookup")
        return jsonify({"error": "server error in /api/lookup", "detail": str(e)}), 500


# --- Main entrypoint (for local runs only) -------------------------------
if __name__ == "__main__":
    # Use PORT from environment (Render sets $PORT). Default 10000 for local dev.
    port = int(os.environ.get("PORT", 10000))
    # Allow enabling debug locally by setting FLASK_DEBUG=1 in env
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"

    # Log dataset summary if your ai module exposes it (non-fatal)
    logger.info("Starting server (host=0.0.0.0 port=%s debug=%s)",
                port, debug_mode)

    # Bind to 0.0.0.0 so Render (and other hosts) can access it
    # In production on Render you should use gunicorn instead of this dev server:
    # Start command (Render): gunicorn server:app --bind 0.0.0.0:$PORT --workers 3
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
