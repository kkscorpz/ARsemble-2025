# server.py
# Lightweight Flask API that wraps ARsemble AI module and serves static UI

from flask import Flask, request, jsonify, send_from_directory, abort
import os
import logging
import traceback
import warnings

# Prefer environment-driven host/port
HOST = os.getenv("ARSSEMBLE_HOST", "0.0.0.0")
PORT = int(os.getenv("ARSSEMBLE_PORT", "10000"))

# Silence noisy warnings about dev server (we control which server we run)
warnings.filterwarnings("ignore", message=".*development server.*")

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("AI-Fallback")
# Reduce chatter from Werkzeug dev server and third-party libs
logging.getLogger("werkzeug").setLevel(logging.WARNING)
logging.getLogger("waitress").setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Defensive import of AI module
try:
    from arsemble_ai import get_ai_response, get_component_details, list_shops
    # expose dataset too if needed
    try:
        from arsemble_ai import data as DATA
    except Exception:
        DATA = None
except Exception as e:
    logger.exception("Failed to import arsemble_ai functions.")
    raise RuntimeError(
        "Can't import get_ai_response/get_component_details/list_shops. Make sure arsemble_ai.py exists and defines those functions."
    ) from e

# Create Flask app (serves ./static by default)
app = Flask(__name__, static_folder="static", static_url_path="")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/", methods=["GET"])
def index_route():
    try:
        return send_from_directory(app.static_folder, "index.html")
    except Exception:
        abort(404)


@app.route("/<path:filename>")
def static_files_route(filename):
    try:
        return send_from_directory(app.static_folder, filename)
    except Exception:
        abort(404)


@app.route("/api/recommend", methods=["POST"])
def recommend_api():
    try:
        body = request.get_json(force=True) or {}
        query = body.get("query", "") or ""
        shop_id = body.get("shop_id")

        # Forward to AI handler
        resp = get_ai_response(query)

        # If structured dict -> transform certain types into plain text for easy UI display
        if isinstance(resp, dict):
            src = resp.get("source", "")

            # Local component details
            if src == "local-lookup" and resp.get("found"):
                comp = resp.get("component", {})
                name = comp.get("name", "Unknown Component")
                ctype = (comp.get("type") or "").upper()
                price = comp.get("price", "N/A")
                specs = comp.get("specs", {}) or {}

                lines = [f"🎯 {name} — {price}",
                         f"Type: {ctype}", "Specifications:"]
                common_order = ["vram", "clock", "power", "tdp", "wattage",
                                "slot", "interface", "capacity", "compatibility"]
                added = set()
                for k in common_order:
                    if k in specs:
                        lines.append(f"  • {k.capitalize()}: {specs[k]}")
                        added.add(k)
                for k, v in specs.items():
                    if k in added:
                        continue
                    lines.append(f"  • {k.capitalize()}: {v}")

                text_output = "\n".join(lines)
                return jsonify({"source": "local-lookup", "text": text_output})

            # Local lookup suggestions (no exact match)
            if src == "local-lookup" and not resp.get("found"):
                suggestions = resp.get("suggestions", []) or []
                if suggestions:
                    lines = ["Could not find an exact match. Close matches:"]
                    for s in suggestions[:8]:
                        lines.append(
                            f"  • {s.get('name')} ({s.get('type')}) — score {s.get('score')}")
                    return jsonify({"source": "local-lookup", "text": "\n".join(lines)})
                return jsonify(resp)

            # Local structured outputs (builds, psu, compatibility) -> send as-is (frontend knows how to render)
            if src in ("local-recommendation", "local-psu", "local-compatibility", "local-list"):
                return jsonify(resp)

            # Shops list
            if src == "local-list" and resp.get("type") == "shops_list":
                return jsonify(resp)

            # Gemini fallback or other dicts with text
            if resp.get("text"):
                return jsonify(resp)

            # default: return whatever we got for debugging
            return jsonify(resp)

        # Plain text fallback
        return jsonify({"source": "local", "text": str(resp)})

    except Exception as e:
        logger.exception("server error in /api/recommend")
        return jsonify({"error": "server error in /api/recommend", "detail": str(e)}), 500


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


@app.route("/api/shops", methods=["GET"])
def get_shops():
    try:
        shops = list_shops(only_public=True)
        return jsonify({"shops": shops})
    except Exception as e:
        logger.exception("server error in /api/shops")
        return jsonify({"error": "server error in /api/shops", "detail": str(e)}), 500


# Entry point: require Waitress to run
if __name__ == "__main__":
    # Prefer Waitress and fail fast if absent so the dev-server warning never appears
    try:
        from waitress import serve
    except Exception as e:
        logger.error(
            "Waitress is not installed or failed to import. Install with: pip install waitress")
        raise e

    logger.info("Starting server with Waitress on %s:%s", HOST, PORT)
    serve(app, host=HOST, port=PORT)
