# server.py — Unified ARsemble AI server (UI + API)
# Adds enhanced compatibility response formatting (✅/❌ verdicts + tappable suggestions)

from flask import Flask, request, jsonify, send_from_directory, abort, Response, stream_with_context
from flask_cors import CORS
from waitress import serve
import os
import logging
import traceback
import json
import threading
from flask_compress import Compress

# Import AI functions
from arsemble_ai import get_ai_response, get_component_details, list_shops
import arsemble_ai as arsemble_ai_mod

# Environment setup
HOST = os.getenv("ARSSEMBLE_HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", os.getenv("ARSSEMBLE_PORT", "10000")))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ARsemble")

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
Compress(app)


# ============================
# 🧠 /api/recommend (main AI)
# ============================
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

        # 🧩 Debug log for development
        try:
            if isinstance(result, dict):
                logger.info("DEBUG /api/recommend keys: %s",
                            ", ".join(result.keys()))
                logger.debug("DEBUG /api/recommend full result: %s",
                             json.dumps(result, default=str))
            else:
                logger.info(
                    "DEBUG /api/recommend returned non-dict result: %s", str(result)[:150])
        except Exception:
            logger.exception("Logging debug failed")

        # ✅ Format compatibility responses cleanly
        if isinstance(result, dict):
            looks_like_compat = any(k in result for k in (
                "verdict", "emoji_verdict", "matches", "suggestions", "reason", "text"))
            if not looks_like_compat:
                # normal non-compatibility query
                return jsonify(result)

            # --- Format the response for UI ---
            emoji = result.get("emoji_verdict") or (
                "✅" if result.get("verdict") == "compatible" else "❌")
            verdict = result.get("verdict") or (
                "compatible" if result.get("found") else "incompatible")
            reason = result.get("reason") or result.get("message") or ""
            text = result.get(
                "text") or f"{emoji} Compatibility: {verdict}. {reason}"

            # Choose which chips to show (motherboards, etc.) — enhanced fallbacks
            match_chips = result.get("matches") or []
            suggestions = result.get("suggestions") or {}
            suggest_for_target = suggestions.get("for_target") or []
            suggest_for_other = suggestions.get("for_other") or []
            legacy_chips = result.get("chips") or []

            # Primary: direct matches (expected when comparing two concrete parts)
            if match_chips:
                chips_to_show = match_chips
            # Secondary: suggestions targeted for the other-side (motherboards for CPU)
            elif suggest_for_target:
                chips_to_show = suggest_for_target
            # Tertiary: legacy chips field
            elif legacy_chips:
                chips_to_show = legacy_chips
            # Quaternary: inspect raw payload for nested suggestions/matches (best-effort)
            else:
                raw = result.get("raw") or {}
                chips_to_show = []
                if isinstance(raw, dict):
                    # raw.suggestions.for_target
                    try:
                        rf = raw.get("suggestions") or {}
                        if rf.get("for_target"):
                            chips_to_show = rf.get("for_target")
                        elif raw.get("matches"):
                            chips_to_show = raw.get("matches")
                        elif raw.get("chips"):
                            chips_to_show = raw.get("chips")
                    except Exception:
                        chips_to_show = []
                # Last fallback: empty list
                chips_to_show = chips_to_show or []

            formatted = {
                "source": "compatibility_resolver",
                "text": text,
                "emoji": emoji,
                "verdict": verdict,
                "reason": reason,
                "target": result.get("target"),
                "target_type": result.get("target_type"),
                "other": result.get("other"),
                "other_type": result.get("other_type"),
                # 👈 UI tapable components (motherboards, GPUs, etc.)
                "chips": chips_to_show,
                "suggestions": {
                    "for_target": suggest_for_target,
                    "for_other": suggest_for_other
                },
                "raw": result
            }

            return jsonify(formatted)

        # fallback: plain text response
        else:
            return jsonify({"source": "local", "text": str(result)})

    except Exception as e:
        logger.exception("Error in /api/recommend")
        return jsonify({
            "error": "server error in /api/recommend",
            "detail": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ===================================
# 🧩 /api/ai/stream — streaming mode
# ===================================
@app.route("/api/ai/stream", methods=["POST", "GET"])
def ai_stream():
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
            # Prefer streaming from arsemble_ai_mod if available
            if hasattr(arsemble_ai_mod, "stream_response"):
                for item in arsemble_ai_mod.stream_response(prompt):
                    if isinstance(item, bytes):
                        yield item + b"\n"
                    elif isinstance(item, str):
                        yield (item + "\n").encode("utf-8")
                    else:
                        yield (json.dumps(item) + "\n").encode("utf-8")
                return

            # fallback: non-streaming
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
            if result is None:
                yield (json.dumps({"error": "no_result"}) + "\n").encode("utf-8")
                return

            if isinstance(result, dict):
                yield (json.dumps(result) + "\n").encode("utf-8")
                return

            text = str(result)
            chunk_size = 256
            for i in range(0, len(text), chunk_size):
                yield (json.dumps({"chunk": i // chunk_size, "text": text[i:i+chunk_size]}) + "\n").encode("utf-8")

        except Exception as e:
            yield (json.dumps({"error": "server_error", "msg": str(e)}) + "\n").encode("utf-8")

    return Response(stream_with_context(gen()), mimetype="application/json")


# ==========================
# 🔍 /api/lookup
# ==========================
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


# ==========================
# 🏬 /api/shops
# ==========================
@app.route("/api/shops", methods=["GET"])
def get_shops():
    try:
        shops = list_shops(only_public=True)
        return jsonify({"shops": shops})
    except Exception as e:
        logger.exception("server error in /api/shops")
        return jsonify({"error": "server error in /api/shops", "detail": str(e)}), 500


# ==========================
# ❤️ Health Check
# ==========================
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "ARsemble AI running!"})


# ==========================
# 🌐 Static Routes
# ==========================
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


# ==========================
# 🚀 Entry Point
# ==========================
if __name__ == "__main__":
    logger.info(f"Starting ARsemble unified server on http://{HOST}:{PORT}")
    serve(app, host=HOST, port=PORT)
