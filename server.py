# server.py — Unified ARsemble AI server (UI + API)
# Serves static files from ./static, loads .env, provides API endpoints and streaming fallback.

from flask import Flask, request, jsonify, send_from_directory, abort, Response, stream_with_context
from flask_cors import CORS
from waitress import serve
import os
import logging
import traceback
import json
import threading
from flask_compress import Compress
from dotenv import load_dotenv

# ----------------------------
# Load .env and compute paths
# ----------------------------
load_dotenv()  # loads environment variables from .env if present

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # project root
STATIC_DIR = os.path.join(BASE_DIR, "static")           # <project>/static

# ----------------------------
# Environment / config
# ----------------------------
HOST = os.getenv("ARSSEMBLE_HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", os.getenv("ARSSEMBLE_PORT", "5000")))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CLIENT_SHOP_ID = os.getenv("CLIENT_SHOP_ID")
DEBUG_MODE = os.getenv("DEBUG", "False").lower() in ("1", "true", "yes")

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ARsemble")

# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
Compress(app)

# ----------------------------
# AI imports (with safe stubs)
# ----------------------------
try:
    from arsemble_ai import get_ai_response, get_component_details, list_shops
    import arsemble_ai as arsemble_ai_mod
    logger.info("Imported arsemble_ai module")
except Exception:
    logger.exception(
        "Could not import arsemble_ai; using stub implementations for local testing")

    def get_ai_response(q):
        return {"text": f"(stub) Received query: {q}"}

    def get_component_details(chip_id):
        return {"found": False, "id": chip_id}

    def list_shops(only_public=True):
        return []

    # create a lightweight module-like object for compatibility
    class _StubMod:
        @staticmethod
        def get_ai_response(q): return get_ai_response(q)
    arsemble_ai_mod = _StubMod()

# ----------------------------
# /api/recommend
# ----------------------------


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

        if isinstance(result, dict):
            looks_like_compat = any(k in result for k in (
                "verdict", "emoji_verdict", "matches", "suggestions", "reason", "text"))
            if not looks_like_compat:
                return jsonify(result)

            emoji = result.get("emoji_verdict") or (
                "✅" if result.get("verdict") == "compatible" else "❌")
            verdict = result.get("verdict") or (
                "compatible" if result.get("found") else "incompatible")
            reason = result.get("reason") or result.get("message") or ""
            text = result.get(
                "text") or f"{emoji} Compatibility: {verdict}. {reason}"

            match_chips = result.get("matches") or []
            suggestions = result.get("suggestions") or {}
            suggest_for_target = suggestions.get("for_target") or []
            suggest_for_other = suggestions.get("for_other") or []
            legacy_chips = result.get("chips") or []

            if match_chips:
                chips_to_show = match_chips
            elif suggest_for_target:
                chips_to_show = suggest_for_target
            elif legacy_chips:
                chips_to_show = legacy_chips
            else:
                raw = result.get("raw") or {}
                chips_to_show = []
                if isinstance(raw, dict):
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
                "chips": chips_to_show,
                "suggestions": {
                    "for_target": suggest_for_target,
                    "for_other": suggest_for_other
                },
                "raw": result
            }
            return jsonify(formatted)
        else:
            return jsonify({"source": "local", "text": str(result)})

    except Exception as e:
        logger.exception("Error in /api/recommend")
        return jsonify({
            "error": "server error in /api/recommend",
            "detail": str(e),
            "traceback": traceback.format_exc()
        }), 500

# ----------------------------
# /api/ai/stream
# ----------------------------


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
            if hasattr(arsemble_ai_mod, "stream_response"):
                for item in arsemble_ai_mod.stream_response(prompt):
                    if isinstance(item, bytes):
                        yield item + b"\n"
                    elif isinstance(item, str):
                        yield (item + "\n").encode("utf-8")
                    else:
                        yield (json.dumps(item) + "\n").encode("utf-8")
                return

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

# ----------------------------
# /api/lookup
# ----------------------------


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

# ----------------------------
# /api/shops
# ----------------------------


@app.route("/api/shops", methods=["GET"])
def get_shops():
    try:
        shops = list_shops(only_public=True)
        return jsonify({"shops": shops})
    except Exception as e:
        logger.exception("server error in /api/shops")
        return jsonify({"error": "server error in /api/shops", "detail": str(e)}), 500

# ----------------------------
# /api/health
# ----------------------------


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "ARsemble AI running!"})

# ----------------------------
# Static routes (serve from ./static)
# ----------------------------


@app.route("/", methods=["GET"])
def index():
    """
    Serve static/index.html if present, otherwise return a small fallback HTML.
    """
    try:
        index_path = os.path.join(app.static_folder, "index.html")
        if os.path.exists(index_path):
            return app.send_static_file("index.html")

        logger.warning(
            "index.html not found at %s — serving fallback HTML", index_path)
        fallback_html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>ARsemble — Fallback Index</title>
  <style>body{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial;padding:2rem;}a{display:block;margin:.5rem 0}code{background:#f1f1f1;padding:.1rem .3rem;border-radius:4px}</style>
</head>
<body>
  <h1>ARsemble — Fallback Index</h1>
  <p>index.html not found in the static folder. Useful endpoints:</p>
  <a href="/api/health"><code>/api/health</code></a>
  <a href="/api/recommend"><code>/api/recommend</code> (GET shows docs)</a>
  <a href="/api/lookup?chip_id=example"><code>/api/lookup?chip_id=example</code></a>
  <a href="/api/shops"><code>/api/shops</code></a>
  <p>To serve your app, place your frontend's <code>index.html</code> at <code>./static/index.html</code>.</p>
</body>
</html>"""
        return Response(fallback_html, mimetype="text/html")
    except Exception:
        logger.exception("Error serving fallback/index")
        abort(500)


@app.route("/<path:filename>", methods=["GET"])
def static_files(filename):
    try:
        full = os.path.join(app.static_folder, filename)
        if os.path.exists(full):
            return send_from_directory(app.static_folder, filename)
        abort(404)
    except Exception:
        abort(404)


# ----------------------------
# Start server
# ----------------------------
if __name__ == "__main__":
    logger.info("Loaded ENV: GEMINI_API_KEY=%s CLIENT_SHOP_ID=%s DEBUG=%s PORT=%s",
                GEMINI_API_KEY, CLIENT_SHOP_ID, DEBUG_MODE, PORT)
    logger.info("Starting ARsemble unified server on http://%s:%s", HOST, PORT)
    logger.info("Open your browser at: http://localhost:%s", PORT)
    serve(app, host=HOST, port=PORT)
