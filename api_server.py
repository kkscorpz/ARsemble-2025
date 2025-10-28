# server.py — Pure Backend API Only (No HTML)
from flask import Flask, request, jsonify
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
# Load .env and configuration
# ----------------------------
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
logger = logging.getLogger("ARsemble-API")

# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
Compress(app)

# ----------------------------
# AI imports
# ----------------------------
try:
    from arsemble_ai import (
        get_ai_response,
        get_component_details,
        list_shops,
        budget_builds,
        get_compatible_components,
        recommend_psu_for_query_with_chips,
        analyze_bottleneck_text,
        lookup_component_by_chip_id
    )
    import arsemble_ai as arsemble_ai_mod
    logger.info("✅ Successfully imported arsemble_ai module")
except Exception as e:
    logger.error(f"❌ Failed to import arsemble_ai: {e}")
    # Provide stub implementations for critical functions

    def get_ai_response(q):
        return {"error": "AI module not available", "text": f"Stub response for: {q}"}

    def get_component_details(chip_id):
        return {"found": False, "error": "AI module not available"}

    def list_shops(only_public=True):
        return []

    def budget_builds(budget, usage="general", top_n=3):
        return []

    def get_compatible_components(query, data):
        return {"found": False, "error": "AI module not available"}

    def recommend_psu_for_query_with_chips(query, data, headroom_percent=30):
        return {"error": "AI module not available"}

    def analyze_bottleneck_text(cpu, gpu, resolution="1080p", settings="high", target_fps=60):
        return "AI module not available"

    def lookup_component_by_chip_id(chip_id):
        return None

# ----------------------------
# API Routes
# ----------------------------


@app.route("/", methods=["GET"])
def api_root():
    """API Welcome Page"""
    return jsonify({
        "service": "ARsemble AI Backend API",
        "version": "1.0",
        "endpoints": {
            "GET /": "This welcome page",
            "POST /api/chat": "Main AI chat endpoint",
            "POST /api/recommend": "AI recommendations",
            "GET /api/lookup": "Component lookup",
            "POST /api/builds/budget": "Budget PC builds",
            "POST /api/compatibility": "Component compatibility check",
            "POST /api/psu/recommend": "PSU recommendations",
            "POST /api/analysis/bottleneck": "CPU/GPU bottleneck analysis",
            "GET /api/shops": "List computer shops",
            "GET /api/health": "Health check"
        }
    })


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "ARsemble AI Backend API",
        "timestamp": json.dumps(str(__import__('datetime').datetime.now()))
    })


@app.route("/api/chat", methods=["POST"])
def chat_api():
    """
    Main AI chat endpoint
    Expected JSON: {"message": "user query", "session_id": "optional"}
    """
    try:
        body = request.get_json(force=True) or {}
        query = (body.get("message") or "").strip()
        session_id = body.get("session_id", "default")

        if not query:
            return jsonify({"error": "Missing 'message' in request"}), 400

        logger.info(f"Chat request - Session: {session_id}, Query: {query}")

        # Get AI response
        result = get_ai_response(query)

        # Add session info
        if isinstance(result, dict):
            result['session_id'] = session_id
            result['user_query'] = query

        return jsonify(result)

    except Exception as e:
        logger.exception("Error in /api/chat")
        return jsonify({
            "error": "server error in /api/chat",
            "detail": str(e)
        }), 500


@app.route("/api/recommend", methods=["POST", "GET"])
def recommend_api():
    """
    AI recommendations endpoint (legacy compatibility)
    """
    if request.method == "GET":
        return jsonify({
            "info": "POST JSON {'query':'your query'} to this endpoint",
            "example": {"query": "Recommend me a ₱30000 gaming PC"}
        })

    try:
        body = request.get_json(force=True) or {}
        query = (body.get("query") or "").strip()
        if not query:
            return jsonify({"error": "Missing 'query' in request"}), 400

        logger.info(f"Recommendation query: {query}")
        result = get_ai_response(query)

        return jsonify(result)

    except Exception as e:
        logger.exception("Error in /api/recommend")
        return jsonify({
            "error": "server error in /api/recommend",
            "detail": str(e)
        }), 500


@app.route("/api/lookup", methods=["POST", "GET"])
def lookup_api():
    """
    Component lookup endpoint
    """
    try:
        if request.method == "GET":
            chip_id = request.args.get(
                "chip_id") or request.args.get("id") or ""
        else:
            body = request.get_json(force=True) or {}
            chip_id = (body.get("chip_id") or body.get("id") or "").strip()

        if not chip_id:
            return jsonify({"found": False, "error": "chip_id required"}), 400

        logger.info(f"Component lookup: {chip_id}")
        details = get_component_details(chip_id)
        return jsonify(details)

    except Exception as e:
        logger.exception("Error in /api/lookup")
        return jsonify({"error": "server error in /api/lookup", "detail": str(e)}), 500


@app.route("/api/builds/budget", methods=["POST"])
def budget_builds_api():
    """
    Generate budget PC builds
    Expected JSON: {"budget": 50000, "usage": "gaming", "top_n": 3}
    """
    try:
        body = request.get_json(force=True) or {}
        budget = body.get("budget")
        usage = body.get("usage", "general")
        top_n = body.get("top_n", 3)

        if budget is None:
            return jsonify({"error": "Missing 'budget' in request"}), 400

        try:
            budget = int(budget)
        except (ValueError, TypeError):
            return jsonify({"error": "Budget must be a number"}), 400

        if budget <= 0:
            return jsonify({"error": "Budget must be positive"}), 400

        logger.info(f"Budget builds request: ₱{budget}, usage: {usage}")

        builds = budget_builds(budget, usage=usage, top_n=top_n)

        return jsonify({
            "budget": budget,
            "usage": usage,
            "builds": builds,
            "count": len(builds)
        })

    except Exception as e:
        logger.exception("Error in /api/builds/budget")
        return jsonify({
            "error": "server error in /api/builds/budget",
            "detail": str(e)
        }), 500


@app.route("/api/compatibility", methods=["POST"])
def compatibility_api():
    """
    Check component compatibility
    Expected JSON: {"query": "compatibility query"}
    """
    try:
        body = request.get_json(force=True) or {}
        query = (body.get("query") or "").strip()

        if not query:
            return jsonify({"error": "Missing 'query' in request"}), 400

        logger.info(f"Compatibility check: {query}")

        # You'll need to import your data or pass it differently
        from arsemble_ai import data as component_data
        result = get_compatible_components(query, component_data)

        return jsonify(result)

    except Exception as e:
        logger.exception("Error in /api/compatibility")
        return jsonify({
            "error": "server error in /api/compatibility",
            "detail": str(e)
        }), 500


@app.route("/api/psu/recommend", methods=["POST"])
def psu_recommend_api():
    """
    PSU recommendation endpoint
    Expected JSON: {"query": "PSU query", "headroom_percent": 30}
    """
    try:
        body = request.get_json(force=True) or {}
        query = (body.get("query") or "").strip()
        headroom_percent = body.get("headroom_percent", 30)

        if not query:
            return jsonify({"error": "Missing 'query' in request"}), 400

        logger.info(
            f"PSU recommendation: {query}, headroom: {headroom_percent}%")

        from arsemble_ai import data as component_data
        result = recommend_psu_for_query_with_chips(
            query,
            component_data,
            headroom_percent=headroom_percent
        )

        return jsonify(result)

    except Exception as e:
        logger.exception("Error in /api/psu/recommend")
        return jsonify({
            "error": "server error in /api/psu/recommend",
            "detail": str(e)
        }), 500


@app.route("/api/analysis/bottleneck", methods=["POST"])
def bottleneck_analysis_api():
    """
    CPU/GPU bottleneck analysis
    Expected JSON: {"cpu": "cpu_name", "gpu": "gpu_name", "resolution": "1080p", "settings": "high", "target_fps": 60}
    """
    try:
        body = request.get_json(force=True) or {}
        cpu_name = body.get("cpu", "").strip()
        gpu_name = body.get("gpu", "").strip()
        resolution = body.get("resolution", "1080p")
        settings = body.get("settings", "high")
        target_fps = body.get("target_fps", 60)

        if not cpu_name or not gpu_name:
            return jsonify({"error": "Missing 'cpu' or 'gpu' in request"}), 400

        logger.info(f"Bottleneck analysis: CPU={cpu_name}, GPU={gpu_name}")

        # Lookup components
        cpu = lookup_component_by_chip_id(cpu_name)
        if not cpu:
            # Try fuzzy matching
            from arsemble_ai import _best_match_in_dataset, data
            cpu_obj, _ = _best_match_in_dataset(
                cpu_name, {"cpu": data.get("cpu", {})})
            cpu = cpu_obj

        gpu = lookup_component_by_chip_id(gpu_name)
        if not gpu:
            from arsemble_ai import _best_match_in_dataset, data
            gpu_obj, _ = _best_match_in_dataset(
                gpu_name, {"gpu": data.get("gpu", {})})
            gpu = gpu_obj

        if not cpu or not gpu:
            return jsonify({
                "error": "Could not find both CPU and GPU components"
            }), 404

        analysis_text = analyze_bottleneck_text(
            cpu, gpu,
            resolution=resolution,
            settings=settings,
            target_fps=target_fps
        )

        return jsonify({
            "cpu": cpu.get('name'),
            "gpu": gpu.get('name'),
            "analysis": analysis_text,
            "resolution": resolution,
            "settings": settings,
            "target_fps": target_fps
        })

    except Exception as e:
        logger.exception("Error in /api/analysis/bottleneck")
        return jsonify({
            "error": "server error in /api/analysis/bottleneck",
            "detail": str(e)
        }), 500


@app.route("/api/shops", methods=["GET"])
def shops_api():
    """
    Get list of computer shops
    Query params: ?public_only=true
    """
    try:
        public_only = request.args.get('public_only', 'true').lower() == 'true'
        shops = list_shops(only_public=public_only)

        return jsonify({
            "shops": shops,
            "count": len(shops),
            "public_only": public_only
        })

    except Exception as e:
        logger.exception("Error in /api/shops")
        return jsonify({
            "error": "server error in /api/shops",
            "detail": str(e)
        }), 500


@app.route("/api/ai/stream", methods=["POST"])
def ai_stream_api():
    """
    Streaming AI response endpoint (if needed)
    """
    try:
        body = request.get_json(force=True) or {}
        prompt = (body.get("prompt") or "").strip()

        if not prompt:
            return jsonify({"error": "Missing 'prompt' in request"}), 400

        def generate():
            try:
                result = get_ai_response(prompt)
                if isinstance(result, dict):
                    yield json.dumps(result) + "\n"
                else:
                    yield json.dumps({"text": str(result)}) + "\n"
            except Exception as e:
                yield json.dumps({"error": str(e)}) + "\n"

        return Response(generate(), mimetype="application/json")

    except Exception as e:
        logger.exception("Error in /api/ai/stream")
        return jsonify({"error": "server error in streaming"}), 500

# ----------------------------
# Error handlers
# ----------------------------


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({"error": "Internal server error"}), 500


# ----------------------------
# Start server
# ----------------------------
if __name__ == "__main__":
    logger.info(
        "🚀 Starting ARsemble Pure Backend API on http://%s:%s", HOST, PORT)
    logger.info("📚 API Documentation available at: http://localhost:%s", PORT)
    logger.info("🔧 Debug mode: %s", DEBUG_MODE)

    if DEBUG_MODE:
        app.run(host=HOST, port=PORT, debug=True)
    else:
        serve(app, host=HOST, port=PORT)
