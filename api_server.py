# api_server.py - Pure Backend API for Render
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import json
from datetime import datetime

# ----------------------------
# Configuration
# ----------------------------
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "10000"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger("ARsemble-API")

# ----------------------------
# Flask App
# ----------------------------
app = Flask(__name__)
CORS(app)

# ----------------------------
# AI Module Import with Better Error Handling
# ----------------------------
AI_AVAILABLE = False
try:
    # Try to import all AI functions
    from arsemble_ai import (
        get_ai_response,
        get_component_details,
        list_shops,
        budget_builds,
        get_compatible_components,
        recommend_psu_for_query_with_chips,
        analyze_bottleneck_text,
        lookup_component_by_chip_id,
        data as component_data
    )
    AI_AVAILABLE = True
    logger.info("✅ arsemble_ai module imported successfully")

except ImportError as e:
    logger.error(f"❌ Failed to import arsemble_ai: {e}")

    # Stub implementations
    def get_ai_response(query):
        return {
            "source": "stub",
            "text": f"AI module not available. Original query: {query}",
            "error": "AI module failed to load"
        }

    def get_component_details(chip_id):
        return {
            "found": False,
            "error": "AI module not available",
            "debug": f"Requested: {chip_id}"
        }

    def list_shops(only_public=True):
        return {
            "smfp_computer": {
                "name": "SMFP Computer",
                "address": "594 J Nepomuceno St, Quiapo, Manila",
                "region": "Metro Manila",
                "public": True
            }
        }

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

    component_data = {}

# ----------------------------
# API Routes
# ----------------------------


@app.route("/", methods=["GET"])
def api_root():
    """API Welcome Page"""
    return jsonify({
        "service": "ARsemble AI Backend API",
        "version": "1.0",
        "status": "running",
        "ai_available": AI_AVAILABLE,
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "GET /": "This welcome page",
            "GET /api/health": "Health check",
            "POST /api/chat": "Main AI chat endpoint",
            "GET /api/lookup?chip_id=...": "Component lookup",
            "POST /api/builds/budget": "Budget PC builds",
            "POST /api/compatibility": "Component compatibility check",
            "POST /api/psu/recommend": "PSU recommendations",
            "POST /api/analysis/bottleneck": "CPU/GPU bottleneck analysis",
            "GET /api/shops": "List computer shops"
        },
        "example_usage": {
            "chat": 'curl -X POST https://your-api.onrender.com/api/chat -H "Content-Type: application/json" -d \'{"message": "Hello"}\'',
            "health": "curl https://your-api.onrender.com/api/health"
        }
    })


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "ARsemble AI Backend API",
        "timestamp": datetime.now().isoformat(),
        "ai_module": "available" if AI_AVAILABLE else "unavailable",
        "environment": {
            "host": HOST,
            "port": PORT,
            "debug_mode": DEBUG_MODE,
            "gemini_key_set": bool(GEMINI_API_KEY)
        }
    })


@app.route("/api/chat", methods=["POST"])
def chat_api():
    """
    Main AI chat endpoint
    Expected JSON: {"message": "user query"}
    """
    try:
        # Parse JSON data
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        data = request.get_json()
        if not data:
            return jsonify({"error": "Empty JSON body"}), 400

        message = data.get("message", "").strip()
        if not message:
            return jsonify({"error": "Missing 'message' field"}), 400

        logger.info(f"Chat request: {message[:100]}...")

        # Get AI response
        response = get_ai_response(message)

        # Ensure response is properly formatted
        if not isinstance(response, dict):
            response = {"source": "ai", "text": str(response)}

        # Add metadata
        response.update({
            "timestamp": datetime.now().isoformat(),
            "user_query": message,
            "ai_available": AI_AVAILABLE
        })

        return jsonify(response)

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({
            "error": "Internal server error in /api/chat",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


@app.route("/api/lookup", methods=["GET"])
def lookup_api():
    """
    Component lookup endpoint
    GET: /api/lookup?chip_id=component_id
    """
    try:
        chip_id = request.args.get("chip_id", "").strip()
        if not chip_id:
            return jsonify({
                "error": "Missing chip_id parameter",
                "example": "/api/lookup?chip_id=amd-ryzen-5-5600x"
            }), 400

        logger.info(f"Component lookup: {chip_id}")
        details = get_component_details(chip_id)

        return jsonify(details)

    except Exception as e:
        logger.error(f"Lookup error: {str(e)}")
        return jsonify({
            "error": "Component lookup failed",
            "message": str(e)
        }), 500


@app.route("/api/builds/budget", methods=["POST"])
def budget_builds_api():
    """
    Generate budget PC builds
    Expected JSON: {"budget": 50000, "usage": "gaming", "top_n": 3}
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        data = request.get_json()
        if not data:
            return jsonify({"error": "Empty JSON body"}), 400

        budget = data.get("budget")
        if budget is None:
            return jsonify({"error": "Missing 'budget' field"}), 400

        try:
            budget = int(budget)
        except (ValueError, TypeError):
            return jsonify({"error": "Budget must be a number"}), 400

        if budget <= 0:
            return jsonify({"error": "Budget must be positive"}), 400

        usage = data.get("usage", "general")
        top_n = data.get("top_n", 3)

        logger.info(f"Budget builds: ₱{budget}, usage: {usage}")

        builds = budget_builds(budget, usage=usage, top_n=top_n)

        return jsonify({
            "budget": budget,
            "usage": usage,
            "builds": builds,
            "count": len(builds),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Budget builds error: {str(e)}")
        return jsonify({
            "error": "Failed to generate budget builds",
            "message": str(e)
        }), 500


@app.route("/api/compatibility", methods=["POST"])
def compatibility_api():
    """
    Check component compatibility
    Expected JSON: {"query": "compatibility query"}
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        data = request.get_json()
        if not data:
            return jsonify({"error": "Empty JSON body"}), 400

        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "Missing 'query' field"}), 400

        logger.info(f"Compatibility check: {query}")

        result = get_compatible_components(query, component_data)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Compatibility error: {str(e)}")
        return jsonify({
            "error": "Compatibility check failed",
            "message": str(e)
        }), 500


@app.route("/api/psu/recommend", methods=["POST"])
def psu_recommend_api():
    """
    PSU recommendation endpoint
    Expected JSON: {"query": "PSU query", "headroom_percent": 30}
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        data = request.get_json()
        if not data:
            return jsonify({"error": "Empty JSON body"}), 400

        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "Missing 'query' field"}), 400

        headroom_percent = data.get("headroom_percent", 30)

        logger.info(f"PSU recommendation: {query}")

        result = recommend_psu_for_query_with_chips(
            query,
            component_data,
            headroom_percent=headroom_percent
        )

        return jsonify(result)

    except Exception as e:
        logger.error(f"PSU recommendation error: {str(e)}")
        return jsonify({
            "error": "PSU recommendation failed",
            "message": str(e)
        }), 500


@app.route("/api/shops", methods=["GET"])
def shops_api():
    """
    Get list of computer shops
    """
    try:
        public_only = request.args.get('public_only', 'true').lower() == 'true'
        shops = list_shops(only_public=public_only)

        return jsonify({
            "shops": shops,
            "count": len(shops),
            "public_only": public_only,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Shops error: {str(e)}")
        return jsonify({
            "error": "Failed to get shops",
            "message": str(e)
        }), 500

# ----------------------------
# Error Handlers
# ----------------------------


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": [
            "/", "/api/health", "/api/chat", "/api/lookup",
            "/api/builds/budget", "/api/compatibility",
            "/api/psu/recommend", "/api/shops"
        ]
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({"error": "Internal server error"}), 500


# ----------------------------
# Startup
# ----------------------------
if __name__ == "__main__":
    logger.info("🚀 Starting ARsemble API Server")
    logger.info(f"📍 Host: {HOST}, Port: {PORT}")
    logger.info(f"🔧 Debug Mode: {DEBUG_MODE}")
    logger.info(
        f"🤖 AI Module: {'✅ Available' if AI_AVAILABLE else '❌ Unavailable'}")

    if not GEMINI_API_KEY:
        logger.warning(
            "⚠️  GEMINI_API_KEY not set - Gemini features may not work")

    # Use production server for Render
    if DEBUG_MODE:
        from waitress import serve
        serve(app, host=HOST, port=PORT)
    else:
        from waitress import serve
        serve(app, host=HOST, port=PORT)
