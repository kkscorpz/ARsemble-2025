# api/api.py — Pure API Service
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import traceback

# Import AI functions
from arsemble_ai import get_ai_response, get_component_details, list_shops, budget_builds as get_budget_builds

# Environment setup
HOST = os.getenv("ARSSEMBLE_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", "10001"))  # Different port from main server

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ARsemble-API")

app = Flask(__name__)
CORS(app)

# ============================
# 🧠 /recommend (main AI)
# ============================


@app.route("/recommend", methods=["POST", "GET"])
def recommend_api():
    if request.method == "GET":
        return jsonify({
            "service": "ARsemble AI API",
            "endpoint": "/recommend",
            "method": "POST",
            "example": {"query": "Recommend me a ₱30000 PC build"}
        })

    try:
        body = request.get_json(force=True) or {}
        query = (body.get("query") or "").strip()
        if not query:
            return jsonify({"error": "Missing 'query' in request"}), 400

        logger.info("AI query: %s", query)
        result = get_ai_response(query)

        if isinstance(result, dict):
            return jsonify(result)
        else:
            return jsonify({"text": str(result)})

    except Exception as e:
        logger.exception("Error in /recommend")
        return jsonify({"error": "server error", "detail": str(e)}), 500

# ==========================
# 🔍 /lookup
# ==========================


@app.route("/lookup", methods=["POST", "GET"])
def lookup_api():
    try:
        if request.method == "GET":
            chip_id = request.args.get(
                "chip_id") or request.args.get("id") or ""
        else:
            body = request.get_json(force=True) or {}
            chip_id = (body.get("chip_id") or body.get("id") or "").strip()

        if not chip_id:
            return jsonify({"error": "chip_id required"}), 400

        details = get_component_details(chip_id)
        return jsonify(details)

    except Exception as e:
        logger.exception("Error in /lookup")
        return jsonify({"error": str(e)}), 500

# ==========================
# 🏬 /shops
# ==========================


@app.route("/shops", methods=["GET"])
def get_shops():
    try:
        shops = list_shops(only_public=True)
        return jsonify({"shops": shops})
    except Exception as e:
        logger.exception("Error in /shops")
        return jsonify({"error": str(e)}), 500

# ==========================
# 💰 /budget-builds
# ==========================


@app.route("/budget-builds", methods=["POST"])
def budget_builds():
    try:
        body = request.get_json(force=True) or {}
        budget = body.get("budget")
        usage = body.get("usage", "gaming")

        if not budget:
            return jsonify({"error": "Budget required"}), 400

        builds = get_budget_builds(int(budget), usage)

        return jsonify({
            "budget": budget,
            "usage": usage,
            "builds": builds
        })

    except Exception as e:
        logger.exception("Error in /budget-builds")
        return jsonify({"error": str(e)}), 500

# ==========================
# 🔧 /compatibility
# ==========================


@app.route("/compatibility", methods=["POST"])
def check_compatibility():
    try:
        body = request.get_json(force=True) or {}
        component1 = body.get("component1")
        component2 = body.get("component2")

        if not component1 or not component2:
            return jsonify({"error": "Both component1 and component2 required"}), 400

        query = f"Are {component1} and {component2} compatible?"
        result = get_ai_response(query)

        return jsonify(result)

    except Exception as e:
        logger.exception("Error in /compatibility")
        return jsonify({"error": str(e)}), 500

# ==========================
# 📊 /components
# ==========================


@app.route("/components", methods=["GET"])
def list_components():
    try:
        component_type = request.args.get("type", "").lower()
        from arsemble_ai import data

        if component_type and component_type in data:
            components = list(data[component_type].keys())
            return jsonify({
                "type": component_type,
                "components": components[:50]  # Limit results
            })
        else:
            # Return all component types
            return jsonify({
                "component_types": list(data.keys()),
                "counts": {key: len(value) for key, value in data.items()}
            })

    except Exception as e:
        logger.exception("Error in /components")
        return jsonify({"error": str(e)}), 500

# ==========================
# ❤️ Health & Root
# ==========================


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "ARsemble AI API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "/recommend": "POST - AI recommendations",
            "/lookup": "GET/POST - Component details",
            "/shops": "GET - PC part shops",
            "/budget-builds": "POST - Budget PC builds",
            "/compatibility": "POST - Check compatibility",
            "/components": "GET - List components",
            "/health": "GET - Health check"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "ARsemble AI API"})


# ==========================
# 🚀 Entry Point
# ==========================
if __name__ == "__main__":
    logger.info(f"🚀 ARsemble API starting on http://{HOST}:{PORT}")
    logger.info("📚 Available endpoints:")
    logger.info("   GET  /          - API documentation")
    logger.info("   POST /recommend - AI recommendations")
    logger.info("   GET  /lookup    - Component details")
    logger.info("   POST /budget-builds - Budget builds")
    logger.info("   GET  /health    - Health check")

    app.run(host=HOST, port=PORT, debug=False)
