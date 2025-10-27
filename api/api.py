# api/api.py — Dedicated ARsemble API Service
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import json

# Import AI functions
from arsemble_ai import (
    get_ai_response,
    get_component_details,
    list_shops,
    budget_builds as get_budget_builds
)

# Environment setup
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "10000"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ARsemble-API")

app = Flask(__name__)
CORS(app)

# ============================
# 🧠 CORE AI ENDPOINTS
# ============================


@app.route("/api/recommend", methods=["POST"])
def recommend():
    """Main AI recommendation endpoint"""
    try:
        body = request.get_json() or {}
        query = body.get("query", "").strip()

        if not query:
            return jsonify({
                "success": False,
                "error": "Query parameter required"
            }), 400

        logger.info(f"API Query: {query}")
        result = get_ai_response(query)

        return jsonify({
            "success": True,
            "query": query,
            "data": result
        })

    except Exception as e:
        logger.error(f"Recommend error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500


@app.route("/api/lookup", methods=["GET", "POST"])
def lookup():
    """Lookup component details"""
    try:
        if request.method == "GET":
            component_id = request.args.get("id", "").strip()
        else:
            body = request.get_json() or {}
            component_id = body.get("id", "").strip()

        if not component_id:
            return jsonify({
                "success": False,
                "error": "Component ID required"
            }), 400

        details = get_component_details(component_id)

        return jsonify({
            "success": True,
            "component_id": component_id,
            "data": details
        })

    except Exception as e:
        logger.error(f"Lookup error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Lookup failed"
        }), 500

# ============================
# 💰 BUDGET BUILDS
# ============================


@app.route("/api/budget-builds", methods=["POST"])
def budget_builds():
    """Generate budget PC builds"""
    try:
        body = request.get_json() or {}
        budget = body.get("budget")
        usage = body.get("usage", "gaming")
        top_n = body.get("top_n", 3)

        if not budget:
            return jsonify({
                "success": False,
                "error": "Budget parameter required"
            }), 400

        builds = get_budget_builds(int(budget), usage, int(top_n))

        return jsonify({
            "success": True,
            "parameters": {
                "budget": budget,
                "usage": usage,
                "top_n": top_n
            },
            "data": {
                "builds": builds,
                "count": len(builds)
            }
        })

    except Exception as e:
        logger.error(f"Budget builds error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Budget build generation failed"
        }), 500

# ============================
# 🔧 COMPONENTS
# ============================


@app.route("/api/components", methods=["GET"])
def list_components():
    """List all components by type"""
    try:
        from arsemble_ai import data

        component_type = request.args.get("type", "").lower()

        if component_type and component_type in data:
            components = list(data[component_type].keys())
            return jsonify({
                "success": True,
                "filters": {"type": component_type},
                "data": {
                    "components": components,
                    "count": len(components)
                }
            })
        else:
            return jsonify({
                "success": True,
                "data": {
                    "component_types": list(data.keys()),
                    "counts": {key: len(value) for key, value in data.items()}
                }
            })

    except Exception as e:
        logger.error(f"Components error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Failed to list components"
        }), 500


@app.route("/api/components/search", methods=["GET"])
def search_components():
    """Search components"""
    try:
        query = request.args.get("q", "").strip()
        component_type = request.args.get("type", "").lower()

        if not query:
            return jsonify({
                "success": False,
                "error": "Search query required"
            }), 400

        from arsemble_ai import data

        results = []
        search_types = [component_type] if component_type else data.keys()

        for comp_type in search_types:
            if comp_type in data:
                for comp_name, comp_data in data[comp_type].items():
                    if query.lower() in comp_name.lower():
                        results.append({
                            "name": comp_name,
                            "type": comp_type,
                            "data": comp_data
                        })

        return jsonify({
            "success": True,
            "search": {
                "query": query,
                "type_filter": component_type,
                "results_count": len(results)
            },
            "data": {
                "results": results[:20]
            }
        })

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Search failed"
        }), 500

# ============================
# 🏬 SHOPS
# ============================


@app.route("/api/shops", methods=["GET"])
def shops():
    """Get PC part shops"""
    try:
        shops_list = list_shops(only_public=True)

        return jsonify({
            "success": True,
            "data": {
                "shops": shops_list,
                "count": len(shops_list)
            }
        })

    except Exception as e:
        logger.error(f"Shops error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Failed to fetch shops"
        }), 500

# ============================
# 🔗 COMPATIBILITY
# ============================


@app.route("/api/compatibility", methods=["POST"])
def compatibility():
    """Check component compatibility"""
    try:
        body = request.get_json() or {}
        component1 = body.get("component1")
        component2 = body.get("component2")

        if not component1 or not component2:
            return jsonify({
                "success": False,
                "error": "Both component1 and component2 required"
            }), 400

        query = f"Check compatibility between {component1} and {component2}"
        result = get_ai_response(query)

        return jsonify({
            "success": True,
            "components": {
                "component1": component1,
                "component2": component2
            },
            "data": result
        })

    except Exception as e:
        logger.error(f"Compatibility error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Compatibility check failed"
        }), 500

# ============================
# 📊 HEALTH & INFO
# ============================


@app.route("/api/health", methods=["GET"])
def health():
    """Health check"""
    try:
        from arsemble_ai import data

        return jsonify({
            "status": "healthy",
            "service": "ARsemble API",
            "version": "1.0.0",
            "database": {
                "component_types": len(data),
                "total_components": sum(len(components) for components in data.values())
            }
        })
    except Exception as e:
        return jsonify({
            "status": "degraded",
            "error": str(e)
        }), 500


@app.route("/", methods=["GET"])
def root():
    """API Documentation"""
    return jsonify({
        "service": "ARsemble AI API",
        "version": "1.0.0",
        "documentation": "Visit /api/health for service status",
        "endpoints": {
            "POST /api/recommend": "AI recommendations",
            "GET /api/lookup": "Component details",
            "POST /api/budget-builds": "Budget PC builds",
            "GET /api/components": "List components",
            "GET /api/components/search": "Search components",
            "GET /api/shops": "PC part shops",
            "POST /api/compatibility": "Check compatibility",
            "GET /api/health": "Health check"
        }
    })

# ============================
# 🚀 START SERVER
# ============================


if __name__ == "__main__":
    logger.info(f"🚀 ARsemble API Service starting on http://{HOST}:{PORT}")
    logger.info("📚 Available Endpoints:")
    logger.info("   GET  /                    - API documentation")
    logger.info("   POST /api/recommend       - AI recommendations")
    logger.info("   GET  /api/lookup          - Component details")
    logger.info("   POST /api/budget-builds   - Budget builds")
    logger.info("   GET  /api/components      - List components")
    logger.info("   GET  /api/health          - Health check")

    app.run(host=HOST, port=PORT, debug=False)
