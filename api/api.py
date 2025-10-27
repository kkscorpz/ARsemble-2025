# api.py - DEDICATED API FOR UNITY
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from arsemble_ai import get_ai_response, budget_builds, get_component_details, list_shops

app = Flask(__name__)
CORS(app)  # ✅ Important for Unity!


@app.route('/api/ask', methods=['POST'])
def ask_ai():
    """Simple AI endpoint for Unity"""
    data = request.get_json()
    question = data.get('question', '')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        response = get_ai_response(question)
        return jsonify({
            'success': True,
            'question': question,
            'answer': str(response)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/budget', methods=['POST'])
def get_budget():
    """Budget builds for Unity"""
    data = request.get_json()
    budget = data.get('budget', 0)
    usage = data.get('usage', 'gaming')

    try:
        builds = budget_builds(int(budget), usage)
        return jsonify({
            'success': True,
            'budget': budget,
            'builds': builds
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ready', 'service': 'ARsemble API'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
