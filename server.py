from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import json
import logging
from typing import Dict, Any, List

# Import your existing backend functions
from arsemble_ai import (
    get_ai_response,
    get_component_details,
    budget_builds,
    get_compatible_components,
    gemini_fallback_with_data,
    make_public_data,
    data  # your components dataset
)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ARsemble-Server")

# ==================== ROUTES ====================


@app.route('/')
def index():
    """Serve the main UI"""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint - handles all AI responses"""
    try:
        data = request.get_json()
        user_input = data.get('message', '').strip()

        if not user_input:
            return jsonify({'error': 'Empty message'}), 400

        logger.info(f"Chat request: {user_input}")

        # Get AI response using your unified handler
        response = get_ai_response(user_input)

        return jsonify(response)

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/components/details', methods=['POST'])
def component_details():
    """Get detailed component information"""
    try:
        data = request.get_json()
        component_query = data.get('component', '').strip()

        if not component_query:
            return jsonify({'error': 'No component specified'}), 400

        details = get_component_details(component_query)
        return jsonify(details)

    except Exception as e:
        logger.error(f"Component details error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/builds/budget', methods=['POST'])
def budget_recommendations():
    """Get budget build recommendations"""
    try:
        data = request.get_json()
        budget = data.get('budget')
        usage = data.get('usage', 'general')
        top_n = data.get('top_n', 3)

        if not budget:
            return jsonify({'error': 'No budget specified'}), 400

        builds = budget_builds(int(budget), usage, int(top_n))

        # Format builds for response
        formatted_builds = []
        for i, build in enumerate(builds):
            formatted_builds.append({
                'id': f'build_{i+1}',
                'total_price': build.get('total_price'),
                'score': build.get('score'),
                'components': {
                    'cpu': build.get('cpu'),
                    'gpu': build.get('gpu'),
                    'motherboard': build.get('motherboard'),
                    'ram': build.get('ram'),
                    'storage': build.get('storage'),
                    'psu': build.get('psu')
                }
            })

        return jsonify({
            'budget': budget,
            'usage': usage,
            'builds': formatted_builds
        })

    except Exception as e:
        logger.error(f"Budget builds error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/components/compatible', methods=['POST'])
def compatibility_check():
    """Check component compatibility"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()

        if not query:
            return jsonify({'error': 'No query specified'}), 400

        compatibility = get_compatible_components(query, data)
        return jsonify(compatibility)

    except Exception as e:
        logger.error(f"Compatibility check error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'ARsemble AI Backend',
        'dataset_summary': {
            'cpu_count': len(data.get('cpu', {})),
            'gpu_count': len(data.get('gpu', {})),
            'motherboard_count': len(data.get('motherboard', {})),
            'ram_count': len(data.get('ram', {})),
            'storage_count': len(data.get('storage', {})),
            'psu_count': len(data.get('psu', {}))
        }
    })

# ==================== ERROR HANDLERS ====================


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ==================== MAIN ====================


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)

    # Create basic HTML template
    with open('templates/index.html', 'w') as f:
        f.write('''<!DOCTYPE html>
<html>
<head>
    <title>ARsemble AI - PC Building Assistant</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .chat-container { max-width: 800px; margin: 0 auto; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .user { background: #e3f2fd; text-align: right; }
        .assistant { background: #f5f5f5; }
        .chip { display: inline-block; background: #2196f3; color: white; padding: 5px 10px; margin: 2px; border-radius: 15px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="chat-container">
        <h1>ARsemble AI - PC Building Assistant</h1>
        <div id="chat-messages"></div>
        <input type="text" id="user-input" placeholder="Ask about PC components..." style="width: 70%; padding: 10px;">
        <button onclick="sendMessage()">Send</button>
    </div>

    <script>
        async function sendMessage() {
            const input = document.getElementById('user-input');
            const message = input.value.trim();
            
            if (!message) return;
            
            // Add user message to chat
            addMessage('user', message);
            input.value = '';
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
                });
                
                const data = await response.json();
                
                if (data.chips) {
                    // Display chips if available
                    let chipHTML = data.chips.map(chip => 
                        `<div class="chip" onclick="handleChipClick('${chip.id}')">${chip.text}</div>`
                    ).join('');
                    addMessage('assistant', data.text + '<br>' + chipHTML);
                } else {
                    addMessage('assistant', data.text);
                }
                
            } catch (error) {
                addMessage('assistant', 'Error: Could not get response');
            }
        }
        
        function addMessage(role, content) {
            const messagesDiv = document.getElementById('chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            messageDiv.innerHTML = content;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function handleChipClick(chipId) {
            // Handle chip clicks - you can implement this based on your needs
            alert('Chip clicked: ' + chipId);
        }
        
        // Allow sending with Enter key
        document.getElementById('user-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') sendMessage();
        });
    </script>
</body>
</html>''')

    # Start the server
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'

    logger.info(f"Starting ARIA AI Server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
