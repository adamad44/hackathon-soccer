"""
SOKR AI - WEB API BACKEND
========================

Flask backend that integrates the AI models with the web frontend.
Provides REST API endpoints for soccer match prediction using both 
neural networks and traditional ML models.

Endpoints:
- GET /api/teams - Get list of available teams
- POST /api/predict - Get match prediction
- GET /api/model-info - Get model information
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import sys
import os
import traceback

# Add core directory to path to import our AI modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

try:
    from neural_predictor import NeuralSoccerPredictor
    from predictor import SoccerPredictor
    from database import get_connection
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("💡 Make sure you're running from the project root directory")

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Global predictors - initialized on first use
neural_predictor = None
traditional_predictor = None

def get_neural_predictor():
    """Initialize neural predictor on first use."""
    global neural_predictor
    if neural_predictor is None:
        try:
            # Change to models directory for loading
            os.chdir('models')
            neural_predictor = NeuralSoccerPredictor('neural_model_deep_mlp.h5')
            os.chdir('..')  # Return to root
        except Exception as e:
            print(f"❌ Error initializing neural predictor: {e}")
            os.chdir('..')  # Make sure we return to root
            return None
    return neural_predictor

def get_traditional_predictor():
    """Initialize traditional ML predictor on first use."""
    global traditional_predictor
    if traditional_predictor is None:
        try:
            # Change to models directory for loading
            os.chdir('models')
            traditional_predictor = SoccerPredictor('model_random_forest.pkl')
            os.chdir('..')  # Return to root
        except Exception as e:
            print(f"❌ Error initializing traditional predictor: {e}")
            os.chdir('..')  # Make sure we return to root
            return None
    return traditional_predictor

@app.route('/api/teams', methods=['GET'])
def get_teams():
    """Get list of available teams from database."""
    try:
        conn = get_connection('soccer_stats')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT home_team FROM epl_matches 
            UNION 
            SELECT DISTINCT away_team FROM epl_matches
            ORDER BY home_team
        """)
        
        teams = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'teams': teams,
            'count': len(teams)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to load teams from database'
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict_match():
    """Predict match outcome using AI models."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        home_team = data.get('homeTeam')
        away_team = data.get('awayTeam')
        model_type = data.get('modelType', 'neural')  # Default to neural network
        
        if not home_team or not away_team:
            return jsonify({
                'success': False,
                'error': 'Both homeTeam and awayTeam are required'
            }), 400
        
        if home_team == away_team:
            return jsonify({
                'success': False,
                'error': 'Home team and away team cannot be the same'
            }), 400
        
        # Get the appropriate predictor
        if model_type == 'neural':
            # Temporarily disabled due to feature mismatch - use traditional as fallback
            print("🔄 Neural network temporarily disabled, using traditional ML")
            model_type = 'traditional'
        
        if model_type == 'traditional':
            predictor = get_traditional_predictor()
            if not predictor or not predictor.predictor:
                return jsonify({
                    'success': False,
                    'error': 'Traditional ML model not available'
                }), 503
        
        # Make prediction
        if model_type == 'neural':
            result = predictor.predict_match(home_team, away_team, verbose=False)
        else:
            result = predictor.predict(home_team, away_team)
        
        if not result:
            return jsonify({
                'success': False,
                'error': 'Prediction failed',
                'message': 'Check if team names are valid'
            }), 400
        
        # Format response based on model type
        if model_type == 'neural':
            # Neural network returns raw probabilities
            probabilities = result.get('probabilities', [0.33, 0.33, 0.34])
            home_win = round(probabilities[0] * 100, 1)
            draw = round(probabilities[1] * 100, 1)
            away_win = round(probabilities[2] * 100, 1)
            
            # Ensure percentages add up to 100
            total = home_win + draw + away_win
            if total != 100.0:
                away_win = round(100.0 - home_win - draw, 1)
            
            confidence = result.get('confidence', 'Medium')
            model_info = f"Neural Network (195K parameters)"
            
        else:
            # Traditional ML returns actual probabilities
            home_win = round(result.get('home_win_prob', 0.33) * 100, 1)
            draw = round(result.get('draw_prob', 0.33) * 100, 1)
            away_win = round(result.get('away_win_prob', 0.34) * 100, 1)
            
            # Ensure percentages add up to 100
            total = home_win + draw + away_win
            if total != 100.0:
                away_win = round(100.0 - home_win - draw, 1)
            
            confidence_score = result.get('confidence', 0.5)
            confidence = 'High' if confidence_score > 0.7 else 'Medium' if confidence_score > 0.5 else 'Low'
            model_info = "Random Forest (Traditional ML)"
        
        return jsonify({
            'success': True,
            'prediction': {
                'homeWin': home_win,
                'draw': draw,
                'awayWin': away_win
            },
            'modelInfo': {
                'type': model_type,
                'name': model_info,
                'confidence': confidence
            },
            'matchup': {
                'homeTeam': home_team,
                'awayTeam': away_team
            }
        })
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Internal server error during prediction'
        }), 500

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get information about available models."""
    try:
        neural_available = False
        traditional_available = False
        
        # Check neural network availability
        try:
            predictor = get_neural_predictor()
            neural_available = predictor is not None and predictor.model is not None
        except:
            pass
        
        # Check traditional ML availability
        try:
            predictor = get_traditional_predictor()
            traditional_available = predictor is not None and predictor.model is not None
        except:
            pass
        
        return jsonify({
            'success': True,
            'models': {
                'neural': {
                    'available': neural_available,
                    'name': 'Deep Learning Neural Network',
                    'description': '195K parameter neural network with 120+ features',
                    'accuracy': '50% (tested on real matches)'
                },
                'traditional': {
                    'available': traditional_available,
                    'name': 'Random Forest',
                    'description': 'Traditional ML with 32 statistical features',
                    'accuracy': '50% (tested on real matches)'
                }
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        'success': True,
        'message': 'Sokr AI API is running',
        'version': '2.0'
    })

@app.route('/frontend/<path:filename>')
def serve_frontend(filename):
    """Serve frontend files."""
    return send_from_directory('frontend', filename)

@app.route('/')
def index():
    """Redirect to frontend."""
    return """
    <h1>🚀 Sokr AI Backend is Running!</h1>
    <p>The API is ready to serve predictions.</p>
    <p><a href="/frontend/index.html">Go to Frontend</a> (Open this to use the web interface)</p>
    <p><a href="/api/health">Health Check</a></p>
    <p><a href="/api/teams">View Teams</a></p>
    """

if __name__ == '__main__':
    print("🚀 Starting Sokr AI Backend...")
    print("📊 Loading AI models...")
    print("🌐 Backend will be available at: http://localhost:5000")
    print("🎯 Frontend will connect to API endpoints")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)