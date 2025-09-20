"""
NEURAL NETWORK PREDICTOR - ADVANCED AI PREDICTION SYSTEM
========================================================

This file provides an advanced neural network prediction interface that can:
- Load trained deep learning models
- Use sophisticated feature engineering
- Make predictions with neural networks
- Test predictions against real match results
- Provide detailed confidence analysis

Educational Focus: Deep learning prediction, model evaluation, and real-world testing.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import os
from datetime import datetime, timedelta
from database import get_connection
from neural_features import create_neural_match_features

class NeuralSoccerPredictor:
    """
    Advanced neural network predictor for soccer matches.
    
    This system uses deep learning models trained with extended feature sets
    to make sophisticated predictions about soccer match outcomes.
    """
    
    def __init__(self, model_path='neural_model_deep_mlp.h5'):
        """
        Initialize the neural network predictor.
        
        Args:
            model_path (str): Path to the trained neural network model
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.team_encoder = None
        self.feature_names = []
        self.teams = []
        
        # Load the neural network components
        self._load_neural_network()
        
        # Get team list for validation
        self._load_teams()
    
    def _load_neural_network(self):
        """Load the trained neural network and associated components."""
        base_name = self.model_path.replace('.h5', '')
        
        try:
            # Load the Keras/TensorFlow model
            print(f"🔄 Loading neural network model: {self.model_path}")
            self.model = keras.models.load_model(self.model_path)
            print(f"✅ Neural network loaded successfully!")
            
            # Load the feature scaler
            scaler_path = f'{base_name}_scaler.pkl'
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"📊 Feature scaler loaded")
            
            # Load the team encoder
            encoder_path = f'{base_name}_team_encoder.pkl'
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    self.team_encoder = pickle.load(f)
                print(f"🏷️  Team encoder loaded")
            
            # Print model architecture summary
            print(f"\n🏗️  Neural Network Architecture:")
            self.model.summary()
            
        except Exception as e:
            print(f"❌ Error loading neural network: {str(e)}")
            print(f"💡 Make sure to train the neural network first by running: python neural_model.py")
            self.model = None
    
    def _load_teams(self):
        """Load list of available teams from database."""
        try:
            conn = get_connection('soccer_stats')
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT DISTINCT home_team FROM epl_matches 
                UNION 
                SELECT DISTINCT away_team FROM epl_matches
                ORDER BY home_team
            """)
            
            self.teams = [row[0] for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            
            print(f"📋 Loaded {len(self.teams)} teams from database")
            
        except Exception as e:
            print(f"❌ Error loading teams: {str(e)}")
            self.teams = []
    
    def predict_match(self, home_team, away_team, verbose=True):
        """
        Predict a single match using the neural network.
        
        Args:
            home_team (str): Name of the home team
            away_team (str): Name of the away team
            verbose (bool): Whether to print detailed information
            
        Returns:
            dict: Prediction results with probabilities and analysis
        """
        if self.model is None:
            print("❌ No neural network model loaded!")
            return None
        
        # Validate team names
        if home_team not in self.teams:
            print(f"❌ Team '{home_team}' not found in database")
            print(f"💡 Available teams: {self.teams[:10]} ...")
            return None
        
        if away_team not in self.teams:
            print(f"❌ Team '{away_team}' not found in database")
            print(f"💡 Available teams: {self.teams[:10]} ...")
            return None
        
        try:
            if verbose:
                print(f"\n🤖 NEURAL NETWORK PREDICTION")
                print(f"=" * 50)
                print(f"🏠 Home Team: {home_team}")
                print(f"✈️  Away Team: {away_team}")
                print(f"🧠 Model: Deep Learning Neural Network")
            
            # Load historical data
            conn = get_connection('soccer_stats')
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM epl_matches ORDER BY match_date")
            matches_data = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if verbose:
                print(f"📊 Using {len(matches_data)} historical matches for analysis")
            
            # Create neural network features
            print(f"🔍 Creating advanced neural features...")
            features, feature_names = create_neural_match_features(home_team, away_team, matches_data)
            
            if features is None:
                print(f"❌ Could not create features for this match")
                return None
            
            if verbose:
                print(f"✅ Created {len(features)} advanced features")
            
            # Scale features if scaler is available
            if self.scaler is not None:
                features_scaled = self.scaler.transform([features])
                if verbose:
                    print(f"📈 Features normalized using trained scaler")
            else:
                features_scaled = [features]
                if verbose:
                    print(f"⚠️  No scaler available - using raw features")
            
            # Make prediction with neural network
            if verbose:
                print(f"🚀 Running neural network prediction...")
            
            predictions = self.model.predict(features_scaled, verbose=0)
            probabilities = predictions[0]  # Get probabilities for this match
            
            # Convert to readable format
            outcomes = ['Away Win', 'Draw', 'Home Win']
            predicted_class = np.argmax(probabilities)
            predicted_outcome = outcomes[predicted_class]
            confidence = probabilities[predicted_class]
            
            # Create detailed result
            result = {
                'home_team': home_team,
                'away_team': away_team,
                'predicted_outcome': predicted_outcome,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'away_win_prob': probabilities[0],
                'draw_prob': probabilities[1], 
                'home_win_prob': probabilities[2],
                'raw_probabilities': probabilities,
                'model_type': 'Neural Network',
                'feature_count': len(features)
            }
            
            if verbose:
                print(f"\n🎯 PREDICTION RESULTS:")
                print(f"=" * 30)
                print(f"🏆 Predicted Outcome: {predicted_outcome}")
                print(f"🎲 Confidence: {confidence*100:.1f}%")
                print(f"\n📊 Detailed Probabilities:")
                print(f"   🏠 Home Win ({home_team}): {probabilities[2]*100:.1f}%")
                print(f"   🤝 Draw: {probabilities[1]*100:.1f}%")
                print(f"   ✈️  Away Win ({away_team}): {probabilities[0]*100:.1f}%")
                
                # Show betting odds
                print(f"\n💰 Betting Odds (if you bet $1):")
                print(f"   🏠 Home Win: ${1/probabilities[2]:.2f}")
                print(f"   🤝 Draw: ${1/probabilities[1]:.2f}")
                print(f"   ✈️  Away Win: ${1/probabilities[0]:.2f}")
                
                # Confidence analysis
                print(f"\n🔍 AI Confidence Analysis:")
                if confidence > 0.6:
                    print(f"   🔥 HIGH CONFIDENCE - AI is very sure about this prediction")
                elif confidence > 0.45:
                    print(f"   ⚡ MEDIUM CONFIDENCE - AI has reasonable certainty")
                else:
                    print(f"   ❓ LOW CONFIDENCE - This is a difficult match to predict")
                
                # Feature insight
                print(f"\n🧠 Neural Network Insights:")
                print(f"   📈 Used {len(features)} advanced features")
                print(f"   🔧 Features include team form, goals, momentum, head-to-head history")
                print(f"   🎯 Neural network processed through multiple hidden layers")
                print(f"   📊 Model trained on thousands of historical matches")
            
            return result
            
        except Exception as e:
            print(f"❌ Error making neural network prediction: {str(e)}")
            return None
    
    def test_prediction_accuracy(self, home_team, away_team, actual_result=None):
        """
        Test a prediction against a known result.
        
        Args:
            home_team (str): Home team name
            away_team (str): Away team name
            actual_result (str): Actual result ('H', 'A', 'D') or ('Home Win', 'Away Win', 'Draw')
        """
        print(f"\n🧪 TESTING NEURAL NETWORK PREDICTION")
        print(f"=" * 60)
        
        # Make prediction
        prediction = self.predict_match(home_team, away_team, verbose=True)
        
        if prediction is None:
            return
        
        # Get actual result if not provided
        if actual_result is None:
            print(f"\n🔍 Looking up actual result from database...")
            actual_result = self._get_actual_result(home_team, away_team)
        
        if actual_result:
            # Normalize actual result format
            if actual_result in ['H', 'Home Win']:
                actual_outcome = 'Home Win'
                actual_code = 'H'
            elif actual_result in ['A', 'Away Win']:
                actual_outcome = 'Away Win'
                actual_code = 'A'
            elif actual_result in ['D', 'Draw']:
                actual_outcome = 'Draw' 
                actual_code = 'D'
            else:
                print(f"❌ Unknown result format: {actual_result}")
                return
            
            print(f"\n⚽ ACTUAL RESULT:")
            print(f"=" * 20)
            print(f"🏆 Actual Outcome: {actual_outcome}")
            
            # Compare prediction vs reality
            print(f"\n🔍 PREDICTION vs REALITY:")
            print(f"=" * 30)
            print(f"🤖 AI Predicted: {prediction['predicted_outcome']}")
            print(f"⚽ Actual Result: {actual_outcome}")
            
            # Check if correct
            is_correct = (prediction['predicted_outcome'] == actual_outcome)
            
            if is_correct:
                print(f"✅ CORRECT PREDICTION! 🎉")
                print(f"🎯 The neural network got it right!")
                print(f"📊 Confidence was: {prediction['confidence']*100:.1f}%")
            else:
                print(f"❌ INCORRECT PREDICTION")
                print(f"😔 The neural network was wrong this time")
                print(f"📊 AI was {prediction['confidence']*100:.1f}% confident in wrong answer")
            
            # Show detailed analysis
            print(f"\n📈 DETAILED ANALYSIS:")
            print(f"=" * 25)
            
            if actual_outcome == 'Home Win':
                actual_prob = prediction['home_win_prob']
            elif actual_outcome == 'Away Win':
                actual_prob = prediction['away_win_prob']
            else:
                actual_prob = prediction['draw_prob']
            
            print(f"🎲 AI gave the actual outcome a {actual_prob*100:.1f}% chance")
            
            if actual_prob > 0.4:
                print(f"💡 The AI did assign reasonable probability to the correct outcome")
            elif actual_prob > 0.25:
                print(f"🤔 The AI thought this outcome was possible but not most likely")
            else:
                print(f"😅 The AI thought this outcome was very unlikely")
            
            return {
                'predicted': prediction['predicted_outcome'],
                'actual': actual_outcome,
                'correct': is_correct,
                'confidence': prediction['confidence'],
                'actual_outcome_probability': actual_prob
            }
        
        else:
            print(f"❌ Could not find actual result for this match")
            return None
    
    def _get_actual_result(self, home_team, away_team):
        """Get actual result for a match from the database."""
        try:
            conn = get_connection('soccer_stats')
            cursor = conn.cursor()
            
            # Look for recent matches between these teams
            cursor.execute("""
                SELECT full_time_result, match_date, home_goals, away_goals
                FROM epl_matches 
                WHERE home_team = %s AND away_team = %s
                ORDER BY match_date DESC
                LIMIT 5
            """, (home_team, away_team))
            
            matches = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if matches:
                # Show available matches and let user choose or take most recent
                print(f"\n📅 Found {len(matches)} recent matches:")
                for i, match in enumerate(matches):
                    result, date, home_goals, away_goals = match
                    print(f"   {i+1}. {date}: {home_team} {home_goals}-{away_goals} {away_team} ({result})")
                
                # Use most recent match
                most_recent = matches[0]
                return most_recent[0]  # full_time_result
            
            return None
            
        except Exception as e:
            print(f"❌ Error looking up actual result: {str(e)}")
            return None


def test_neural_prediction_system():
    """
    Test the neural network prediction system on real matches.
    """
    print(f"🚀 NEURAL NETWORK PREDICTION TESTING SYSTEM")
    print(f"=" * 70)
    
    # Create neural predictor
    predictor = NeuralSoccerPredictor()
    
    if predictor.model is None:
        print(f"❌ No neural network model available!")
        print(f"💡 Please train the neural network first by running: python neural_model.py")
        return
    
    print(f"\n🎯 TESTING PREDICTIONS ON REAL MATCHES")
    print(f"=" * 50)
    
    # Test cases - famous matches we can verify
    test_matches = [
        ("Arsenal", "Chelsea"),
        ("Liverpool", "Man City"),
        ("Man United", "Tottenham"),
        ("Newcastle", "Brighton"),
        ("Aston Villa", "West Ham")
    ]
    
    correct_predictions = 0
    total_predictions = 0
    
    for home_team, away_team in test_matches:
        print(f"\n" + "="*70)
        print(f"🔍 TESTING: {home_team} vs {away_team}")
        print(f"="*70)
        
        # Test the prediction
        result = predictor.test_prediction_accuracy(home_team, away_team)
        
        if result:
            total_predictions += 1
            if result['correct']:
                correct_predictions += 1
        
        print(f"\n⏸️  Press Enter to continue to next match...")
        input()
    
    # Final summary
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"\n🏆 FINAL NEURAL NETWORK PERFORMANCE:")
        print(f"=" * 50)
        print(f"✅ Correct Predictions: {correct_predictions}/{total_predictions}")
        print(f"🎯 Neural Network Accuracy: {accuracy:.1f}%")
        print(f"🎲 Random Chance: 33.3%")
        print(f"📈 Improvement over Random: {accuracy - 33.3:.1f} percentage points")
        
        if accuracy > 50:
            print(f"🔥 EXCELLENT! Neural network significantly outperforms random chance!")
        elif accuracy > 40:
            print(f"👍 GOOD! Neural network performs well above random!")
        elif accuracy >= 33.3:
            print(f"📊 FAIR! Neural network performs at or slightly above random chance!")
        else:
            print(f"⚠️  NEEDS WORK! Neural network underperforming - needs more training!")


def quick_neural_prediction(home_team, away_team):
    """
    Quick function to make a neural network prediction.
    
    Args:
        home_team (str): Home team name
        away_team (str): Away team name
    """
    predictor = NeuralSoccerPredictor()
    
    if predictor.model is None:
        print("❌ No neural network model available!")
        print("💡 Train the model first: python neural_model.py")
        return None
    
    return predictor.predict_match(home_team, away_team)


if __name__ == "__main__":
    print("🧠 NEURAL NETWORK SOCCER PREDICTION SYSTEM")
    print("=" * 60)
    
    # Test with a specific match
    print("🔍 Testing neural network on a specific match...")
    
    # Let's test Arsenal vs Chelsea
    predictor = NeuralSoccerPredictor()
    
    if predictor.model is not None:
        # Make a prediction and test it
        result = predictor.test_prediction_accuracy("Arsenal", "Chelsea")
        
        print(f"\n🎉 Neural network test completed!")
        
        # Test a few more quick predictions
        print(f"\n🚀 Additional quick predictions:")
        quick_matches = [("Liverpool", "Man City"), ("Tottenham", "Newcastle")]
        
        for home, away in quick_matches:
            print(f"\n" + "-"*40)
            quick_result = predictor.predict_match(home, away, verbose=False)
            if quick_result:
                print(f"🏠 {home} vs ✈️  {away}")
                print(f"🎯 Prediction: {quick_result['predicted_outcome']} ({quick_result['confidence']*100:.1f}%)")
    
    else:
        print("❌ Neural network not ready!")
        print("💡 Please run 'python neural_model.py' first to train the models")