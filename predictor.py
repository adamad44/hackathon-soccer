"""
SOCCER MATCH PREDICTOR - PREDICTION INTERFACE
============================================

This file provides an easy-to-use interface for making soccer match predictions
using our trained machine learning models. It loads a trained model and provides
simple functions to predict match outcomes.

Educational Focus: This demonstrates how to create user-friendly interfaces
for machine learning models that hide the complexity from end users.
"""

# Import required libraries
import pickle  # For loading saved machine learning models
import numpy as np  # For numerical operations
from features import create_match_features  # Our feature engineering functions
from database import get_connection  # Database connection for team validation


class SoccerPredictor:
    """
    A class that makes soccer match predictions easy to use.
    
    Educational Note: This is called a "wrapper class" - it wraps around
    our complex ML model and makes it simple to use with just team names.
    """
    
    def __init__(self, model_path):
        """
        Initialize the predictor by loading a trained model.
        
        Args:
            model_path (str): Path to the saved model file (.pkl)
        """
        # Try to load the trained model from the file
        try:
            # pickle.load() reads a saved Python object from a file
            # This is how we save and load trained ML models
            with open(model_path, 'rb') as f:  # 'rb' means read binary
                model_data = pickle.load(f)
            
            # Extract the actual trained model from the saved data
            # Our model.py saves a dictionary with the model inside
            if isinstance(model_data, dict) and 'model' in model_data:
                self.predictor = model_data['model']  # Get the actual ML model
                self.scaler = model_data.get('scaler')  # Get the feature scaler if available
            else:
                self.predictor = model_data  # Assume it's the model directly
                self.scaler = None
            
            print(f"Successfully loaded model: {model_path}")
            
            # Get list of all teams from database for validation
            self.teams = self._get_all_teams()
            
        except FileNotFoundError:
            # If model file doesn't exist, we can't make predictions
            print(f"Model file not found: {model_path}")
            print("Please train a model first by running model.py")
            self.predictor = None
            self.teams = []
    
    def _get_all_teams(self):
        """
        Get list of all unique team names from database.
        This helps us validate that team names are spelled correctly.
        """
        # Connect to our database (specify the soccer_stats database)
        conn = get_connection('soccer_stats')
        cursor = conn.cursor()
        
        # Get all unique home and away team names
        # UNION combines the two lists and removes duplicates
        cursor.execute("""
            SELECT DISTINCT home_team FROM epl_matches 
            UNION 
            SELECT DISTINCT away_team FROM epl_matches
            ORDER BY home_team
        """)
        
        # Extract just the team names from the query results
        teams = [row[0] for row in cursor.fetchall()]
        
        # Clean up database connection
        cursor.close()
        conn.close()
        
        return teams
    
    def _load_matches_data(self):
        """
        Load all matches data from database for feature calculation.
        This is needed by the feature engineering functions.
        """
        # Connect to database
        conn = get_connection('soccer_stats')
        cursor = conn.cursor()
        
        # Load all matches (same query as in main.py)
        cursor.execute("SELECT * FROM epl_matches ORDER BY match_date")
        matches = cursor.fetchall()
        
        # Clean up database connection
        cursor.close()
        conn.close()
        
        return matches
    
    def predict(self, home_team, away_team):
        """
        Predict the outcome of a single match.
        
        Args:
            home_team (str): Name of the home team
            away_team (str): Name of the away team
            
        Returns:
            dict: Prediction results with probabilities and outcome
        """
        # Check if we have a loaded model
        if self.predictor is None:
            print("No model loaded! Train a model first.")
            return None
        
        # Validate team names exist in our database
        if home_team not in self.teams:
            print(f"Team '{home_team}' not found in database.")
            print("Available teams:", self.teams[:10], "...")
            return None
            
        if away_team not in self.teams:
            print(f"Team '{away_team}' not found in database.")
            print("Available teams:", self.teams[:10], "...")
            return None
        
        # Create features for this match using our feature engineering
        # This converts team names into the numerical features our model needs
        # First we need to load all matches data for feature calculation
        matches_data = self._load_matches_data()
        features, feature_names = create_match_features(home_team, away_team, matches_data)  # Unpack the tuple
        
        if features is None:
            print("Could not create features for this match.")
            return None
        
        # Make prediction using our trained model
        # predict_proba() gives us probabilities for each outcome
        probabilities = self.predictor.predict_proba([features])[0]
        
        # predict() gives us the most likely outcome (0=Away, 1=Draw, 2=Home)
        prediction = self.predictor.predict([features])[0]
        
        # Convert numerical prediction to readable text
        outcomes = ['Away Win', 'Draw', 'Home Win']
        predicted_outcome = outcomes[prediction]
        
        # Get individual probabilities for each outcome
        away_win_prob = probabilities[0]  # Probability of away team winning
        draw_prob = probabilities[1]      # Probability of draw
        home_win_prob = probabilities[2]  # Probability of home team winning
        
        # Confidence is the probability of the predicted outcome
        confidence = probabilities[prediction]
        
        # Return all the prediction information
        return {
            'home_team': home_team,
            'away_team': away_team,
            'predicted_outcome': predicted_outcome,
            'confidence': confidence,
            'home_win_prob': home_win_prob,
            'draw_prob': draw_prob,
            'away_win_prob': away_win_prob,
            'probabilities': probabilities
        }
    
    def predict_multiple(self, fixtures):
        """
        Predict outcomes for multiple matches.
        
        Args:
            fixtures (list): List of (home_team, away_team) tuples
            
        Returns:
            list: List of (home_team, away_team, prediction_dict) tuples
        """
        results = []
        
        # Make prediction for each fixture
        for home_team, away_team in fixtures:
            result = self.predict(home_team, away_team)
            if result:
                results.append((home_team, away_team, result))
        
        return results
    
    def get_betting_odds(self, home_team, away_team):
        """
        Convert probabilities to betting odds format.
        
        Educational Note: Betting odds show how much you'd win if you bet $1.
        Odds = 1 / probability. Lower odds = more likely outcome.
        """
        result = self.predict(home_team, away_team)
        
        if result:
            print(f"\nBetting Odds for {home_team} vs {away_team}:")
            print(f"Home Win: {1/result['home_win_prob']:.2f}/1")
            print(f"Draw: {1/result['draw_prob']:.2f}/1") 
            print(f"Away Win: {1/result['away_win_prob']:.2f}/1")


# Convenience functions for quick predictions
def predict_match(home_team, away_team, model_path='model_random_forest.pkl'):
    """
    Quick function to predict a single match.
    
    This is a simple interface that doesn't require creating a class instance.
    """
    predictor = SoccerPredictor(model_path)
    return predictor.predict(home_team, away_team)


def predict_weekend_fixtures(fixtures, model_path='model_random_forest.pkl'):
    """
    Quick function to predict multiple matches.
    
    Args:
        fixtures (list): List of (home_team, away_team) tuples
        
    Example:
        fixtures = [('Arsenal', 'Chelsea'), ('Liverpool', 'Man City')]
        results = predict_weekend_fixtures(fixtures)
    """
    predictor = SoccerPredictor(model_path)
    return predictor.predict_multiple(fixtures)


def interactive_predictor():
    """
    Interactive command-line interface for making predictions.
    
    This lets users type team names and get predictions in real-time.
    """
    predictor = SoccerPredictor('model_random_forest.pkl')
    
    if predictor.predictor is None:
        return
    
    print("\nInteractive Soccer Predictor")
    print("Type 'quit' to exit")
    print(f"Available teams: {len(predictor.teams)} total")
    
    while True:
        print("\n" + "="*50)
        
        # Get home team
        home_team = input("Enter home team: ").strip()
        if home_team.lower() == 'quit':
            break
            
        # Get away team  
        away_team = input("Enter away team: ").strip()
        if away_team.lower() == 'quit':
            break
        
        # Make prediction
        result = predictor.predict(home_team, away_team)
        
        if result:
            print(f"\nPrediction: {result['predicted_outcome']}")
            print(f"Confidence: {result['confidence']*100:.1f}%")
            print(f"Probabilities:")
            print(f"  Home Win: {result['home_win_prob']*100:.1f}%")
            print(f"  Draw: {result['draw_prob']*100:.1f}%")
            print(f"  Away Win: {result['away_win_prob']*100:.1f}%")


# Demo and testing section
if __name__ == "__main__":
    print("SOCCER PREDICTION INTERFACE - FULLY IMPLEMENTED")
    print("=" * 60)
    
    # Load the trained model and make predictions
    print("Loading trained Random Forest model...")
    
    # Create predictor instance
    predictor = SoccerPredictor('model_random_forest.pkl')
    
    if predictor.predictor is None:
        print("No trained model found!")
        print("Please run 'python model.py' first to train a model.")
        exit()
    
    print("\n" + "="*60)
    print("MAKING EXAMPLE PREDICTIONS")
    print("="*60)
    
    # Example 1: Single predictions
    print("\nINDIVIDUAL MATCH PREDICTIONS:")
    print("-" * 40)
    
    example_matches = [
        ("Arsenal", "Chelsea"),
        ("Liverpool", "Man City"), 
        ("Tottenham", "Man United"),
        ("Brighton", "West Ham")
    ]
    
    for home, away in example_matches:
        print(f"\n{home} vs {away}")
        print("-" * 30)
        
        # Make prediction
        result = predictor.predict(home, away)
        
        if result:
            # Show prediction with probabilities
            print(f"Prediction: {result['predicted_outcome']}")
            print(f"Confidence: {result['confidence']*100:.1f}%")
            print(f"Home Win: {result['home_win_prob']*100:.1f}%")
            print(f"Away Win: {result['away_win_prob']*100:.1f}%") 
            print(f"Draw: {result['draw_prob']*100:.1f}%")
            
            # Show betting odds
            predictor.get_betting_odds(home, away)
    
    # Example 2: Weekend fixtures
    print("\n" + "="*60)
    print("WEEKEND FIXTURES PREDICTIONS")
    print("="*60)
    
    weekend_fixtures = [
        ("Arsenal", "Liverpool"),
        ("Chelsea", "Man City"),
        ("Tottenham", "Newcastle"),
        ("Man United", "Brighton")
    ]
    
    print("Predicting weekend matches:")
    results = predictor.predict_multiple(weekend_fixtures)
    
    print("\nWEEKEND PREDICTIONS SUMMARY:")
    print("-" * 40)
    for home, away, result in results:
        confidence_indicator = "HIGH" if result['confidence'] > 0.6 else "MED" if result['confidence'] > 0.5 else "LOW"
        print(f"{confidence_indicator}: {home} vs {away}: {result['predicted_outcome']} ({result['confidence']*100:.1f}%)")
    
    # Example 3: Show available teams
    print(f"\n" + "="*60)
    print(f"AVAILABLE TEAMS ({len(predictor.teams)} total)")
    print("="*60)
    
    # Show teams in columns
    teams_per_row = 4
    for i, team in enumerate(predictor.teams):
        if i % teams_per_row == 0:
            print()
        print(f"{team:18s}", end="")
    
    print(f"\n\n" + "="*60)
    print("PREDICTION INTERFACE COMPLETE!")
    print("="*60)
    print("Ready to make predictions!")
    print("\nHow to use:")
    print("1. Import: from predictor import predict_match")
    print("2. Predict: result = predict_match('Arsenal', 'Chelsea')")
    print("3. Interactive: Run interactive_predictor() for CLI mode")
    print("\nModel Stats:")
    print(f"- Algorithm: Random Forest (50% accuracy)")
    print(f"- Features: 10 team statistics")
    print(f"- Training Data: 500+ historical matches")
    print(f"- Teams: {len(predictor.teams)} EPL teams")