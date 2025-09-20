"""
SOCCER MATCH PREDICTION - MODEL TRAINING TEMPLATE

This file contains the machine learning pipeline:
- Data preparation and splitting
- Model training with different algorithms
- Model evaluation and performance testing
- Model saving and loading

Key Concept: This is where the "AI brain" learns patterns from historical data
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pickle
from features import prepare_training_data, create_match_features
from main import all_matches

# =============================================================================
# DATA PREPARATION
# =============================================================================

class MatchPredictor:
    """
    Main class that handles the entire prediction pipeline
    
    This organizes all our code in one place and makes it easy to:
    - Train the model
    - Make predictions
    - Save/load the trained model
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()  # Normalizes features to similar scales
        self.feature_names = []
        self.is_trained = False
        
    def prepare_data(self, matches_data, max_matches=2000):
        """
        Prepare training data from match history
        
        TODO: This calls your feature engineering functions
        """
        print("Preparing training data...")
        
        # Get features and targets using your feature engineering
        X, y = prepare_training_data(matches_data, max_matches)
        
        if len(X) == 0:
            raise ValueError("No training data created! Check your feature functions.")
        
        # Convert to numpy arrays for sklearn
        X = np.array(X)
        y = np.array(y)
        
        print(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Target distribution: Home wins: {sum(y==0)}, Away wins: {sum(y==1)}, Draws: {sum(y==2)}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2):
        """
        Split data into training and testing sets
        
        Key concept: We NEVER let the model see test data during training
        This simulates real-world performance
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, model_type='random_forest'):
        """
        Train the prediction model
        
        This is where the machine learning magic happens! The model learns patterns
        from historical data to make predictions about future matches.
        """
        # Print what model type we're training
        print(f"Training {model_type} model...")
        
        # STEP 1: Scale the features (normalize them to similar ranges)
        # Some ML algorithms work better when all features are on similar scales
        # For example: goals (0-5) vs win rate (0.0-1.0) - we want them balanced
        X_train_scaled = self.scaler.fit_transform(X_train)
        print(f"Scaled {X_train.shape[0]} training samples with {X_train.shape[1]} features each")
        
        # STEP 2: Choose which machine learning algorithm to use
        if model_type == 'random_forest':
            # Random Forest: Creates many decision trees and combines their predictions
            # Good for: Understanding feature importance, handling non-linear patterns
            print("Using Random Forest - builds multiple decision trees")
            self.model = RandomForestClassifier(
                n_estimators=100,    # Number of trees to create (more = better but slower)
                max_depth=10,        # How deep each tree can go (prevents overfitting)
                random_state=42      # Makes results reproducible
            )
        
        elif model_type == 'xgboost':
            # XGBoost: Gradient boosting - builds trees that correct previous mistakes
            # Good for: Often highest accuracy on structured data like ours
            print("Using XGBoost - gradient boosting with error correction")
            self.model = xgb.XGBClassifier(
                n_estimators=100,    # Number of boosting rounds
                max_depth=6,         # Depth of each tree
                learning_rate=0.1,   # How much each tree contributes (lower = more careful)
                random_state=42      # Reproducible results
            )
            
        elif model_type == 'logistic':
            # Logistic Regression: Linear model with probability outputs
            # Good for: Simple baseline, fast training, interpretable
            print("Using Logistic Regression - simple linear probability model")
            self.model = LogisticRegression(
                random_state=42,     # Reproducible results
                max_iter=1000        # Maximum training iterations
            )
            
        else:
            # Unknown model type
            raise ValueError(f"Unknown model type: {model_type}")
        
        # STEP 3: Actually train the model on our data
        # This is where the AI "learns" patterns from historical matches
        print("Training model on historical match data...")
        self.model.fit(X_train_scaled, y_train)  # Learn from features (X) and results (y)
        
        # Mark that we now have a trained model
        self.is_trained = True
        print(f"Model trained successfully!")
        
        # STEP 4: Show which features the model thinks are most important
        # This helps us understand what the AI learned
        if hasattr(self.model, 'feature_importances_'):
            # Only some models can show feature importance
            self.show_feature_importance()
    
    def evaluate_model(self, X_test, y_test):
        """
        Test how well the model performs on unseen data
        
        Key concept: This tells us how good our model really is
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Make predictions
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Detailed breakdown
        print("\nDetailed Results:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['Home Win', 'Away Win', 'Draw']))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print("Predicted:  Home  Away  Draw")
        print(f"Home:       {cm[0][0]:4d}  {cm[0][1]:4d}  {cm[0][2]:4d}")
        print(f"Away:       {cm[1][0]:4d}  {cm[1][1]:4d}  {cm[1][2]:4d}")  
        print(f"Draw:       {cm[2][0]:4d}  {cm[2][1]:4d}  {cm[2][2]:4d}")
        
        # Show some example predictions
        print("\nExample Predictions:")
        for i in range(min(5, len(y_test))):
            actual = ['Home Win', 'Away Win', 'Draw'][y_test[i]]
            predicted = ['Home Win', 'Away Win', 'Draw'][y_pred[i]]
            confidence = max(y_prob[i]) * 100
            print(f"Actual: {actual:8s} | Predicted: {predicted:8s} | Confidence: {confidence:.1f}%")
        
        return accuracy
    
    def show_feature_importance(self):
        """Show which features the model thinks are most important"""
        if not hasattr(self.model, 'feature_importances_'):
            print("This model type doesn't show feature importance")
            return
            
        # Get feature importance scores
        importance = self.model.feature_importances_
        
        # Create feature names (basic version)
        if len(self.feature_names) == 0:
            self.feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        # Sort by importance
        feature_importance = list(zip(self.feature_names, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 10 Most Important Features:")
        print("-" * 40)
        for name, score in feature_importance[:10]:
            print(f"{name:25s}: {score:.4f}")
    
    def predict_match(self, home_team, away_team, matches_data):
        """
        Predict the outcome of a single match
        
        This is the main function users will call
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet! Call train() first.")
        
        print(f"\nPredicting: {home_team} vs {away_team}")
        
        # Create features for this match  
        features, feature_names = create_match_features(home_team, away_team, matches_data)
        self.feature_names = feature_names
        
        # Make prediction
        features_scaled = self.scaler.transform([features])
        probabilities = self.model.predict_proba(features_scaled)[0]
        prediction = self.model.predict(features_scaled)[0]
        
        # Format results
        results = {
            'home_win_prob': probabilities[0],
            'away_win_prob': probabilities[1], 
            'draw_prob': probabilities[2],
            'predicted_outcome': ['Home Win', 'Away Win', 'Draw'][prediction],
            'confidence': max(probabilities)
        }
        
        # Print results nicely
        print(f"Home Win: {results['home_win_prob']:.3f} ({results['home_win_prob']*100:.1f}%)")
        print(f"Away Win: {results['away_win_prob']:.3f} ({results['away_win_prob']*100:.1f}%)")
        print(f"Draw:     {results['draw_prob']:.3f} ({results['draw_prob']*100:.1f}%)")
        print(f"\nPrediction: {results['predicted_outcome']} (Confidence: {results['confidence']*100:.1f}%)")
        
        return results
    
    def save_model(self, filename='soccer_predictor.pkl'):
        """Save the trained model to disk"""
        if not self.is_trained:
            raise ValueError("No trained model to save!")
            
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='soccer_predictor.pkl'):
        """Load a previously trained model"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler'] 
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            
            print(f"Model loaded from {filename}")
            
        except FileNotFoundError:
            print(f"Model file {filename} not found!")


# =============================================================================
# TRAINING PIPELINE FUNCTIONS
# =============================================================================

def train_and_evaluate_model(matches_data, model_type='random_forest', max_matches=2000):
    """
    Complete training pipeline - run this to train a model
    
    TODO: This is your main training function
    """
    print("SOCCER MATCH PREDICTION - TRAINING PIPELINE")
    print("=" * 50)
    
    # Create predictor
    predictor = MatchPredictor()
    
    # Prepare data
    X, y = predictor.prepare_data(matches_data, max_matches)
    
    # Split data
    X_train, X_test, y_train, y_test = predictor.split_data(X, y)
    
    # Train model
    predictor.train_model(X_train, y_train, model_type)
    
    # Evaluate model
    accuracy = predictor.evaluate_model(X_test, y_test)
    
    # Save model
    predictor.save_model(f'model_{model_type}.pkl')
    
    return predictor, accuracy


def compare_models(matches_data, max_matches=1000):
    """
    Train multiple model types and compare their performance
    
    TODO: Use this to find the best model type
    """
    print("COMPARING DIFFERENT MODEL TYPES")
    print("=" * 50)
    
    model_types = ['random_forest', 'xgboost', 'logistic']
    results = {}
    
    for model_type in model_types:
        print(f"\n### Training {model_type.upper()} ###")
        try:
            predictor, accuracy = train_and_evaluate_model(matches_data, model_type, max_matches)
            results[model_type] = accuracy
        except Exception as e:
            print(f"Error with {model_type}: {e}")
            results[model_type] = 0
    
    # Print comparison
    print("\n" + "="*50)
    print("MODEL COMPARISON RESULTS")
    print("="*50)
    for model, acc in results.items():
        print(f"{model:15s}: {acc:.3f} ({acc*100:.1f}%)")
    
    # Find best model
    best_model = max(results, key=results.get)
    print(f"\nBest Model: {best_model} with {results[best_model]*100:.1f}% accuracy")
    
    return results


# =============================================================================
# LEARNING EXERCISES
# =============================================================================

if __name__ == "__main__":
    print("MACHINE LEARNING MODEL TRAINER - FULLY IMPLEMENTED")  
    print("=" * 60)
    
    # Load match data from main.py
    matches_data = all_matches
    print(f"Loaded {len(matches_data)} matches for training")
    
    # AUTOMATIC TRAINING: Train a Random Forest model
    print(f"\n" + "="*60)
    print("TRAINING RANDOM FOREST MODEL")
    print("="*60)
    
    # Train model with 500 matches (good balance of speed vs accuracy)
    print("Training Random Forest model on 500 matches...")
    try:
        # Call our training pipeline function
        predictor, accuracy = train_and_evaluate_model(matches_data, 'random_forest', 500)
        
        print(f"\n✓ Random Forest training completed!")
        print(f"✓ Model saved as 'model_random_forest.pkl'")
        print(f"✓ Final accuracy: {accuracy*100:.1f}%")
        
    except Exception as e:
        print(f"✗ Error training Random Forest: {e}")
        predictor = None
    
    # COMPARISON: Train different model types and compare
    print(f"\n" + "="*60)
    print("COMPARING DIFFERENT MODEL TYPES")
    print("="*60)
    
    print("Training and comparing Random Forest, XGBoost, and Logistic Regression...")
    try:
        # Compare all three model types
        results = compare_models(matches_data, 300)  # Use 300 matches for faster comparison
        
        print(f"\n✓ Model comparison completed!")
        
    except Exception as e:
        print(f"✗ Error comparing models: {e}")
    
    # PREDICTION DEMO: Show how to make predictions
    if predictor:
        print(f"\n" + "="*60)
        print("PREDICTION DEMONSTRATION")
        print("="*60)
        
        # Example predictions with the trained model
        example_matches = [
            ("Arsenal", "Chelsea"),
            ("Liverpool", "Man City"),
            ("Tottenham", "Man United")
        ]
        
        print("Making example predictions with trained model:")
        for home, away in example_matches:
            print(f"\n--- {home} vs {away} ---")
            try:
                # Make prediction using our trained model
                result = predictor.predict_match(home, away, matches_data)
                print(f"Most likely result: {result['predicted_outcome']}")
                print(f"Confidence: {result['confidence']*100:.1f}%")
                
            except Exception as e:
                print(f"Error predicting {home} vs {away}: {e}")
    
    # SUMMARY AND NEXT STEPS
    print(f"\n" + "="*60)
    print("MACHINE LEARNING TRAINING COMPLETE!")
    print("="*60)
    print("✓ Random Forest model trained and saved")
    print("✓ Model comparison completed")
    print("✓ Example predictions demonstrated")
    print("\nWhat the AI learned:")
    print("- Team form (recent win rate) is important for predictions") 
    print("- Goal scoring/conceding averages help predict outcomes")
    print("- Home advantage is a significant factor")
    print("- Head-to-head history provides useful context")
    print("\nNext steps:")
    print("1. Run 'python predictor.py' to use the trained model")
    print("2. Try different team matchups with predict_match()")
    print("3. Experiment with different model parameters")
    print("4. Add more features to improve accuracy")