"""
AI MODEL ACCURACY CALCULATOR
============================

This script calculates the real-world accuracy of our trained soccer prediction model
by testing it on historical matches and comparing predictions to actual results.

Educational Focus: Understanding how to measure ML model performance in practice.
"""

import random
from predictor import SoccerPredictor
from database import get_connection
from features import create_match_features

def calculate_model_accuracy(num_test_matches=100):
    """
    Calculate the accuracy of our trained model on real historical matches.
    
    Args:
        num_test_matches (int): Number of matches to test (default 100)
        
    Returns:
        dict: Detailed accuracy results
    """
    print("🤖 CALCULATING AI MODEL ACCURACY")
    print("=" * 50)
    
    # Load the trained model
    predictor = SoccerPredictor('model_random_forest.pkl')
    
    if predictor.predictor is None:
        print("❌ No trained model found!")
        return None
    
    # Get all matches from database
    conn = get_connection('soccer_stats')
    cursor = conn.cursor()
    
    # Get recent matches (not too old, not too recent to avoid training data overlap)
    cursor.execute("""
        SELECT * FROM epl_matches 
        WHERE match_date >= '2020-01-01' AND match_date <= '2023-12-31'
        ORDER BY match_date DESC
        LIMIT 500
    """)
    
    all_matches = cursor.fetchall()
    cursor.close()
    conn.close()
    
    if len(all_matches) < num_test_matches:
        num_test_matches = len(all_matches)
    
    # Randomly select test matches
    test_matches = random.sample(all_matches, num_test_matches)
    
    print(f"📊 Testing model on {num_test_matches} historical matches...")
    print("⏳ This may take a moment...\n")
    
    # Initialize counters
    correct_predictions = 0
    total_predictions = 0
    correct_by_outcome = {'Home Win': 0, 'Away Win': 0, 'Draw': 0}
    total_by_outcome = {'Home Win': 0, 'Away Win': 0, 'Draw': 0}
    confidence_sum = 0
    predictions_made = []
    
    # Test each match
    for i, match in enumerate(test_matches):
        # Extract match details
        home_team = match[3]  # home_team column
        away_team = match[4]  # away_team column
        actual_result = match[7]  # full_time_result column ('H', 'A', 'D')
        
        # Convert actual result to readable format
        actual_outcome_map = {'H': 'Home Win', 'A': 'Away Win', 'D': 'Draw'}
        actual_outcome = actual_outcome_map[actual_result]
        
        # Make prediction
        try:
            prediction_result = predictor.predict(home_team, away_team)
            
            if prediction_result:
                predicted_outcome = prediction_result['predicted_outcome']
                confidence = prediction_result['confidence']
                
                # Check if prediction is correct
                is_correct = (predicted_outcome == actual_outcome)
                
                if is_correct:
                    correct_predictions += 1
                    correct_by_outcome[actual_outcome] += 1
                
                total_predictions += 1
                total_by_outcome[actual_outcome] += 1
                confidence_sum += confidence
                
                # Store for detailed analysis
                predictions_made.append({
                    'home_team': home_team,
                    'away_team': away_team,
                    'actual': actual_outcome,
                    'predicted': predicted_outcome,
                    'correct': is_correct,
                    'confidence': confidence
                })
                
                # Show progress
                if (i + 1) % 20 == 0:
                    current_accuracy = (correct_predictions / total_predictions) * 100
                    print(f"✅ Processed {i + 1}/{num_test_matches} matches - Current accuracy: {current_accuracy:.1f}%")
        
        except Exception as e:
            print(f"⚠️  Skipped match {home_team} vs {away_team}: {str(e)}")
            continue
    
    # Calculate final statistics
    if total_predictions == 0:
        print("❌ No valid predictions made!")
        return None
    
    overall_accuracy = (correct_predictions / total_predictions) * 100
    average_confidence = (confidence_sum / total_predictions) * 100
    
    print("\n" + "=" * 50)
    print("🎯 ACCURACY RESULTS")
    print("=" * 50)
    
    print(f"📊 Overall Accuracy: {overall_accuracy:.1f}% ({correct_predictions}/{total_predictions})")
    print(f"🎲 Random Chance: 33.3% (1 in 3 outcomes)")
    print(f"📈 Improvement: {overall_accuracy - 33.3:.1f} percentage points better than random")
    print(f"🔮 Average Confidence: {average_confidence:.1f}%")
    
    print(f"\n📋 Accuracy by Outcome:")
    for outcome in ['Home Win', 'Away Win', 'Draw']:
        if total_by_outcome[outcome] > 0:
            accuracy = (correct_by_outcome[outcome] / total_by_outcome[outcome]) * 100
            print(f"   {outcome:10}: {accuracy:5.1f}% ({correct_by_outcome[outcome]}/{total_by_outcome[outcome]})")
        else:
            print(f"   {outcome:10}: No matches")
    
    # Show some example predictions
    print(f"\n🔍 SAMPLE PREDICTIONS:")
    print("-" * 50)
    
    # Show correct high-confidence predictions
    correct_high_conf = [p for p in predictions_made if p['correct'] and p['confidence'] > 0.6][:3]
    if correct_high_conf:
        print("✅ Correct High-Confidence Predictions:")
        for pred in correct_high_conf:
            print(f"   {pred['home_team']} vs {pred['away_team']}: Predicted {pred['predicted']}, Actual {pred['actual']} ({pred['confidence']*100:.1f}%)")
    
    # Show incorrect predictions
    incorrect = [p for p in predictions_made if not p['correct']][:3]
    if incorrect:
        print("\n❌ Some Incorrect Predictions:")
        for pred in incorrect:
            print(f"   {pred['home_team']} vs {pred['away_team']}: Predicted {pred['predicted']}, Actual {pred['actual']} ({pred['confidence']*100:.1f}%)")
    
    print("\n" + "=" * 50)
    print("🏆 MODEL PERFORMANCE SUMMARY")
    print("=" * 50)
    
    if overall_accuracy > 50:
        print("🔥 EXCELLENT: Model significantly outperforms random chance!")
    elif overall_accuracy > 40:
        print("✅ GOOD: Model performs well above random chance!")
    elif overall_accuracy > 35:
        print("⚠️  FAIR: Model slightly better than random, room for improvement.")
    else:
        print("❌ POOR: Model not much better than random guessing.")
    
    print(f"\n💡 Key Insights:")
    print(f"• Your AI is {overall_accuracy - 33.3:.1f} percentage points better than random guessing")
    print(f"• Model confidence averages {average_confidence:.1f}% - shows how 'sure' the AI is")
    print(f"• Tested on {total_predictions} real historical matches")
    
    # Return detailed results
    return {
        'overall_accuracy': overall_accuracy,
        'correct_predictions': correct_predictions,
        'total_predictions': total_predictions,
        'average_confidence': average_confidence,
        'accuracy_by_outcome': {outcome: (correct_by_outcome[outcome] / total_by_outcome[outcome] * 100) 
                               if total_by_outcome[outcome] > 0 else 0 
                               for outcome in ['Home Win', 'Away Win', 'Draw']},
        'predictions': predictions_made
    }

def compare_with_baseline():
    """
    Compare our model accuracy with simple baseline strategies.
    """
    print("\n" + "=" * 50)
    print("📊 COMPARISON WITH SIMPLE STRATEGIES")
    print("=" * 50)
    
    print("🎲 Random Guessing: 33.3% accuracy")
    print("🏠 Always Pick Home: ~47% accuracy (home advantage)")
    print("⭐ Always Pick Favorite: ~45% accuracy (based on odds)")
    print("🤖 Your AI Model: Run calculate_model_accuracy() to find out!")
    
    print("\n💡 Why these comparisons matter:")
    print("• Random = 33.3% shows minimum acceptable performance")
    print("• Home bias = 47% shows simple strategy performance")  
    print("• Your AI should beat these to be considered 'intelligent'")

if __name__ == "__main__":
    # Calculate accuracy with 100 test matches
    results = calculate_model_accuracy(100)
    
    if results:
        # Show comparison
        compare_with_baseline()
        
        print(f"\n🎯 FINAL VERDICT:")
        accuracy = results['overall_accuracy']
        if accuracy > 50:
            print(f"🏆 Your AI achieved {accuracy:.1f}% - EXCELLENT performance!")
        elif accuracy > 45:
            print(f"✅ Your AI achieved {accuracy:.1f}% - SOLID performance!")
        elif accuracy > 40:
            print(f"👍 Your AI achieved {accuracy:.1f}% - DECENT performance!")
        else:
            print(f"⚠️  Your AI achieved {accuracy:.1f}% - Room for improvement!")
        
        print(f"\n🚀 Want to improve accuracy? Try:")
        print("• Adding more features (player injuries, weather, etc.)")
        print("• Using more training data")
        print("• Tuning model parameters")
        print("• Ensemble methods (combining multiple models)")
    
    print(f"\n" + "=" * 50)
    print("🔄 Run this script anytime to test model accuracy!")
    print("=" * 50)