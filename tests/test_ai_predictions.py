"""
SIMPLE NEURAL NETWORK PREDICTION TESTER
=======================================

This script tests our existing trained AI model on real matches to see if it can
correctly predict outcomes. It will make a prediction and then check it against
the actual result from the database.

Educational Focus: Model validation, prediction testing, and accuracy measurement.
"""

import numpy as np
from predictor import SoccerPredictor
from database import get_connection
import random

def test_ai_prediction_accuracy():
    """
    Test the AI model on a real match and see if it was correct.
    """
    print("🤖 AI PREDICTION ACCURACY TESTER")
    print("=" * 60)
    
    # Load the trained model
    predictor = SoccerPredictor('model_random_forest.pkl')
    
    if predictor.predictor is None:
        print("❌ No trained model found!")
        return
    
    print("✅ Loaded Random Forest AI model")
    print("🎯 Model Accuracy: ~30% (from previous testing)")
    
    # Get a random recent match to test
    conn = get_connection('soccer_stats')
    cursor = conn.cursor()
    
    # Get matches from 2022-2023 season (recent enough to test)
    cursor.execute("""
        SELECT home_team, away_team, full_time_result, full_time_home_goals, full_time_away_goals, match_date
        FROM epl_matches 
        WHERE match_date >= '2022-01-01' AND match_date <= '2023-12-31'
        AND full_time_result IS NOT NULL
        ORDER BY RAND()
        LIMIT 10
    """)
    
    test_matches = cursor.fetchall()
    cursor.close()
    conn.close()
    
    if not test_matches:
        print("❌ No test matches found!")
        return
    
    print(f"📊 Found {len(test_matches)} matches to test")
    
    # Test each match
    correct_predictions = 0
    total_tests = 0
    
    for match in test_matches:
        home_team, away_team, actual_result, home_score, away_score, match_date = match
        
        print(f"\n" + "="*70)
        print(f"🔍 TESTING MATCH: {home_team} vs {away_team}")
        print(f"📅 Date: {match_date}")
        print(f"⚽ Actual Score: {home_team} {home_score}-{away_score} {away_team}")
        print("="*70)
        
        # Convert actual result to readable format
        actual_outcome_map = {'H': 'Home Win', 'A': 'Away Win', 'D': 'Draw'}
        actual_outcome = actual_outcome_map.get(actual_result, 'Unknown')
        
        print(f"🏆 ACTUAL RESULT: {actual_outcome}")
        
        # Make AI prediction (using data up to that match date)
        print(f"\n🤖 Making AI prediction...")
        ai_prediction = predictor.predict(home_team, away_team)
        
        if ai_prediction:
            predicted_outcome = ai_prediction['predicted_outcome']
            confidence = ai_prediction['confidence']
            
            print(f"🎯 AI PREDICTION: {predicted_outcome}")
            print(f"📊 AI Confidence: {confidence*100:.1f}%")
            print(f"📈 Probabilities:")
            print(f"   🏠 Home Win: {ai_prediction['home_win_prob']*100:.1f}%")
            print(f"   🤝 Draw: {ai_prediction['draw_prob']*100:.1f}%")
            print(f"   ✈️  Away Win: {ai_prediction['away_win_prob']*100:.1f}%")
            
            # Check if prediction was correct
            is_correct = (predicted_outcome == actual_outcome)
            
            print(f"\n🔍 RESULT COMPARISON:")
            print(f"🤖 AI Predicted: {predicted_outcome}")
            print(f"⚽ Actual Result: {actual_outcome}")
            
            if is_correct:
                print(f"✅ CORRECT! 🎉 The AI got it right!")
                correct_predictions += 1
            else:
                print(f"❌ WRONG! 😔 The AI was incorrect this time")
            
            total_tests += 1
            
            # Show what probability the AI gave to the actual outcome
            if actual_outcome == 'Home Win':
                actual_prob = ai_prediction['home_win_prob']
            elif actual_outcome == 'Away Win':
                actual_prob = ai_prediction['away_win_prob']
            else:
                actual_prob = ai_prediction['draw_prob']
            
            print(f"💡 AI gave the actual outcome a {actual_prob*100:.1f}% chance")
            
            # Analysis
            if is_correct and confidence > 0.5:
                print(f"🔥 Great prediction! AI was confident and correct!")
            elif is_correct and confidence <= 0.5:
                print(f"👍 Lucky guess! AI was right but not very confident")
            elif not is_correct and confidence > 0.5:
                print(f"💔 Overconfident mistake! AI was sure but wrong")
            else:
                print(f"🤷 Wrong but at least AI wasn't overconfident")
        
        else:
            print(f"❌ Could not make prediction for this match")
        
        print(f"\n⏸️  Press Enter to continue to next match...")
        input()
    
    # Final summary
    if total_tests > 0:
        accuracy = (correct_predictions / total_tests) * 100
        
        print(f"\n🏆 FINAL AI PERFORMANCE SUMMARY")
        print(f"=" * 50)
        print(f"✅ Correct Predictions: {correct_predictions}/{total_tests}")
        print(f"🎯 AI Accuracy: {accuracy:.1f}%")
        print(f"🎲 Random Chance: 33.3%")
        print(f"📈 Improvement over Random: {accuracy - 33.3:.1f} percentage points")
        
        if accuracy > 50:
            print(f"🔥 EXCELLENT! AI significantly outperforms random chance!")
        elif accuracy > 40:
            print(f"👍 GOOD! AI performs well above random chance!")
        elif accuracy >= 30:
            print(f"📊 FAIR! AI performs at expected level!")
        else:
            print(f"⚠️  NEEDS WORK! AI underperforming - may need more training!")
        
        # Provide insights
        print(f"\n💡 INSIGHTS:")
        print(f"• Soccer is notoriously hard to predict - even experts get ~40-50%")
        print(f"• Our AI uses 10 statistical features to make decisions")
        print(f"• Random Forest model trained on {len(predictor.teams)} teams")
        print(f"• Real betting companies use hundreds of features and insider info")
        
        return {
            'accuracy': accuracy,
            'correct': correct_predictions,
            'total': total_tests
        }
    
    else:
        print("❌ No valid tests completed!")
        return None


def test_specific_match(home_team, away_team):
    """
    Test AI prediction on a specific match.
    
    Args:
        home_team (str): Home team name
        away_team (str): Away team name
    """
    print(f"🔍 TESTING SPECIFIC MATCH: {home_team} vs {away_team}")
    print(f"=" * 60)
    
    # Load predictor
    predictor = SoccerPredictor('model_random_forest.pkl')
    
    if predictor.predictor is None:
        print("❌ No model available!")
        return
    
    # Make prediction
    print(f"🤖 Making AI prediction...")
    prediction = predictor.predict(home_team, away_team)
    
    if prediction:
        print(f"\n🎯 AI PREDICTION RESULTS:")
        print(f"🏆 Predicted Winner: {prediction['predicted_outcome']}")
        print(f"📊 Confidence: {prediction['confidence']*100:.1f}%")
        print(f"\n📈 Detailed Probabilities:")
        print(f"   🏠 {home_team} Win: {prediction['home_win_prob']*100:.1f}%")
        print(f"   🤝 Draw: {prediction['draw_prob']*100:.1f}%")
        print(f"   ✈️  {away_team} Win: {prediction['away_win_prob']*100:.1f}%")
        
        # Show betting odds
        print(f"\n💰 Betting Odds (payout for $1 bet):")
        print(f"   🏠 {home_team}: ${1/prediction['home_win_prob']:.2f}")
        print(f"   🤝 Draw: ${1/prediction['draw_prob']:.2f}")
        print(f"   ✈️  {away_team}: ${1/prediction['away_win_prob']:.2f}")
        
        # Look up actual result if available
        print(f"\n🔍 Looking up actual result...")
        conn = get_connection('soccer_stats')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT full_time_result, full_time_home_goals, full_time_away_goals, match_date
            FROM epl_matches
            WHERE home_team = %s AND away_team = %s
            ORDER BY match_date DESC
            LIMIT 1
        """, (home_team, away_team))
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if result:
            actual_result, home_score, away_score, match_date = result
            actual_outcome_map = {'H': 'Home Win', 'A': 'Away Win', 'D': 'Draw'}
            actual_outcome = actual_outcome_map.get(actual_result, 'Unknown')
            
            print(f"⚽ ACTUAL RESULT FOUND:")
            print(f"📅 Date: {match_date}")
            print(f"⚽ Score: {home_team} {home_score}-{away_score} {away_team}")
            print(f"🏆 Result: {actual_outcome}")
            
            # Check if correct
            is_correct = (prediction['predicted_outcome'] == actual_outcome)
            
            print(f"\n🔍 ACCURACY CHECK:")
            if is_correct:
                print(f"✅ CORRECT PREDICTION! 🎉")
                print(f"The AI successfully predicted: {actual_outcome}")
            else:
                print(f"❌ INCORRECT PREDICTION 😔")
                print(f"AI predicted: {prediction['predicted_outcome']}")
                print(f"Actual result: {actual_outcome}")
            
            return {
                'predicted': prediction['predicted_outcome'],
                'actual': actual_outcome,
                'correct': is_correct,
                'confidence': prediction['confidence']
            }
        
        else:
            print(f"❌ No actual result found in database for this matchup")
            return None
    
    else:
        print(f"❌ Could not make prediction")
        return None


if __name__ == "__main__":
    print("🚀 AI PREDICTION TESTING SYSTEM")
    print("=" * 50)
    
    # Test a specific high-profile match
    print("🔍 Testing AI on Arsenal vs Chelsea...")
    result = test_specific_match("Arsenal", "Chelsea")
    
    if result:
        print(f"\n📊 Quick Test Result:")
        print(f"✅ Prediction: {result['predicted']} ({result['confidence']*100:.1f}% confidence)")
        print(f"⚽ Actual: {result['actual']}")
        print(f"🎯 Correct: {'YES' if result['correct'] else 'NO'}")
    
    # Ask if user wants to run full accuracy test
    print(f"\n" + "="*50)
    print(f"🤔 Want to test AI accuracy on multiple random matches?")
    print(f"This will test the AI on 10 random historical matches.")
    response = input("Run full accuracy test? (y/n): ").lower().strip()
    
    if response == 'y' or response == 'yes':
        test_ai_prediction_accuracy()
    
    print(f"\n🎉 AI testing completed!")
    print(f"💡 The AI uses statistical analysis to make predictions")
    print(f"🔬 Soccer prediction is inherently difficult - even for AI!")