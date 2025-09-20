"""
WEEKEND FIXTURES PREDICTOR
=========================

Predict multiple matches at once - perfect for weekend fixtures!
Edit the matches list below and run to get all predictions.
"""

from predictor import predict_weekend_fixtures

def predict_weekend():
    """Predict a full set of weekend matches"""
    
    # 🎯 EDIT THESE FIXTURES - ADD YOUR OWN MATCHES:
    weekend_matches = [
        ("Arsenal", "Liverpool"),      # Arsenal at home vs Liverpool away
        ("Chelsea", "Man City"),       # Chelsea at home vs Man City away
        ("Tottenham", "Man United"),   # Tottenham at home vs Man United away
        ("Newcastle", "Brighton"),     # Newcastle at home vs Brighton away
        ("West Ham", "Everton"),       # West Ham at home vs Everton away
    ]
    
    print("🗓️ WEEKEND FIXTURES PREDICTIONS")
    print("=" * 60)
    print(f"Predicting {len(weekend_matches)} matches...")
    print()
    
    # Get all predictions
    results = predict_weekend_fixtures(weekend_matches)
    
    print("📋 RESULTS:")
    print("-" * 60)
    
    for home, away, prediction in results:
        # Determine confidence level
        confidence = prediction['confidence']
        if confidence > 0.6:
            confidence_icon = "🔥 HIGH"
        elif confidence > 0.5:
            confidence_icon = "⚠️  MED "
        else:
            confidence_icon = "❓ LOW "
        
        # Show result
        outcome = prediction['predicted_outcome']
        print(f"{confidence_icon} | {home:12} vs {away:12} → {outcome:8} ({confidence*100:.1f}%)")
    
    print("-" * 60)
    
    # Show best bets (highest confidence)
    print("\n💰 BEST BETS (highest confidence):")
    print("-" * 40)
    
    # Sort by confidence
    sorted_results = sorted(results, key=lambda x: x[2]['confidence'], reverse=True)
    
    for i, (home, away, prediction) in enumerate(sorted_results[:3]):
        confidence = prediction['confidence']
        outcome = prediction['predicted_outcome']
        
        if outcome == 'Home Win':
            odds = 1 / prediction['home_win_prob']
            bet_team = home
        elif outcome == 'Away Win':
            odds = 1 / prediction['away_win_prob']
            bet_team = away
        else:
            odds = 1 / prediction['draw_prob']
            bet_team = "Draw"
        
        print(f"{i+1}. {bet_team:12} | Confidence: {confidence*100:.1f}% | Odds: ${odds:.2f}")
    
    print("\n" + "=" * 60)
    print("✅ WEEKEND PREDICTIONS COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    predict_weekend()
    
    print("\n🎲 Want different matches?")
    print("Edit the 'weekend_matches' list in this script!")
    
    print("\n💡 How to add matches:")
    print('("Home Team", "Away Team"),  # ← Add lines like this')
    print('("Liverpool", "Chelsea"),    # ← Liverpool at home')
    print('("Man City", "Arsenal"),     # ← Man City at home')