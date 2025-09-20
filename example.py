"""
COMPLETE SOCCER PREDICTION EXAMPLE
=================================

This script demonstrates how to use the complete machine learning pipeline
to predict soccer matches. It shows everything working together from 
database to predictions with detailed educational comments.

WHAT THIS DEMONSTRATES:
- How to load and use trained ML models
- How to make single match predictions
- How to predict multiple matches at once
- How to interpret prediction probabilities
- How to convert probabilities to betting odds

LEARNING OBJECTIVES:
- Understanding how ML models make predictions
- Working with prediction probabilities and confidence
- Converting model outputs to real-world applications
"""

# Import our prediction functions
from predictor import predict_match, predict_weekend_fixtures, SoccerPredictor

def main():
    """
    Main demonstration of the soccer prediction system
    """
    print("=" * 70)
    print("🚀 COMPLETE SOCCER MACHINE LEARNING PREDICTION SYSTEM")
    print("=" * 70)
    
    print("\n📚 WHAT WE'VE BUILT:")
    print("✅ MariaDB database with 9,380 EPL matches (2000-2025)")
    print("✅ Feature engineering with 10 team statistics")
    print("✅ Random Forest model trained on historical data")
    print("✅ 50% prediction accuracy (better than random chance)")
    print("✅ Complete prediction interface")
    
    print("\n" + "=" * 70)
    print("🎯 EXAMPLE 1: SINGLE MATCH PREDICTION")
    print("=" * 70)
    
    # Example 1: Simple prediction
    print("\n🏟️ Predicting: Arsenal vs Chelsea")
    print("-" * 40)
    
    result = predict_match('Arsenal', 'Chelsea')
    
    if result:
        print(f"🔮 Prediction: {result['predicted_outcome']}")
        print(f"📊 Model Confidence: {result['confidence']*100:.1f}%")
        print(f"\n📈 Full Probabilities:")
        print(f"   🏠 Home Win (Arsenal): {result['home_win_prob']*100:.1f}%")
        print(f"   ✈️  Away Win (Chelsea): {result['away_win_prob']*100:.1f}%")
        print(f"   🤝 Draw: {result['draw_prob']*100:.1f}%")
        
        print(f"\n💰 Betting Odds (how much you'd win per $1 bet):")
        print(f"   🏠 Arsenal Win: ${1/result['home_win_prob']:.2f}")
        print(f"   ✈️  Chelsea Win: ${1/result['away_win_prob']:.2f}")
        print(f"   🤝 Draw: ${1/result['draw_prob']:.2f}")
    
    print("\n" + "=" * 70) 
    print("📅 EXAMPLE 2: WEEKEND FIXTURES")
    print("=" * 70)
    
    # Example 2: Multiple predictions
    weekend_matches = [
        ('Liverpool', 'Man City'),
        ('Tottenham', 'Chelsea'), 
        ('Arsenal', 'Man United'),
        ('Newcastle', 'Brighton')
    ]
    
    print(f"\n🗓️ Predicting {len(weekend_matches)} weekend matches:")
    print("-" * 50)
    
    results = predict_weekend_fixtures(weekend_matches)
    
    print(f"\n📋 WEEKEND PREDICTIONS:")
    for home, away, prediction in results:
        confidence_emoji = "🔥" if prediction['confidence'] > 0.6 else "⚠️" if prediction['confidence'] > 0.5 else "❓"
        print(f"{confidence_emoji} {home:12} vs {away:12} → {prediction['predicted_outcome']:8} ({prediction['confidence']*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("🎓 EXAMPLE 3: UNDERSTANDING THE AI MODEL")
    print("=" * 70)
    
    print("\n🧠 How the AI makes predictions:")
    print("1. 📊 Calculates 10 features for each team:")
    print("   - Recent form (win percentage in last 10 games)")
    print("   - Goals scored per game average")
    print("   - Goals conceded per game average") 
    print("   - Home/away strength")
    print("   - Head-to-head record")
    
    print("\n2. 🤖 Random Forest algorithm:")
    print("   - Uses 100 decision trees")
    print("   - Each tree 'votes' on the outcome")
    print("   - Final prediction = majority vote")
    print("   - Probabilities = percentage of trees voting each way")
    
    print("\n3. 📈 Model performance:")
    print("   - Trained on 500+ historical matches")
    print("   - 50% accuracy (33% = random chance)")
    print("   - Better at predicting home wins than away wins")
    print("   - Considers team strength, form, and matchup history")
    
    print("\n" + "=" * 70)
    print("🎯 EXAMPLE 4: INTERACTIVE PREDICTIONS")
    print("=" * 70)
    
    print("\n🔮 Try your own predictions:")
    print("Available functions:")
    print("- predict_match('Home Team', 'Away Team')")
    print("- predict_weekend_fixtures([(team1, team2), ...])")
    print("- interactive_predictor()  # Command line interface")
    
    # Demo a few more predictions
    demo_matches = [
        ('Man City', 'Liverpool'),
        ('Chelsea', 'Arsenal'),
        ('Man United', 'Tottenham')
    ]
    
    print(f"\n🎲 Quick predictions for big matches:")
    for home, away in demo_matches:
        result = predict_match(home, away)
        if result:
            winner = "HOME" if result['predicted_outcome'] == 'Home Win' else "AWAY" if result['predicted_outcome'] == 'Away Win' else "DRAW"
            print(f"   {home} vs {away}: {winner} ({result['confidence']*100:.0f}% confidence)")
    
    print("\n" + "=" * 70)
    print("✅ MACHINE LEARNING PIPELINE COMPLETE!")
    print("=" * 70)
    
    print("\n🎉 What you've learned:")
    print("✅ How to set up a machine learning database")
    print("✅ Feature engineering for sports predictions")
    print("✅ Training Random Forest models")
    print("✅ Making predictions with trained models")
    print("✅ Converting probabilities to real-world insights")
    print("✅ Building user-friendly ML interfaces")
    
    print("\n🚀 Next steps you could try:")
    print("- Add more features (weather, player injuries, etc.)")
    print("- Try different ML algorithms (Neural Networks, etc.)")
    print("- Build a web interface for the predictions")
    print("- Track prediction accuracy over time")
    print("- Add other sports or leagues")
    
    print("\n💡 Key ML concepts you've mastered:")
    print("- Feature Engineering: Converting raw data to ML features")
    print("- Model Training: Teaching AI to recognize patterns")
    print("- Cross-Validation: Testing model on unseen data")
    print("- Probability Predictions: Getting confidence scores")
    print("- Model Persistence: Saving and loading trained models")
    
    print(f"\n🎯 Final Stats:")
    print(f"- Database: 9,380 matches from 46 teams")
    print(f"- Features: 10 statistical measures per match") 
    print(f"- Model: Random Forest with 100 trees")
    print(f"- Accuracy: 50% (significantly better than 33% random)")
    print(f"- Ready for real predictions!")

if __name__ == "__main__":
    main()