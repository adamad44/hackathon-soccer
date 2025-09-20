"""
CUSTOM PREDICTION SCRIPT
=======================

Use this script to predict any match you want!
Just edit the team names below and run the script.
"""

from predictor import predict_match

def predict_custom_match(home_team, away_team):
    """Make a prediction for any match"""
    print("=" * 50)
    print(f"🏟️ PREDICTING: {home_team} (home) vs {away_team} (away)")
    print("=" * 50)
    
    result = predict_match(home_team, away_team)
    
    if result:
        print(f"🔮 PREDICTION: {result['predicted_outcome']}")
        print(f"📊 CONFIDENCE: {result['confidence']*100:.1f}%")
        print()
        print("📈 FULL BREAKDOWN:")
        print(f"   🏠 {home_team} Win: {result['home_win_prob']*100:.1f}%")
        print(f"   ✈️  {away_team} Win: {result['away_win_prob']*100:.1f}%")
        print(f"   🤝 Draw: {result['draw_prob']*100:.1f}%")
        print()
        print("💰 BETTING ODDS (payout per $1 bet):")
        print(f"   🏠 {home_team}: ${1/result['home_win_prob']:.2f}")
        print(f"   ✈️  {away_team}: ${1/result['away_win_prob']:.2f}")
        print(f"   🤝 Draw: ${1/result['draw_prob']:.2f}")
    else:
        print("❌ Could not make prediction - check team names!")
    
    print("=" * 50)

if __name__ == "__main__":
    # 🎯 EDIT THESE TEAM NAMES TO PREDICT YOUR MATCH:
    home_team = "Chelsea"      # ← Change this to your home team
    away_team = "Arsenal"      # ← Change this to your away team
    
    # Make the prediction
    predict_custom_match(home_team, away_team)
    
    print("\n🎲 Want to predict more matches?")
    print("Edit the team names above and run this script again!")
    
    print("\n📝 Available teams:")
    print("Arsenal, Chelsea, Liverpool, Man City, Man United, Tottenham,")
    print("Newcastle, Brighton, West Ham, Everton, Leicester, Wolves,")
    print("Crystal Palace, Fulham, Bournemouth, Brentford, Aston Villa...")
    print("(and 39 more teams!)")