"""
SIMPLE SOCCER PREDICTION LAUNCHER
================================

This script provides the easiest way to start making soccer predictions.
Just run this file and follow the prompts!
"""

from predictor import predict_match, SoccerPredictor

def main():
    print("SOCCER PREDICTION SYSTEM")
    print("=" * 40)
    
    # Show available teams first
    predictor = SoccerPredictor('model_random_forest.pkl')
    
    if predictor.predictor is None:
        print("ERROR: No trained model found!")
        print("Please run 'python model.py' first to train a model.")
        return
    
    print(f"Available teams ({len(predictor.teams)} total):")
    print("-" * 40)
    
    # Show teams in a nice format
    for i, team in enumerate(predictor.teams):
        if i % 3 == 0:
            print()
        print(f"{team:18s}", end="")
    
    print("\n\n" + "=" * 40)
    print("MAKING PREDICTIONS")
    print("=" * 40)
    
    # Example predictions
    examples = [
        ("Arsenal", "Chelsea"),
        ("Liverpool", "Man City"),
        ("Man United", "Tottenham")
    ]
    
    for home, away in examples:
        print(f"\n{home} vs {away}")
        print("-" * 30)
        
        result = predict_match(home, away)
        
        if result:
            print(f"Prediction: {result['predicted_outcome']}")
            print(f"Confidence: {result['confidence']*100:.1f}%")
            print(f"Home Win: {result['home_win_prob']*100:.1f}%")
            print(f"Away Win: {result['away_win_prob']*100:.1f}%")
            print(f"Draw: {result['draw_prob']*100:.1f}%")
    
    print("\n" + "=" * 40)
    print("HOW TO USE:")
    print("=" * 40)
    print("1. from predictor import predict_match")
    print("2. result = predict_match('Arsenal', 'Chelsea')")
    print("3. print(result)")
    
    print("\nOr try your own team:")
    print("python -c \"from predictor import predict_match; print(predict_match('Arsenal', 'Liverpool'))\"")

if __name__ == "__main__":
    main()