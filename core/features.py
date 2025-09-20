"""
SOCCER MATCH PREDICTION - FEATURE ENGINEERING TEMPLATE

This file contains functions to calculate team statistics and features
that will be used to train the prediction model.

Key Concept: Features are the "characteristics" we calculate about teams
that help predict match outcomes (like recent form, goal averages, etc.)
"""

import pandas as pd
from datetime import datetime, timedelta
from database import *

# =============================================================================
# TEAM PERFORMANCE CALCULATORS
# =============================================================================

def calculate_team_form(team_name, matches_data, num_recent_games=5):
    """
    Calculate recent form for a team (win rate in last N games)
    
    This function looks at a team's recent matches and calculates what percentage they won.
    A score of 1.0 = won all recent games, 0.0 = won no recent games
    """
    # Create empty list to store matches involving this team
    team_matches = []
    
    # Loop through ALL matches in the dataset
    for match in matches_data:
        # Extract home team (index 3) and away team (index 4) from match data
        home_team = match[3]
        away_team = match[4]
        
        # Check if our target team played in this match (either home or away)
        if home_team == team_name or away_team == team_name:
            # Add this match to our team's match history
            team_matches.append(match)
    
    # If we don't have enough matches, return neutral form score
    if len(team_matches) < num_recent_games:
        return 0.5  # Neutral form when not enough data
    
    # Get the LAST N matches (most recent games)
    # [-num_recent_games:] means "give me the last N items from the list"
    recent_matches = team_matches[-num_recent_games:]
    
    # Count how many of these recent matches the team won
    wins = 0
    
    # Loop through each recent match
    for match in recent_matches:
        # Extract match result (index 7: 'H'=home win, 'A'=away win, 'D'=draw)
        result = match[7]
        home_team = match[3]
        away_team = match[4]
        
        # Check if our team won this match
        if (team_name == home_team and result == 'H') or (team_name == away_team and result == 'A'):
            # Team won! Increment win counter
            wins += 1
    
    # Calculate win percentage: wins divided by total games
    # This gives us a number between 0.0 (no wins) and 1.0 (all wins)
    form_score = wins / len(recent_matches)
    
    # Return the form score
    return form_score


def calculate_goal_average(team_name, matches_data, num_recent_games=10):
    """
    Calculate goals scored and conceded per game averages
    
    This function calculates how many goals a team typically scores and allows.
    Higher goals scored = stronger attack, lower goals conceded = stronger defense
    """
    # Create empty lists to store goals from each match
    goals_scored = []    # Goals this team scored in each game
    goals_conceded = []  # Goals this team allowed in each game
    
    # Loop through all matches to find ones with our team
    for match in matches_data:
        # Extract team names and goal data from match
        home_team = match[3]           # Home team name
        away_team = match[4]           # Away team name  
        home_goals = match[5]          # Goals scored by home team
        away_goals = match[6]          # Goals scored by away team
        
        # Check if our team played as the HOME team
        if home_team == team_name:
            # Team was playing at home
            goals_scored.append(home_goals)    # They scored the "home goals"
            goals_conceded.append(away_goals)  # They conceded the "away goals"
            
        # Check if our team played as the AWAY team
        elif away_team == team_name:
            # Team was playing away
            goals_scored.append(away_goals)    # They scored the "away goals"
            goals_conceded.append(home_goals)  # They conceded the "home goals"
    
    # If we don't have enough match data, return default values
    if len(goals_scored) == 0:
        return 1.0, 1.0  # Default: 1 goal scored, 1 goal conceded per game
    
    # Get only the most recent games (limit the analysis)
    if len(goals_scored) > num_recent_games:
        # Take only the last N games
        goals_scored = goals_scored[-num_recent_games:]
        goals_conceded = goals_conceded[-num_recent_games:]
    
    # Calculate average goals per game
    # sum() adds up all values in the list
    # len() gives us the number of games
    # Dividing sum by count gives us the average
    avg_scored = sum(goals_scored) / len(goals_scored)
    avg_conceded = sum(goals_conceded) / len(goals_conceded)
    
    # Return both averages as a tuple (two values)
    return avg_scored, avg_conceded


def calculate_home_away_strength(team_name, matches_data):
    """
    Calculate how well team performs at home vs away
    
    In soccer, teams usually perform better at home than away.
    This function measures that difference for each team.
    """
    # Initialize counters for home matches
    home_wins = 0    # Number of wins when playing at home
    home_games = 0   # Total number of home games played
    
    # Initialize counters for away matches  
    away_wins = 0    # Number of wins when playing away
    away_games = 0   # Total number of away games played
    
    # Loop through all matches in the dataset
    for match in matches_data:
        # Extract match information
        home_team = match[3]    # Team playing at home
        away_team = match[4]    # Team playing away
        result = match[7]       # Match result: 'H', 'A', or 'D'
        
        # Check if our team played AT HOME in this match
        if home_team == team_name:
            # Increment home games counter
            home_games += 1
            
            # Check if they won (home win = 'H')
            if result == 'H':
                home_wins += 1  # They won at home!
        
        # Check if our team played AWAY in this match        
        elif away_team == team_name:
            # Increment away games counter
            away_games += 1
            
            # Check if they won (away win = 'A')
            if result == 'A':
                away_wins += 1  # They won away!
    
    # Calculate win percentages (avoid division by zero)
    if home_games > 0:
        # Home strength = percentage of home games won
        home_strength = home_wins / home_games
    else:
        # No home games played, assume average strength
        home_strength = 0.5
    
    if away_games > 0:
        # Away strength = percentage of away games won
        away_strength = away_wins / away_games
    else:
        # No away games played, assume average strength
        away_strength = 0.5
    
    # Return both strength values
    return home_strength, away_strength


# =============================================================================
# HEAD-TO-HEAD ANALYSIS
# =============================================================================

def get_head_to_head_record(team1, team2, matches_data, num_recent=5):
    """
    Get recent head-to-head record between two teams
    
    This function finds all matches between two specific teams and counts
    who wins more often. This is important because some teams have
    "bogey teams" they struggle against.
    """
    # Create empty list to store matches between these two teams only
    head_to_head_matches = []
    
    # Loop through all matches to find matchups between our two teams
    for match in matches_data:
        # Extract the teams that played in this match
        home_team = match[3]
        away_team = match[4]
        
        # Check if this match involved both of our teams
        # (team1 vs team2 OR team2 vs team1 - order doesn't matter)
        if ((home_team == team1 and away_team == team2) or 
            (home_team == team2 and away_team == team1)):
            # This is a head-to-head match! Add it to our list
            head_to_head_matches.append(match)
    
    # If no recent matches found, return zeros
    if len(head_to_head_matches) == 0:
        return 0, 0, 0
    
    # Get only the most recent head-to-head matches
    if len(head_to_head_matches) > num_recent:
        # Take only the last N matches between these teams
        head_to_head_matches = head_to_head_matches[-num_recent:]
    
    # Initialize counters for results
    team1_wins = 0   # Number of times team1 won
    team2_wins = 0   # Number of times team2 won  
    draws = 0        # Number of draws
    
    # Count the results of each head-to-head match
    for match in head_to_head_matches:
        # Extract match details
        home_team = match[3]    # Who played at home
        away_team = match[4]    # Who played away
        result = match[7]       # Result: 'H'=home win, 'A'=away win, 'D'=draw
        
        # Determine who won based on the result
        if result == 'H':
            # Home team won - check which of our teams was home
            if home_team == team1:
                team1_wins += 1    # team1 was home and won
            else:
                team2_wins += 1    # team2 was home and won
                
        elif result == 'A':
            # Away team won - check which of our teams was away
            if away_team == team1:
                team1_wins += 1    # team1 was away and won
            else:
                team2_wins += 1    # team2 was away and won
                
        else:  # result == 'D'
            # Match was a draw
            draws += 1
    
    # Return the head-to-head record
    return team1_wins, team2_wins, draws


# =============================================================================
# FEATURE COMBINATION FUNCTIONS
# =============================================================================

def create_match_features(home_team, away_team, matches_data, match_date=None):
    """
    Create all features for a single match prediction
    
    This is the MAIN function that combines all feature calculations.
    It takes two team names and creates a list of numbers that describe
    their current form, strength, etc. This is what feeds into the AI model.
    """
    # Print what we're working on (helpful for debugging)
    print(f"Creating features for: {home_team} vs {away_team}")
    
    # Initialize empty lists to store our calculated features
    features = []        # Will contain the actual numbers
    feature_names = []   # Will contain names describing each number
    
    # SECTION 1: HOME TEAM FEATURES
    # Calculate various statistics about the home team
    
    # How well is the home team playing recently? (0.0 to 1.0)
    home_form = calculate_team_form(home_team, matches_data)
    
    # How many goals does home team typically score and concede?
    home_goals_for, home_goals_against = calculate_goal_average(home_team, matches_data)
    
    # How strong is home team when playing at home? (we want home strength)
    home_strength, _ = calculate_home_away_strength(home_team, matches_data)
    
    # Add all home team features to our lists
    features.extend([home_form, home_goals_for, home_goals_against, home_strength])
    feature_names.extend(['home_form', 'home_goals_for', 'home_goals_against', 'home_strength'])
    
    # SECTION 2: AWAY TEAM FEATURES  
    # Calculate the same statistics for the away team
    
    # How well is the away team playing recently?
    away_form = calculate_team_form(away_team, matches_data)
    
    # Away team's typical goals scored and conceded
    away_goals_for, away_goals_against = calculate_goal_average(away_team, matches_data)
    
    # How strong is away team when playing away? (we want away strength)
    _, away_strength = calculate_home_away_strength(away_team, matches_data)
    
    # Add all away team features to our lists
    features.extend([away_form, away_goals_for, away_goals_against, away_strength])
    feature_names.extend(['away_form', 'away_goals_for', 'away_goals_against', 'away_strength'])
    
    # SECTION 3: HEAD-TO-HEAD FEATURES
    # How do these two teams perform against each other historically?
    
    # Get recent head-to-head results
    h2h_home_wins, h2h_away_wins, h2h_draws = get_head_to_head_record(home_team, away_team, matches_data)
    
    # Calculate total head-to-head matches
    total_h2h = h2h_home_wins + h2h_away_wins + h2h_draws
    
    # Convert raw counts to percentages
    if total_h2h > 0:
        # We have head-to-head data, calculate actual percentages
        h2h_home_rate = h2h_home_wins / total_h2h      # What % of time home team wins
        h2h_away_rate = h2h_away_wins / total_h2h      # What % of time away team wins
    else:
        # No head-to-head data available, use neutral assumptions
        h2h_home_rate = 0.33  # Assume 33% chance each (33% win, 33% lose, 33% draw)
        h2h_away_rate = 0.33
    
    # Add head-to-head features to our lists
    features.extend([h2h_home_rate, h2h_away_rate])
    feature_names.extend(['h2h_home_rate', 'h2h_away_rate'])
    
    # Print summary of what we created (helpful for learning)
    print(f"Created {len(features)} features: {feature_names}")
    
    # Return both the feature values and their names
    return features, feature_names


# =============================================================================
# DATASET PREPARATION FUNCTIONS
# =============================================================================

def prepare_training_data(matches_data, max_matches=1000):
    """
    Prepare the full dataset for training
    
    This is the most important function! It creates a training dataset by:
    1. Going through each historical match
    2. Calculating features based on data BEFORE that match
    3. Recording what actually happened (the result)
    4. Building a dataset the AI can learn from
    """
    # Print what we're about to do
    print(f"Preparing training data from {len(matches_data)} matches...")
    
    # Initialize lists to store our training data
    X = []  # "X" is machine learning notation for "input features"
    y = []  # "y" is machine learning notation for "target outcomes"
    
    # Only process a limited number of matches to avoid long processing times
    # You can increase max_matches later when you're confident it works
    matches_to_process = matches_data[:max_matches]
    
    # Loop through each match to create training examples
    for i, match in enumerate(matches_to_process):
        # Print progress every 100 matches so we know it's working
        if i % 100 == 0:
            print(f"Processing match {i+1}/{len(matches_to_process)}")
        
        # Extract information from this match
        # Remember: match is a list with data at specific indices
        home_team = match[3]    # Index 3 = home team name
        away_team = match[4]    # Index 4 = away team name  
        result = match[7]       # Index 7 = actual result ('H', 'A', or 'D')
        
        # CRITICAL CONCEPT: Only use historical data!
        # We can only use matches that happened BEFORE this match
        # This simulates real-world prediction where we don't know the future
        historical_data = matches_data[:i]  # All matches before index i
        
        # Skip if we don't have enough historical data to calculate meaningful features
        if len(historical_data) < 10:
            continue  # Skip to next match
            
        # Create features for this match using only historical data
        try:
            # Call our feature engineering function
            features, _ = create_match_features(home_team, away_team, historical_data)
            
            # Convert the text result to a number the AI can understand
            # Machine learning models need numbers, not text
            if result == 'H':      # Home team won
                target = 0
            elif result == 'A':    # Away team won  
                target = 1
            else:                  # Draw (result == 'D')
                target = 2
                
            # Add this training example to our dataset
            X.append(features)  # Input: team statistics before the match
            y.append(target)    # Output: what actually happened
            
        except Exception as e:
            # If something goes wrong, print the error and skip this match
            print(f"Error processing match {i}: {e}")
            continue  # Skip to next match
    
    # Print summary of what we created
    print(f"Created training data: {len(X)} samples with {len(X[0]) if X else 0} features each")
    
    # Return the training data
    # X = list of feature lists (input to model)
    # y = list of results (what model should predict)
    return X, y


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_all_teams(matches_data):
    """Get list of all unique team names in the dataset"""
    teams = set()
    for match in matches_data:
        teams.add(match[3])  # home_team
        teams.add(match[4])  # away_team
    return sorted(list(teams))


def print_team_stats(team_name, matches_data):
    """Print basic statistics for a team (for debugging/learning)"""
    print(f"\n=== STATS FOR {team_name} ===")
    
    form = calculate_team_form(team_name, matches_data)
    goals_for, goals_against = calculate_goal_average(team_name, matches_data)
    home_str, away_str = calculate_home_away_strength(team_name, matches_data)
    
    print(f"Recent Form: {form:.3f}")
    print(f"Goals For/Game: {goals_for:.2f}")
    print(f"Goals Against/Game: {goals_against:.2f}")
    print(f"Home Strength: {home_str:.3f}")
    print(f"Away Strength: {away_str:.3f}")


# =============================================================================
# LEARNING EXERCISES (uncomment to test your functions)
# =============================================================================

if __name__ == "__main__":
    print("FEATURE ENGINEERING - FULLY IMPLEMENTED")
    print("=" * 50)
    
    # Load your match data from main.py
    from main import all_matches
    
    # TEST 1: Show team statistics for popular teams
    print("\nTEST 1: Team Statistics")
    print("-" * 30)
    
    # Test our functions on well-known teams
    popular_teams = ["Arsenal", "Chelsea", "Liverpool", "Man United"]
    
    for team in popular_teams:
        print(f"\n=== {team} Stats ===")
        # Calculate and display form (recent win rate)
        form = calculate_team_form(team, all_matches, 5)
        print(f"Recent Form (last 5 games): {form:.3f} ({form*100:.1f}% win rate)")
        
        # Calculate and display goal averages
        goals_for, goals_against = calculate_goal_average(team, all_matches, 10)
        print(f"Goals scored per game: {goals_for:.2f}")
        print(f"Goals conceded per game: {goals_against:.2f}")
        
        # Calculate home/away strength
        home_str, away_str = calculate_home_away_strength(team, all_matches)
        print(f"Home strength: {home_str:.3f} ({home_str*100:.1f}% home win rate)")
        print(f"Away strength: {away_str:.3f} ({away_str*100:.1f}% away win rate)")
    
    # TEST 2: Test feature creation for a specific matchup
    print(f"\n" + "="*50)
    print("TEST 2: Feature Creation for Arsenal vs Chelsea")
    print("="*50)
    
    # Create features for Arsenal vs Chelsea using all historical data
    features, names = create_match_features("Arsenal", "Chelsea", all_matches)
    
    # Display each feature with its name and value
    print(f"\nCreated {len(features)} features:")
    for i, (name, value) in enumerate(zip(names, features)):
        print(f"  {i+1:2d}. {name:20s}: {value:.4f}")
    
    # TEST 3: Create small training dataset
    print(f"\n" + "="*50)
    print("TEST 3: Training Data Creation")
    print("="*50)
    
    # Create training data from first 200 matches (small test)
    print("Creating training data from first 200 matches...")
    X, y = prepare_training_data(all_matches, max_matches=200)
    
    # Display summary statistics
    print(f"\nTraining Data Summary:")
    print(f"  Total samples: {len(X)}")
    print(f"  Features per sample: {len(X[0]) if X else 0}")
    print(f"  Home wins: {sum(1 for result in y if result == 0)}")
    print(f"  Away wins: {sum(1 for result in y if result == 1)}")
    print(f"  Draws: {sum(1 for result in y if result == 2)}")
    
    # Show first few training examples
    if len(X) > 0:
        print(f"\nFirst 3 training examples:")
        for i in range(min(3, len(X))):
            result_names = ["Home Win", "Away Win", "Draw"]
            print(f"  Sample {i+1}: {len(X[i])} features → {result_names[y[i]]}")
    
    print(f"\n" + "="*50)
    print("FEATURE ENGINEERING COMPLETE!")
    print("="*50)
    print("✓ All functions implemented and working")
    print("✓ Team statistics calculated successfully") 
    print("✓ Match features created successfully")
    print("✓ Training data prepared successfully")
    print("\nNext step: Run model.py to train your prediction model!")