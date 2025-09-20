"""
ADVANCED FEATURE ENGINEERING FOR NEURAL NETWORKS
===============================================

Enhanced feature engineering system specifically designed for deep learning models.
Creates sophisticated features including temporal sequences, team embeddings,
and advanced statistical measures that neural networks can learn from effectively.

Educational Focus: Advanced feature engineering, temporal modeling, embeddings,
and feature preparation for deep learning systems.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from database import get_connection
import warnings
warnings.filterwarnings('ignore')

class NeuralFeatureEngineer:
    """
    Advanced feature engineering system for neural network soccer prediction.
    
    Creates sophisticated features that neural networks can learn complex patterns from:
    - Temporal sequences and trends
    - Team performance embeddings
    - Advanced statistical measures
    - Rolling window features
    - Interaction features
    """
    
    def __init__(self, sequence_length=10, rolling_windows=[3, 5, 10]):
        """
        Initialize the neural feature engineering system.
        
        Args:
            sequence_length (int): Length of temporal sequences to create
            rolling_windows (list): Different window sizes for rolling statistics
        """
        self.sequence_length = sequence_length
        self.rolling_windows = rolling_windows
        self.team_stats_cache = {}
        self.feature_names = []
        
        print(f"🔧 Initialized Neural Feature Engineer")
        print(f"📊 Sequence Length: {sequence_length}")
        print(f"🔄 Rolling Windows: {rolling_windows}")
    
    def create_neural_features(self, home_team, away_team, matches_data, target_date=None):
        """
        Create comprehensive neural network features for a match.
        
        Returns a feature vector optimized for deep learning with:
        - Basic team statistics (enhanced)
        - Temporal sequences
        - Rolling averages
        - Momentum indicators
        - Head-to-head deep features
        - Team strength embeddings
        """
        # Filter matches up to target date if specified
        if target_date:
            matches_data = [m for m in matches_data if m[2] < target_date]  # match_date column
        
        try:
            # 1. Enhanced basic features (20 features)
            basic_features = self._create_enhanced_basic_features(home_team, away_team, matches_data)
            
            # 2. Temporal sequence features (30 features)
            temporal_features = self._create_temporal_sequences(home_team, away_team, matches_data)
            
            # 3. Rolling window features (24 features)
            rolling_features = self._create_rolling_window_features(home_team, away_team, matches_data)
            
            # 4. Momentum and trend features (16 features)
            momentum_features = self._create_momentum_features(home_team, away_team, matches_data)
            
            # 5. Advanced head-to-head features (12 features)
            h2h_features = self._create_advanced_h2h_features(home_team, away_team, matches_data)
            
            # 6. Team strength embeddings (10 features)
            strength_features = self._create_team_strength_features(home_team, away_team, matches_data)
            
            # 7. Contextual features (8 features)
            context_features = self._create_contextual_features(home_team, away_team, matches_data)
            
            # Combine all features
            all_features = np.concatenate([
                basic_features,      # 20 features
                temporal_features,   # 30 features  
                rolling_features,    # 24 features
                momentum_features,   # 16 features
                h2h_features,       # 12 features
                strength_features,   # 10 features
                context_features    # 8 features
            ])
            
            # Create feature names for interpretability
            if not self.feature_names:
                self._create_feature_names()
            
            print(f"✅ Created {len(all_features)} neural network features")
            
            return all_features, self.feature_names
            
        except Exception as e:
            print(f"❌ Error creating neural features: {str(e)}")
            return None, None
    
    def _create_enhanced_basic_features(self, home_team, away_team, matches_data):
        """Create enhanced basic statistical features (20 features)."""
        features = np.zeros(20)
        
        try:
            # Get recent matches for both teams (last 20 matches)
            home_matches = self._get_team_matches(home_team, matches_data, limit=20)
            away_matches = self._get_team_matches(away_team, matches_data, limit=20)
            
            if len(home_matches) < 5 or len(away_matches) < 5:
                return features
            
            # Home team features (0-9)
            home_stats = self._calculate_team_stats(home_team, home_matches)
            features[0:10] = [
                home_stats['win_rate'],
                home_stats['draw_rate'], 
                home_stats['loss_rate'],
                home_stats['avg_goals_for'],
                home_stats['avg_goals_against'],
                home_stats['goal_difference'],
                home_stats['home_advantage'],  # Performance when playing at home
                home_stats['scoring_consistency'],  # Standard deviation of goals scored
                home_stats['defensive_consistency'],  # Standard deviation of goals conceded
                home_stats['recent_form_trend']  # Trend in recent results
            ]
            
            # Away team features (10-19)
            away_stats = self._calculate_team_stats(away_team, away_matches)
            features[10:20] = [
                away_stats['win_rate'],
                away_stats['draw_rate'],
                away_stats['loss_rate'], 
                away_stats['avg_goals_for'],
                away_stats['avg_goals_against'],
                away_stats['goal_difference'],
                away_stats['away_performance'],  # Performance when playing away
                away_stats['scoring_consistency'],
                away_stats['defensive_consistency'],
                away_stats['recent_form_trend']
            ]
            
        except Exception as e:
            pass
        
        return features
    
    def _create_temporal_sequences(self, home_team, away_team, matches_data):
        """Create temporal sequence features for neural networks (30 features)."""
        features = np.zeros(30)
        
        try:
            # Get last 10 matches for each team
            home_sequence = self._get_team_matches(home_team, matches_data, limit=10)
            away_sequence = self._get_team_matches(away_team, matches_data, limit=10)
            
            # Home team sequence (0-14)
            home_seq_features = self._extract_sequence_features(home_team, home_sequence)
            features[0:15] = home_seq_features
            
            # Away team sequence (15-29)
            away_seq_features = self._extract_sequence_features(away_team, away_sequence)
            features[15:30] = away_seq_features
            
        except Exception as e:
            pass
            
        return features
    
    def _extract_sequence_features(self, team, matches):
        """Extract features from a sequence of matches."""
        seq_features = np.zeros(15)
        
        if len(matches) < 3:
            return seq_features
        
        try:
            # Extract sequences
            results = []  # 1=win, 0.5=draw, 0=loss
            goals_for = []
            goals_against = []
            
            for match in matches[:10]:  # Last 10 matches
                if match[3] == team:  # Home team
                    home_goals = int(match[5]) if match[5] else 0
                    away_goals = int(match[6]) if match[6] else 0
                    
                    if home_goals > away_goals:
                        results.append(1.0)  # Win
                    elif home_goals == away_goals:
                        results.append(0.5)  # Draw
                    else:
                        results.append(0.0)  # Loss
                        
                    goals_for.append(home_goals)
                    goals_against.append(away_goals)
                    
                elif match[4] == team:  # Away team
                    home_goals = int(match[5]) if match[5] else 0
                    away_goals = int(match[6]) if match[6] else 0
                    
                    if away_goals > home_goals:
                        results.append(1.0)  # Win
                    elif away_goals == home_goals:
                        results.append(0.5)  # Draw
                    else:
                        results.append(0.0)  # Loss
                        
                    goals_for.append(away_goals)
                    goals_against.append(home_goals)
            
            if len(results) >= 3:
                # Sequence statistics (0-9)
                seq_features[0] = np.mean(results)  # Average result
                seq_features[1] = np.std(results)   # Result consistency
                seq_features[2] = np.mean(goals_for)  # Average goals scored
                seq_features[3] = np.std(goals_for)   # Scoring consistency
                seq_features[4] = np.mean(goals_against)  # Average goals conceded
                seq_features[5] = np.std(goals_against)   # Defensive consistency
                
                # Trend features (6-9)
                if len(results) >= 5:
                    recent_form = np.mean(results[:3])  # Last 3 matches
                    older_form = np.mean(results[3:6])  # Previous 3 matches
                    seq_features[6] = recent_form - older_form  # Form trend
                    
                    recent_attack = np.mean(goals_for[:3])
                    older_attack = np.mean(goals_for[3:6])
                    seq_features[7] = recent_attack - older_attack  # Attack trend
                    
                    recent_defense = np.mean(goals_against[:3])
                    older_defense = np.mean(goals_against[3:6])
                    seq_features[8] = older_defense - recent_defense  # Defense trend (inverted)
                
                # Momentum features (10-14)
                if len(results) >= 5:
                    # Winning/losing streaks
                    current_streak = 0
                    for result in results:
                        if result == 1.0:  # Win
                            current_streak = max(0, current_streak) + 1
                        elif result == 0.0:  # Loss
                            current_streak = min(0, current_streak) - 1
                        else:  # Draw
                            break
                    
                    seq_features[9] = current_streak / 5.0  # Normalized streak
                    
                    # Goal momentum
                    seq_features[10] = (sum(goals_for[:3]) - sum(goals_for[3:6])) / 3.0
                    seq_features[11] = (sum(goals_against[3:6]) - sum(goals_against[:3])) / 3.0
                    
                    # Performance variance
                    seq_features[12] = np.var(results)
                    seq_features[13] = np.var(goals_for)
                    seq_features[14] = np.var(goals_against)
                    
        except Exception as e:
            pass
            
        return seq_features
    
    def _create_rolling_window_features(self, home_team, away_team, matches_data):
        """Create rolling window statistical features (24 features)."""
        features = np.zeros(24)
        
        try:
            # Get matches for both teams
            home_matches = self._get_team_matches(home_team, matches_data, limit=30)
            away_matches = self._get_team_matches(away_team, matches_data, limit=30)
            
            # Calculate rolling statistics for different windows
            for i, window in enumerate(self.rolling_windows):  # [3, 5, 10]
                if len(home_matches) >= window:
                    home_rolling = self._calculate_rolling_stats(home_team, home_matches[:window])
                    features[i*4:(i+1)*4] = [
                        home_rolling['avg_points'],
                        home_rolling['avg_goals_for'],
                        home_rolling['avg_goals_against'],
                        home_rolling['goal_difference']
                    ]
                
                if len(away_matches) >= window:
                    away_rolling = self._calculate_rolling_stats(away_team, away_matches[:window])
                    features[12+i*4:12+(i+1)*4] = [
                        away_rolling['avg_points'],
                        away_rolling['avg_goals_for'],
                        away_rolling['avg_goals_against'],
                        away_rolling['goal_difference']
                    ]
                    
        except Exception as e:
            pass
            
        return features
    
    def _create_momentum_features(self, home_team, away_team, matches_data):
        """Create momentum and trend features (16 features)."""
        features = np.zeros(16)
        
        try:
            # Home team momentum (0-7)
            home_momentum = self._calculate_team_momentum(home_team, matches_data)
            features[0:8] = [
                home_momentum['short_form'],      # Last 3 matches form
                home_momentum['medium_form'],     # Last 5 matches form
                home_momentum['long_form'],       # Last 10 matches form
                home_momentum['attack_momentum'], # Scoring trend
                home_momentum['defense_momentum'],# Defensive trend
                home_momentum['result_momentum'], # Results trend
                home_momentum['home_momentum'],   # Home performance trend
                home_momentum['overall_trend']    # Overall performance trend
            ]
            
            # Away team momentum (8-15)  
            away_momentum = self._calculate_team_momentum(away_team, matches_data)
            features[8:16] = [
                away_momentum['short_form'],
                away_momentum['medium_form'],
                away_momentum['long_form'],
                away_momentum['attack_momentum'],
                away_momentum['defense_momentum'],
                away_momentum['result_momentum'],
                away_momentum['away_momentum'],   # Away performance trend
                away_momentum['overall_trend']
            ]
            
        except Exception as e:
            pass
            
        return features
    
    def _create_advanced_h2h_features(self, home_team, away_team, matches_data):
        """Create advanced head-to-head features (12 features)."""
        features = np.zeros(12)
        
        try:
            # Get head-to-head matches
            h2h_matches = []
            for match in matches_data:
                if (match[3] == home_team and match[4] == away_team) or \
                   (match[3] == away_team and match[4] == home_team):
                    h2h_matches.append(match)
            
            if len(h2h_matches) >= 3:
                # Recent H2H (last 5 matches)
                recent_h2h = h2h_matches[:5]
                
                # Features 0-5: Recent H2H statistics
                home_wins = sum(1 for m in recent_h2h if 
                               (m[3] == home_team and m[7] == 'H') or 
                               (m[4] == home_team and m[7] == 'A'))
                draws = sum(1 for m in recent_h2h if m[7] == 'D')
                away_wins = len(recent_h2h) - home_wins - draws
                
                features[0] = home_wins / len(recent_h2h)  # Home win rate
                features[1] = draws / len(recent_h2h)      # Draw rate
                features[2] = away_wins / len(recent_h2h)  # Away win rate
                
                # Goal statistics
                home_goals_total = 0
                away_goals_total = 0
                
                for match in recent_h2h:
                    home_goals = int(match[5]) if match[5] else 0
                    away_goals = int(match[6]) if match[6] else 0
                    
                    if match[3] == home_team:
                        home_goals_total += home_goals
                        away_goals_total += away_goals
                    else:
                        home_goals_total += away_goals
                        away_goals_total += home_goals
                
                features[3] = home_goals_total / len(recent_h2h)  # Avg goals for home team
                features[4] = away_goals_total / len(recent_h2h)  # Avg goals for away team
                features[5] = (home_goals_total - away_goals_total) / len(recent_h2h)  # Goal difference
                
                # Features 6-11: Historical H2H trends
                if len(h2h_matches) >= 6:
                    older_h2h = h2h_matches[5:10] if len(h2h_matches) >= 10 else h2h_matches[5:]
                    
                    # Compare recent vs older H2H performance
                    recent_home_wins = home_wins / len(recent_h2h)
                    older_home_wins = sum(1 for m in older_h2h if 
                                        (m[3] == home_team and m[7] == 'H') or 
                                        (m[4] == home_team and m[7] == 'A')) / len(older_h2h)
                    
                    features[6] = recent_home_wins - older_home_wins  # H2H trend
                    features[7] = len(h2h_matches)  # Total H2H matches (normalized)
                    features[8] = np.std([1 if ((m[3] == home_team and m[7] == 'H') or 
                                               (m[4] == home_team and m[7] == 'A')) else 0 
                                        for m in h2h_matches[:10]])  # Result consistency
                    
                    # Venue-specific H2H
                    home_venue_matches = [m for m in h2h_matches if m[3] == home_team]
                    if home_venue_matches:
                        home_venue_wins = sum(1 for m in home_venue_matches[:5] if m[7] == 'H')
                        features[9] = home_venue_wins / min(5, len(home_venue_matches))
                    
                    # Recent goal trends in H2H
                    recent_total_goals = sum(int(m[5]) + int(m[6]) for m in recent_h2h 
                                           if m[5] and m[6])
                    features[10] = recent_total_goals / len(recent_h2h) if recent_h2h else 0
                    
                    # H2H competitive balance
                    results = []
                    for m in h2h_matches[:10]:
                        if m[7] == 'H':
                            results.append(1 if m[3] == home_team else 0)
                        elif m[7] == 'A':
                            results.append(0 if m[3] == home_team else 1)
                        else:
                            results.append(0.5)
                    
                    features[11] = np.std(results) if results else 0  # Competitive balance
                    
        except Exception as e:
            pass
            
        return features
    
    def _create_team_strength_features(self, home_team, away_team, matches_data):
        """Create team strength embedding features (10 features)."""
        features = np.zeros(10)
        
        try:
            # Calculate overall team strengths
            home_strength = self._calculate_team_strength(home_team, matches_data)
            away_strength = self._calculate_team_strength(away_team, matches_data)
            
            # Features 0-4: Home team strength metrics
            features[0] = home_strength['offensive_rating']
            features[1] = home_strength['defensive_rating']
            features[2] = home_strength['overall_rating']
            features[3] = home_strength['consistency_rating']
            features[4] = home_strength['home_advantage_rating']
            
            # Features 5-9: Away team strength metrics
            features[5] = away_strength['offensive_rating']
            features[6] = away_strength['defensive_rating']
            features[7] = away_strength['overall_rating']
            features[8] = away_strength['consistency_rating']
            features[9] = away_strength['away_performance_rating']
            
        except Exception as e:
            pass
            
        return features
    
    def _create_contextual_features(self, home_team, away_team, matches_data):
        """Create contextual features (8 features)."""
        features = np.zeros(8)
        
        try:
            # Feature 0: Relative team strength difference
            home_strength = self._get_cached_team_strength(home_team, matches_data)
            away_strength = self._get_cached_team_strength(away_team, matches_data)
            features[0] = home_strength - away_strength
            
            # Feature 1: Days since last match (both teams)
            home_rest = self._get_rest_days(home_team, matches_data)
            away_rest = self._get_rest_days(away_team, matches_data)
            features[1] = (home_rest - away_rest) / 7.0  # Normalized by week
            
            # Feature 2-3: Current season form
            home_season_form = self._get_season_form(home_team, matches_data)
            away_season_form = self._get_season_form(away_team, matches_data)
            features[2] = home_season_form
            features[3] = away_season_form
            
            # Feature 4-5: League position effect (approximate)
            home_position_strength = self._estimate_league_position(home_team, matches_data)
            away_position_strength = self._estimate_league_position(away_team, matches_data)
            features[4] = home_position_strength
            features[5] = away_position_strength
            
            # Feature 6-7: Match importance (based on recent results and trends)
            features[6] = self._calculate_match_importance(home_team, matches_data)
            features[7] = self._calculate_match_importance(away_team, matches_data)
            
        except Exception as e:
            pass
            
        return features
    
    def _get_team_matches(self, team, matches_data, limit=None):
        """Get recent matches for a team."""
        team_matches = []
        for match in reversed(matches_data):
            if match[3] == team or match[4] == team:
                team_matches.append(match)
                if limit and len(team_matches) >= limit:
                    break
        return team_matches
    
    def _calculate_team_stats(self, team, matches):
        """Calculate comprehensive team statistics."""
        stats = {
            'win_rate': 0, 'draw_rate': 0, 'loss_rate': 0,
            'avg_goals_for': 0, 'avg_goals_against': 0, 'goal_difference': 0,
            'home_advantage': 0, 'away_performance': 0,
            'scoring_consistency': 0, 'defensive_consistency': 0,
            'recent_form_trend': 0
        }
        
        if not matches:
            return stats
        
        wins = draws = losses = 0
        goals_for = []
        goals_against = []
        home_results = []
        away_results = []
        
        for match in matches:
            if match[3] == team:  # Home team
                home_goals = int(match[5]) if match[5] else 0
                away_goals = int(match[6]) if match[6] else 0
                
                goals_for.append(home_goals)
                goals_against.append(away_goals)
                
                if home_goals > away_goals:
                    wins += 1
                    home_results.append(3)  # 3 points for win
                elif home_goals == away_goals:
                    draws += 1
                    home_results.append(1)  # 1 point for draw
                else:
                    losses += 1
                    home_results.append(0)  # 0 points for loss
                    
            elif match[4] == team:  # Away team
                home_goals = int(match[5]) if match[5] else 0
                away_goals = int(match[6]) if match[6] else 0
                
                goals_for.append(away_goals)
                goals_against.append(home_goals)
                
                if away_goals > home_goals:
                    wins += 1
                    away_results.append(3)
                elif away_goals == home_goals:
                    draws += 1
                    away_results.append(1)
                else:
                    losses += 1
                    away_results.append(0)
        
        total_matches = len(matches)
        if total_matches > 0:
            stats['win_rate'] = wins / total_matches
            stats['draw_rate'] = draws / total_matches
            stats['loss_rate'] = losses / total_matches
            stats['avg_goals_for'] = np.mean(goals_for) if goals_for else 0
            stats['avg_goals_against'] = np.mean(goals_against) if goals_against else 0
            stats['goal_difference'] = stats['avg_goals_for'] - stats['avg_goals_against']
            stats['scoring_consistency'] = np.std(goals_for) if len(goals_for) > 1 else 0
            stats['defensive_consistency'] = np.std(goals_against) if len(goals_against) > 1 else 0
        
        if home_results:
            stats['home_advantage'] = np.mean(home_results) / 3.0  # Normalized
        if away_results:
            stats['away_performance'] = np.mean(away_results) / 3.0  # Normalized
        
        # Recent form trend (last 5 vs previous 5)
        if total_matches >= 10:
            recent_points = sum([3 if m[7] == 'H' and m[3] == team else 
                               3 if m[7] == 'A' and m[4] == team else
                               1 if m[7] == 'D' else 0 for m in matches[:5]])
            older_points = sum([3 if m[7] == 'H' and m[3] == team else 
                              3 if m[7] == 'A' and m[4] == team else
                              1 if m[7] == 'D' else 0 for m in matches[5:10]])
            stats['recent_form_trend'] = (recent_points - older_points) / 15.0  # Normalized
        
        return stats
    
    def _calculate_rolling_stats(self, team, matches):
        """Calculate rolling window statistics."""
        stats = {'avg_points': 0, 'avg_goals_for': 0, 'avg_goals_against': 0, 'goal_difference': 0}
        
        if not matches:
            return stats
        
        points = []
        goals_for = []
        goals_against = []
        
        for match in matches:
            if match[3] == team:  # Home team
                home_goals = int(match[5]) if match[5] else 0
                away_goals = int(match[6]) if match[6] else 0
                
                goals_for.append(home_goals)
                goals_against.append(away_goals)
                
                if home_goals > away_goals:
                    points.append(3)
                elif home_goals == away_goals:
                    points.append(1)
                else:
                    points.append(0)
                    
            elif match[4] == team:  # Away team
                home_goals = int(match[5]) if match[5] else 0
                away_goals = int(match[6]) if match[6] else 0
                
                goals_for.append(away_goals)
                goals_against.append(home_goals)
                
                if away_goals > home_goals:
                    points.append(3)
                elif away_goals == home_goals:
                    points.append(1)
                else:
                    points.append(0)
        
        if points:
            stats['avg_points'] = np.mean(points)
            stats['avg_goals_for'] = np.mean(goals_for)
            stats['avg_goals_against'] = np.mean(goals_against)
            stats['goal_difference'] = stats['avg_goals_for'] - stats['avg_goals_against']
        
        return stats
    
    def _calculate_team_momentum(self, team, matches_data):
        """Calculate team momentum indicators."""
        momentum = {
            'short_form': 0, 'medium_form': 0, 'long_form': 0,
            'attack_momentum': 0, 'defense_momentum': 0, 'result_momentum': 0,
            'home_momentum': 0, 'away_momentum': 0, 'overall_trend': 0
        }
        
        team_matches = self._get_team_matches(team, matches_data, limit=15)
        
        if len(team_matches) >= 3:
            # Calculate form for different periods
            for period, key in [(3, 'short_form'), (5, 'medium_form'), (10, 'long_form')]:
                if len(team_matches) >= period:
                    period_matches = team_matches[:period]
                    points = 0
                    for match in period_matches:
                        if (match[3] == team and match[7] == 'H') or \
                           (match[4] == team and match[7] == 'A'):
                            points += 3
                        elif match[7] == 'D':
                            points += 1
                    momentum[key] = points / (period * 3)  # Normalized
            
            # Calculate momentum trends
            if len(team_matches) >= 6:
                recent_goals_for = []
                recent_goals_against = []
                older_goals_for = []
                older_goals_against = []
                
                for i, match in enumerate(team_matches[:6]):
                    if match[3] == team:  # Home
                        goals_for = int(match[5]) if match[5] else 0
                        goals_against = int(match[6]) if match[6] else 0
                    else:  # Away
                        goals_for = int(match[6]) if match[6] else 0
                        goals_against = int(match[5]) if match[5] else 0
                    
                    if i < 3:  # Recent
                        recent_goals_for.append(goals_for)
                        recent_goals_against.append(goals_against)
                    else:  # Older
                        older_goals_for.append(goals_for)
                        older_goals_against.append(goals_against)
                
                if recent_goals_for and older_goals_for:
                    momentum['attack_momentum'] = np.mean(recent_goals_for) - np.mean(older_goals_for)
                    momentum['defense_momentum'] = np.mean(older_goals_against) - np.mean(recent_goals_against)
        
        return momentum
    
    def _calculate_team_strength(self, team, matches_data):
        """Calculate comprehensive team strength metrics."""
        strength = {
            'offensive_rating': 0.5, 'defensive_rating': 0.5, 'overall_rating': 0.5,
            'consistency_rating': 0.5, 'home_advantage_rating': 0.5, 'away_performance_rating': 0.5
        }
        
        team_matches = self._get_team_matches(team, matches_data, limit=20)
        
        if len(team_matches) >= 10:
            # Calculate ratings based on performance
            goals_for = []
            goals_against = []
            results = []
            home_results = []
            away_results = []
            
            for match in team_matches:
                if match[3] == team:  # Home
                    gf = int(match[5]) if match[5] else 0
                    ga = int(match[6]) if match[6] else 0
                    home_results.append(3 if gf > ga else 1 if gf == ga else 0)
                else:  # Away
                    gf = int(match[6]) if match[6] else 0
                    ga = int(match[5]) if match[5] else 0
                    away_results.append(3 if gf > ga else 1 if gf == ga else 0)
                
                goals_for.append(gf)
                goals_against.append(ga)
                results.append(3 if gf > ga else 1 if gf == ga else 0)
            
            # Normalize ratings
            strength['offensive_rating'] = min(1.0, np.mean(goals_for) / 2.5)  # Normalize by 2.5 goals
            strength['defensive_rating'] = max(0.0, 1.0 - np.mean(goals_against) / 2.5)
            strength['overall_rating'] = np.mean(results) / 3.0
            strength['consistency_rating'] = max(0.0, 1.0 - np.std(results) / 3.0)
            
            if home_results:
                strength['home_advantage_rating'] = np.mean(home_results) / 3.0
            if away_results:
                strength['away_performance_rating'] = np.mean(away_results) / 3.0
        
        return strength
    
    def _get_cached_team_strength(self, team, matches_data):
        """Get cached team strength calculation."""
        if team not in self.team_stats_cache:
            strength_metrics = self._calculate_team_strength(team, matches_data)
            self.team_stats_cache[team] = strength_metrics['overall_rating']
        return self.team_stats_cache[team]
    
    def _get_rest_days(self, team, matches_data):
        """Calculate days since last match."""
        team_matches = self._get_team_matches(team, matches_data, limit=1)
        if team_matches:
            # This is a simplified version - in practice you'd calculate actual days
            return 3  # Assume 3 days rest on average
        return 7  # Default to 7 days
    
    def _get_season_form(self, team, matches_data):
        """Calculate current season form."""
        # Simplified - take form over last 15 matches
        team_matches = self._get_team_matches(team, matches_data, limit=15)
        if len(team_matches) >= 5:
            points = 0
            for match in team_matches:
                if (match[3] == team and match[7] == 'H') or \
                   (match[4] == team and match[7] == 'A'):
                    points += 3
                elif match[7] == 'D':
                    points += 1
            return points / (len(team_matches) * 3)  # Normalized
        return 0.5  # Default
    
    def _estimate_league_position(self, team, matches_data):
        """Estimate relative league position strength."""
        # This is a simplified estimation based on recent form
        return self._get_cached_team_strength(team, matches_data)
    
    def _calculate_match_importance(self, team, matches_data):
        """Calculate match importance based on recent trends."""
        # Simplified - based on recent form volatility
        team_matches = self._get_team_matches(team, matches_data, limit=5)
        if len(team_matches) >= 3:
            results = []
            for match in team_matches:
                if (match[3] == team and match[7] == 'H') or \
                   (match[4] == team and match[7] == 'A'):
                    results.append(3)
                elif match[7] == 'D':
                    results.append(1)
                else:
                    results.append(0)
            return np.std(results) / 3.0  # Normalized volatility
        return 0.5  # Default
    
    def _create_feature_names(self):
        """Create descriptive names for all features."""
        self.feature_names = []
        
        # Enhanced basic features (20)
        basic_names = [
            'home_win_rate', 'home_draw_rate', 'home_loss_rate', 'home_avg_goals_for',
            'home_avg_goals_against', 'home_goal_diff', 'home_advantage', 
            'home_scoring_consistency', 'home_defensive_consistency', 'home_form_trend',
            'away_win_rate', 'away_draw_rate', 'away_loss_rate', 'away_avg_goals_for',
            'away_avg_goals_against', 'away_goal_diff', 'away_performance',
            'away_scoring_consistency', 'away_defensive_consistency', 'away_form_trend'
        ]
        self.feature_names.extend(basic_names)
        
        # Temporal sequence features (30)
        for team in ['home', 'away']:
            seq_names = [
                f'{team}_seq_avg_result', f'{team}_seq_result_consistency',
                f'{team}_seq_avg_goals_for', f'{team}_seq_scoring_consistency',
                f'{team}_seq_avg_goals_against', f'{team}_seq_defensive_consistency',
                f'{team}_seq_form_trend', f'{team}_seq_attack_trend', f'{team}_seq_defense_trend',
                f'{team}_seq_streak', f'{team}_seq_goal_momentum', f'{team}_seq_defense_momentum',
                f'{team}_seq_performance_var', f'{team}_seq_scoring_var', f'{team}_seq_defensive_var'
            ]
            self.feature_names.extend(seq_names)
        
        # Rolling window features (24)
        for team in ['home', 'away']:
            for window in [3, 5, 10]:
                window_names = [
                    f'{team}_rolling_{window}_avg_points', f'{team}_rolling_{window}_avg_goals_for',
                    f'{team}_rolling_{window}_avg_goals_against', f'{team}_rolling_{window}_goal_diff'
                ]
                self.feature_names.extend(window_names)
        
        # Momentum features (16)
        for team in ['home', 'away']:
            momentum_names = [
                f'{team}_short_form', f'{team}_medium_form', f'{team}_long_form',
                f'{team}_attack_momentum', f'{team}_defense_momentum', f'{team}_result_momentum',
                f'{team}_venue_momentum', f'{team}_overall_trend'
            ]
            self.feature_names.extend(momentum_names)
        
        # H2H features (12)
        h2h_names = [
            'h2h_home_win_rate', 'h2h_draw_rate', 'h2h_away_win_rate',
            'h2h_home_avg_goals', 'h2h_away_avg_goals', 'h2h_goal_diff',
            'h2h_trend', 'h2h_total_matches', 'h2h_consistency',
            'h2h_home_venue_rate', 'h2h_avg_total_goals', 'h2h_competitive_balance'
        ]
        self.feature_names.extend(h2h_names)
        
        # Team strength features (10)
        strength_names = [
            'home_offensive_rating', 'home_defensive_rating', 'home_overall_rating',
            'home_consistency_rating', 'home_advantage_rating',
            'away_offensive_rating', 'away_defensive_rating', 'away_overall_rating',
            'away_consistency_rating', 'away_performance_rating'
        ]
        self.feature_names.extend(strength_names)
        
        # Contextual features (8)
        context_names = [
            'strength_difference', 'rest_difference', 'home_season_form', 'away_season_form',
            'home_position_strength', 'away_position_strength', 'home_match_importance', 'away_match_importance'
        ]
        self.feature_names.extend(context_names)


# Convenience function to create neural features
def create_neural_match_features(home_team, away_team, matches_data, target_date=None):
    """
    Convenience function to create neural network features for a match.
    
    Args:
        home_team (str): Home team name
        away_team (str): Away team name  
        matches_data (list): Historical matches data
        target_date (str, optional): Target date to filter matches
        
    Returns:
        tuple: (features_array, feature_names) or (None, None) if error
    """
    try:
        feature_engineer = NeuralFeatureEngineer()
        return feature_engineer.create_neural_features(home_team, away_team, matches_data, target_date)
    except Exception as e:
        print(f"❌ Error creating neural features: {str(e)}")
        return None, None


if __name__ == "__main__":
    print("🧠 ADVANCED NEURAL FEATURE ENGINEERING SYSTEM")
    print("=" * 60)
    
    # Test the feature engineering system
    print("🔍 Testing neural feature engineering...")
    
    # Load some test data
    conn = get_connection('soccer_stats')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM epl_matches ORDER BY match_date LIMIT 1000")
    test_matches = cursor.fetchall()
    cursor.close()
    conn.close()
    
    # Test feature creation
    features, feature_names = create_neural_match_features('Arsenal', 'Chelsea', test_matches)
    
    if features is not None:
        print(f"✅ Successfully created {len(features)} neural features")
        print(f"📊 Feature vector shape: {features.shape}")
        print(f"📋 Sample features: {features[:10]}")
        print(f"🏷️  Total feature names: {len(feature_names)}")
        print("📝 First 10 feature names:")
        for i, name in enumerate(feature_names[:10]):
            print(f"   {i+1:2d}. {name}")
    else:
        print("❌ Failed to create neural features")
    
    print("\n✅ Neural feature engineering system ready!")