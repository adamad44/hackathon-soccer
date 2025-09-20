"""
ADVANCED NEURAL NETWORK MODEL FOR SOCCER PREDICTION
===================================================

This file implements a sophisticated deep learning system for soccer match prediction
using TensorFlow/Keras with complex neural network architectures, extended training
capabilities, and advanced monitoring.

Educational Focus: Deep learning, neural networks, extended training, regularization,
optimization techniques, and advanced AI concepts.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Embedding, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from database import get_connection
from features import create_match_features

class AdvancedSoccerNeuralNetwork:
    """
    Sophisticated neural network system for soccer prediction with extended training capabilities.
    
    Features:
    - Deep neural networks with multiple architectures
    - Extended training with monitoring
    - Advanced regularization techniques
    - Team embeddings for better representation
    - Temporal sequence modeling
    - Ensemble predictions
    """
    
    def __init__(self, architecture='deep_mlp'):
        """
        Initialize the advanced neural network system.
        
        Args:
            architecture (str): Type of neural network architecture
                - 'deep_mlp': Deep Multi-Layer Perceptron
                - 'wide_deep': Wide & Deep architecture
                - 'embedding': Team embedding based model
                - 'ensemble': Ensemble of multiple models
        """
        self.architecture = architecture
        self.model = None
        self.scaler = StandardScaler()
        self.team_encoder = LabelEncoder()
        self.training_history = None
        self.model_metrics = {}
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Configure TensorFlow for optimal performance
        tf.config.run_functions_eagerly(False)
        
        print(f"🚀 Initialized Advanced Soccer Neural Network")
        print(f"📊 Architecture: {architecture.upper()}")
        print(f"🔧 TensorFlow Version: {tf.__version__}")
        print(f"💻 GPU Available: {len(tf.config.experimental.list_physical_devices('GPU')) > 0}")
    
    def create_advanced_features(self, matches_data):
        """
        Create advanced features for neural network training.
        
        This creates more sophisticated features than the basic system:
        - Temporal sequences (last N matches)
        - Team embeddings
        - Advanced statistical features
        - Rolling averages and trends
        """
        print("🔍 Creating advanced features for neural network...")
        
        features = []
        labels = []
        team_features = []
        
        # Get all unique teams for encoding
        all_teams = set()
        for match in matches_data:
            all_teams.add(match[3])  # home_team
            all_teams.add(match[4])  # away_team
        
        all_teams = sorted(list(all_teams))
        self.team_encoder.fit(all_teams)
        
        successful_features = 0
        
        for i, match in enumerate(matches_data):
            if i % 500 == 0:
                print(f"   Processing match {i}/{len(matches_data)}...")
            
            home_team = match[3]
            away_team = match[4]
            actual_result = match[7]  # 'H', 'A', 'D'
            
            # Skip if no result
            if not actual_result or actual_result not in ['H', 'A', 'D']:
                continue
            
            try:
                # Create basic features using existing function
                basic_features, feature_names = create_match_features(home_team, away_team, matches_data[:i])
                
                if basic_features is None:
                    continue
                
                # Add team encoding features
                home_team_encoded = self.team_encoder.transform([home_team])[0]
                away_team_encoded = self.team_encoder.transform([away_team])[0]
                
                # Create advanced temporal features
                advanced_features = self._create_temporal_features(home_team, away_team, matches_data[:i])
                
                # Combine all features
                combined_features = np.concatenate([
                    basic_features,  # Basic statistical features (10)
                    [home_team_encoded, away_team_encoded],  # Team encodings (2)
                    advanced_features  # Advanced temporal features (20)
                ])
                
                features.append(combined_features)
                team_features.append([home_team_encoded, away_team_encoded])
                
                # Convert result to numerical label
                label_map = {'A': 0, 'D': 1, 'H': 2}  # Away=0, Draw=1, Home=2
                labels.append(label_map[actual_result])
                
                successful_features += 1
                
            except Exception as e:
                continue
        
        print(f"✅ Created {successful_features} feature sets with {len(combined_features)} features each")
        
        return np.array(features), np.array(labels), np.array(team_features)
    
    def _create_temporal_features(self, home_team, away_team, historical_matches):
        """
        Create temporal features based on recent match history.
        
        This analyzes trends and patterns over time for both teams.
        """
        # Initialize features array (20 features)
        temporal_features = np.zeros(20)
        
        if len(historical_matches) < 10:
            return temporal_features
        
        try:
            # Get recent matches for both teams (last 10 matches)
            home_recent = []
            away_recent = []
            
            for match in reversed(historical_matches[-100:]):  # Look at last 100 matches
                if match[3] == home_team or match[4] == home_team:
                    home_recent.append(match)
                    if len(home_recent) >= 10:
                        break
            
            for match in reversed(historical_matches[-100:]):
                if match[3] == away_team or match[4] == away_team:
                    away_recent.append(match)
                    if len(away_recent) >= 10:
                        break
            
            # Feature 0-4: Home team recent form (last 5 matches)
            home_form = self._calculate_recent_form(home_team, home_recent[:5])
            temporal_features[0:5] = home_form
            
            # Feature 5-9: Away team recent form (last 5 matches)
            away_form = self._calculate_recent_form(away_team, away_recent[:5])
            temporal_features[5:10] = away_form
            
            # Feature 10-12: Goal trends (last 5 matches)
            home_goal_trend = self._calculate_goal_trend(home_team, home_recent[:5])
            away_goal_trend = self._calculate_goal_trend(away_team, away_recent[:5])
            temporal_features[10:13] = home_goal_trend
            temporal_features[13:16] = away_goal_trend
            
            # Feature 16-19: Head-to-head recent history
            h2h_recent = self._get_recent_h2h(home_team, away_team, historical_matches)
            temporal_features[16:20] = h2h_recent
            
        except Exception as e:
            pass  # Return zeros if any error
        
        return temporal_features
    
    def _calculate_recent_form(self, team, recent_matches):
        """Calculate recent form features for a team."""
        form = np.zeros(5)  # [wins, draws, losses, goals_for, goals_against]
        
        for match in recent_matches:
            if match[3] == team:  # Home team
                home_goals = int(match[5]) if match[5] else 0
                away_goals = int(match[6]) if match[6] else 0
                
                if home_goals > away_goals:
                    form[0] += 1  # Win
                elif home_goals == away_goals:
                    form[1] += 1  # Draw
                else:
                    form[2] += 1  # Loss
                
                form[3] += home_goals    # Goals for
                form[4] += away_goals    # Goals against
                
            elif match[4] == team:  # Away team
                home_goals = int(match[5]) if match[5] else 0
                away_goals = int(match[6]) if match[6] else 0
                
                if away_goals > home_goals:
                    form[0] += 1  # Win
                elif away_goals == home_goals:
                    form[1] += 1  # Draw
                else:
                    form[2] += 1  # Loss
                
                form[3] += away_goals    # Goals for
                form[4] += home_goals    # Goals against
        
        return form
    
    def _calculate_goal_trend(self, team, recent_matches):
        """Calculate goal scoring trends."""
        trend = np.zeros(3)  # [avg_goals_for, avg_goals_against, goal_difference]
        
        goals_for = []
        goals_against = []
        
        for match in recent_matches:
            if match[3] == team:  # Home team
                home_goals = int(match[5]) if match[5] else 0
                away_goals = int(match[6]) if match[6] else 0
                goals_for.append(home_goals)
                goals_against.append(away_goals)
            elif match[4] == team:  # Away team
                home_goals = int(match[5]) if match[5] else 0
                away_goals = int(match[6]) if match[6] else 0
                goals_for.append(away_goals)
                goals_against.append(home_goals)
        
        if goals_for:
            trend[0] = np.mean(goals_for)
            trend[1] = np.mean(goals_against)
            trend[2] = trend[0] - trend[1]  # Goal difference
        
        return trend
    
    def _get_recent_h2h(self, home_team, away_team, historical_matches):
        """Get recent head-to-head statistics."""
        h2h = np.zeros(4)  # [home_wins, away_wins, draws, total_matches]
        
        for match in reversed(historical_matches[-50:]):  # Last 50 matches
            if (match[3] == home_team and match[4] == away_team) or \
               (match[3] == away_team and match[4] == home_team):
                
                h2h[3] += 1  # Total matches
                
                if match[7] == 'H':
                    if match[3] == home_team:
                        h2h[0] += 1  # Home team win
                    else:
                        h2h[1] += 1  # Away team win
                elif match[7] == 'A':
                    if match[4] == away_team:
                        h2h[1] += 1  # Away team win
                    else:
                        h2h[0] += 1  # Home team win
                elif match[7] == 'D':
                    h2h[2] += 1  # Draw
        
        return h2h
    
    def build_neural_network(self, input_shape, num_classes=3):
        """
        Build the neural network architecture based on selected type.
        
        Args:
            input_shape (int): Number of input features
            num_classes (int): Number of output classes (3 for Home/Draw/Away)
        """
        print(f"🏗️  Building {self.architecture.upper()} neural network...")
        
        if self.architecture == 'deep_mlp':
            self.model = self._build_deep_mlp(input_shape, num_classes)
        elif self.architecture == 'wide_deep':
            self.model = self._build_wide_deep(input_shape, num_classes)
        elif self.architecture == 'embedding':
            self.model = self._build_embedding_model(input_shape, num_classes)
        elif self.architecture == 'ensemble':
            self.model = self._build_ensemble_model(input_shape, num_classes)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        # Print model summary
        print("\n📋 Neural Network Architecture:")
        self.model.summary()
        
        return self.model
    
    def _build_deep_mlp(self, input_shape, num_classes):
        """
        Build a deep Multi-Layer Perceptron with advanced regularization.
        
        This is a sophisticated deep learning model with:
        - Multiple hidden layers with varying sizes
        - Batch normalization for stable training
        - Dropout for regularization
        - Advanced activation functions
        """
        model = Sequential([
            # Input layer
            Input(shape=(input_shape,)),
            
            # First hidden block
            Dense(512, activation='relu', name='dense_1'),
            BatchNormalization(name='bn_1'),
            Dropout(0.3, name='dropout_1'),
            
            # Second hidden block
            Dense(256, activation='relu', name='dense_2'),
            BatchNormalization(name='bn_2'),
            Dropout(0.4, name='dropout_2'),
            
            # Third hidden block
            Dense(128, activation='relu', name='dense_3'),
            BatchNormalization(name='bn_3'),
            Dropout(0.3, name='dropout_3'),
            
            # Fourth hidden block
            Dense(64, activation='relu', name='dense_4'),
            BatchNormalization(name='bn_4'),
            Dropout(0.2, name='dropout_4'),
            
            # Fifth hidden block
            Dense(32, activation='relu', name='dense_5'),
            Dropout(0.2, name='dropout_5'),
            
            # Output layer
            Dense(num_classes, activation='softmax', name='output')
        ])
        
        # Compile with advanced optimizer
        optimizer = keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            decay=0.0001
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_categorical_crossentropy']
        )
        
        return model
    
    def _build_wide_deep(self, input_shape, num_classes):
        """
        Build a Wide & Deep neural network architecture.
        
        This combines:
        - Wide component: Linear model for memorization
        - Deep component: Neural network for generalization
        """
        # Input layer
        inputs = Input(shape=(input_shape,), name='features')
        
        # Wide component (linear)
        wide = Dense(num_classes, activation='linear', name='wide_component')(inputs)
        
        # Deep component
        deep = Dense(256, activation='relu', name='deep_1')(inputs)
        deep = BatchNormalization(name='deep_bn_1')(deep)
        deep = Dropout(0.3, name='deep_dropout_1')(deep)
        
        deep = Dense(128, activation='relu', name='deep_2')(deep)
        deep = BatchNormalization(name='deep_bn_2')(deep)
        deep = Dropout(0.3, name='deep_dropout_2')(deep)
        
        deep = Dense(64, activation='relu', name='deep_3')(deep)
        deep = Dropout(0.2, name='deep_dropout_3')(deep)
        
        deep = Dense(num_classes, activation='linear', name='deep_component')(deep)
        
        # Combine wide and deep
        combined = layers.Add(name='wide_deep_combine')([wide, deep])
        outputs = layers.Activation('softmax', name='output')(combined)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_embedding_model(self, input_shape, num_classes):
        """
        Build a model with team embeddings.
        
        This learns dense representations of teams that capture their characteristics.
        """
        # Feature inputs
        feature_input = Input(shape=(input_shape - 2,), name='features')  # Exclude team encodings
        
        # Team inputs
        home_team_input = Input(shape=(1,), name='home_team')
        away_team_input = Input(shape=(1,), name='away_team')
        
        # Team embeddings
        num_teams = len(self.team_encoder.classes_)
        embedding_dim = min(50, num_teams // 2)  # Embedding dimension
        
        home_embedding = Embedding(num_teams, embedding_dim, name='home_embedding')(home_team_input)
        away_embedding = Embedding(num_teams, embedding_dim, name='away_embedding')(away_team_input)
        
        # Flatten embeddings
        home_flat = layers.Flatten(name='home_flat')(home_embedding)
        away_flat = layers.Flatten(name='away_flat')(away_embedding)
        
        # Combine all inputs
        combined = Concatenate(name='combine_inputs')([feature_input, home_flat, away_flat])
        
        # Deep layers
        x = Dense(256, activation='relu', name='dense_1')(combined)
        x = BatchNormalization(name='bn_1')(x)
        x = Dropout(0.3, name='dropout_1')(x)
        
        x = Dense(128, activation='relu', name='dense_2')(x)
        x = BatchNormalization(name='bn_2')(x)
        x = Dropout(0.3, name='dropout_2')(x)
        
        x = Dense(64, activation='relu', name='dense_3')(x)
        x = Dropout(0.2, name='dropout_3')(x)
        
        outputs = Dense(num_classes, activation='softmax', name='output')(x)
        
        model = Model(inputs=[feature_input, home_team_input, away_team_input], outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_ensemble_model(self, input_shape, num_classes):
        """
        Build an ensemble of different neural network architectures.
        
        Note: For simplicity, this returns a single deep model.
        In practice, you would train multiple models and combine predictions.
        """
        return self._build_deep_mlp(input_shape, num_classes)
    
    def train_extended(self, X, y, epochs=100, batch_size=32, validation_split=0.2, 
                      patience=15, save_path='neural_model.h5'):
        """
        Train the neural network with extended training capabilities.
        
        Args:
            X: Feature matrix
            y: Target labels
            epochs: Maximum number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data for validation
            patience: Early stopping patience
            save_path: Path to save the best model
        """
        print(f"🚀 Starting extended neural network training...")
        print(f"📊 Training Configuration:")
        print(f"   • Architecture: {self.architecture.upper()}")
        print(f"   • Max Epochs: {epochs}")
        print(f"   • Batch Size: {batch_size}")
        print(f"   • Validation Split: {validation_split*100:.1f}%")
        print(f"   • Early Stopping Patience: {patience}")
        print(f"   • Training Samples: {len(X)}")
        print(f"   • Features per Sample: {X.shape[1]}")
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Setup callbacks for extended training
        callbacks_list = [
            # Early stopping to prevent overfitting
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint to save best model
            callbacks.ModelCheckpoint(
                save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Custom callback for progress monitoring
            TrainingProgressCallback()
        ]
        
        print(f"\n🎯 Beginning training with {len(callbacks_list)} callbacks...")
        
        # Train the model
        start_time = datetime.now()
        
        history = self.model.fit(
            X_scaled, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks_list,
            verbose=1,
            shuffle=True
        )
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        print(f"\n✅ Training completed!")
        print(f"⏱️  Total training time: {training_duration}")
        print(f"📈 Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"📊 Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        
        # Store training history
        self.training_history = history
        
        # Calculate and store metrics
        self._calculate_training_metrics(history)
        
        # Save training artifacts
        self._save_training_artifacts(save_path, history)
        
        return history
    
    def _calculate_training_metrics(self, history):
        """Calculate comprehensive training metrics."""
        self.model_metrics = {
            'final_train_accuracy': history.history['accuracy'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1],
            'best_val_accuracy': max(history.history['val_accuracy']),
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'best_val_loss': min(history.history['val_loss']),
            'epochs_trained': len(history.history['accuracy']),
            'overfitting_score': history.history['accuracy'][-1] - history.history['val_accuracy'][-1]
        }
    
    def _save_training_artifacts(self, model_path, history):
        """Save all training artifacts."""
        base_name = model_path.replace('.h5', '')
        
        # Save scaler
        with open(f'{base_name}_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save team encoder
        with open(f'{base_name}_team_encoder.pkl', 'wb') as f:
            pickle.dump(self.team_encoder, f)
        
        # Save training history
        with open(f'{base_name}_history.pkl', 'wb') as f:
            pickle.dump(history.history, f)
        
        # Save model metrics
        with open(f'{base_name}_metrics.pkl', 'wb') as f:
            pickle.dump(self.model_metrics, f)
        
        print(f"💾 Saved training artifacts to {base_name}_*")
    
    def plot_training_progress(self, save_plot=True):
        """
        Plot comprehensive training progress visualization.
        """
        if self.training_history is None:
            print("❌ No training history available. Train the model first.")
            return
        
        history = self.training_history.history
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Neural Network Training Progress', fontsize=16, fontweight='bold')
        
        # Plot 1: Training & Validation Accuracy
        axes[0, 0].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0, 0].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy Progress')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Training & Validation Loss
        axes[0, 1].plot(history['loss'], label='Training Loss', linewidth=2)
        axes[0, 1].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss Progress')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Learning Rate (if available)
        if 'lr' in history:
            axes[1, 0].plot(history['lr'], label='Learning Rate', linewidth=2, color='red')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Learning Rate\nNot Tracked', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Plot 4: Training Summary Statistics
        epochs = len(history['accuracy'])
        best_val_acc = max(history['val_accuracy'])
        best_epoch = history['val_accuracy'].index(best_val_acc) + 1
        final_acc = history['val_accuracy'][-1]
        
        axes[1, 1].axis('off')
        summary_text = f"""
Training Summary:
• Total Epochs: {epochs}
• Best Val Accuracy: {best_val_acc:.4f}
• Best Epoch: {best_epoch}
• Final Val Accuracy: {final_acc:.4f}
• Architecture: {self.architecture.upper()}
• Overfitting: {self.model_metrics.get('overfitting_score', 0):.4f}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, 
                        fontsize=12, verticalalignment='center', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(f'neural_training_progress_{self.architecture}.png', 
                       dpi=300, bbox_inches='tight')
            print("📊 Training progress plot saved!")
        
        plt.show()
    
    def evaluate_model(self, X_test, y_test):
        """
        Comprehensive model evaluation.
        """
        print("🔍 Evaluating neural network model...")
        
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        predictions = self.model.predict(X_test_scaled)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, predicted_classes)
        
        print(f"🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        class_names = ['Away Win', 'Draw', 'Home Win']
        report = classification_report(y_test, predicted_classes, 
                                     target_names=class_names, output_dict=True)
        
        print("\n📊 Detailed Classification Report:")
        print(classification_report(y_test, predicted_classes, target_names=class_names))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, predicted_classes)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Neural Network Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'neural_confusion_matrix_{self.architecture}.png', dpi=300)
        plt.show()
        
        return accuracy, report, cm


class TrainingProgressCallback(callbacks.Callback):
    """Custom callback to monitor training progress in real-time."""
    
    def __init__(self):
        super().__init__()
        self.epoch_start_time = None
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = datetime.now()
        if epoch % 10 == 0:  # Print progress every 10 epochs
            print(f"\n🔄 Starting Epoch {epoch + 1}...")
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:  # Print detailed progress every 10 epochs
            duration = datetime.now() - self.epoch_start_time
            print(f"✅ Epoch {epoch + 1} complete in {duration.total_seconds():.2f}s")
            print(f"   Training Accuracy: {logs.get('accuracy', 0):.4f}")
            print(f"   Validation Accuracy: {logs.get('val_accuracy', 0):.4f}")
            print(f"   Training Loss: {logs.get('loss', 0):.4f}")
            print(f"   Validation Loss: {logs.get('val_loss', 0):.4f}")


def train_neural_network_system():
    """
    Main function to train the complete neural network system.
    """
    print("🚀 ADVANCED NEURAL NETWORK TRAINING SYSTEM")
    print("=" * 60)
    
    # Load data from database
    print("📊 Loading soccer match data...")
    conn = get_connection('soccer_stats')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM epl_matches ORDER BY match_date")
    matches_data = cursor.fetchall()
    cursor.close()
    conn.close()
    
    print(f"✅ Loaded {len(matches_data)} matches from database")
    
    # Train different neural network architectures
    architectures = ['deep_mlp', 'wide_deep']  # Start with these two
    
    for arch in architectures:
        print(f"\n" + "="*60)
        print(f"🏗️  Training {arch.upper()} Architecture")
        print("="*60)
        
        # Create neural network
        nn = AdvancedSoccerNeuralNetwork(architecture=arch)
        
        # Create features
        X, y, team_features = nn.create_advanced_features(matches_data)
        
        if len(X) == 0:
            print(f"❌ No features created for {arch}. Skipping...")
            continue
        
        print(f"📊 Dataset: {len(X)} samples, {X.shape[1]} features")
        print(f"📋 Class distribution: {np.bincount(y)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Build model
        nn.build_neural_network(X.shape[1])
        
        # Train with extended training
        history = nn.train_extended(
            X_train, y_train,
            epochs=150,  # Extended training
            batch_size=64,
            validation_split=0.25,
            patience=20,
            save_path=f'neural_model_{arch}.h5'
        )
        
        # Plot training progress
        nn.plot_training_progress(save_plot=True)
        
        # Evaluate model
        accuracy, report, cm = nn.evaluate_model(X_test, y_test)
        
        print(f"\n🏆 {arch.upper()} Final Results:")
        print(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Best Val Accuracy: {nn.model_metrics['best_val_accuracy']:.4f}")
        print(f"   Epochs Trained: {nn.model_metrics['epochs_trained']}")
        
        print(f"\n✅ {arch.upper()} training completed!")
    
    print(f"\n" + "="*60)
    print("🎉 NEURAL NETWORK TRAINING SYSTEM COMPLETE!")
    print("="*60)
    print("All models trained and saved. Check the generated files:")
    print("• neural_model_*.h5 - Trained models")
    print("• *_scaler.pkl - Feature scalers")
    print("• *_history.pkl - Training histories")
    print("• *.png - Training plots and confusion matrices")


if __name__ == "__main__":
    train_neural_network_system()