"""
Advanced Model Training Module
Trains multiple ML models with hyperparameter optimization and comprehensive evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import optuna
import joblib
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our preprocessing module
import sys
sys.path.append('src')
from data.preprocessing import DataPreprocessor

class ModelTrainer:
    """Comprehensive model training with multiple algorithms and optimization."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.preprocessor = None
        self.feature_names = None
        self.results = {}
        
    def load_and_prepare_data(self, data_path: str):
        """Load and prepare data for training."""
        print("Loading and preparing data...")
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Load and process data
        df = preprocessor.load_data(data_path)
        df_clean = preprocessor.clean_data(df)
        df_engineered = preprocessor.engineer_features(df_clean)
        
        # Prepare features
        X, y = preprocessor.prepare_features(df_engineered)
        
        # Fit preprocessor
        preprocessor.fit_preprocessor(X)
        X_transformed = preprocessor.transform_data(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.preprocessor = preprocessor
        self.feature_names = preprocessor.feature_names
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Target distribution in training set:\n{y_train.value_counts()}")
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, n_trials=50):
        """Train Random Forest with Optuna optimization."""
        print("\n=== Training Random Forest ===")
        
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': self.random_state
            }
            
            # Create model
            model = RandomForestClassifier(**params)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                scoring='f1_weighted'
            )
            
            return cv_scores.mean()
        
        # Optimize hyperparameters
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Train final model with best parameters
        best_params = study.best_params
        best_params['random_state'] = self.random_state
        
        rf_model = RandomForestClassifier(**best_params)
        rf_model.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred = rf_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred, average='weighted'
        )
        
        self.models['RandomForest'] = {
            'model': rf_model,
            'params': best_params,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'feature_importance': rf_model.feature_importances_
        }
        
        print(f"Random Forest - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        print(f"Best parameters: {best_params}")
        
        return rf_model
    
    def train_xgboost(self, n_trials=50):
        """Train XGBoost with Optuna optimization."""
        print("\n=== Training XGBoost ===")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': self.random_state
            }
            
            model = xgb.XGBClassifier(**params)
            
            # Encode labels for XGBoost
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(self.y_train)
            
            cv_scores = cross_val_score(
                model, self.X_train, y_encoded,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                scoring='f1_weighted'
            )
            
            return cv_scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        best_params['random_state'] = self.random_state
        
        xgb_model = xgb.XGBClassifier(**best_params)
        
        # Encode labels for training
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(self.y_train)
        y_test_encoded = label_encoder.transform(self.y_test)
        
        xgb_model.fit(self.X_train, y_train_encoded)
        
        y_pred_encoded = xgb_model.predict(self.X_test)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred, average='weighted'
        )
        
        self.models['XGBoost'] = {
            'model': xgb_model,
            'params': best_params,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'feature_importance': xgb_model.feature_importances_
        }
        
        print(f"XGBoost - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        print(f"Best parameters: {best_params}")
        
        return xgb_model
    
    def train_lightgbm(self, n_trials=50):
        """Train LightGBM with Optuna optimization."""
        print("\n=== Training LightGBM ===")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': self.random_state
            }
            
            model = lgb.LGBMClassifier(**params)
            
            # Encode labels for LightGBM
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(self.y_train)
            
            cv_scores = cross_val_score(
                model, self.X_train, y_encoded,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                scoring='f1_weighted'
            )
            
            return cv_scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        best_params['random_state'] = self.random_state
        
        lgb_model = lgb.LGBMClassifier(**best_params)
        
        # Encode labels for training
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(self.y_train)
        y_test_encoded = label_encoder.transform(self.y_test)
        
        lgb_model.fit(self.X_train, y_train_encoded)
        
        y_pred_encoded = lgb_model.predict(self.X_test)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred, average='weighted'
        )
        
        self.models['LightGBM'] = {
            'model': lgb_model,
            'params': best_params,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'feature_importance': lgb_model.feature_importances_
        }
        
        print(f"LightGBM - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        print(f"Best parameters: {best_params}")
        
        return lgb_model
    
    def train_all_models(self, n_trials=50):
        """Train all models and select the best one."""
        print("Training all models...")
        
        # Train each model
        self.train_random_forest(n_trials)
        self.train_xgboost(n_trials)
        self.train_lightgbm(n_trials)
        
        # Find best model
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['f1_score'])
        self.best_model = self.models[best_model_name]['model']
        self.best_score = self.models[best_model_name]['f1_score']
        
        print(f"\n=== Model Comparison ===")
        for name, results in self.models.items():
            print(f"{name}:")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  Precision: {results['precision']:.4f}")
            print(f"  Recall: {results['recall']:.4f}")
            print(f"  F1-Score: {results['f1_score']:.4f}")
        
        print(f"\nBest model: {best_model_name} (F1: {self.best_score:.4f})")
        
        return self.best_model
    
    def evaluate_model(self, model, model_name):
        """Comprehensive model evaluation."""
        print(f"\n=== Evaluating {model_name} ===")
        
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)
        
        # Ensure both y_test and y_pred are strings for consistent evaluation
        y_test_str = self.y_test.astype(str)
        y_pred_str = y_pred.astype(str)
        
        # Metrics
        accuracy = accuracy_score(y_test_str, y_pred_str)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test_str, y_pred_str, average='weighted'
        )
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test_str, y_pred_str))
        
        # Confusion matrix
        cm = confusion_matrix(y_test_str, y_pred_str)
        print(f"\nConfusion Matrix:\n{cm}")
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm
        }
        
        return self.results[model_name]
    
    def get_feature_importance(self, model, model_name):
        """Get feature importance for a model."""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            
            if self.feature_names and len(self.feature_names) == len(importance):
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                print(f"\n=== {model_name} Feature Importance (Top 10) ===")
                print(importance_df.head(10))
                
                return importance_df
            else:
                print(f"\n=== {model_name} Feature Importance ===")
                print(f"Number of features: {len(importance)}")
                print(f"Top 10 feature importances: {importance[:10]}")
        
        return None
    
    def save_models(self, output_dir='models/saved_models'):
        """Save all trained models and metadata."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save preprocessor
        if self.preprocessor:
            joblib.dump(self.preprocessor, f'{output_dir}/preprocessor.joblib')
        
        # Save models
        for name, model_info in self.models.items():
            model_path = f'{output_dir}/{name.lower()}_model.joblib'
            joblib.dump(model_info['model'], model_path)
            print(f"Saved {name} model to {model_path}")
        
        # Save best model
        if self.best_model:
            joblib.dump(self.best_model, f'{output_dir}/best_model.joblib')
            print(f"Saved best model to {output_dir}/best_model.joblib")
        
        # Save results and metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'best_model': max(self.models.keys(), key=lambda x: self.models[x]['f1_score']),
            'best_score': self.best_score,
            'model_results': {
                name: {
                    'accuracy': info['accuracy'],
                    'precision': info['precision'],
                    'recall': info['recall'],
                    'f1_score': info['f1_score']
                } for name, info in self.models.items()
            },
            'data_info': {
                'train_shape': self.X_train.shape,
                'test_shape': self.X_test.shape,
                'feature_count': len(self.feature_names) if self.feature_names else 0
            }
        }
        
        with open(f'{output_dir}/training_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved training metadata to {output_dir}/training_metadata.json")
    
    def create_ensemble_model(self):
        """Create an ensemble of the best models."""
        print("\n=== Creating Ensemble Model ===")
        
        # Select top 2 models based on F1 score
        sorted_models = sorted(
            self.models.items(), 
            key=lambda x: x[1]['f1_score'], 
            reverse=True
        )
        
        top_models = [model_info['model'] for _, model_info in sorted_models[:2]]
        
        # Create ensemble predictions
        ensemble_probs = []
        
        for model in top_models:
            probs = model.predict_proba(self.X_test)
            ensemble_probs.append(probs)
        
        # Average probabilities
        ensemble_prob = np.mean(ensemble_probs, axis=0)
        
        # Convert to class predictions
        ensemble_pred_class = np.argmax(ensemble_prob, axis=1)
        
        # Convert numeric predictions back to string labels
        label_encoder = LabelEncoder()
        label_encoder.fit(self.y_train)
        ensemble_pred = label_encoder.inverse_transform(ensemble_pred_class)
        
        # Evaluate ensemble
        accuracy = accuracy_score(self.y_test, ensemble_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, ensemble_pred, average='weighted'
        )
        
        print(f"Ensemble - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': ensemble_pred,
            'probabilities': ensemble_prob
        }

def main():
    """Main training function."""
    trainer = ModelTrainer(random_state=42)
    
    # Load and prepare data
    trainer.load_and_prepare_data('data/raw/enhanced_sti_data.csv')
    
    # Train all models
    best_model = trainer.train_all_models(n_trials=30)
    
    # Evaluate best model
    best_model_name = max(trainer.models.keys(), key=lambda x: trainer.models[x]['f1_score'])
    trainer.evaluate_model(best_model, best_model_name)
    
    # Get feature importance
    trainer.get_feature_importance(best_model, best_model_name)
    
    # Create ensemble
    ensemble_results = trainer.create_ensemble_model()
    
    # Save models
    trainer.save_models()
    
    print("\n=== Training Complete ===")
    print(f"Best model: {best_model_name}")
    print(f"Best F1 score: {trainer.best_score:.4f}")

if __name__ == "__main__":
    main() 