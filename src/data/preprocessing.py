"""
Data Preprocessing Module
Handles data cleaning, feature engineering, and preparation for ML models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Comprehensive data preprocessing for STI prediction."""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.preprocessor = None
        self.feature_names = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load and perform initial data inspection."""
        print(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        print(f"Missing values:\n{df.isnull().sum()}")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataset by handling missing values and outliers."""
        print("Cleaning data...")
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed_duplicates = initial_rows - len(df_clean)
        print(f"Removed {removed_duplicates} duplicate rows")
        
        # Handle outliers in numeric columns
        df_clean = self._handle_outliers(df_clean)
        
        # Validate data ranges
        df_clean = self._validate_data_ranges(df_clean)
        
        print(f"Cleaned data shape: {df_clean.shape}")
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate strategies."""
        # For categorical columns, fill with mode
        categorical_cols = ['sexually_active', 'partner_history', 'infection_location', 
                          'antibody_test', 'antibiotic_treatment']
        
        for col in categorical_cols:
            if col in df.columns and df[col].isnull().any():
                mode_value = df[col].mode()[0]
                df[col] = df[col].fillna(mode_value)
                print(f"Filled missing values in {col} with mode: {mode_value}")
        
        # For numeric columns, fill with median
        numeric_cols = ['age', 'body_temp', 'white_blood_cell_count']
        for col in numeric_cols:
            if col in df.columns and df[col].isnull().any():
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                print(f"Filled missing values in {col} with median: {median_value}")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method."""
        numeric_cols = ['age', 'body_temp', 'white_blood_cell_count']
        
        for col in numeric_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                
                outliers_removed = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers_removed > 0:
                    print(f"Capped {outliers_removed} outliers in {col}")
        
        return df
    
    def _validate_data_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and correct data ranges for medical data."""
        # Age validation (16-100)
        if 'age' in df.columns:
            df['age'] = df['age'].clip(16, 100)
        
        # Body temperature validation (35-42)
        if 'body_temp' in df.columns:
            df['body_temp'] = df['body_temp'].clip(35.0, 42.0)
        
        # White blood cell count validation (1000-20000)
        if 'white_blood_cell_count' in df.columns:
            df['white_blood_cell_count'] = df['white_blood_cell_count'].clip(1000, 20000)
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features for better model performance."""
        print("Engineering features...")
        
        df_engineered = df.copy()
        
        # Age categories
        df_engineered['age_group'] = pd.cut(
            df_engineered['age'], 
            bins=[0, 18, 25, 35, 50, 100], 
            labels=['Teen', 'Young Adult', 'Adult', 'Middle Age', 'Senior']
        )
        
        # Temperature categories
        df_engineered['temp_category'] = pd.cut(
            df_engineered['body_temp'],
            bins=[35, 36.5, 37.5, 38.5, 42],
            labels=['Low', 'Normal', 'Elevated', 'High']
        )
        
        # WBC categories
        df_engineered['wbc_category'] = pd.cut(
            df_engineered['white_blood_cell_count'],
            bins=[0, 4000, 11000, 20000],
            labels=['Low', 'Normal', 'High']
        )
        
        # Risk factors
        df_engineered['high_risk_age'] = (df_engineered['age'] >= 18) & (df_engineered['age'] <= 35)
        df_engineered['high_risk_activity'] = (df_engineered['sexually_active'] == 'Yes') & \
                                            (df_engineered['partner_history'] == 'Non-monogamous')
        
        # Symptom severity (if symptoms column exists)
        if 'sti_symptoms' in df_engineered.columns:
            df_engineered['symptom_count'] = df_engineered['sti_symptoms'].str.count(',') + 1
            df_engineered['symptom_severity'] = pd.cut(
                df_engineered['symptom_count'],
                bins=[0, 1, 2, 3, 10],
                labels=['Mild', 'Moderate', 'Severe', 'Very Severe']
            )
        
        # Medical indicators
        df_engineered['fever'] = df_engineered['body_temp'] > 37.5
        df_engineered['elevated_wbc'] = df_engineered['white_blood_cell_count'] > 11000
        
        print(f"Added {len(df_engineered.columns) - len(df.columns)} new features")
        return df_engineered
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'sti_name') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for modeling."""
        print("Preparing features for modeling...")
        
        # Remove target and non-predictive columns
        columns_to_drop = [target_col, 'sti_symptoms', 'timestamp']
        feature_cols = [col for col in df.columns if col not in columns_to_drop]
        
        X = df[feature_cols].copy()
        y = df[target_col]
        
        print(f"Feature columns: {feature_cols}")
        print(f"Target distribution:\n{y.value_counts()}")
        
        return X, y
    
    def create_preprocessing_pipeline(self, X: pd.DataFrame) -> ColumnTransformer:
        """Create a comprehensive preprocessing pipeline."""
        print("Creating preprocessing pipeline...")
        
        # Define feature types
        numeric_features = ['age', 'body_temp', 'white_blood_cell_count']
        categorical_features = ['sexually_active', 'partner_history', 'infection_location', 
                             'antibody_test', 'antibiotic_treatment', 'age_group', 
                             'temp_category', 'wbc_category', 'symptom_severity']
        boolean_features = ['high_risk_age', 'high_risk_activity', 'fever', 'elevated_wbc']
        
        # Ensure all features exist
        numeric_features = [col for col in numeric_features if col in X.columns]
        categorical_features = [col for col in categorical_features if col in X.columns]
        boolean_features = [col for col in boolean_features if col in X.columns]
        
        # Create transformers
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # For boolean features, we'll handle them differently
        boolean_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=False))
        ])
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )
        
        self.preprocessor = preprocessor
        print(f"Created preprocessor with {len(numeric_features)} numeric, "
              f"{len(categorical_features)} categorical, and {len(boolean_features)} boolean features")
        
        return preprocessor
    
    def fit_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Fit the preprocessing pipeline."""
        if self.preprocessor is None:
            self.create_preprocessing_pipeline(X)
        
        print("Fitting preprocessor...")
        self.preprocessor.fit(X)
        
        # Get feature names after preprocessing
        feature_names = []
        for name, trans, cols in self.preprocessor.transformers:
            if hasattr(trans, 'get_feature_names_out'):
                try:
                    feature_names.extend(trans.get_feature_names_out(cols))
                except:
                    # Fallback to original column names
                    feature_names.extend(cols)
            else:
                feature_names.extend(cols)
        
        self.feature_names = feature_names
        print(f"Preprocessor fitted with {len(feature_names)} features")
        
        return self.preprocessor
    
    def transform_data(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted preprocessor."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_preprocessor first.")
        
        return self.preprocessor.transform(X)
    
    def save_preprocessor(self, filepath: str):
        """Save the fitted preprocessor."""
        if self.preprocessor is None:
            raise ValueError("No preprocessor to save. Fit preprocessor first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.preprocessor, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str):
        """Load a fitted preprocessor."""
        self.preprocessor = joblib.load(filepath)
        print(f"Preprocessor loaded from {filepath}")
    
    def get_feature_importance_dataframe(self, feature_importance: np.ndarray) -> pd.DataFrame:
        """Create a DataFrame with feature importance scores."""
        if self.feature_names is None:
            raise ValueError("Feature names not available. Fit preprocessor first.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df

def main():
    """Example usage of the DataPreprocessor."""
    preprocessor = DataPreprocessor()
    
    # Load data
    df = preprocessor.load_data('data/raw/enhanced_sti_data.csv')
    
    # Clean data
    df_clean = preprocessor.clean_data(df)
    
    # Engineer features
    df_engineered = preprocessor.engineer_features(df_clean)
    
    # Prepare features
    X, y = preprocessor.prepare_features(df_engineered)
    
    # Create and fit preprocessor
    preprocessor.fit_preprocessor(X)
    
    # Transform data
    X_transformed = preprocessor.transform_data(X)
    
    print(f"Final transformed data shape: {X_transformed.shape}")
    
    # Save preprocessor
    preprocessor.save_preprocessor('models/saved_models/preprocessor.joblib')
    
    # Save processed data
    df_engineered.to_csv('data/processed/processed_sti_data.csv', index=False)
    print("Processed data saved to data/processed/processed_sti_data.csv")

if __name__ == "__main__":
    main() 