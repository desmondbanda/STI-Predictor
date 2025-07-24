"""
Unit tests for STI Predictor models and data processing.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append('src')

from data.preprocessing import DataPreprocessor
from models.train_models import ModelTrainer

class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class."""
    
    def setUp(self):
        """Set up test data."""
        self.preprocessor = DataPreprocessor()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'sti_name': ['Chlamydia', 'Gonorrhea', 'Syphilis', 'HPV', 'HIV'],
            'sti_symptoms': ['Symptom1', 'Symptom2', 'Symptom3', 'Symptom4', 'Symptom5'],
            'sexually_active': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'body_temp': [36.8, 37.2, 36.5, 37.8, 38.1],
            'white_blood_cell_count': [7500, 8500, 9200, 7800, 6500],
            'partner_history': ['Monogamous', 'Non-monogamous', 'Monogamous', 'Non-monogamous', 'Monogamous'],
            'infection_location': ['Genital', 'Oral', 'Anal', 'Throat', 'Other'],
            'antibody_test': ['Positive', 'Negative', 'Inconclusive', 'Positive', 'Negative'],
            'antibiotic_treatment': ['Yes', 'No', 'Yes', 'No', 'Yes']
        })
    
    def test_load_data(self):
        """Test data loading functionality."""
        # Mock file reading
        with patch('pandas.read_csv') as mock_read:
            mock_read.return_value = self.sample_data
            result = self.preprocessor.load_data('dummy_path.csv')
            
            self.assertEqual(len(result), 5)
            self.assertEqual(list(result.columns), list(self.sample_data.columns))
    
    def test_clean_data(self):
        """Test data cleaning functionality."""
        # Add some missing values and outliers
        dirty_data = self.sample_data.copy()
        dirty_data.loc[0, 'age'] = np.nan
        dirty_data.loc[1, 'body_temp'] = 50.0  # Outlier
        
        cleaned_data = self.preprocessor.clean_data(dirty_data)
        
        # Check that missing values are handled
        self.assertEqual(cleaned_data.isnull().sum().sum(), 0)
        
        # Check that outliers are capped
        self.assertLessEqual(cleaned_data['body_temp'].max(), 42.0)
    
    def test_engineer_features(self):
        """Test feature engineering functionality."""
        engineered_data = self.preprocessor.engineer_features(self.sample_data)
        
        # Check that new features are added
        expected_new_features = ['age_group', 'temp_category', 'wbc_category', 
                               'high_risk_age', 'high_risk_activity', 'fever', 'elevated_wbc']
        
        for feature in expected_new_features:
            self.assertIn(feature, engineered_data.columns)
    
    def test_prepare_features(self):
        """Test feature preparation for modeling."""
        X, y = self.preprocessor.prepare_features(self.sample_data)
        
        # Check that target is separated
        self.assertIn('sti_name', y.name)
        self.assertNotIn('sti_name', X.columns)
        
        # Check that symptoms column is removed
        self.assertNotIn('sti_symptoms', X.columns)
    
    def test_create_preprocessing_pipeline(self):
        """Test preprocessing pipeline creation."""
        engineered_data = self.preprocessor.engineer_features(self.sample_data)
        X, y = self.preprocessor.prepare_features(engineered_data)
        
        pipeline = self.preprocessor.create_preprocessing_pipeline(X)
        
        self.assertIsNotNone(pipeline)
        self.assertEqual(len(pipeline.transformers), 3)  # num, cat, bool
    
    def test_fit_preprocessor(self):
        """Test preprocessor fitting."""
        engineered_data = self.preprocessor.engineer_features(self.sample_data)
        X, y = self.preprocessor.prepare_features(engineered_data)
        
        fitted_preprocessor = self.preprocessor.fit_preprocessor(X)
        
        self.assertIsNotNone(fitted_preprocessor)
        self.assertIsNotNone(self.preprocessor.feature_names)
    
    def test_transform_data(self):
        """Test data transformation."""
        engineered_data = self.preprocessor.engineer_features(self.sample_data)
        X, y = self.preprocessor.prepare_features(engineered_data)
        
        self.preprocessor.fit_preprocessor(X)
        transformed_data = self.preprocessor.transform_data(X)
        
        self.assertIsInstance(transformed_data, np.ndarray)
        self.assertEqual(transformed_data.shape[0], X.shape[0])

class TestModelTrainer(unittest.TestCase):
    """Test cases for ModelTrainer class."""
    
    def setUp(self):
        """Set up test data and trainer."""
        self.trainer = ModelTrainer(random_state=42)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'age': [25, 30, 35, 40, 45] * 20,  # 100 samples
            'sti_name': ['Chlamydia', 'Gonorrhea', 'Syphilis', 'HPV', 'HIV'] * 20,
            'sti_symptoms': ['Symptom1'] * 100,
            'sexually_active': ['Yes', 'No'] * 50,
            'body_temp': np.random.normal(37, 0.5, 100),
            'white_blood_cell_count': np.random.normal(7500, 2000, 100),
            'partner_history': ['Monogamous', 'Non-monogamous'] * 50,
            'infection_location': ['Genital', 'Oral', 'Anal', 'Throat', 'Other'] * 20,
            'antibody_test': ['Positive', 'Negative', 'Inconclusive'] * 33 + ['Positive'],
            'antibiotic_treatment': ['Yes', 'No'] * 50
        })
    
    @patch('src.data.preprocessing.DataPreprocessor')
    def test_load_and_prepare_data(self, mock_preprocessor):
        """Test data loading and preparation."""
        # Mock the preprocessor
        mock_prep = MagicMock()
        mock_preprocessor.return_value = mock_prep
        
        # Mock the data processing steps
        mock_prep.load_data.return_value = self.sample_data
        mock_prep.clean_data.return_value = self.sample_data
        mock_prep.engineer_features.return_value = self.sample_data
        mock_prep.prepare_features.return_value = (self.sample_data.drop('sti_name', axis=1), self.sample_data['sti_name'])
        mock_prep.fit_preprocessor.return_value = MagicMock()
        mock_prep.transform_data.return_value = np.random.rand(80, 10)  # Mock transformed data
        
        # Mock train_test_split
        with patch('sklearn.model_selection.train_test_split') as mock_split:
            mock_split.return_value = (
                np.random.rand(80, 10),  # X_train
                np.random.rand(20, 10),  # X_test
                pd.Series(['Chlamydia'] * 80),  # y_train
                pd.Series(['Gonorrhea'] * 20)   # y_test
            )
            
            X_train, X_test, y_train, y_test = self.trainer.load_and_prepare_data('dummy_path.csv')
            
            self.assertEqual(X_train.shape[0], 80)
            self.assertEqual(X_test.shape[0], 20)
    
    def test_calculate_risk_score(self):
        """Test risk score calculation."""
        user_data = {
            'age': 25,
            'sexually_active': 'Yes',
            'partner_history': 'Non-monogamous',
            'body_temp': 38.0,
            'white_blood_cell_count': 12000
        }
        
        risk_score = self.trainer.calculate_risk_score(user_data)
        
        # Should be high risk due to multiple factors
        self.assertGreater(risk_score, 5)
        self.assertLessEqual(risk_score, 10)
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = ['Chlamydia', 'Gonorrhea']
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
        
        # Mock test data
        self.trainer.X_test = np.random.rand(2, 10)
        self.trainer.y_test = pd.Series(['Chlamydia', 'Gonorrhea'])
        
        results = self.trainer.evaluate_model(mock_model, 'TestModel')
        
        self.assertIn('accuracy', results)
        self.assertIn('precision', results)
        self.assertIn('recall', results)
        self.assertIn('f1_score', results)
    
    def test_get_feature_importance(self):
        """Test feature importance extraction."""
        # Mock model with feature importances
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.3, 0.2, 0.1, 0.4])
        
        # Mock feature names
        self.trainer.feature_names = ['feature1', 'feature2', 'feature3', 'feature4']
        
        importance_df = self.trainer.get_feature_importance(mock_model, 'TestModel')
        
        self.assertIsInstance(importance_df, pd.DataFrame)
        self.assertEqual(len(importance_df), 4)
        self.assertIn('feature', importance_df.columns)
        self.assertIn('importance', importance_df.columns)

class TestDataQuality(unittest.TestCase):
    """Test cases for data quality checks."""
    
    def test_data_validation(self):
        """Test data validation rules."""
        # Test age validation
        invalid_age_data = pd.DataFrame({
            'age': [15, 25, 35, 101],  # Invalid ages
            'body_temp': [36.8, 37.2, 36.5, 37.8],
            'white_blood_cell_count': [7500, 8500, 9200, 7800]
        })
        
        preprocessor = DataPreprocessor()
        validated_data = preprocessor._validate_data_ranges(invalid_age_data)
        
        # Check that ages are clipped to valid range
        self.assertGreaterEqual(validated_data['age'].min(), 16)
        self.assertLessEqual(validated_data['age'].max(), 100)
    
    def test_outlier_detection(self):
        """Test outlier detection and handling."""
        # Create data with outliers
        outlier_data = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'body_temp': [36.8, 37.2, 50.0, 37.8, 38.1],  # Outlier at 50.0
            'white_blood_cell_count': [7500, 8500, 9200, 7800, 6500]
        })
        
        preprocessor = DataPreprocessor()
        cleaned_data = preprocessor._handle_outliers(outlier_data)
        
        # Check that outliers are capped
        self.assertLessEqual(cleaned_data['body_temp'].max(), 42.0)
    
    def test_missing_value_handling(self):
        """Test missing value handling."""
        # Create data with missing values
        missing_data = pd.DataFrame({
            'age': [25, np.nan, 35, 40, 45],
            'sexually_active': ['Yes', 'No', np.nan, 'No', 'Yes'],
            'body_temp': [36.8, 37.2, 36.5, 37.8, 38.1],
            'white_blood_cell_count': [7500, 8500, 9200, 7800, 6500]
        })
        
        preprocessor = DataPreprocessor()
        cleaned_data = preprocessor._handle_missing_values(missing_data)
        
        # Check that missing values are filled
        self.assertEqual(cleaned_data.isnull().sum().sum(), 0)

class TestModelPerformance(unittest.TestCase):
    """Test cases for model performance metrics."""
    
    def test_accuracy_calculation(self):
        """Test accuracy calculation."""
        y_true = ['Chlamydia', 'Gonorrhea', 'Syphilis', 'HPV', 'HIV']
        y_pred = ['Chlamydia', 'Gonorrhea', 'Syphilis', 'HPV', 'HIV']  # Perfect prediction
        
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_true, y_pred)
        
        self.assertEqual(accuracy, 1.0)
    
    def test_f1_score_calculation(self):
        """Test F1 score calculation."""
        y_true = ['Chlamydia', 'Gonorrhea', 'Syphilis', 'HPV', 'HIV']
        y_pred = ['Chlamydia', 'Gonorrhea', 'Syphilis', 'HPV', 'HIV']
        
        from sklearn.metrics import f1_score
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        self.assertEqual(f1, 1.0)
    
    def test_confusion_matrix(self):
        """Test confusion matrix calculation."""
        y_true = ['Chlamydia', 'Gonorrhea', 'Syphilis', 'HPV', 'HIV']
        y_pred = ['Chlamydia', 'Gonorrhea', 'Syphilis', 'HPV', 'HIV']
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Should be diagonal matrix for perfect predictions
        self.assertEqual(cm.shape, (5, 5))
        self.assertEqual(np.trace(cm), 5)  # All diagonal elements should be 1

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2) 