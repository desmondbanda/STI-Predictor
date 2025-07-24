"""
Enhanced STI Data Generator
Generates realistic synthetic data for STI prediction model training.
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class STIDataGenerator:
    """Generate realistic synthetic STI data for ML training."""
    
    def __init__(self, seed=42):
        self.fake = Faker()
        Faker.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # STI definitions with realistic patterns
        self.sti_definitions = {
            'Chlamydia': {
                'age_range': (16, 45),
                'symptoms': ['Unusual discharge', 'Painful urination', 'Lower abdominal pain'],
                'common_locations': ['Vagina', 'Penis', 'Anus', 'Throat'],
                'temp_range': (36.5, 37.8),
                'wbc_range': (4000, 12000),
                'prevalence': 0.35
            },
            'Gonorrhea': {
                'age_range': (18, 50),
                'symptoms': ['Unusual discharge', 'Painful urination', 'Genital sores'],
                'common_locations': ['Penis', 'Vagina', 'Anus', 'Throat'],
                'temp_range': (36.8, 38.2),
                'wbc_range': (5000, 15000),
                'prevalence': 0.25
            },
            'Syphilis': {
                'age_range': (20, 55),
                'symptoms': ['Genital sores', 'Rash', 'Flu-like symptoms'],
                'common_locations': ['Penis', 'Vagina', 'Mouth', 'Anus'],
                'temp_range': (36.5, 37.5),
                'wbc_range': (3500, 11000),
                'prevalence': 0.15
            },
            'HPV': {
                'age_range': (16, 60),
                'symptoms': ['Genital warts', 'Unusual discharge'],
                'common_locations': ['Penis', 'Vagina', 'Anus'],
                'temp_range': (36.0, 37.2),
                'wbc_range': (3000, 10000),
                'prevalence': 0.20
            },
            'HIV': {
                'age_range': (18, 65),
                'symptoms': ['Flu-like symptoms', 'Fatigue', 'Weight loss'],
                'common_locations': ['Systemic'],
                'temp_range': (36.8, 38.5),
                'wbc_range': (2000, 8000),
                'prevalence': 0.05
            }
        }
        
    def generate_demographics(self, n_samples):
        """Generate realistic demographic data."""
        data = []
        
        for _ in range(n_samples):
            # Age distribution based on STI prevalence
            age = np.random.normal(28, 8)
            age = max(16, min(65, int(age)))
            
            # Gender distribution
            gender = np.random.choice(['Male', 'Female'], p=[0.6, 0.4])
            
            # Sexual activity patterns
            sexually_active = np.random.choice(['Yes', 'No'], p=[0.85, 0.15])
            
            # Partner history based on age and activity
            if sexually_active == 'Yes':
                if age < 25:
                    partner_history = np.random.choice(['Monogamous', 'Non-monogamous'], p=[0.4, 0.6])
                else:
                    partner_history = np.random.choice(['Monogamous', 'Non-monogamous'], p=[0.6, 0.4])
            else:
                partner_history = 'Monogamous'
            
            data.append({
                'age': age,
                'gender': gender,
                'sexually_active': sexually_active,
                'partner_history': partner_history
            })
        
        return pd.DataFrame(data)
    
    def generate_medical_data(self, n_samples):
        """Generate realistic medical measurements."""
        data = []
        
        for _ in range(n_samples):
            # Body temperature (normal range with some variation)
            base_temp = 36.8
            temp_variation = np.random.normal(0, 0.5)
            body_temp = base_temp + temp_variation
            body_temp = max(35.5, min(39.0, body_temp))
            
            # White blood cell count (normal range 4000-11000)
            wbc_base = 7500
            wbc_variation = np.random.normal(0, 2000)
            wbc_count = wbc_base + wbc_variation
            wbc_count = max(2000, min(15000, int(wbc_count)))
            
            # Antibody test results
            antibody_test = np.random.choice(['Positive', 'Negative', 'Inconclusive'], p=[0.3, 0.6, 0.1])
            
            # Antibiotic treatment history
            antibiotic_treatment = np.random.choice(['Yes', 'No'], p=[0.4, 0.6])
            
            data.append({
                'body_temp': round(body_temp, 1),
                'white_blood_cell_count': wbc_count,
                'antibody_test': antibody_test,
                'antibiotic_treatment': antibiotic_treatment
            })
        
        return pd.DataFrame(data)
    
    def generate_sti_data(self, n_samples):
        """Generate STI-specific data with realistic patterns."""
        sti_data = []
        
        for _ in range(n_samples):
            # Select STI based on prevalence
            sti_name = np.random.choice(
                list(self.sti_definitions.keys()),
                p=[self.sti_definitions[sti]['prevalence'] for sti in self.sti_definitions.keys()]
            )
            
            sti_info = self.sti_definitions[sti_name]
            
            # Generate age-appropriate data
            age = np.random.normal(
                (sti_info['age_range'][0] + sti_info['age_range'][1]) / 2,
                (sti_info['age_range'][1] - sti_info['age_range'][0]) / 6
            )
            age = max(sti_info['age_range'][0], min(sti_info['age_range'][1], int(age)))
            
            # Generate symptoms based on STI type
            num_symptoms = np.random.poisson(2.5) + 1  # 1-4 symptoms
            symptoms = np.random.choice(sti_info['symptoms'], size=min(num_symptoms, len(sti_info['symptoms'])), replace=False)
            symptoms_str = ', '.join(symptoms)
            
            # Generate location based on STI type
            infection_location = np.random.choice(sti_info['common_locations'])
            
            # Generate medical measurements with STI-specific patterns
            temp_offset = np.random.normal(0.5, 0.3) if sti_name in ['Gonorrhea', 'HIV'] else np.random.normal(0, 0.2)
            body_temp = 36.8 + temp_offset
            body_temp = max(35.5, min(39.0, body_temp))
            
            wbc_offset = np.random.normal(2000, 1000) if sti_name in ['Gonorrhea', 'Chlamydia'] else np.random.normal(0, 500)
            wbc_count = 7500 + wbc_offset
            wbc_count = max(2000, min(15000, int(wbc_count)))
            
            # Sexual activity patterns
            sexually_active = np.random.choice(['Yes', 'No'], p=[0.9, 0.1])
            
            # Partner history
            if sexually_active == 'Yes':
                partner_history = np.random.choice(['Monogamous', 'Non-monogamous'], p=[0.3, 0.7])
            else:
                partner_history = 'Monogamous'
            
            # Antibody test patterns
            if sti_name == 'HIV':
                antibody_test = np.random.choice(['Positive', 'Negative'], p=[0.8, 0.2])
            else:
                antibody_test = np.random.choice(['Positive', 'Negative', 'Inconclusive'], p=[0.4, 0.5, 0.1])
            
            # Treatment patterns
            antibiotic_treatment = np.random.choice(['Yes', 'No'], p=[0.6, 0.4])
            
            sti_data.append({
                'age': age,
                'sti_name': sti_name,
                'sti_symptoms': symptoms_str,
                'sexually_active': sexually_active,
                'body_temp': round(body_temp, 1),
                'white_blood_cell_count': wbc_count,
                'partner_history': partner_history,
                'infection_location': infection_location,
                'antibody_test': antibody_test,
                'antibiotic_treatment': antibiotic_treatment
            })
        
        return pd.DataFrame(sti_data)
    
    def generate_comprehensive_dataset(self, n_samples=1000):
        """Generate comprehensive STI dataset with realistic patterns."""
        print(f"Generating {n_samples} synthetic STI records...")
        
        # Generate STI-specific data
        df = self.generate_sti_data(n_samples)
        
        # Add additional features
        df['timestamp'] = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
        df['risk_score'] = self.calculate_risk_score(df)
        df['symptom_severity'] = df['sti_symptoms'].str.count(',').add(1)
        
        # Add some noise and realistic variations
        df = self.add_realistic_noise(df)
        
        print(f"Generated dataset with {len(df)} records")
        print(f"STI distribution:\n{df['sti_name'].value_counts()}")
        
        return df
    
    def calculate_risk_score(self, df):
        """Calculate risk score based on multiple factors."""
        risk_scores = []
        
        for _, row in df.iterrows():
            score = 0
            
            # Age factor
            if 18 <= row['age'] <= 35:
                score += 2
            elif 16 <= row['age'] <= 17 or 36 <= row['age'] <= 45:
                score += 1
            
            # Sexual activity
            if row['sexually_active'] == 'Yes':
                score += 2
            
            # Partner history
            if row['partner_history'] == 'Non-monogamous':
                score += 2
            
            # Medical factors
            if row['body_temp'] > 37.5:
                score += 1
            if row['white_blood_cell_count'] > 10000:
                score += 1
            
            # STI-specific risk
            sti_risk = {'HIV': 5, 'Syphilis': 4, 'Gonorrhea': 3, 'Chlamydia': 2, 'HPV': 1}
            score += sti_risk.get(row['sti_name'], 0)
            
            risk_scores.append(min(score, 10))  # Cap at 10
        
        return risk_scores
    
    def add_realistic_noise(self, df):
        """Add realistic noise and variations to the data."""
        # Add some missing values (realistic for medical data)
        missing_mask = np.random.random(len(df)) < 0.05
        df.loc[missing_mask, 'antibody_test'] = 'Inconclusive'
        
        # Add some outliers (realistic medical cases)
        outlier_mask = np.random.random(len(df)) < 0.02
        df.loc[outlier_mask, 'body_temp'] = np.random.uniform(39.0, 40.0, sum(outlier_mask))
        
        return df
    
    def save_dataset(self, df, filepath):
        """Save dataset to CSV with metadata."""
        df.to_csv(filepath, index=False)
        
        # Save metadata
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'total_records': len(df),
            'sti_distribution': df['sti_name'].value_counts().to_dict(),
            'features': list(df.columns),
            'data_quality': {
                'missing_values': df.isnull().sum().to_dict(),
                'duplicates': int(df.duplicated().sum())
            }
        }
        
        metadata_file = filepath.replace('.csv', '_metadata.json')
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset saved to {filepath}")
        print(f"Metadata saved to {metadata_file}")

def main():
    """Generate and save the enhanced STI dataset."""
    generator = STIDataGenerator(seed=42)
    
    # Generate comprehensive dataset
    df = generator.generate_comprehensive_dataset(n_samples=2000)
    
    # Save to data directory
    output_path = 'data/raw/enhanced_sti_data.csv'
    generator.save_dataset(df, output_path)
    
    # Print summary statistics
    print("\n=== Dataset Summary ===")
    print(f"Total records: {len(df)}")
    print(f"Features: {list(df.columns)}")
    print(f"\nSTI Distribution:")
    print(df['sti_name'].value_counts())
    print(f"\nAge distribution:")
    print(df['age'].describe())
    print(f"\nRisk score distribution:")
    print(df['risk_score'].describe())

if __name__ == "__main__":
    main() 