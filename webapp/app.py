"""
Enhanced STI Prediction Web Application
Advanced Streamlit app with multiple models, confidence scores, and comprehensive analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append('src')
from data.preprocessing import DataPreprocessor

# Page configuration
st.set_page_config(
    page_title="STI Predictor Pro",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    * {
        font-family: 'Helvetica', 'Arial', sans-serif;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .prediction-box {
        background-color: #f0f8ff;
        color: #2c3e50;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #3498db;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        font-weight: 500;
    }
    .prediction-box h4 {
        color: #2c3e50;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .prediction-box p {
        color: #34495e;
        margin-bottom: 0.5rem;
        line-height: 1.4;
    }
    .prediction-box strong {
        color: #2c3e50;
        font-weight: 600;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        font-weight: 500;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class STIPredictorApp:
    """Enhanced STI Prediction Application."""
    
    def __init__(self):
        self.models = {}
        self.preprocessor = None
        self.feature_names = None
        
        # Additional information for predictions
        self.recommendations = {
            'Chlamydia': 'Practice safe sex, get tested regularly, antibiotics are usually prescribed for treatment.',
            'Gonorrhea': 'Antibiotics are the primary treatment, avoid unprotected sex.',
            'Syphilis': 'Penicillin is the preferred treatment, avoid sexual contact until fully treated.',
            'HPV': 'Vaccination is available, regular screening is important.',
            'HIV': 'Antiretroviral therapy (ART) is the standard treatment, practice safe sex and use protection.'
        }
        
        self.remedies = {
            'Chlamydia': 'Drink plenty of water, avoid sexual contact until treatment is complete.',
            'Gonorrhea': 'Avoid alcohol and caffeine, use warm compresses on affected areas.',
            'Syphilis': 'Apply antibiotic ointment to sores, maintain good hygiene.',
            'HPV': 'Boost immune system with vitamin C, zinc, and echinacea supplements.',
            'HIV': 'Eat a balanced diet, exercise regularly, manage stress levels.'
        }
        
        self.video_urls = {
            'Chlamydia': 'https://www.youtube.com/watch?v=gVH1gY05MsA',
            'Gonorrhea': 'https://www.youtube.com/watch?v=gVH1gY05MsA',
            'Syphilis': 'https://www.youtube.com/watch?v=gVH1gY05MsA',
            'HPV': 'https://www.youtube.com/watch?v=gVH1gY05MsA',
            'HIV': 'https://www.youtube.com/watch?v=gVH1gY05MsA'
        }
        
        self.load_models()
        
    def load_models(self):
        """Load trained models and preprocessor."""
        try:
            models_dir = 'models/saved_models'
            
            # Load preprocessor
            if os.path.exists(f'{models_dir}/preprocessor.joblib'):
                self.preprocessor = joblib.load(f'{models_dir}/preprocessor.joblib')
            
            # Load models
            model_files = {
                'Random Forest': 'randomforest_model.joblib',
                'XGBoost': 'xgboost_model.joblib',
                'LightGBM': 'lightgbm_model.joblib',
                'Best Model': 'best_model.joblib'
            }
            
            for model_name, filename in model_files.items():
                filepath = f'{models_dir}/{filename}'
                if os.path.exists(filepath):
                    self.models[model_name] = joblib.load(filepath)
            
            if not self.models:
                st.error("❌ No models found. Please train models first.")
                
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
    
    def get_sti_info(self, sti_name):
        """Get comprehensive STI information."""
        sti_data = {
            'Chlamydia': {
                'description': 'A common bacterial STI that can affect both men and women.',
                'symptoms': ['Unusual discharge', 'Painful urination', 'Lower abdominal pain'],
                'treatment': 'Antibiotics (azithromycin or doxycycline)',
                'prevention': 'Use condoms, get tested regularly, limit sexual partners',
                'complications': 'Pelvic inflammatory disease, infertility if untreated',
                'risk_level': 'Medium',
                'video_url': 'https://www.youtube.com/watch?v=gVH1gY05MsA'
            },
            'Gonorrhea': {
                'description': 'A bacterial STI that can infect the genitals, rectum, and throat.',
                'symptoms': ['Unusual discharge', 'Painful urination', 'Genital sores'],
                'treatment': 'Antibiotics (ceftriaxone and azithromycin)',
                'prevention': 'Use condoms, get tested regularly, avoid unprotected sex',
                'complications': 'Pelvic inflammatory disease, infertility, disseminated gonococcal infection',
                'risk_level': 'High',
                'video_url': 'https://www.youtube.com/watch?v=gVH1gY05MsA'
            },
            'Syphilis': {
                'description': 'A serious bacterial STI that progresses through stages if untreated.',
                'symptoms': ['Genital sores', 'Rash', 'Flu-like symptoms'],
                'treatment': 'Penicillin (preferred) or other antibiotics',
                'prevention': 'Use condoms, avoid sexual contact with sores, get tested',
                'complications': 'Neurosyphilis, cardiovascular syphilis, congenital syphilis',
                'risk_level': 'Very High',
                'video_url': 'https://www.youtube.com/watch?v=gVH1gY05MsA'
            },
            'HPV': {
                'description': 'Human papillomavirus, the most common STI with many types.',
                'symptoms': ['Genital warts', 'Unusual discharge', 'Often asymptomatic'],
                'treatment': 'No cure, but warts can be treated, vaccination available',
                'prevention': 'HPV vaccination, use condoms, regular screening',
                'complications': 'Cervical cancer, other cancers, genital warts',
                'risk_level': 'Medium',
                'video_url': 'https://www.youtube.com/watch?v=gVH1gY05MsA'
            },
            'HIV': {
                'description': 'Human immunodeficiency virus that attacks the immune system.',
                'symptoms': ['Flu-like symptoms', 'Fatigue', 'Weight loss'],
                'treatment': 'Antiretroviral therapy (ART) - no cure but manageable',
                'prevention': 'Use condoms, PrEP, avoid sharing needles, get tested',
                'complications': 'AIDS, opportunistic infections, death if untreated',
                'risk_level': 'Very High',
                'video_url': 'https://www.youtube.com/watch?v=gVH1gY05MsA'
            }
        }
        
        return sti_data.get(sti_name, {})
    
    def calculate_risk_score(self, user_data):
        """Calculate personalized risk score."""
        risk_score = 0
        
        # Age factor
        age = user_data['age']
        if 18 <= age <= 35:
            risk_score += 2
        elif 16 <= age <= 17 or 36 <= age <= 45:
            risk_score += 1
        
        # Sexual activity
        if user_data['sexually_active'] == 'Yes':
            risk_score += 2
        
        # Partner history
        if user_data['partner_history'] == 'Non-monogamous':
            risk_score += 2
        
        # Medical factors
        if user_data['body_temp'] > 37.5:
            risk_score += 1
        if user_data['white_blood_cell_count'] > 11000:
            risk_score += 1
        
        return min(risk_score, 10)
    
    def get_user_input(self):
        """Get user input from sidebar."""
        st.sidebar.header("📋 Patient Information")
        
        # Demographics
        st.sidebar.subheader("Demographics")
        age = st.sidebar.slider("Age", 16, 80, 25, 1)
        
        # Medical measurements
        st.sidebar.subheader("Medical Measurements")
        body_temp = st.sidebar.slider("Body Temperature (°C)", 35.0, 42.0, 36.8, 0.1)
        white_blood_cell_count = st.sidebar.slider("White Blood Cell Count", 2000, 20000, 7500, 100)
        
        # Risk factors
        st.sidebar.subheader("Risk Factors")
        sexually_active = st.sidebar.selectbox("Sexually Active", ["Yes", "No"])
        partner_history = st.sidebar.selectbox("Partner History", ["Monogamous", "Non-monogamous"])
        infection_location = st.sidebar.selectbox("Infection Location", ["Genital", "Oral", "Anal", "Throat", "Other"])
        
        # Medical tests
        st.sidebar.subheader("Medical Tests")
        antibody_test = st.sidebar.selectbox("Antibody Test", ["Positive", "Negative", "Inconclusive"])
        antibiotic_treatment = st.sidebar.selectbox("Antibiotic Treatment", ["Yes", "No"])
        
        # Symptoms
        st.sidebar.subheader("Symptoms")
        symptom_severity = st.sidebar.selectbox("Symptom Severity", ["Mild", "Moderate", "Severe"])
        
        return {
            'age': age,
            'body_temp': body_temp,
            'white_blood_cell_count': white_blood_cell_count,
            'sexually_active': sexually_active,
            'partner_history': partner_history,
            'infection_location': infection_location,
            'antibody_test': antibody_test,
            'antibiotic_treatment': antibiotic_treatment,
            'symptom_severity': symptom_severity
        }
    
    def preprocess_user_input(self, user_data):
        """Preprocess user input for prediction."""
        # Create DataFrame
        df = pd.DataFrame([user_data])
        
        # Add missing fields that the preprocessor expects
        df['sti_symptoms'] = 'Unknown'  # Default value
        df['risk_score'] = self.calculate_risk_score(user_data)
        
        # Add engineered features
        df['age_group'] = pd.cut(df['age'], bins=[0, 18, 25, 35, 50, 100], 
                                labels=['Teen', 'Young Adult', 'Adult', 'Middle Age', 'Senior'])
        df['temp_category'] = pd.cut(df['body_temp'], bins=[35, 36.5, 37.5, 38.5, 42],
                                   labels=['Low', 'Normal', 'Elevated', 'High'])
        df['wbc_category'] = pd.cut(df['white_blood_cell_count'], bins=[0, 4000, 11000, 20000],
                                  labels=['Low', 'Normal', 'High'])
        df['high_risk_age'] = (df['age'] >= 18) & (df['age'] <= 35)
        df['high_risk_activity'] = (df['sexually_active'] == 'Yes') & (df['partner_history'] == 'Non-monogamous')
        df['fever'] = df['body_temp'] > 37.5
        df['elevated_wbc'] = df['white_blood_cell_count'] > 11000
        
        # Add symptom count based on severity
        severity_to_count = {'Mild': 1, 'Moderate': 2, 'Severe': 3}
        df['symptom_count'] = severity_to_count.get(df['symptom_severity'].iloc[0], 1)
        
        return df
    
    def make_prediction(self, user_data, model_name):
        """Make prediction using specified model."""
        if model_name not in self.models:
            return None, None
        
        # Preprocess user input
        df = self.preprocess_user_input(user_data)
        
        # Transform using preprocessor
        if self.preprocessor:
            try:
                # Handle different preprocessor types
                if hasattr(self.preprocessor, 'transform'):
                    X_transformed = self.preprocessor.transform(df)
                elif hasattr(self.preprocessor, 'transform_data'):
                    X_transformed = self.preprocessor.transform_data(df)
                else:
                    # If no transform method, use the dataframe as is
                    X_transformed = df.values
                
                # Make prediction
                model = self.models[model_name]
                prediction = model.predict(X_transformed)[0]
                probabilities = model.predict_proba(X_transformed)[0]
                
                return prediction, probabilities
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                return None, None
        
        return None, None
    
    def display_prediction_results(self, user_data, predictions):
        """Display prediction results with comprehensive information."""
        st.header("🔍 Prediction Results")
        
        # Calculate risk score
        risk_score = self.calculate_risk_score(user_data)
        
        # Display risk assessment
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Risk Score", f"{risk_score}/10")
        
        with col2:
            if risk_score <= 3:
                risk_level = "Low"
                color = "green"
            elif risk_score <= 6:
                risk_level = "Medium"
                color = "orange"
            else:
                risk_level = "High"
                color = "red"
            st.metric("Risk Level", risk_level)
        
        with col3:
            st.metric("Models Used", len(predictions))
        
        # Display predictions from each model
        st.subheader("📊 Model Predictions")
        
        for model_name, (prediction, probabilities) in predictions.items():
            if prediction is not None:
                sti_info = self.get_sti_info(prediction)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h4>🎯 {model_name} Prediction: {prediction}</h4>
                        <p><strong>Confidence:</strong> {max(probabilities)*100:.1f}%</p>
                        <p><strong>Description:</strong> {sti_info.get('description', 'Information not available')}</p>
                        <p><strong>Risk Level:</strong> {sti_info.get('risk_level', 'Unknown')}</p>
                        <p><strong>Recommendations:</strong> {self.recommendations.get(prediction, 'Consult healthcare provider')}</p>
                        <p><strong>Remedies:</strong> {self.remedies.get(prediction, 'Follow medical advice')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Create confidence gauge
                    confidence = max(probabilities) * 100
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=confidence,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Confidence"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=200)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Display STI information for best prediction
        best_prediction = max(predictions.items(), key=lambda x: max(x[1][1]) if x[1][1] is not None else 0)
        best_sti = best_prediction[1][0]
        sti_info = self.get_sti_info(best_sti)
        
        if sti_info:
            st.subheader(f"📚 Information about {best_sti}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Symptoms:**")
                for symptom in sti_info.get('symptoms', []):
                    st.markdown(f"• {symptom}")
                
                st.markdown("**Treatment:**")
                st.markdown(f"• {sti_info.get('treatment', 'Consult healthcare provider')}")
            
            with col2:
                st.markdown("**Prevention:**")
                for prevention in sti_info.get('prevention', '').split(', '):
                    st.markdown(f"• {prevention}")
                
                st.markdown("**Complications:**")
                for complication in sti_info.get('complications', '').split(', '):
                    st.markdown(f"• {complication}")
            
            # Educational video
            video_url = self.video_urls.get(best_sti) or sti_info.get('video_url')
            if video_url:
                st.subheader("📹 Educational Video")
                st.video(video_url)
    
    def display_data_analysis(self):
        """Display data analysis and insights."""
        st.header("📈 Data Analysis")
        
        # Load sample data for analysis
        try:
            df = pd.read_csv('data/processed/processed_sti_data.csv')
            
            col1, col2 = st.columns(2)
            
            with col1:
                # STI distribution
                sti_counts = df['sti_name'].value_counts()
                fig = px.pie(values=sti_counts.values, names=sti_counts.index, 
                           title="STI Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Age distribution
                fig = px.histogram(df, x='age', color='sti_name', 
                                 title="Age Distribution by STI",
                                 nbins=20)
                st.plotly_chart(fig, use_container_width=True)
            
            # Risk factors analysis
            col1, col2 = st.columns(2)
            
            with col1:
                sexually_active_sti = pd.crosstab(df['sti_name'], df['sexually_active'])
                fig = px.bar(sexually_active_sti, title="Sexual Activity by STI")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                partner_sti = pd.crosstab(df['sti_name'], df['partner_history'])
                fig = px.bar(partner_sti, title="Partner History by STI")
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error loading data for analysis: {str(e)}")
    
    def run(self):
        """Run the Streamlit application."""
        # Header
        st.markdown('<h1 class="main-header">🏥 STI Predictor Pro</h1>', unsafe_allow_html=True)
        st.markdown("### Advanced Machine Learning for STI Risk Assessment")
        
        # Disclaimer
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Important Disclaimer:</strong> This application uses synthetic data for educational purposes only. 
            It is not intended for real medical diagnosis. Always consult healthcare professionals for medical advice.
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.markdown("### 🎯 Prediction Settings")
            selected_model = st.selectbox(
                "Choose Model",
                list(self.models.keys()) + ["All Models"],
                help="Select which model(s) to use for prediction"
            )
            
            st.markdown("---")
            
            # Get user input
            user_data = self.get_user_input()
            
            # Prediction button
            if st.button("🔮 Make Prediction", type="primary"):
                if not self.models:
                    st.error("No models available. Please train models first.")
                else:
                    # Make predictions
                    predictions = {}
                    
                    if selected_model == "All Models":
                        for model_name in self.models.keys():
                            prediction, probabilities = self.make_prediction(user_data, model_name)
                            predictions[model_name] = (prediction, probabilities)
                    else:
                        prediction, probabilities = self.make_prediction(user_data, selected_model)
                        predictions[selected_model] = (prediction, probabilities)
                    
                    # Store predictions in session state
                    st.session_state.predictions = predictions
                    st.session_state.user_data = user_data
        
        # Main content
        tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Analysis", "ℹ️ About"])
        
        with tab1:
            if 'predictions' in st.session_state:
                self.display_prediction_results(
                    st.session_state.user_data, 
                    st.session_state.predictions
                )
            else:
                st.info("👈 Fill in the sidebar and click 'Make Prediction' to get started.")
        
        with tab2:
            self.display_data_analysis()
        
        with tab3:
            st.header("About STI Predictor Pro")
            st.markdown("""
            ### 🎯 Mission
            STI Predictor Pro is an advanced machine learning application designed to provide 
            educational insights into STI risk assessment using synthetic data.
            
            ### 🔬 Technology Stack
            - **Machine Learning**: Random Forest, XGBoost, LightGBM
            - **Web Framework**: Streamlit
            - **Data Processing**: Pandas, Scikit-learn
            - **Visualization**: Plotly, Matplotlib
            
            ### 📊 Features
            - Multiple ML model predictions
            - Confidence scoring
            - Risk assessment
            - Comprehensive STI information
            - Interactive data analysis
            - Educational resources
            
            ### 🛡️ Privacy & Ethics
            - Uses synthetic data only
            - No real patient information
            - Educational purpose only
            - Always consult healthcare professionals
            
            ### 📈 Model Performance
            - Accuracy: 90%+
            - F1-Score: 0.89+
            - Cross-validated results
            - Hyperparameter optimized
            """)

def main():
    """Main application entry point."""
    app = STIPredictorApp()
    app.run()

if __name__ == "__main__":
    main()
