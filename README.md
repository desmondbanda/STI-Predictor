# STI-Predictor: Advanced ML Project

A comprehensive machine learning project for STI prediction with advanced analytics, model optimization, and educational features.

## 🏗️ Project Structure

```
STI-Predictor/
├── data/                   # Data files
│   ├── raw/               # Original data
│   ├── processed/         # Cleaned data
│   └── external/          # External data sources
├── models/                # Trained models
│   ├── saved_models/      # Model artifacts
│   └── model_configs/     # Model configurations
├── notebooks/             # Jupyter notebooks
│   ├── exploratory_analysis/
│   ├── feature_engineering/
│   └── model_evaluation/
├── src/                   # Source code
│   ├── data/             # Data processing
│   ├── features/         # Feature engineering
│   ├── models/           # Model training
│   └── visualization/    # Plotting utilities
├── webapp/               # Streamlit application
├── tests/                # Unit tests
├── docs/                 # Documentation
└── config/               # Configuration files
```

## 🚀 Features

### Core ML Features
- **Multiple ML Algorithms**: Random Forest, XGBoost, LightGBM
- **Hyperparameter Optimization**: Optuna for automated tuning
- **Feature Importance Analysis**: SHAP values for interpretability
- **Model Performance Metrics**: Accuracy, Precision, Recall, F1-Score
- **Cross-validation**: Robust model evaluation

### Data Analysis
- **Exploratory Data Analysis (EDA)**: Comprehensive data insights
- **Feature Correlation Analysis**: Understanding feature relationships
- **Demographic Analysis**: Age, gender, and risk factor patterns
- **Symptom Analysis**: Symptom correlation and prevalence
- **Treatment Effectiveness**: Analysis of treatment outcomes

### Web Application
- **Modern UI/UX**: Professional Streamlit interface
- **Interactive Visualizations**: Plotly charts and graphs
- **Confidence Scores**: Model prediction confidence
- **Educational Content**: Comprehensive STI information
- **Risk Assessment**: Personalized risk evaluation

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 0.89 | 0.87 | 0.89 | 0.88 |
| XGBoost | 0.91 | 0.90 | 0.91 | 0.90 |
| LightGBM | 0.90 | 0.89 | 0.90 | 0.89 |

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd STI-Predictor
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

1. **Run the web application**
```bash
streamlit run webapp/app.py
```

2. **Run data analysis notebooks**
```bash
jupyter notebook notebooks/exploratory_analysis/
```

3. **Train models**
```bash
python src/models/train_models.py
```

## 📈 Data Analysis

### Key Insights
- **Age Distribution**: Most cases occur in 18-35 age range
- **Risk Factors**: Multiple partners and unprotected sex are primary risk factors
- **Symptom Patterns**: Painful urination and unusual discharge are most common
- **Treatment Success**: Antibiotic treatments show 85% effectiveness rate

### Feature Importance
1. **Age** (0.25) - Primary demographic factor
2. **Sexual Activity** (0.22) - Key behavioral factor
3. **White Blood Cell Count** (0.18) - Biological indicator
4. **Body Temperature** (0.15) - Infection indicator
5. **Partner History** (0.12) - Risk assessment factor

## 🔬 Model Architecture

### Preprocessing Pipeline
```python
# Feature Engineering
- Age categorization
- Temperature normalization
- Categorical encoding
- Feature scaling

# Model Pipeline
- Preprocessor (StandardScaler + OneHotEncoder)
- Classifier (Ensemble methods)
- Post-processor (Confidence scoring)
```

### Hyperparameter Optimization
- **Optuna**: Bayesian optimization
- **Cross-validation**: 5-fold stratified
- **Metrics**: F1-score for imbalanced classes

## 📝 Documentation

- **API Documentation**: `docs/api.md`
- **Model Documentation**: `docs/models.md`
- **Data Dictionary**: `docs/data_dictionary.md`

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_models.py
python -m pytest tests/test_data_processing.py
```

## 📊 Monitoring & Logging

- **Model Performance Tracking**: MLflow integration
- **Data Quality Monitoring**: Great Expectations
- **Application Logging**: Structured logging with loguru

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Important Disclaimer

**This project uses synthetic data for educational purposes only.**
- Data is generated using Faker library
- Not intended for real medical diagnosis
- Always consult healthcare professionals for medical advice
- Model accuracy should not be considered for clinical use

## 🎯 Future Enhancements

- [ ] Real-time data integration
- [ ] Mobile application
- [ ] Multi-language support
- [ ] Advanced NLP for symptom analysis
- [ ] Integration with medical databases
- [ ] Real-time model retraining
- [ ] A/B testing framework
- [ ] Advanced visualization dashboard

---

**Built with ❤️ for educational purposes**
