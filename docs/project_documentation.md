# STI Predictor Pro - Complete Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Installation & Setup](#installation--setup)
4. [Data Pipeline](#data-pipeline)
5. [Model Training](#model-training)
6. [Web Application](#web-application)
7. [Testing](#testing)
8. [Deployment](#deployment)
9. [Troubleshooting](#troubleshooting)
10. [Contributing](#contributing)

## Project Overview

STI Predictor Pro is a comprehensive machine learning project that demonstrates advanced ML engineering practices. The project includes:

- **Multiple ML Algorithms**: Random Forest, XGBoost, LightGBM
- **Hyperparameter Optimization**: Optuna for automated tuning
- **Feature Engineering**: Comprehensive preprocessing pipeline
- **Web Application**: Modern Streamlit interface
- **Data Analysis**: Interactive visualizations and insights
- **Testing**: Comprehensive unit tests
- **Documentation**: Complete technical documentation

### Key Features
- ✅ Synthetic data generation with realistic patterns
- ✅ Advanced data preprocessing and feature engineering
- ✅ Multiple ML models with hyperparameter optimization
- ✅ Model ensemble and confidence scoring
- ✅ Interactive web application with modern UI
- ✅ Comprehensive data analysis and visualization
- ✅ Unit testing and quality assurance
- ✅ Professional documentation and deployment guides

## Architecture

### Project Structure
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

### Technology Stack
- **Python 3.8+**: Core programming language
- **Scikit-learn**: Machine learning algorithms
- **XGBoost & LightGBM**: Advanced gradient boosting
- **Optuna**: Hyperparameter optimization
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Pandas & NumPy**: Data manipulation
- **Joblib**: Model persistence
- **Pytest**: Testing framework

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for version control)

### Step-by-Step Setup

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

4. **Verify installation**
```bash
python -c "import streamlit, sklearn, xgboost, lightgbm; print('All packages installed successfully!')"
```

### Environment Variables
Create a `.env` file in the root directory:
```env
# Model settings
RANDOM_STATE=42
N_TRIALS=50
TEST_SIZE=0.2

# Data settings
DATA_PATH=data/raw/enhanced_sti_data.csv
MODEL_PATH=models/saved_models/

# Web app settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
```

## Data Pipeline

### Data Generation
The project uses synthetic data generated with realistic patterns:

```python
# Generate enhanced dataset
python src/data/generate_data.py
```

**Features Generated:**
- Age distribution (16-65 years)
- Body temperature (35-42°C)
- White blood cell count (2000-20000)
- Sexual activity patterns
- Partner history
- Medical test results
- STI-specific symptoms

### Data Preprocessing
Comprehensive preprocessing pipeline:

```python
# Run preprocessing
python src/data/preprocessing.py
```

**Preprocessing Steps:**
1. **Data Cleaning**
   - Handle missing values
   - Remove duplicates
   - Cap outliers using IQR method
   - Validate data ranges

2. **Feature Engineering**
   - Age categorization
   - Temperature categories
   - WBC categories
   - Risk factor indicators
   - Medical indicators

3. **Feature Transformation**
   - StandardScaler for numeric features
   - OneHotEncoder for categorical features
   - Boolean features handling

## Model Training

### Training Process
```bash
# Train all models
python src/models/train_models.py
```

### Model Algorithms

#### 1. Random Forest
- **Advantages**: Robust, handles mixed data types, feature importance
- **Hyperparameters**: n_estimators, max_depth, min_samples_split
- **Optimization**: Optuna with 50 trials

#### 2. XGBoost
- **Advantages**: High performance, handles missing values
- **Hyperparameters**: learning_rate, max_depth, subsample
- **Optimization**: Bayesian optimization

#### 3. LightGBM
- **Advantages**: Fast training, memory efficient
- **Hyperparameters**: num_leaves, learning_rate, feature_fraction
- **Optimization**: Optuna with cross-validation

### Model Evaluation
- **Cross-validation**: 5-fold stratified
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Feature importance**: SHAP values
- **Confidence scoring**: Probability estimates

### Expected Performance
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 0.89 | 0.87 | 0.89 | 0.88 |
| XGBoost | 0.91 | 0.90 | 0.91 | 0.90 |
| LightGBM | 0.90 | 0.89 | 0.90 | 0.89 |

## Web Application

### Running the App
```bash
streamlit run webapp/app.py
```

### Features
- **Multiple Model Support**: Choose from different algorithms
- **Confidence Scoring**: Visual confidence gauges
- **Risk Assessment**: Personalized risk scoring
- **Educational Content**: Comprehensive STI information
- **Interactive Analysis**: Data visualizations
- **Modern UI**: Professional design with CSS styling

### Application Structure
```
webapp/
├── app.py              # Main application
├── components/         # Reusable components
├── utils/             # Utility functions
└── assets/            # Static assets
```

### Key Components
1. **STIPredictorApp Class**
   - Model loading and management
   - User input processing
   - Prediction generation
   - Results display

2. **Data Analysis Tab**
   - Interactive visualizations
   - Statistical insights
   - Feature correlations

3. **About Tab**
   - Project information
   - Technology stack
   - Disclaimers

## Testing

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_models.py
python -m pytest tests/test_data_processing.py

# Run with coverage
python -m pytest --cov=src tests/
```

### Test Categories
1. **Data Preprocessing Tests**
   - Data loading
   - Cleaning operations
   - Feature engineering
   - Pipeline creation

2. **Model Training Tests**
   - Model initialization
   - Training process
   - Evaluation metrics
   - Feature importance

3. **Data Quality Tests**
   - Validation rules
   - Outlier detection
   - Missing value handling

4. **Performance Tests**
   - Accuracy calculation
   - F1-score computation
   - Confusion matrix

### Test Coverage
- **Target**: >90% code coverage
- **Areas**: Core functionality, edge cases, error handling
- **Automation**: CI/CD pipeline integration

## Deployment

### Local Deployment
1. **Generate data and train models**
```bash
python src/data/generate_data.py
python src/models/train_models.py
```

2. **Run web application**
```bash
streamlit run webapp/app.py
```

3. **Access application**
   - URL: http://localhost:8501
   - Default port: 8501

### Production Deployment

#### Option 1: Streamlit Cloud
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy automatically

#### Option 2: Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "webapp/app.py", "--server.port=8501"]
```

#### Option 3: Heroku
1. Create `Procfile`:
```
web: streamlit run webapp/app.py --server.port=$PORT --server.address=0.0.0.0
```

2. Deploy to Heroku:
```bash
heroku create sti-predictor-pro
git push heroku main
```

### Environment Configuration
- **Development**: Local environment with debug mode
- **Staging**: Test environment with sample data
- **Production**: Optimized environment with real models

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Solution: Add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

#### 2. Model Loading Errors
```bash
# Solution: Train models first
python src/models/train_models.py
```

#### 3. Memory Issues
```bash
# Solution: Reduce data size or use sampling
python src/data/generate_data.py --n_samples 1000
```

#### 4. Streamlit Connection Issues
```bash
# Solution: Check port availability
streamlit run webapp/app.py --server.port 8502
```

### Performance Optimization
1. **Data Processing**
   - Use chunked processing for large datasets
   - Implement caching for repeated operations
   - Optimize memory usage with data types

2. **Model Training**
   - Use early stopping for gradient boosting
   - Implement parallel processing
   - Optimize hyperparameter search space

3. **Web Application**
   - Implement lazy loading for visualizations
   - Cache model predictions
   - Optimize CSS and JavaScript

### Debug Mode
```bash
# Enable debug logging
export STREAMLIT_DEBUG=true
streamlit run webapp/app.py --logger.level=debug
```

## Contributing

### Development Setup
1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

### Code Standards
- **PEP 8**: Python style guide
- **Type hints**: Function annotations
- **Docstrings**: Comprehensive documentation
- **Tests**: Unit tests for new features

### Pull Request Process
1. **Fork and clone**
2. **Create feature branch**
3. **Make changes**
4. **Add tests**
5. **Update documentation**
6. **Submit PR**

### Code Review Checklist
- [ ] Code follows PEP 8
- [ ] Functions have type hints
- [ ] New features have tests
- [ ] Documentation is updated
- [ ] No breaking changes
- [ ] Performance impact considered

## Additional Resources

### Documentation
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Optuna Documentation](https://optuna.readthedocs.io/)

### Learning Resources
- [Machine Learning Engineering](https://mlops.community/)
- [Data Science Best Practices](https://www.datascience.com/)
- [Python Testing](https://docs.pytest.org/)

### Community
- [GitHub Issues](https://github.com/your-repo/issues)
- [Discord Community](https://discord.gg/ml-community)
- [Stack Overflow](https://stackoverflow.com/)

---

**Note**: This project uses synthetic data for educational purposes only. Always consult healthcare professionals for medical advice. 