# Industrial Vibration Prediction System

A modular machine learning system for predicting vibration patterns in industrial roller mills using advanced feature engineering and ensemble methods.

## 🏭 Project Overview

This system transforms raw industrial sensor data into actionable vibration predictions, enabling:
- **Predictive Maintenance**: Early detection of equipment anomalies
- **Operational Optimization**: Data-driven insights for mill performance
- **Cost Reduction**: Prevention of unplanned downtime

## 📊 Key Features

- **Modular Architecture**: Clean separation of concerns for maintainability
- **Advanced Feature Engineering**: Rolling statistics, temporal features, and domain-specific transformations
- **Multiple ML Models**: Comprehensive comparison of regression algorithms
- **Robust Evaluation**: Statistical significance testing and business metrics
- **Production Ready**: Configuration management and scalable pipeline design

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd modularized_code

# Install dependencies
pip install -r requirements.txt

# Optional: Install advanced ML libraries
pip install xgboost catboost
```

### Command Line Usage (Recommended)

```bash
# Run complete pipeline
python scripts/run_complete_pipeline.py data/vibration_data.csv

# Run only data processing
python scripts/run_data_pipeline.py data/vibration_data.csv --save-data

# Custom configuration
python scripts/run_complete_pipeline.py data/vibration_data.csv \
  --output-dir results/ \
  --models random_forest xgboost \
  --equipment-count 20 \
  --maintenance-budget 1000000
```

### Programmatic Usage

```python
from src.pipeline.orchestrator import run_complete_pipeline

# Run complete end-to-end workflow
results = run_complete_pipeline(
    data_path="data/vibration_data.csv",
    output_dir="results/",
    annual_equipment_count=10,
    annual_maintenance_budget=500000
)

# Access results
best_model = results['ml_pipeline_results']['recommendation']['recommended_model']
business_insights = results['ml_pipeline_results']['business_analysis']
```

## 📁 Project Structure

```
modularized_code/
├── src/
│   ├── config/          # Configuration management
│   ├── data/            # Data loading and preprocessing
│   ├── features/        # Feature engineering and selection
│   ├── models/          # ML model implementations
│   ├── evaluation/      # Model evaluation and metrics
│   ├── utils/           # Utility functions and plotting
│   └── pipeline/        # End-to-end pipeline orchestration
├── scripts/             # Standalone execution scripts
├── tests/              # Unit tests
└── results/            # Generated results and models
```

## 🔧 Configuration

The system uses centralized configuration in `src/config/settings.py`. Key configurations:

```python
# Data processing
DATA_CONFIG = {
    "data_path": "full_data/",
    "target_frequency": "5T",  # 5-minute resampling
    "test_size": 0.2,
}

# Feature engineering
FEATURE_CONFIG = {
    "rolling_windows": [3, 6, 12],  # 15min, 30min, 1hr
    "key_patterns": ["POWER", "PRESSURE", "CURRENT", "FLOW"],
}

# Model selection
MODEL_CONFIG = {
    "max_features": 50,
    "cv_folds": 5,
    "scoring_metric": "r2",
}
```

## 📈 Supported Models

- **Linear Models**: Linear Regression, Ridge, Lasso, Huber
- **Tree-Based**: Decision Tree, Random Forest, Gradient Boosting
- **Advanced**: XGBoost, CatBoost (if installed)
- **Distance-Based**: K-Nearest Neighbors

## 🔍 Usage Examples

### Data Processing Pipeline
```python
from src.pipeline.data_pipeline import DataPipeline

# Initialize data pipeline
data_pipeline = DataPipeline()

# Run complete data processing
X_train, X_test, y_train, y_test = data_pipeline.run_complete_pipeline(
    data_path="data/vibration_data.csv",
    target_column="vibration",
    test_size=0.2,
    selection_strategy="balanced"
)
```

### Machine Learning Pipeline
```python
from src.pipeline.ml_pipeline import MLPipeline

# Initialize ML pipeline
ml_pipeline = MLPipeline()
ml_pipeline.setup_data(X_train, y_train, X_test, y_test)

# Train and evaluate models
training_results = ml_pipeline.train_models()
evaluation_results = ml_pipeline.evaluate_models()
business_analysis = ml_pipeline.generate_business_analysis()

# Get best model recommendation
recommendation = ml_pipeline.get_best_model_recommendation()
```

### Individual Components
```python
from src.features.engineer import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.evaluation.visualizer import create_evaluation_plots

# Feature engineering
engineer = FeatureEngineer()
df_features = engineer.engineer_features(df_clean, "vibration")

# Model training
trainer = ModelTrainer()
models = trainer.train_multiple_models(X_train, y_train, X_test, y_test)

# Create evaluation plots
plots = create_evaluation_plots(y_train, train_pred, y_test, test_pred)
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📋 Development Workflow

This project follows a modular development approach:

1. **PR 1**: Foundation & Configuration ✅
2. **PR 2**: Data Layer (loading, cleaning, preprocessing) ✅
3. **PR 3**: Feature Engineering ✅
4. **PR 4**: Feature Analysis (importance, selection) ✅
5. **PR 5**: Model Infrastructure ✅
6. **PR 6**: Evaluation System ✅
7. **PR 7**: Pipeline Integration ✅
8. **PR 8**: Advanced Features & Documentation ✅

**Status**: 🎉 **COMPLETE** - All 8 PRs successfully implemented!

## 📊 Results

The system typically achieves:
- **R² Score**: 0.85-0.95 on test data
- **RMSE**: <0.005 mm/s prediction error
- **Processing Speed**: <5 minutes for complete pipeline
- **Model Comparison**: 6-8 different algorithms evaluated

## 🔧 Customization

### Adding New Models
```python
# In src/config/model_config.py
new_model_config = {
    "my_model": {
        "class": MyModelClass,
        "params": {"param1": value1},
        "requires_scaling": True,
        "description": "My custom model",
    }
}
```

### Custom Feature Engineering
```python
# In src/features/engineer.py
def create_custom_features(df, target_col):
    # Your custom feature engineering logic
    return df_with_new_features
```

## 📝 Contributing

1. Follow the modular architecture
2. Add comprehensive tests for new features
3. Update configuration files as needed
4. Document new functionality
5. Follow PEP 8 style guidelines

## 📧 Contact

**Author**: Paulo Cesar da Silva Junior  
**Institution**: Universidade Federal do Ceará (UFC)  
**Advisor**: Profa. Dra. Rosineide Fernando da Paz

## 📄 License

This project is developed as part of academic research at UFC and is available for educational and research purposes.

---

*This system represents a complete transformation from notebook-based analysis to production-ready modular code, enabling scalable and maintainable machine learning workflows for industrial applications.*