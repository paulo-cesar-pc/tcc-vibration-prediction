# Execution Scripts

This directory contains standalone execution scripts for running the vibration prediction pipeline components.

## Available Scripts

### 1. `run_complete_pipeline.py`
Executes the complete end-to-end pipeline from raw data to business insights.

**Usage:**
```bash
python scripts/run_complete_pipeline.py path/to/data.csv
```

**Options:**
- `--output-dir, -o`: Output directory for results
- `--target, -t`: Target column name (default: vibration)
- `--test-size`: Test set size fraction (default: 0.2)
- `--selection-strategy`: Feature selection strategy (balanced/best_performance/minimal)
- `--models`: Specific models to train (space-separated list)
- `--equipment-count`: Number of equipment units for business analysis
- `--maintenance-budget`: Annual maintenance budget for ROI analysis
- `--config`: Path to configuration JSON file
- `--log-level`: Logging level (DEBUG/INFO/WARNING/ERROR)
- `--quiet, -q`: Suppress console output

**Examples:**
```bash
# Basic usage
python scripts/run_complete_pipeline.py data/vibration_data.csv

# With custom output directory and models
python scripts/run_complete_pipeline.py data/vibration_data.csv \
  --output-dir results/ \
  --models random_forest xgboost catboost

# With business parameters
python scripts/run_complete_pipeline.py data/vibration_data.csv \
  --equipment-count 20 \
  --maintenance-budget 1000000 \
  --selection-strategy best_performance
```

### 2. `run_data_pipeline.py`
Executes only the data processing pipeline (loading, cleaning, preprocessing, feature engineering, selection).

**Usage:**
```bash
python scripts/run_data_pipeline.py path/to/data.csv
```

**Options:**
- `--output-dir, -o`: Output directory for results
- `--target, -t`: Target column name (default: vibration)
- `--test-size`: Test set size fraction (default: 0.2)
- `--selection-strategy`: Feature selection strategy
- `--config`: Path to configuration JSON file
- `--save-data`: Save processed data to CSV files
- `--log-level`: Logging level
- `--quiet, -q`: Suppress console output

**Examples:**
```bash
# Basic usage
python scripts/run_data_pipeline.py data/vibration_data.csv

# Save processed data files
python scripts/run_data_pipeline.py data/vibration_data.csv \
  --output-dir data_outputs/ \
  --save-data

# Custom configuration
python scripts/run_data_pipeline.py data/vibration_data.csv \
  --config config/pipeline_config.json \
  --selection-strategy minimal
```

## Output Structure

### Complete Pipeline Output (`run_complete_pipeline.py`)
```
pipeline_outputs/
├── complete_results.json          # Complete pipeline results
├── executive_report.json          # Executive summary
├── execution_log.txt              # Detailed execution log
├── data_pipeline/                 # Data processing results
│   ├── pipeline_state.pkl
│   ├── pipeline_summary.json
│   └── selected_features.json
└── ml_pipeline/                   # ML pipeline results
    ├── evaluation_results.json
    ├── validation_results.json
    ├── business_insights.json
    └── pipeline_metadata.json
```

### Data Pipeline Output (`run_data_pipeline.py`)
```
data_pipeline_outputs/
├── pipeline_state.pkl             # Complete pipeline state
├── pipeline_summary.json          # Execution summary
├── selected_features.json         # Selected features list
└── (optional CSV files if --save-data)
    ├── raw_data.csv
    ├── clean_data.csv
    ├── engineered_features.csv
    ├── train_data.csv
    └── test_data.csv
```

## Configuration Files

You can provide custom configuration via JSON files:

**Example configuration (`config/pipeline_config.json`):**
```json
{
  "data_pipeline": {
    "cleaning": {
      "remove_outliers": true,
      "outlier_method": "iqr",
      "outlier_threshold": 3.0
    },
    "feature_engineering": {
      "rolling_windows": [5, 10, 20],
      "create_temporal_features": true,
      "create_lag_features": true
    }
  },
  "ml_pipeline": {
    "save_plots": true,
    "plots_output_dir": "plots/"
  }
}
```

## Requirements

Ensure you have installed all dependencies:
```bash
pip install -r requirements.txt
```

## Error Handling

Both scripts include comprehensive error handling and logging:

- **Input validation**: Checks for file existence, valid parameters
- **Execution logging**: Detailed logs saved to files
- **Graceful failures**: Clear error messages and cleanup
- **Interrupt handling**: Clean shutdown on Ctrl+C

## Performance Considerations

For large datasets:
- Use `--log-level WARNING` to reduce log output
- Consider using `--models` to train only specific models
- Use `--selection-strategy minimal` for faster feature selection
- Monitor memory usage during feature engineering

## Integration with Original Notebook

These scripts replicate the functionality of `simple_vibration_prediction.ipynb` but with:
- Improved error handling and logging
- Modular, reusable components
- Configurable parameters
- Production-ready architecture
- Comprehensive output and reporting

The modularized code maintains the same analytical capabilities while providing:
- Better code organization
- Enhanced maintainability
- Automated pipeline execution
- Standardized outputs
- Business-ready insights