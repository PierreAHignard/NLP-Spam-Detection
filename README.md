# Air Quality ML Pipeline

A machine learning pipeline for air quality prediction using time-series data with geographic features.

## Overview

[Describe your project objectives and business context here]

## Dataset Description

[Provide details about your dataset:
- Data source and collection period
- Number of records and features
- Key variables and their meanings
- Data quality notes]

## Project Structure

[Describe your project organization here]

## Installation

```bash
# Extract the project files
cd air_quality

# Install dependencies and package with uv
uv sync --extra dev

# Verify installation
uv run python scripts/run_tests.py --quick
```

## Usage

### Basic Pipeline

```bash
# Run basic pipeline
uv run python scripts/run_pipeline.py

# Different feature selection methods
uv run python scripts/run_pipeline.py --model linear --method rfe --n-features 15
```

### Advanced Models

[Add your advanced models usage here]

### MLflow Experiment Tracking

[Add your MLflow usage here]

## Key Findings

[Summarize your main analytical findings here:
- Key patterns discovered in the data
- Most important features for prediction
- Model performance comparisons
- Business insights and recommendations]

## Model Performance

[Document your model results:
- Performance metrics (RMSE, MAE, R²)
- Cross-validation results
- Feature importance rankings
- Comparison between different models]

## Methodology

[Describe your analytical approach:
- Data preprocessing steps
- Feature engineering strategy
- Model selection rationale
- Evaluation methodology]

## Authors

[Add team member names and task distribution:
- **Student 1 Name**: [Specific tasks and contributions]
- **Student 2 Name**: [Specific tasks and contributions]

Example:
- **Alice Dupont**: Data preprocessing, feature engineering, model evaluation
- **Bob Martin**: Model training, hyperparameter optimization, documentation]
