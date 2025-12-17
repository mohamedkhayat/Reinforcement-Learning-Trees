# RLT Flask Application

This is a Flask web application for comparing Reinforcement Learning Trees (RLT) with Random Forest models.

## Features

- **Dataset Selection**: Choose from multiple available datasets (classification and regression)
- **Model Comparison**: Train both RLT and Random Forest with the same random state
- **Performance Metrics**: Display metrics for both models
  - Classification: Accuracy, Precision, Recall, F1-Score
  - Regression: MSE, RMSE, MAE, R² Score
- **Feature Importance Visualization**: Side-by-side comparison of feature importances
- **Interactive UI**: Modern, responsive web interface

## Installation

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Make sure the RLT package is installed:

```bash
cd ..
pip install -e .
```

## Running the Application

```bash
cd app
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

1. Select a dataset from the dropdown menu
2. Configure parameters (random state, number of trees, test size)
3. Click "Run Comparison" to train both models
4. View the results:
   - Performance metrics for both models
   - Feature importance comparison plots
   - Detailed feature importance table

## Available Datasets

- **Breast Cancer Wisconsin** - Classification
- **Boston Housing** - Regression
- **Red Wine Quality** - Classification
- **White Wine Quality** - Classification
- **Auto MPG** - Regression
- **Concrete Strength** - Regression
- **Sonar** - Classification
