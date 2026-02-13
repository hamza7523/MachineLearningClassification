# XGBoost Machine Learning Projects

## Summary

This repository contains two comprehensive machine learning implementations using XGBoost (eXtreme Gradient Boosting) for different prediction tasks. The projects demonstrate advanced data preprocessing, feature engineering, model optimization, and GPU acceleration techniques. The first project focuses on binary classification with cross-validation and ROC analysis, while the second implements regression with GPU-accelerated training for improved performance.

## Table of Contents

- [Project Overview](#project-overview)
- [Projects](#projects)
- [Technologies & Tools](#technologies--tools)
- [Installation & Setup](#installation--setup)
- [Project 1: XGBoost Classification](#project-1-xgboost-classification)
- [Project 2: XGBoost Regression (GPU-Accelerated)](#project-2-xgboost-regression-gpu-accelerated)
- [Model Performance](#model-performance)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)
- [References](#references)

## Project Overview

### What is XGBoost?

XGBoost (eXtreme Gradient Boosting) is a powerful machine learning algorithm that:
- Uses gradient boosting framework for decision trees
- Provides excellent performance on structured/tabular data
- Handles missing values automatically
- Supports parallel processing and GPU acceleration
- Prevents overfitting through regularization

### Business Applications

These implementations can be applied to:
- **Classification**: Customer churn prediction, fraud detection, medical diagnosis, credit scoring
- **Regression**: Price prediction, demand forecasting, sales estimation, risk assessment

## Projects

### Project 1: XGBoost Classification with Cross-Validation

**Objective**: Binary classification with robust validation and performance visualization

**Key Features**:
- Stratified K-Fold cross-validation (5 folds)
- Correlation-based feature selection
- ROC curve analysis
- Probability predictions for ranking

**Use Cases**: 
- Customer conversion prediction
- Fraud detection
- Medical diagnosis
- Loan default prediction

### Project 2: XGBoost Regression with GPU Acceleration

**Objective**: Price prediction using GPU-accelerated training

**Key Features**:
- GPU-accelerated training (`tree_method='gpu_hist'`)
- Standard scaling for numeric features
- Pipeline architecture for reproducibility
- RMSE evaluation metric
- Execution time tracking

**Use Cases**:
- Real estate price prediction
- Stock price forecasting
- Demand estimation
- Revenue prediction

## Technologies & Tools

### Core Libraries

**Machine Learning**:
- **XGBoost 2.0+**: Gradient boosting implementation
- **scikit-learn**: Data preprocessing and model evaluation
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation

**Visualization**:
- **Matplotlib**: ROC curve plotting

**Performance**:
- **CUDA/GPU**: Hardware acceleration for training

### Development Environment

- **Python**: 3.6+
- **Jupyter Notebook**: Interactive development
- **Kaggle Kernels**: Cloud-based execution environment
- **GPU**: NVIDIA GPU with CUDA support (for regression project)

## Installation & Setup

### Prerequisites

- Python 3.6 or higher
- pip package manager
- NVIDIA GPU with CUDA (optional, for GPU acceleration)

### Step 1: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv xgboost_env
source xgboost_env/bin/activate  # On Windows: xgboost_env\Scripts\activate

# Install required packages
pip install pandas numpy xgboost scikit-learn matplotlib

# For GPU support (optional)
pip install xgboost[gpu]
```

**requirements.txt**:
```
pandas>=1.3.0
numpy>=1.21.0
xgboost>=2.0.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
jupyter>=1.0.0
```

### Step 2: Clone Repository

```bash
git clone https://github.com/yourusername/xgboost-ml-projects.git
cd xgboost-ml-projects
```

### Step 3: Verify GPU Setup (Optional)

```python
import xgboost as xgb

# Check XGBoost build info
print(xgb.build_info())

# Verify GPU availability
dmatrix = xgb.DMatrix(data=[[1, 2], [3, 4]], label=[0, 1])
params = {'tree_method': 'gpu_hist'}
xgb.train(params, dmatrix, num_boost_round=1)
print("GPU is working!")
```

## Project 1: XGBoost Classification

### Overview

Binary classification project with comprehensive data preprocessing, feature engineering, and model validation.

### Dataset

**Files**:
- `train_set.csv`: Training data with features and target variable
- `test_set.csv`: Test data for predictions

**Features**:
- Multiple numeric features (exact count depends on dataset)
- Target variable: `Y` (binary: 0 or 1)
- RecordId: Unique identifier for each record

### Methodology

#### Step 1: Data Loading and Preparation

```python
import pandas as pd

# Load datasets
train_data = pd.read_csv('train_set.csv')
test_data = pd.read_csv('test_set.csv')

# Separate features and target
X_train = train_data.drop(['Y', 'RecordId'], axis=1)
y_train = train_data['Y']
X_test = test_data.drop('RecordId', axis=1)
```

#### Step 2: Missing Value Imputation

**Strategy**: Mean imputation for numeric features

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
```

**Why Mean Imputation?**:
- Simple and effective for numeric data
- Preserves distribution characteristics
- Doesn't introduce bias for symmetric distributions

#### Step 3: Correlation-Based Feature Selection

**Objective**: Remove highly correlated features to reduce multicollinearity

```python
# Calculate correlation matrix
correlation_matrix = X_train_df.corr().abs()

# Set correlation threshold
correlation_threshold = 0.88

# Identify and drop correlated features
to_drop = [column for column in upper_triangle.columns 
           if any(upper_triangle[column] > correlation_threshold)]
```

**Benefits**:
- Reduces model complexity
- Prevents redundancy
- Improves training speed
- Reduces overfitting risk

**Threshold Selection**: 0.88 chosen to balance between:
- Removing truly redundant features
- Retaining valuable information

#### Step 4: Feature Normalization

**Method**: MinMaxScaler (scales features to [0, 1] range)

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

**Why MinMaxScaler?**:
- XGBoost is tree-based (doesn't strictly require scaling)
- Helps with model convergence
- Useful for neural network ensembles
- Prevents feature dominance

#### Step 5: Model Configuration

**XGBoost Hyperparameters**:

```python
XGBClassifier(
    n_estimators=11000,        # Number of boosting rounds
    learning_rate=0.004,       # Step size shrinkage (small for stability)
    max_depth=3,               # Maximum tree depth (prevents overfitting)
    subsample=0.73,            # Sample ratio for training instances
    colsample_bytree=0.73,     # Sample ratio for features
    eval_metric='logloss',     # Evaluation metric
    gamma=0.1,                 # Minimum loss reduction for split
    reg_alpha=0.01,            # L1 regularization
    reg_lambda=1               # L2 regularization
)
```

**Hyperparameter Rationale**:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| n_estimators | 11000 | Large ensemble for better performance |
| learning_rate | 0.004 | Small steps for stability |
| max_depth | 3 | Shallow trees prevent overfitting |
| subsample | 0.73 | Random sampling reduces overfitting |
| colsample_bytree | 0.73 | Feature sampling for diversity |
| gamma | 0.1 | Conservative splitting (reduces complexity) |
| reg_alpha | 0.01 | L1 regularization for sparsity |
| reg_lambda | 1.0 | L2 regularization for smoothness |

#### Step 6: Stratified K-Fold Cross-Validation

**Purpose**: Robust performance evaluation with balanced class distribution

```python
from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
    
    xgb.fit(X_train_fold, y_train_fold)
    val_accuracy = xgb.score(X_val_fold, y_val_fold)
    print(f"Fold {fold + 1} Accuracy: {val_accuracy:.2f}")
```

**Why Stratified K-Fold?**:
- Maintains class distribution in each fold
- Provides reliable performance estimates
- Detects overfitting early
- 5 folds balance between bias and variance

#### Step 7: Prediction and Output

**Probability Predictions**:
```python
# Predict probabilities (useful for ranking and threshold tuning)
y_test_pred_prob = xgb.predict_proba(X_test)[:, 1]

# Create submission
submission = pd.DataFrame({
    'RecordId': test_data['RecordId'],
    'Y': y_test_pred_prob
})
submission.to_csv('XGBoosttest1.csv', index=False)
```

#### Step 8: ROC Curve Analysis

**Purpose**: Visualize model's discriminative ability

```python
from sklearn.metrics import roc_curve, auc

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_val_fold, y_val_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.plot(fpr, tpr, label=f'XGBoost ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line (random classifier)
```

**ROC-AUC Interpretation**:
- 0.5: Random guessing
- 0.7-0.8: Acceptable
- 0.8-0.9: Excellent
- 0.9+: Outstanding

### Model Performance (Classification)

**Cross-Validation Results**:
- 5-Fold Stratified CV
- Average Accuracy: Reported per fold
- Consistent performance across folds indicates robustness

**Output**:
- `XGBoosttest1.csv`: Prediction probabilities for test set
- ROC curve visualization

## Project 2: XGBoost Regression (GPU-Accelerated)

### Overview

Price prediction project leveraging GPU acceleration for faster training on large datasets.

### Dataset

**Files**:
- `train.csv`: Training data with features and target price
- `test.csv`: Test data for predictions

**Features**:
- Numeric features (automatically detected)
- Target variable: `price_doc`
- row ID: Unique identifier

### Methodology

#### Step 1: Data Loading with Timing

```python
import time

start_load_time = time.time()
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(f"Loading time: {time.time() - start_load_time:.2f} seconds")
```

**Performance Tracking**: All major steps are timed for optimization analysis

#### Step 2: Feature Preparation

```python
# Separate features and target
X_train = train.drop(columns=['price_doc'])
y_train = train['price_doc']
X_test = test

# Identify numeric columns
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
```

#### Step 3: Preprocessing Pipeline

**Architecture**: sklearn Pipeline for reproducibility

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols)
    ]
)
```

**Why StandardScaler?**:
- Centers data (mean=0, std=1)
- More robust to outliers than MinMaxScaler
- Common choice for regression tasks

#### Step 4: GPU-Accelerated XGBoost Configuration

**Key Parameters for GPU**:

```python
XGBRegressor(
    n_estimators=1200,
    learning_rate=0.0041,
    max_depth=18,              # Deeper trees for complex patterns
    subsample=0.77,
    colsample_bytree=0.77,
    reg_alpha=0.1,
    reg_lambda=1.0,
    gamma=0,
    eval_metric='rmse',        # Root Mean Squared Error
    tree_method='gpu_hist',    # GPU acceleration
    predictor='gpu_predictor'  # GPU prediction
)
```

**GPU Acceleration Benefits**:
- 10-20x faster training on large datasets
- Efficient histogram-based algorithm
- Lower memory footprint
- Identical results to CPU version

**Hyperparameter Differences from Classification**:

| Parameter | Classification | Regression | Reason |
|-----------|---------------|------------|--------|
| max_depth | 3 | 18 | Regression needs deeper trees for continuous values |
| learning_rate | 0.004 | 0.0041 | Similar conservative approach |
| eval_metric | logloss | rmse | Appropriate for task type |

#### Step 5: Model Training

```python
# Create complete pipeline
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(...))
])

# Train model
start_train_time = time.time()
xgb_pipeline.fit(X_train, y_train)
print(f"Training time: {time.time() - start_train_time:.2f} seconds")
```

#### Step 6: Prediction and Evaluation

```python
# Predict on test set
y_pred = xgb_pipeline.predict(X_test)

# Calculate training RMSE
train_pred = xgb_pipeline.predict(X_train)
rmse = np.sqrt(mean_squared_error(y_train, train_pred))
print(f"Training RMSE: {rmse:.10f}")
```

**RMSE Interpretation**:
- Lower is better
- Same units as target variable (price)
- More sensitive to large errors than MAE

#### Step 7: Submission Creation

```python
submission = pd.DataFrame({
    'row ID': test['row ID'], 
    'price_doc': y_pred
})
submission.to_csv('xgbregressor_submission11.csv', index=False)
```

### Model Performance (Regression)

**Metrics**:
- Training RMSE: High precision (10 decimal places reported)
- GPU Training Time: Significantly reduced vs. CPU
- Total Execution Time: Tracked for optimization

**Output**:
- `xgbregressor_submission11.csv`: Predicted prices for test set

## Usage

### Running Classification Project

```bash
# In Jupyter Notebook or Python script
jupyter notebook newml.ipynb

# Or run as Python script
python classification_xgboost.py
```

**Expected Output**:
```
Handling missing values...
Applying correlation filter...
Normalizing data...
Performing Stratified K-Fold Cross-Validation...
Fold 1 Accuracy: 0.87
Fold 2 Accuracy: 0.85
Fold 3 Accuracy: 0.86
Fold 4 Accuracy: 0.88
Fold 5 Accuracy: 0.86

Average Stratified K-Fold Accuracy: 0.86

Predicting on the test set...
Submission file 'XGBoosttest1.csv' created successfully.
Generating ROC curve...
```

### Running Regression Project

```bash
# In Jupyter Notebook
jupyter notebook xgboost-best-perfomance.ipynb

# Or as Python script
python regression_xgboost.py
```

**Expected Output**:
```
Starting XGBRegressor Process with GPU...
1. Loading datasets...
   Datasets loaded. Train shape: (10000, 20), Test shape: (5000, 19)
   Loading time: 2.34 seconds

2. Preparing features and target...
   Numeric columns: ['feature1', 'feature2', ...]

3. Creating preprocessing pipeline...

4. Configuring GPU-accelerated XGBRegressor pipeline...

5. Training XGBRegressor model...
   Model training completed. Training time: 45.67 seconds

6. Making predictions...
   Predictions completed. Prediction time: 1.23 seconds
   Sample predictions: [250000.5, 380000.3, ...]

7. Creating submission file...
   Submission file 'xgbregressor_submission11.csv' created successfully

8. Calculating Training RMSE...
   Training RMSE: 12345.6789012345

Total Execution Time: 52.45 seconds
```

### Customizing Hyperparameters

**For Classification**:
```python
# Adjust for faster training (lower accuracy)
XGBClassifier(
    n_estimators=1000,      # Reduce from 11000
    learning_rate=0.01,     # Increase from 0.004
    max_depth=5,            # Increase from 3
)

# Adjust for higher accuracy (slower training)
XGBClassifier(
    n_estimators=15000,     # Increase from 11000
    learning_rate=0.002,    # Decrease from 0.004
    max_depth=3,            # Keep shallow
    reg_lambda=2.0,         # Stronger regularization
)
```

**For Regression**:
```python
# CPU version (no GPU available)
XGBRegressor(
    tree_method='hist',           # CPU histogram
    predictor='cpu_predictor',    # CPU prediction
    # ... other parameters same
)

# More aggressive training
XGBRegressor(
    n_estimators=2000,
    max_depth=25,
    learning_rate=0.01,
)
```

## Project Structure

```
xgboost-ml-projects/
├── classification/
│   ├── newml.ipynb                      # Jupyter notebook (classification)
│   ├── classification_xgboost.py        # Python script version
│   ├── train_set.csv                    # Training data
│   ├── test_set.csv                     # Test data
│   └── XGBoosttest1.csv                 # Predictions output
├── regression/
│   ├── xgboost-best-perfomance.ipynb   # Jupyter notebook (regression)
│   ├── regression_xgboost.py            # Python script version
│   ├── train.csv                        # Training data
│   ├── test.csv                         # Test data
│   └── xgbregressor_submission11.csv    # Predictions output
├── notebooks/
│   ├── exploratory_analysis.ipynb       # EDA notebook
│   └── hyperparameter_tuning.ipynb      # Tuning experiments
├── utils/
│   ├── preprocessing.py                 # Reusable preprocessing functions
│   ├── evaluation.py                    # Evaluation metrics
│   └── visualization.py                 # Plotting utilities
├── models/
│   ├── xgb_classifier.pkl              # Saved classification model
│   └── xgb_regressor.pkl               # Saved regression model
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
└── .gitignore                           # Git ignore file
```

## Best Practices

### 1. Data Preprocessing

**Always**:
- Handle missing values before training
- Scale/normalize features appropriately
- Remove highly correlated features
- Check for data leakage

**Never**:
- Fit scalers on test data
- Use information from test set during training
- Ignore data quality issues

### 2. Model Training

**Do**:
- Use cross-validation for reliable estimates
- Track training time and resource usage
- Save models for reproducibility
- Log hyperparameters and metrics

**Don't**:
- Overfit on training data
- Ignore validation performance
- Use default parameters without tuning
- Train without monitoring

### 3. Hyperparameter Tuning

**Recommended Approaches**:

```python
# Grid Search
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [500, 1000, 1500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1]
}

grid_search = GridSearchCV(
    XGBClassifier(),
    param_grid,
    cv=5,
    scoring='roc_auc'
)
```

```python
# Random Search (faster)
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [100, 500, 1000, 2000],
    'max_depth': [3, 5, 7, 10, 15],
    'learning_rate': [0.001, 0.01, 0.05, 0.1],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9]
}

random_search = RandomizedSearchCV(
    XGBClassifier(),
    param_dist,
    n_iter=50,
    cv=5,
    random_state=42
)
```

### 4. Model Evaluation

**Classification Metrics**:
- Accuracy: Overall correctness
- Precision: Positive prediction accuracy
- Recall: Positive class coverage
- F1-Score: Harmonic mean of precision/recall
- ROC-AUC: Discrimination ability

**Regression Metrics**:
- RMSE: Root Mean Squared Error
- MAE: Mean Absolute Error
- R²: Coefficient of determination
- MAPE: Mean Absolute Percentage Error

### 5. Production Deployment

```python
# Save trained model
import pickle

with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_pipeline, f)

# Load for inference
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

predictions = model.predict(new_data)
```

## Troubleshooting

### Common Issues

**Issue**: GPU not detected for regression model
```bash
# Solution: Verify CUDA installation
nvidia-smi

# Install GPU-enabled XGBoost
pip uninstall xgboost
pip install xgboost[gpu]

# Or use CPU version
tree_method='hist'  # Instead of 'gpu_hist'
```

**Issue**: Memory error during training
```python
# Solution: Reduce dataset size or batch processing
# Use sample for large datasets
X_train_sample = X_train.sample(frac=0.5, random_state=42)

# Or reduce n_estimators
n_estimators=500  # Instead of 11000
```

**Issue**: Poor model performance
```python
# Solution: Check for:
# 1. Data quality issues
print(X_train.isnull().sum())
print(X_train.describe())

# 2. Class imbalance (classification)
print(y_train.value_counts())

# 3. Feature importance
import matplotlib.pyplot as plt
xgb.get_booster().feature_importances_
```

**Issue**: Overfitting (high train accuracy, low validation accuracy)
```python
# Solution: Increase regularization
XGBClassifier(
    max_depth=3,        # Reduce from 5
    reg_alpha=0.1,      # Increase from 0.01
    reg_lambda=2.0,     # Increase from 1.0
    subsample=0.7,      # Reduce from 0.8
)
```

**Issue**: Correlation filter removing too many features
```python
# Solution: Increase threshold
correlation_threshold = 0.95  # Instead of 0.88

# Or use variance threshold
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X_train = selector.fit_transform(X_train)
```

## Future Enhancements

### Short-term (Next 3 Months)
- [ ] Implement automated hyperparameter tuning (Optuna, Hyperopt)
- [ ] Add feature importance analysis and visualization
- [ ] Create SHAP values for model interpretability
- [ ] Implement early stopping to prevent overfitting
- [ ] Add logging and experiment tracking (MLflow, Weights & Biases)

### Medium-term (6-12 Months)
- [ ] Build ensemble models (XGBoost + LightGBM + CatBoost)
- [ ] Implement automated feature engineering (Featuretools)
- [ ] Add A/B testing framework for model comparison
- [ ] Create REST API for model deployment (FastAPI)
- [ ] Develop real-time prediction pipeline
- [ ] Add model monitoring and drift detection

### Long-term (1+ Years)
- [ ] Deploy to cloud platforms (AWS SageMaker, Google AI Platform)
- [ ] Implement AutoML pipeline
- [ ] Add neural network ensemble
- [ ] Build custom loss functions for domain-specific problems
- [ ] Create MLOps pipeline with CI/CD
- [ ] Implement federated learning for privacy

## Performance Optimization Tips

### 1. Training Speed

```python
# Use histogram-based algorithm
tree_method='hist'  # Faster than 'auto'

# Reduce data size
subsample=0.7       # Sample 70% of data
colsample_bytree=0.7  # Sample 70% of features

# Enable parallel processing
n_jobs=-1           # Use all CPU cores
```

### 2. Memory Usage

```python
# Use sparse matrices for high-dimensional data
from scipy.sparse import csr_matrix
X_train_sparse = csr_matrix(X_train)

# Reduce precision
XGBClassifier(
    tree_method='hist',
    max_bin=256  # Reduce from default 256
)
```

### 3. Prediction Speed

```python
# Use predictor parameter
predictor='cpu_predictor'  # or 'gpu_predictor'

# Batch predictions
batch_size = 1000
predictions = []
for i in range(0, len(X_test), batch_size):
    batch = X_test[i:i+batch_size]
    predictions.extend(model.predict(batch))
```

## Model Interpretability

### Feature Importance

```python
import matplotlib.pyplot as plt

# Get feature importance
importance = xgb.feature_importances_
features = X_train.columns

# Plot
plt.figure(figsize=(10, 6))
plt.barh(features, importance)
plt.xlabel('Importance Score')
plt.title('XGBoost Feature Importance')
plt.show()
```

### SHAP Values

```python
import shap

# Create explainer
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_train)

# Summary plot
shap.summary_plot(shap_values, X_train)

# Force plot for single prediction
shap.force_plot(explainer.expected_value, shap_values[0], X_train.iloc[0])
```

## Comparison with Other Algorithms

| Algorithm | Speed | Accuracy | Interpretability | Scalability |
|-----------|-------|----------|------------------|-------------|
| **XGBoost** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Random Forest | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| CatBoost | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Neural Networks | ⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| Logistic Regression | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## References

### Documentation
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)

### Tutorials & Resources
- [XGBoost Python Package](https://xgboost.readthedocs.io/en/stable/python/)
- [Hyperparameter Tuning Guide](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html)
## Acknowledgments

- XGBoost development team for the excellent gradient boosting library
- scikit-learn community for preprocessing and evaluation tools
- Kaggle for providing the platform and datasets
- Open-source community for continuous improvements

---

**Built with 🚀 using XGBoost, scikit-learn, and Python**
