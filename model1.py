# Student CGPA Prediction - Complete Pipeline

**Author:** adamya1231  
**Date:** 2025-11-18  
**Objective:** Predict student CGPA using 5 machine learning models

---

## 📚 Step 1: Import Libraries

```python
# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Utilities
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ All libraries imported successfully!")
```

---

## 📂 Step 2: Load Dataset

```python
# Load your dataset (update the filename)
df = pd.read_excel('Students_Academic_Performance.xlsx')  # or pd.read_csv('filename.csv')

print(f"Dataset Shape: {df.shape}")
print(f"Total Records: {df.shape[0]}")
print(f"Total Features: {df.shape[1]}")
print("\nFirst 5 rows:")
df.head()
```

---

## 🧹 Step 3: Data Cleaning

```python
# Check for missing values
print("Missing Values:")
print(df.isnull().sum())

# Fill missing values
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("\n✅ Missing values handled!")

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")
if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print("✅ Duplicates removed!")

# Convert CGPA and SGPA from 4.0 to 10.0 scale
if 'CGPA' in df.columns:
    df['CGPA'] = df['CGPA'] * 2.5
    print(f"\n✅ CGPA converted to 10.0 scale: Range [{df['CGPA'].min():.2f}, {df['CGPA'].max():.2f}]")

if 'SGPA' in df.columns:
    df['SGPA'] = df['SGPA'] * 2.5
    print(f"✅ SGPA converted to 10.0 scale: Range [{df['SGPA'].min():.2f}, {df['SGPA'].max():.2f}]")
```

---

## 🔢 Step 4: Encode Categorical Variables

```python
# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical columns: {len(categorical_cols)}")
print(f"Numerical columns: {len(numerical_cols)}")

# Encode categorical variables
df_encoded = df.copy()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le

print(f"\n✅ Encoded {len(categorical_cols)} categorical columns!")
```

---

## 📊 Step 5: Data Visualization - CGPA Distribution

```python
plt.figure(figsize=(10, 6))
plt.hist(df_encoded['CGPA'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel('CGPA (10.0 scale)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of CGPA', fontsize=14, fontweight='bold')
plt.axvline(df_encoded['CGPA'].mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {df_encoded["CGPA"].mean():.2f}')
plt.legend(fontsize=11)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Mean CGPA: {df_encoded['CGPA'].mean():.2f}")
print(f"Median CGPA: {df_encoded['CGPA'].median():.2f}")
print(f"Std Dev: {df_encoded['CGPA'].std():.2f}")
```

---

## 📈 Step 6: Correlation Analysis

```python
# Calculate correlation with CGPA
correlations = df_encoded.corr()['CGPA'].sort_values(ascending=False)

print("Top 15 Features Correlated with CGPA:")
print(correlations.head(16))  # 16 to include CGPA itself

# Plot top correlations
plt.figure(figsize=(10, 8))
top_features = correlations.head(16)[1:]  # Exclude CGPA itself
colors = ['green' if x > 0 else 'red' for x in top_features.values]
plt.barh(range(len(top_features)), top_features.values, color=colors, alpha=0.7)
plt.yticks(range(len(top_features)), top_features.index, fontsize=10)
plt.xlabel('Correlation Coefficient', fontsize=12)
plt.title('Top 15 Features Correlated with CGPA', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 🔥 Step 7: Feature Importance (Random Forest)

```python
# Prepare data for feature importance
X_temp = df_encoded.drop(['CGPA'], axis=1)
y_temp = df_encoded['CGPA']

# Train Random Forest to get feature importance
rf_temp = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_temp.fit(X_temp, y_temp)

# Create feature importance dataframe
feature_importance = pd.DataFrame({
    'Feature': X_temp.columns,
    'Importance': rf_temp.feature_importances_
}).sort_values('Importance', ascending=False)

print("Top 15 Most Important Features:")
print(feature_importance.head(15))

# Plot feature importance
plt.figure(figsize=(10, 8))
top_15 = feature_importance.head(15)
plt.barh(range(len(top_15)), top_15['Importance'].values, color='coral', alpha=0.7)
plt.yticks(range(len(top_15)), top_15['Feature'].values, fontsize=10)
plt.xlabel('Feature Importance', fontsize=12)
plt.title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# Save important features for later use
IMPORTANT_FEATURES = feature_importance.head(10)['Feature'].tolist()
print(f"\n✅ Identified {len(IMPORTANT_FEATURES)} most important features for user input")
```

---

## 📉 Step 8: Scatter Plots (Top Features vs CGPA)

```python
# Get top 4 numerical features for visualization
top_numerical = []
for feat in IMPORTANT_FEATURES:
    if df_encoded[feat].dtype in ['float64', 'int64']:
        top_numerical.append(feat)
        if len(top_numerical) == 4:
            break

# Create scatter plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, feature in enumerate(top_numerical):
    axes[idx].scatter(df_encoded[feature], df_encoded['CGPA'], alpha=0.5, s=30)
    axes[idx].set_xlabel(feature, fontsize=11)
    axes[idx].set_ylabel('CGPA', fontsize=11)
    axes[idx].set_title(f'CGPA vs {feature}', fontsize=12, fontweight='bold')
    axes[idx].grid(alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df_encoded[feature], df_encoded['CGPA'], 1)
    p = np.poly1d(z)
    axes[idx].plot(df_encoded[feature], p(df_encoded[feature]), 
                   "r--", alpha=0.8, linewidth=2, label='Trend')
    axes[idx].legend()

plt.tight_layout()
plt.show()
```

---

## ✂️ Step 9: Data Splitting

```python
# Prepare features and target
X = df_encoded.drop(['CGPA'], axis=1)
y = df_encoded['CGPA']

print(f"Feature Matrix Shape: {X.shape}")
print(f"Target Vector Shape: {y.shape}")

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✅ Data Split Complete!")
print(f"   Training samples: {X_train.shape[0]}")
print(f"   Testing samples: {X_test.shape[0]}")

# Feature Scaling (for linear models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save median values for user input imputation
median_values = X_train.median().to_dict()

print(f"✅ Feature scaling complete!")
print(f"✅ Saved median values for {len(median_values)} features")
```

---

## 🤖 Step 10: Model Training Setup

```python
# Function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a model"""
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Print results
    print(f"\n{model_name}")
    print("="*70)
    print(f"{'Metric':<15} {'Train':<20} {'Test':<20}")
    print("-"*70)
    print(f"{'RMSE':<15} {train_rmse:<20.4f} {test_rmse:<20.4f}")
    print(f"{'MAE':<15} {train_mae:<20.4f} {test_mae:<20.4f}")
    print(f"{'R² Score':<15} {train_r2:<20.4f} {test_r2:<20.4f}")
    
    return {
        'Model': model_name,
        'Train_RMSE': train_rmse,
        'Test_RMSE': test_rmse,
        'Train_MAE': train_mae,
        'Test_MAE': test_mae,
        'Train_R2': train_r2,
        'Test_R2': test_r2
    }

# Store results
results = []
models = {}

print("🚀 Training 5 Models...")
print("="*70)
```

---

## 🎯 Model 1: Linear Regression

```python
print("\n1️⃣ Training Linear Regression (Baseline)...")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
result = evaluate_model(lr, X_train_scaled, X_test_scaled, y_train, y_test, 'Linear Regression')
results.append(result)
models['Linear Regression'] = lr
```

---

## 🎯 Model 2: Ridge Regression

```python
print("\n2️⃣ Training Ridge Regression (Regularized)...")
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train_scaled, y_train)
result = evaluate_model(ridge, X_train_scaled, X_test_scaled, y_train, y_test, 'Ridge Regression')
results.append(result)
models['Ridge Regression'] = ridge
```

---

## 🎯 Model 3: Random Forest

```python
print("\n3️⃣ Training Random Forest (Tree-based Ensemble)...")
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
result = evaluate_model(rf, X_train, X_test, y_train, y_test, 'Random Forest')
results.append(result)
models['Random Forest'] = rf
```

---

## 🎯 Model 4: XGBoost

```python
print("\n4️⃣ Training XGBoost (Advanced Boosting)...")
xgboost = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgboost.fit(X_train, y_train)
result = evaluate_model(xgboost, X_train, X_test, y_train, y_test, 'XGBoost')
results.append(result)
models['XGBoost'] = xgboost
```

---

## 🎯 Model 5: LightGBM

```python
print("\n5️⃣ Training LightGBM (Fast and Efficient)...")
lgbm = lgb.LGBMRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgbm.fit(X_train, y_train)
result = evaluate_model(lgbm, X_train, X_test, y_train, y_test, 'LightGBM')
results.append(result)
models['LightGBM'] = lgbm

print("\n✅ All 5 models trained successfully!")
```

---

## 📊 Step 11: Model Comparison Summary

```python
# Create results dataframe
results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)
print(results_df.to_string(index=False))

# Find best model
best_idx = results_df['Test_R2'].idxmax()
best_model_name = results_df.loc[best_idx, 'Model']
best_r2 = results_df.loc[best_idx, 'Test_R2']
best_rmse = results_df.loc[best_idx, 'Test_RMSE']
best_mae = results_df.loc[best_idx, 'Test_MAE']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   R² Score: {best_r2:.4f}")
print(f"   RMSE: {best_rmse:.4f}")
print(f"   MAE: {best_mae:.4f}")
```

---

## 📊 Step 12: Visualize Model Comparison

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# RMSE Comparison
axes[0].bar(results_df['Model'], results_df['Test_RMSE'], color='coral', alpha=0.8)
axes[0].set_xlabel('Model', fontsize=11)
axes[0].set_ylabel('RMSE', fontsize=11)
axes[0].set_title('Test RMSE (Lower is Better)', fontsize=13, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', alpha=0.3)

# MAE Comparison
axes[1].bar(results_df['Model'], results_df['Test_MAE'], color='skyblue', alpha=0.8)
axes[1].set_xlabel('Model', fontsize=11)
axes[1].set_ylabel('MAE', fontsize=11)
axes[1].set_title('Test MAE (Lower is Better)', fontsize=13, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', alpha=0.3)

# R² Comparison
axes[2].bar(results_df['Model'], results_df['Test_R2'], color='lightgreen', alpha=0.8)
axes[2].set_xlabel('Model', fontsize=11)
axes[2].set_ylabel('R² Score', fontsize=11)
axes[2].set_title('Test R² Score (Higher is Better)', fontsize=13, fontweight='bold')
axes[2].tick_params(axis='x', rotation=45)
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 📊 Step 13: Train vs Test Performance

```python
# Create comparison plot for train vs test
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

x = np.arange(len(results_df))
width = 0.35

# RMSE - Train vs Test
axes[0].bar(x - width/2, results_df['Train_RMSE'], width, label='Train', alpha=0.8)
axes[0].bar(x + width/2, results_df['Test_RMSE'], width, label='Test', alpha=0.8)
axes[0].set_xlabel('Model', fontsize=11)
axes[0].set_ylabel('RMSE', fontsize=11)
axes[0].set_title('RMSE: Train vs Test', fontsize=13, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(results_df['Model'], rotation=45, ha='right')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# MAE - Train vs Test
axes[1].bar(x - width/2, results_df['Train_MAE'], width, label='Train', alpha=0.8)
axes[1].bar(x + width/2, results_df['Test_MAE'], width, label='Test', alpha=0.8)
axes[1].set_xlabel('Model', fontsize=11)
axes[1].set_ylabel('MAE', fontsize=11)
axes[1].set_title('MAE: Train vs Test', fontsize=13, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(results_df['Model'], rotation=45, ha='right')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

# R² - Train vs Test
axes[2].bar(x - width/2, results_df['Train_R2'], width, label='Train', alpha=0.8)
axes[2].bar(x + width/2, results_df['Test_R2'], width, label='Test', alpha=0.8)
axes[2].set_xlabel('Model', fontsize=11)
axes[2].set_ylabel('R² Score', fontsize=11)
axes[2].set_title('R² Score: Train vs Test', fontsize=13, fontweight='bold')
axes[2].set_xticks(x)
axes[2].set_xticklabels(results_df['Model'], rotation=45, ha='right')
axes[2].legend()
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 🎓 Step 14: User Input Prediction Function

```python
def predict_cgpa(user_input, model_name=None):
    """
    Predict CGPA from user input with median imputation
    
    Parameters:
    -----------
    user_input : dict
        Dictionary with user-provided feature values
    model_name : str (optional)
        Name of model to use. If None, uses best model
    
    Returns:
    --------
    float : Predicted CGPA (0-10 scale)
    """
    
    # Use best model if not specified
    if model_name is None:
        model_name = best_model_name
    
    # Get the model
    model = models[model_name]
    
    # Create full feature vector with median imputation
    full_input = pd.DataFrame([median_values])
    
    # Update with user values
    for key, value in user_input.items():
        if key in full_input.columns:
            full_input[key] = value
    
    # Ensure correct column order
    full_input = full_input[X_train.columns]
    
    # Make prediction (use scaled data for linear models)
    if model_name in ['Linear Regression', 'Ridge Regression']:
        full_input_scaled = scaler.transform(full_input)
        prediction = model.predict(full_input_scaled)[0]
    else:
        prediction = model.predict(full_input)[0]
    
    # Clip to valid range
    prediction = np.clip(prediction, 0, 10)
    
    return prediction

print("✅ Prediction function created!")
print(f"✅ Default model for predictions: {best_model_name}")
```

---

## 🧪 Step 15: Test Prediction with Example

```python
# Example user input (adjust feature names based on your dataset)
# Check your actual feature names by running: print(IMPORTANT_FEATURES)

example_input = {
    # Add your top important features here
    # Example format (adjust to your actual column names):
    # 'SGPA': 8.5,
    # 'Class_Attendance': 85,
    # 'Daily_Study_Hours': 4,
}

print("📝 Important Features Identified:")
print("="*50)
for i, feat in enumerate(IMPORTANT_FEATURES[:10], 1):
    print(f"{i:2d}. {feat}")

print("\n⚠️  Update the example_input dictionary above with actual feature names")
print("    and values, then uncomment the prediction code below.")

# Uncomment below after adding your features
"""
print("\nExample User Input:")
print("="*50)
for key, value in example_input.items():
    print(f"  {key}: {value}")

# Test with all 5 models
print("\n" + "="*50)
print("PREDICTIONS FROM ALL MODELS")
print("="*50)

for model_name in models.keys():
    predicted_cgpa = predict_cgpa(example_input, model_name)
    print(f"{model_name:<20s}: {predicted_cgpa:.2f} / 10.0")

# Best model prediction
print("\n" + "="*50)
print(f"🏆 BEST MODEL PREDICTION ({best_model_name})")
print("="*50)
predicted_cgpa = predict_cgpa(example_input)
print(f"🎓 Predicted CGPA: {predicted_cgpa:.2f} / 10.0")

# Performance category
if predicted_cgpa >= 9.0:
    category = "Outstanding 🌟"
elif predicted_cgpa >= 8.0:
    category = "Excellent ⭐"
elif predicted_cgpa >= 7.0:
    category = "Very Good ✨"
elif predicted_cgpa >= 6.0:
    category = "Good 👍"
elif predicted_cgpa >= 5.0:
    category = "Average 📚"
else:
    category = "Needs Improvement 💪"

print(f"Performance Category: {category}")
print("="*50)
"""
```

---

## 💾 Step 16: Save Models and Artifacts

```python
import pickle
import os

# Create models directory
os.makedirs('models', exist_ok=True)

# Save best model
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(models[best_model_name], f)

# Save all models
with open('models/all_models.pkl', 'wb') as f:
    pickle.dump(models, f)

# Save scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save median values
with open('models/median_values.pkl', 'wb') as f:
    pickle.dump(median_values, f)

# Save feature names
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)

# Save important features
with open('models/important_features.pkl', 'wb') as f:
    pickle.dump(IMPORTANT_FEATURES, f)

# Save label encoders
with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

# Save results
results_df.to_csv('models/model_results.csv', index=False)

print("✅ All models and artifacts saved successfully!")
print("\nSaved files in 'models/' directory:")
print("  📁 best_model.pkl")
print("  📁 all_models.pkl")
print("  📁 scaler.pkl")
print("  📁 median_values.pkl")
print("  📁 feature_names.pkl")
print("  📁 important_features.pkl")
print("  📁 label_encoders.pkl")
print("  📁 model_results.csv")
```

---

## 🎉 Final Summary

```python
print("\n" + "="*80)
print("🎉 CGPA PREDICTION PIPELINE COMPLETED!")
print("="*80)
print(f"\n📊 Dataset Statistics:")
print(f"   • Total Students: {df.shape[0]}")
print(f"   • Total Features: {df.shape[1]}")
print(f"   • Training Samples: {X_train.shape[0]}")
print(f"   • Testing Samples: {X_test.shape[0]}")

print(f"\n🤖 Models Trained: 5")
print("   1. Linear Regression")
print("   2. Ridge Regression")
print("   3. Random Forest")
print("   4. XGBoost")
print("   5. LightGBM")

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   • R² Score: {best_r2:.4f}")
print(f"   • RMSE: {best_rmse:.4f}")
print(f"   • MAE: {best_mae:.4f}")

print(f"\n🎯 Key Features for User Input: {len(IMPORTANT_FEATURES)}")
for i, feat in enumerate(IMPORTANT_FEATURES[:8], 1):
    print(f"   {i}. {feat}")

print("\n✅ All models saved and ready for deployment!")
print("="*80)

# Display final results table
print("\n📊 FINAL MODEL COMPARISON:")
print(results_df.to_string(index=False))
```

---

## 📝 Notes & Next Steps

**To use this notebook:**
1. ✅ Update the dataset filename in Step 2
2. ✅ Run all cells in order
3. ✅ Check the IMPORTANT_FEATURES list in Step 15
4. ✅ Update the example_input dictionary with actual feature names
5. ✅ Uncomment the prediction code in Step 15 to test

**Key Points:**
- CGPA automatically converted from 4.0 to 10.0 scale (×2.5)
- Missing values filled with median (numerical) or mode (categorical)
- 5 models trained: 2 linear, 3 tree-based
- User needs to provide only 8-10 key features
- Other features automatically filled with median values

**Next Steps:**
- Fine-tune hyperparameters
- Create web interface (Flask/Streamlit)
- Deploy the best model
- Add cross-validation
- Create API endpoints

---