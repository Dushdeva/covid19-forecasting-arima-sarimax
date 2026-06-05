"""
Heart Disease Prediction
-------------------------------------------
Reads heart.csv, performs EDA, trains multiple models,
saves graphs and generates an HTML report in the Report/ folder.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)

# ------------------------------
# 1. Create folders and load data
# ------------------------------
os.makedirs("dataset", exist_ok=True)      # just in case
os.makedirs("Report", exist_ok=True)

# Load data (assuming heart.csv is already inside dataset/)
df = pd.read_csv("dataset/heart.csv")

print("Data loaded successfully!")
print(f"Shape: {df.shape}")
print(df.head())

# ------------------------------
# 2. Basic data exploration
# ------------------------------
# Save a summary to a text file inside Report
with open("Report/data_summary.txt", "w") as f:
    f.write("DATA SUMMARY\n")
    f.write("="*50 + "\n")
    f.write(f"Shape: {df.shape}\n")
    f.write(f"Columns: {list(df.columns)}\n\n")
    f.write("Missing values:\n")
    f.write(str(df.isnull().sum()) + "\n\n")
    f.write("Statistical description:\n")
    f.write(str(df.describe()))

# ------------------------------
# 3. Visualizations (saved as PNG)
# ------------------------------
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# 3.1 Target variable distribution
plt.figure()
sns.countplot(x='output', data=df, palette='Set2')
plt.title('Distribution of Heart Disease (0 = No, 1 = Yes)')
plt.savefig('Report/target_distribution.png')
plt.close()

# 3.2 Correlation heatmap
plt.figure(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.savefig('Report/correlation_heatmap.png')
plt.close()

# 3.3 Age distribution by target
plt.figure()
sns.histplot(data=df, x='age', hue='output', kde=True, bins=20, palette='viridis')
plt.title('Age Distribution by Heart Disease Status')
plt.savefig('Report/age_distribution.png')
plt.close()

# 3.4 Boxplots for key numerical features
num_features = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
for feat in num_features:
    plt.figure()
    sns.boxplot(x='output', y=feat, data=df, palette='Set1')
    plt.title(f'{feat} vs Heart Disease')
    plt.savefig(f'Report/boxplot_{feat}.png')
    plt.close()

# 3.5 Correlation with target (bar plot)
corr_with_target = corr['output'].sort_values(ascending=False)
plt.figure()
sns.barplot(x=corr_with_target.index, y=corr_with_target.values, palette='coolwarm')
plt.xticks(rotation=90)
plt.title('Correlation of Each Feature with Target')
plt.savefig('Report/corr_with_target.png')
plt.close()

# ------------------------------
# 4. Prepare data for modeling
# ------------------------------
X = df.drop('output', axis=1)
y = df['output']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features (important for Logistic Regression and SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# 5. Train models and evaluate
# ------------------------------
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(probability=True, random_state=42)
}

results = []
best_model = None
best_acc = 0

for name, model in models.items():
    # Train on scaled data (except Random Forest works fine with unscaled, but we use scaled for consistency)
    if name == 'Random Forest':
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_proba)
    cv_scores = cross_val_score(model, X_train_scaled if name!='Random Forest' else X_train,
                                y_train, cv=5, scoring='accuracy')
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'ROC-AUC': roc,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std()
    })
    
    if acc > best_acc:
        best_acc = acc
        best_model = name
    
    # Confusion matrix plot for each model
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(f'Report/cm_{name.replace(" ", "_")}.png')
    plt.close()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc:.2f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend()
    plt.savefig(f'Report/roc_{name.replace(" ", "_")}.png')
    plt.close()

# Create DataFrame of results
results_df = pd.DataFrame(results)
results_df.to_csv("Report/model_comparison.csv", index=False)

# Feature importance for Random Forest (if trained)
rf_model = models['Random Forest']
importances = rf_model.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,6))
feat_imp.plot(kind='bar', color='teal')
plt.title('Random Forest Feature Importance')
plt.ylabel('Importance')
plt.savefig('Report/feature_importance.png')
plt.close()

# ------------------------------
# 6. Generate HTML Report
# ------------------------------
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Heart Disease Prediction Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #2980b9; }}
        .container {{ max-width: 1200px; margin: auto; background: white; padding: 20px; border-radius: 10px; }}
        .summary {{ background: #ecf0f1; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background: #3498db; color: white; }}
        .good {{ color: green; font-weight: bold; }}
        img {{ max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #ccc; }}
        .img-row {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
        .img-card {{ width: 48%; margin-bottom: 20px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Heart Disease Prediction Project</h1>
    <p><strong>Dataset:</strong> heart.csv (303 samples, 13 features + target)</p>
    <p><strong>Goal:</strong> Predict presence of heart disease (output = 1) using clinical parameters.</p>

    <div class="summary">
        <h2>Data Overview</h2>
        <p>Shape: {df.shape[0]} rows, {df.shape[1]} columns</p>
        <p>Missing values: {df.isnull().sum().sum()} (none in this dataset)</p>
        <p>Target distribution: No disease = {(y==0).sum()} , Disease = {(y==1).sum()}</p>
        <p>Best performing model: <span class="good">{best_model}</span> with accuracy = {best_acc:.2f}</p>
    </div>

    <h2>Exploratory Visualizations</h2>
    <div class="img-row">
        <div class="img-card"><img src="target_distribution.png" alt="Target distribution"></div>
        <div class="img-card"><img src="age_distribution.png" alt="Age distribution"></div>
    </div>
    <div><img src="correlation_heatmap.png" alt="Correlation heatmap"></div>
    <div><img src="corr_with_target.png" alt="Correlation with target"></div>
    
    <h2>Boxplots of Key Features</h2>
    <div class="img-row">
        <div class="img-card"><img src="boxplot_age.png" alt="age boxplot"></div>
        <div class="img-card"><img src="boxplot_trtbps.png" alt="trtbps boxplot"></div>
    </div>
    <div class="img-row">
        <div class="img-card"><img src="boxplot_chol.png" alt="chol boxplot"></div>
        <div class="img-card"><img src="boxplot_thalachh.png" alt="thalachh boxplot"></div>
    </div>

    <h2>Model Performance Comparison</h2>
    {results_df.to_html(index=False, classes='table')}

    <h2>Confusion Matrices & ROC Curves</h2>
    <div class="img-row">
        <div class="img-card"><img src="cm_Logistic_Regression.png" alt="Logistic CM"></div>
        <div class="img-card"><img src="roc_Logistic_Regression.png" alt="Logistic ROC"></div>
    </div>
    <div class="img-row">
        <div class="img-card"><img src="cm_Random_Forest.png" alt="RF CM"></div>
        <div class="img-card"><img src="roc_Random_Forest.png" alt="RF ROC"></div>
    </div>
    <div class="img-row">
        <div class="img-card"><img src="cm_Support_Vector_Machine.png" alt="SVM CM"></div>
        <div class="img-card"><img src="roc_Support_Vector_Machine.png" alt="SVM ROC"></div>
    </div>

    <h2>Feature Importance (Random Forest)</h2>
    <img src="feature_importance.png" alt="Feature Importance">

    <footer style="margin-top: 30px; text-align: center; color: gray;">
        Report generated automatically by main.py | Heart Disease Prediction Project
    </footer>
</div>
</body>
</html>
"""

with open("Report/report.html", "w") as f:
    f.write(html_content)

print("\nAll done!")
print("Graphs and report saved inside the 'Report' folder.")
print("Open 'Report/report.html' in your browser to view the full analysis.")
