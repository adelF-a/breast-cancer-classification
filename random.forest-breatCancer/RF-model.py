import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

# 1. Load Data
raw_data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(raw_data.data, raw_data.target, test_size=0.2, random_state=42)

# 2. Build the Random Forest Pipeline
# n_estimators=100 creates 100 decision trees to vote
pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 3. Train the Model
pipeline_rf.fit(X_train, y_train)

# 4. Predictions & Probabilities
y_pred = pipeline_rf.predict(X_test)
y_probs = pipeline_rf.predict_proba(X_test)[:, 1]

# 5. Validation Figures
plt.figure(figsize=(12, 5))

# Subplot 1: Confusion Matrix
plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
            xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
plt.title('RF Confusion Matrix')

# Subplot 2: ROC Curve
plt.subplot(1, 2, 2)
fpr, tpr, _ = roc_curve(y_test, y_probs)
auc_val = roc_auc_score(y_test, y_probs)
plt.plot(fpr, tpr, color='rebeccapurple', label=f'AUC = {auc_val:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('RF ROC Curve')
plt.legend()

plt.tight_layout()
plt.savefig('rf_validation_results.png')
plt.show()

# 6. Feature Importance Analysis
rf_model = pipeline_rf.named_steps['rf']
importances = pd.Series(rf_model.feature_importances_, index=raw_data.feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x=importances.values[:10], y=importances.index[:10], palette='viridis')
plt.title('Top 10 Most Important Features (Random Forest)')
plt.savefig('rf_feature_importance.png')
plt.show()

print(f"Final AUC Score: {auc_val:.4f}")
print("\n--- Top Features ---")
print(importances.head(5))