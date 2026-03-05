import pandas as pd
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

# 1. Load data
raw_data = load_breast_cancer()
df = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
df['target'] = raw_data.target

# 2. Correlation Plot
correlation = df.corr()['target'].sort_values(ascending=False)
plt.figure(figsize=(10, 10))
sns.barplot(x=correlation.values, y=correlation.index, hue=correlation.index, palette='viridis', legend=False)
plt.title('Feature Correlation with Target')
plt.savefig('1_correlation_plot.png') # Saves to folder
plt.close() # Closes plot to save memory

# 3. Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='mean radius', y='mean texture', hue='target', alpha=0.7)
plt.title('Separation: Radius vs Texture')
plt.savefig('2_scatter_plot.png')
plt.close()

# 4. Training
X_train, X_test, y_train, y_test = train_test_split(raw_data.data, raw_data.target, test_size=0.2, random_state=42)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_probs = pipeline.predict_proba(X_test)[:, 1]

# 5. Confusion Matrix Plot
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
plt.title('Confusion Matrix')
plt.savefig('3_confusion_matrix.png')
plt.close()

# 6. ROC Curve Plot
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
auc_value = roc_auc_score(y_test, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {auc_value:.4f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('4_roc_curve.png')
plt.close()

print("All figures saved to folder. Model training complete.")




# 1. Access the logistic regression part of the pipeline
model = pipeline.named_steps['classifier']

# 2. Get the feature names
feature_names = raw_data.feature_names

# 3. Create a Series to view them clearly
weights = pd.Series(model.coef_[0], index=feature_names).sort_values(ascending=False)

print("\n--- The Model's Learned Weights ---")
print(weights)
print(f"Model Bias (Intercept): {model.intercept_[0]}")

# Create a comparison table
comparison = pd.DataFrame({
    'Correlation': df.corr()['target'].drop('target'),
    'Weight': pd.Series(model.coef_[0], index=raw_data.feature_names)
})

print(comparison.sort_values(by='Weight', ascending=False))


import joblib

# Save the pipeline to a file
joblib.dump(pipeline, 'cancer_model_pipeline.pkl')

print("Model saved as cancer_model_pipeline.pkl")