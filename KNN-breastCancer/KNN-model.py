import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

# 1. Load Data
raw_data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(raw_data.data, raw_data.target, test_size=0.2, random_state=42)

# 2. Build and Train k-NN Pipeline
pipeline_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])
pipeline_knn.fit(X_train, y_train)

# 3. Get Predictions and Probabilities
y_pred = pipeline_knn.predict(X_test)
y_probs = pipeline_knn.predict_proba(X_test)[:, 1]

# 4. Validation Fig 1: Confusion Matrix
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
plt.title('k-NN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('knn_confusion_matrix.png')
plt.show()

# 5. Validation Fig 2: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
auc_value = roc_auc_score(y_test, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='forestgreen', lw=2, label=f'k-NN ROC (AUC = {auc_value:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('k-NN ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('knn_roc_curve.png')
plt.show()

print(f"k-NN Accuracy: {pipeline_knn.score(X_test, y_test):.4f}")
print(f"k-NN AUC Score: {auc_value:.4f}")


import joblib
joblib.dump(pipeline_knn, 'knn_cancer_model.pkl')