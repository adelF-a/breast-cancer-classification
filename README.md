# Breast Cancer Classification Project

This project explores three different machine learning models to classify breast cancer tumors as **Malignant** or **Benign** using the UCI Breast Cancer Wisconsin dataset.

## 🏆 Model Performance Comparison
We compared a "weighted expert," a "neighbor vote," and a "wisdom of the crowd" approach.

| Model | AUC Score | Best Use Case |
| :--- | :--- | :--- |
| **Logistic Regression** | **0.9974** | High interpretability & best performance for this data. |
| **Random Forest** | 0.9953 | Capturing complex, non-linear patterns. |
| **k-Nearest Neighbors** | ~0.9500 | Identifying similar patient "profiles." |

## 🧠 What the Models Learned
* **Logistic Regression:** Identified `worst texture` and `radius error` as the strongest indicators of malignancy.
* **Random Forest:** Found that `worst area` and `concave points` were the most important features across 100 decision trees.

## 🛠️ How to Use
1. Clone the repo.
2. Ensure `scikit-learn`, `pandas`, and `seaborn` are installed.
3. Run `breastcnser-logisticModel-sklearn.py` to see the winning model.

## 📅 Project Status
Completed on March 5, 2026. 
*Next steps: Deploying the Logistic Regression model into a web interface.*