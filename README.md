# ML_WEEK4_B01
# 🚀 ML-Internship

> **Name:** Muhammad Fahad  
> **Email:** [![Email](https://img.shields.io/badge/Email-mfk21927@gmail.com-red?style=flat-square&logo=gmail&logoColor=white)](mailto:mfk21927@gmail.com)  
> **LinkedIn:** [![LinkedIn](https://img.shields.io/badge/LinkedIn-Muhammad%20Fahad-blue?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/muhammad-fahad-087057293)  
> **Start Date:** 20-12-2025  

---

![Internship](https://img.shields.io/badge/Status-Active-blue?style=for-the-badge)
![Batch](https://img.shields.io/badge/Batch-B01-orange?style=for-the-badge)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-0.25-orange?logo=scikitlearn&logoColor=white)]
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Project Overview
This repository documents my **Week 4 Machine Learning Internship tasks**, focused on **classification algorithms**.  
It includes **Logistic Regression, KNN, Decision Tree, and Multi-class Classification**, with model evaluation, visualizations, and model saving.

---

## 📈 Week 4 Tasks Overview

| Task | Title | Dataset | Status |
| :--- | :--- | :--- | :--- |
| 4.1 | Logistic Regression | Breast Cancer Wisconsin | ✅ Completed |
| 4.2 | KNN Classification | Iris | ✅ Completed |
| 4.3 | Decision Tree Classifier | Iris (or Wine Quality) | ✅ Completed |
| 4.4 | Multi-class Classification | Digits | ✅ Completed |

---

## ✅ Task Details

### **Task 4.1: Logistic Regression (Binary Classification)**

- **Dataset:** Breast Cancer Wisconsin (`sklearn.datasets.load_breast_cancer`)  
- **Steps Implemented:**  
  - Data exploration with `df.head()` and `df.info()`  
  - Check class distribution  
  - Train-test split and feature scaling (`StandardScaler`)  
  - Train `LogisticRegression`  
  - Predictions and metrics: **Accuracy, Precision, Recall, F1-score, ROC-AUC**  
  - Plotted **Confusion Matrix** using Seaborn and **ROC Curve**  
  - Saved model as `.pkl`  

**Files:** `logistic_regression.py`, `logistic_regression.pkl`  

**Visuals:**  

Confusion Matrix  
![Confusion Matrix](visuals/cm_LR.png)  

ROC Curve  
![ROC Curve](visuals/roc_curve_logi.png)  

---

### **Task 4.2: K-Nearest Neighbors (KNN) Classification**

- **Dataset:** Iris (`sklearn.datasets.load_iris`)  
- **Steps Implemented:**  
  - Selected 2 features for visualization  
  - Tested K = 1,3,5,7,11,15 with **Euclidean** and **Manhattan** metrics  
  - Calculated accuracy for each K and plotted **Accuracy vs K**  
  - Visualized **decision boundary** for K=5  
  - Generated comparison table and identified optimal K  

**Files:** `knn_classification.py`  

**Visuals:**  

Accuracy vs K  
![Accuracy vs K](visuals/acc_vs_k_value.png)  

Decision Boundary (K=5)  
![Decision Boundary](visuals/knn_decission_bound.png)  

Comparison Table  
![Comparison Table](visuals/comp_table_knn.PNG)  

---

### **Task 4.3: Decision Tree Classifier**

- **Dataset:** Iris (or Wine Quality)  
- **Steps Implemented:**  
  - Trained Decision Tree with `max_depth = 3,5,10,None`  
  - Compared accuracy to find optimal depth  
  - Visualized **Tree Structure** (`plot_tree`)  
  - Extracted **feature importances** and plotted bar chart  
  - Saved **best model** as `.pkl` and exported tree as image  

**Files:** `decision_tree.py`, `best_decision_tree.pkl`, `decision_tree.png`  

**Visuals:**  

Decision Tree  
![Decision Tree](decission_tree_structure.png)  

Feature Importance  
![Feature Importance](visuals/features_d_tree.png)  

---

### **Task 4.4: Multi-class Classification & Evaluation Metrics**

- **Dataset:** Digits (`sklearn.datasets.load_digits`)  
- **Steps Implemented:**  
  - Train-test split and feature scaling  
  - Trained **Logistic Regression (OVR)**, **KNN**, and **Decision Tree**  
  - Evaluated with **classification_report**, **confusion matrix**, and **accuracy**  
  - Plotted side-by-side **confusion matrices**  
  - Created **accuracy comparison bar chart**  

**Files:** `multiclass_classification.py`  

**Visuals:**  

Confusion Matrices  
![Confusion Matrices](visuals/multiclass_comparison.png.png)  

Accuracy Comparison  
![Accuracy Comparison](visuals/model_acc_comparison.png.png)  

---

## 🧠 ML Projects

- Logistic Regression (Binary)  
- KNN Classification (Optimal K)  
- Decision Tree (Pruned, Feature Importance)  
- Multi-class Classification (Digits Dataset)  

---

## 💻 Tech Stack
* **Languages:** Python, Markdown  
* **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn  
* **Tools:** Git, VS Code, Pickle, 

---

## 📜 License
This project is licensed under the MIT License.
