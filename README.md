# 🌸 Iris Flower Classification (Pro Version)

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)  
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)](https://scikit-learn.org/stable/)  
[![Status](https://img.shields.io/badge/status-completed-brightgreen)]()  

This project classifies iris flowers into **Setosa, Versicolor, and Virginica** using multiple ML models.  
It demonstrates a complete **end-to-end ML workflow**:  
**EDA → Preprocessing → Model Training → Hyperparameter Tuning → Evaluation → Visualization.**

---

## ✨ What's New in the Pro Version
- 📊 **Data Visualization** (pairplots, feature distributions)  
- 🔄 **Cross-validation (5-fold CV)** for reliable evaluation  
- 🔍 **Hyperparameter Tuning** with GridSearchCV (KNN, Decision Tree)  
- 🧠 Added **Support Vector Machine (SVM)** → Best model  
- 🎨 **Decision Boundary Plot** (first 2 features)  
- 📑 Clearer results summary & professional structure  

---

## 📊 Results

### Cross-validation Accuracy
| Model              | CV Accuracy |
|---------------------|-------------|
| **SVM**             | **0.9667** |
| Logistic Regression | 0.9600      |
| KNN (k=5)           | 0.9600      |
| Decision Tree (depth=4) | 0.9533 |

### Best Hyperparameters
```python
{'KNN': {'clf__n_neighbors': 5}, 'DecisionTree': {'max_depth': 4}}
