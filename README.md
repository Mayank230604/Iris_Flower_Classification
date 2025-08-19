# ​ Iris Flower Classification (Pro Version)

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange) ![Status](https://img.shields.io/badge/status-completed-brightgreen)

This project classifies iris flowers into **Setosa, Versicolor, and Virginica** using multiple ML models.  
It demonstrates a complete **end-to-end ML workflow**: EDA → Preprocessing → Model Training → Hyperparameter Tuning → Evaluation → Visualization.

---

##  Pro Version Highlights
-  Data visualizations (pairplots, feature distributions)  
-  Cross-validation (5-fold CV)  
-  Hyperparameter tuning (GridSearchCV for KNN & Decision Tree)  
-  Support Vector Machine (SVM) added and best-performing  
-  Decision boundary plot (first two features)  
-  Professional structure with README, summary report, and organized folders  

---

##  Results

### Cross-Validation Accuracy
| Model                   | CV Accuracy |
|--------------------------|-------------|
| **SVM**                  | **0.9667**  |
| Logistic Regression      | 0.9600      |
| KNN (k=5)                | 0.9600      |
| Decision Tree (depth=4)  | 0.9533      |

**Best Hyperparameters:**
```python
{'KNN': {'clf__n_neighbors': 5}, 'DecisionTree': {'max_depth': 4}}
````

**Best Mode**l: **SVM**, achieving **97% test accuracy**

---

## Visual Highlights

| Pairplot                          | Confusion Matrix (SVM)                            | Decision Boundary (SVM)                             |
| --------------------------------- | ------------------------------------------------- | --------------------------------------------------- |
| ![Pairplot](figures/pairplot.png) | ![Confusion Matrix](figures/confusion_matrix.png) | ![Decision Boundary](figures/decision_boundary.png) |


## Future Enhancements

* Deploy via **Streamlit** or **Flask**
* Try ensembles like **Random Forest** or **XGBoost**
* Use **PCA** for dimensionality reduction and clean visuals
* Add **unit tests** and CI/CD for reproducibility

---

## Run Instructions

```bash
git clone https://github.com/Mayank230604/Iris_Flower_Classification-.git
cd Iris_Flower_Classification-
python -m venv venv
# Windows (PowerShell):
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
pip install -r requirements.txt
python src/iris_classification.py
```

