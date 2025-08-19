# Iris Flower Classification (Pro Version)

This project classifies iris flowers into Setosa, Versicolor, and Virginica using multiple ML models.

## What's New in Pro Version
- Data visualization (pairplots, feature distributions)
- Cross-validation (5-fold CV)
- Hyperparameter tuning (GridSearchCV for KNN, Decision Tree)
- Extra model: Support Vector Machine (SVM)
- Decision boundary plot (first 2 features)
- Clearer results summary

## Results
- Cross-validation Accuracies:
                    CV Accuracy
SVM                    0.966667
LogisticRegression     0.960000
KNN                    0.960000
DecisionTree           0.953333

- Best hyperparameters:
{'KNN': {'clf__n_neighbors': 5}, 'DecisionTree': {'max_depth': 4}}

- Best Model: SVM

Confusion matrix and plots are saved in `figures/`.

## Run Instructions
```bash
pip install -r requirements.txt
python src/iris_classification.py
```
# Iris_Flower_Classification-
