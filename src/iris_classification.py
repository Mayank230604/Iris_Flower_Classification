# Iris Classification Pro Version Script
# Trains Logistic Regression, KNN (with tuning), Decision Tree (with tuning), and SVM
# Evaluates with cross-validation and test split

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def main():
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    target_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42, stratify=y)

    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier())
        ]),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC())
        ])
    }

    # Cross-validation
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5)
        print(f"{name}: CV Accuracy = {scores.mean():.4f}")

    # Hyperparameter tuning
    knn_params = {"clf__n_neighbors": [3, 5, 7, 9]}
    grid_knn = GridSearchCV(models["KNN"], knn_params, cv=5)
    grid_knn.fit(X_train, y_train)
    print("Best KNN params:", grid_knn.best_params_)

    dt_params = {"max_depth": [2, 3, 4, 5, None]}
    grid_dt = GridSearchCV(models["DecisionTree"], dt_params, cv=5)
    grid_dt.fit(X_train, y_train)
    print("Best DecisionTree params:", grid_dt.best_params_)

    # Fit best models
    final_models = {
        "LogisticRegression": models["LogisticRegression"].fit(X_train, y_train),
        "KNN": grid_knn.best_estimator_.fit(X_train, y_train),
        "DecisionTree": grid_dt.best_estimator_.fit(X_train, y_train),
        "SVM": models["SVM"].fit(X_train, y_train)
    }

    for name, model in final_models.items():
        y_pred = model.predict(X_test)
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))

    # Confusion matrix for best model (by CV)
    best_model_name = "SVM"  # update if needed
    best_model = final_models[best_model_name]
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
    plt.figure()
    disp.plot(values_format='d')
    plt.title(f"Confusion Matrix — {best_model_name}")
    plt.tight_layout()
    plt.savefig("figures/confusion_matrix.png")

if __name__ == "__main__":
    main()
