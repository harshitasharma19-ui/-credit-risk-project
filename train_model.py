import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    precision_score,
    recall_score,
    f1_score
)

# MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ---------------- CREATE DATA ----------------
np.random.seed(42)

data = pd.DataFrame({
    "age": np.random.randint(18, 60, 1000),
    "income": np.random.randint(20000, 100000, 1000),
    "credit_score": np.random.randint(300, 850, 1000),
    "loan_amount": np.random.randint(5000, 50000, 1000),
    "dti": np.random.randint(10, 50, 1000),
    "employment": np.random.randint(0, 20, 1000),
    "dependents": np.random.randint(0, 5, 1000),
    "education": np.random.randint(0, 3, 1000)
})

# ---------------- TARGET ----------------
data["risk"] = (
    (data["credit_score"] < 600) |
    (data["dti"] > 40) |
    (data["income"] < 30000)
).astype(int)

# ---------------- FEATURES ----------------
X = data.drop("risk", axis=1)
y = data["risk"]

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------- MODELS ----------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

best_model = None
best_accuracy = 0
best_model_name = ""

print("\n================ MODEL ACCURACIES ================\n")

# ---------------- TRAIN & TEST ----------------
for name, model in models.items():

    # TRAIN
    model.fit(X_train, y_train)

    # PREDICT
    y_pred = model.predict(X_test)

    # METRICS
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{name}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # SAVE BEST MODEL
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

# ---------------- SAVE BEST MODEL ----------------
pickle.dump(best_model, open("model_new.pkl", "wb"))

print("\n==================================================")
print(f"Best Model: {best_model_name}")
print(f"Best Accuracy: {best_accuracy:.4f}")
print("Best Model Saved Successfully!")
print("==================================================")

# ---------------- CONFUSION MATRIX ----------------
y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")

# ---------------- ROC CURVE ----------------
RocCurveDisplay.from_estimator(best_model, X_test, y_test)

plt.title("ROC Curve")
plt.savefig("roc_curve.png")

# ---------------- FEATURE IMPORTANCE ----------------
if best_model_name == "Random Forest":

    importances = best_model.feature_importances_

    feature_names = X.columns

    plt.figure(figsize=(10,5))

    sns.barplot(
        x=importances,
        y=feature_names
    )

    plt.title("Feature Importance")

    plt.savefig("feature_importance.png")

print("\nGraphs Saved Successfully!")