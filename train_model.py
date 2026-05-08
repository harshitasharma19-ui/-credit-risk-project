import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODELS ----------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

best_model = None
best_accuracy = 0

print("\nMODEL ACCURACIES:\n")

# ---------------- TRAIN & TEST ----------------
for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"{name}: {acc:.4f}")

    # SAVE BEST MODEL
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# ---------------- SAVE BEST MODEL ----------------
pickle.dump(best_model, open("model_new.pkl", "wb"))

print("\nBest Model Saved Successfully!")
print("Best Accuracy:", best_accuracy)