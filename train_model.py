import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

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

# ---------------- TRAIN ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# ---------------- SAVE ----------------
pickle.dump(model, open("model.pkl", "wb"))

print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))