import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle

# Dummy data (later replace with real dataset)
data = {
    "age": [25, 45, 35, 50, 23, 40],
    "income": [50000, 100000, 75000, 120000, 40000, 90000],
    "loan": [1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df[["age", "income"]]
y = df["loan"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig("confusion_matrix.png")
plt.show()

# Save model
pickle.dump(model, open("model.pkl", "wb"))