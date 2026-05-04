from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return "Server Running ✅"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]

    prediction = model.predict([data])
    prob = model.predict_proba([data])[0][1]

    return jsonify({
        "prediction": int(prediction[0]),
        "probability": float(prob)
    })

if __name__ == "__main__":
    app.run(debug=True)