from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("loan_prediction_model.pkl")


@app.route("/")
def home():
    return """
    <h2>Loan Prediction API is Running 🚀</h2>
    <p>Send a POST request to <b>/predict</b></p>
    """


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        df = pd.DataFrame([{
            " no_of_dependents": data["no_of_dependents"],
            " education": data["education"],
            " self_employed": data["self_employed"],
            " income_annum": data["income_annum"],
            " loan_amount": data["loan_amount"],
            " loan_term": data["loan_term"],
            " cibil_score": data["cibil_score"],
            " residential_assets_value": data["residential_assets_value"],
            " commercial_assets_value": data["commercial_assets_value"],
            " luxury_assets_value": data["luxury_assets_value"],
            " bank_asset_value": data["bank_asset_value"]
        }])

        prediction = model.predict(df)[0]

        return jsonify({
            "prediction": str(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
