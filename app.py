import os
import joblib
import pandas as pd
import gradio as gr

# ===========================
# Load Model
# ===========================
try:
    model = joblib.load("loan_prediction_model.pkl")
except Exception as e:
    print(f"Model Loading Error: {e}")
    model = None


# ===========================
# Prediction Function
# ===========================
def predict_loan_status(
    no_of_dependents,
    education,
    self_employed,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value,
):

    if model is None:
        return "❌ Model could not be loaded."

    try:
        no_of_dependents = int(no_of_dependents)
        education = int(education)
        self_employed = int(self_employed)
        income_annum = float(income_annum)
        loan_amount = float(loan_amount)
        loan_term = int(loan_term)
        cibil_score = int(cibil_score)
        residential_assets_value = float(residential_assets_value)
        commercial_assets_value = float(commercial_assets_value)
        luxury_assets_value = float(luxury_assets_value)
        bank_asset_value = float(bank_asset_value)
    except (ValueError, TypeError):
        return "❌ Please enter valid values."

    numeric_values = [
        no_of_dependents,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value,
    ]

    if any(v < 0 for v in numeric_values):
        return "❌ Negative values are not allowed."

    if not (300 <= cibil_score <= 900):
        return "❌ CIBIL Score must be between 300 and 900."

    # DataFrame with same feature names used during training
    input_df = pd.DataFrame([[
        no_of_dependents,
        education,
        self_employed,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    ]], columns=[
        " no_of_dependents",
        " education",
        " self_employed",
        " income_annum",
        " loan_amount",
        " loan_term",
        " cibil_score",
        " residential_assets_value",
        " commercial_assets_value",
        " luxury_assets_value",
        " bank_asset_value"
    ])

    try:
        prediction = model.predict(input_df)[0]

        confidence_text = ""
        try:
            probability = model.predict_proba(input_df)[0]
            confidence = round(max(probability) * 100, 2)
            confidence_text = f"\n\nConfidence: {confidence}%"
        except:
            pass

        if prediction == 1:
            return f"""✅ LOAN APPROVED

The applicant is eligible for the loan.{confidence_text}
"""
        else:
            return f"""❌ LOAN REJECTED

The applicant is not eligible for the loan.{confidence_text}
"""

    except Exception as e:
        return f"❌ Prediction Error\n\n{e}"


# ===========================
# UI
# ===========================

description = """
# 🏦 Loan Approval Prediction System

Fill in all applicant details and click **Predict**.

This prediction is made using a trained **Random Forest Classifier**.
"""

interface = gr.Interface(
    fn=predict_loan_status,

    inputs=[
        gr.Number(label="Number of Dependents"),

        gr.Dropdown(
            choices=[("Graduate", 1), ("Not Graduate", 0)],
            value=1,
            label="Education"
        ),

        gr.Dropdown(
            choices=[("Yes", 1), ("No", 0)],
            value=0,
            label="Self Employed"
        ),

        gr.Number(label="Annual Income"),
        gr.Number(label="Loan Amount"),
        gr.Number(label="Loan Term"),
        gr.Number(label="CIBIL Score (300-900)"),
        gr.Number(label="Residential Assets Value"),
        gr.Number(label="Commercial Assets Value"),
        gr.Number(label="Luxury Assets Value"),
        gr.Number(label="Bank Asset Value"),
    ],

    outputs=gr.Textbox(
        label="Prediction Result",
        lines=6
    ),

    title="🏦 Loan Approval Prediction",
    description=description
)

if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
