# app.py

# --- CHANGED BLOCK START ---
import gradio as gr
import joblib
import spaces

# We load the model once when the app starts
deployed_lr = joblib.load('my_first_ml_model.pkl')

# --- ZERO-GPU DECORATOR AND PREDICTION LOGIC ---
@spaces.GPU
def predict_rent(size_of_prop):
    # The model expects a 2D array: [[size]]
    prediction = deployed_lr.predict([[size_of_prop]])
    # Extract the single prediction value and format it
    return f"Estimated Rent: {prediction[0]:.2f}"

# Create the web interface
interface = gr.Interface(
    fn=predict_rent,
    inputs=gr.Number(label="Please Enter the Size of Your Property for rent"),
    outputs=gr.Text(label="Predicted Rent"),
    title="Property Rent Predictor",
    description="Enter the property size to get a rent estimate powered by Machine Learning."
)

if __name__ == "__main__":
    interface.launch()
# --- CHANGED BLOCK END ---