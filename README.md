# 🏦 Loan Approval Prediction System (Deployment Repository)

## 📖 About the Project
The **Loan Approval Prediction System** is a machine learning application designed to predict whether an applicant's loan will be approved or rejected based on 11 personal and financial metrics (such as CIBIL Score, Annual Income, Loan Amount, and Asset Values). It utilizes a **Random Forest Classifier** to analyze the applicant's profile and assess their eligibility.

⚠️ **Important Note:** This repository is specifically structured and configured for **Render Deployment/Hosting**. 

If you are looking for the complete machine learning workflow—including data exploration, feature selection, outlier removal, model training, and the raw dataset—please visit the Google Colab Notebook linked below:
👉 **[Full Source Code & Dataset (Google Colab)]([YOUR_COLAB_LINK_HERE](https://colab.research.google.com/drive/1Mb4VK3x1xwRpS4GFFrO0WP9-gGz5rh4C#scrollTo=aAzTNXgmOfB1))**

## 👨‍💻 Developer
**Aleesha**
* **LinkedIn:** [Connect with me]
* **GitHub:** [Check out my projects](https://github.com/Deepak-1765)

## 🛠️ Tools & Technologies Used
* **Machine Learning:** Scikit-learn (Random Forest Classifier)
* **Data Processing:** Pandas, NumPy
* **Model Serialization:** Joblib
* **Web UI Framework:** Gradio
* **Cloud Hosting:** Render
* **Development Environment:** Google Colab / Jupyter

---

## 🚀 How to Host Your Own Model on Render
You can use this repository as a template to host your own machine learning models on the web for free. Follow these step-by-step instructions:

### Step 1: Prepare Your Code
1. **Clone this repository** to your local machine.
2. **Push the code** to a new, public repository on your personal GitHub account.
3. **Train your model** (you can use the provided Colab link above) and export it as a `.pkl` file using `joblib`.
4. Place your `loan_prediction_model.pkl` file into the root of your cloned repository.
5. **Configure `app.py`**: Open the `app.py` file and update it so that the input parameters match the exact features your model was trained on. Ensure you also update the code to load your specific `.pkl` file.

### Step 2: Deploy on Render
1. Go to [Render.com](https://render.com/) and Log In or Sign Up.
2. Click on the **New** button in the dashboard and select **Web Service**.
3. Connect your GitHub account and select the repository you just created.
4. Fill in the deployment details:
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `python app.py`
5. Scroll down and select the **Free** instance type.
6. Click **Create Web Service**. 

Boom! 💥 Render will build your environment and give you a live URL. Your machine learning project is now live on the internet!

---

## 🌐 Live Demo
Check out the live, working version of this project hosted on Render:
👉 **[Loan Approval Prediction System - Live App](https://loan-prediction-system-yi7p.onrender.com)**
