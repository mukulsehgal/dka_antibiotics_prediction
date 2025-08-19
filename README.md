 DKA Antibiotics Prediction

This project uses a **Random Forest machine learning model** to predict whether patients with **Diabetic Ketoacidosis (DKA)** will require **antibiotic therapy**.  
It was developed as part of a research project exploring the role of **artificial intelligence (AI) in clinical decision-making**.

---

## 🚀 Live App

👉 [Try the Streamlit App](https://dka-antibiotics-prediction.streamlit.app)  
Upload your dataset (CSV or Excel) and get real-time predictions.

---

## 📂 Repository Structure

.
├── app.py # Streamlit app code
├── requirements.txt # Dependencies
├── rf_antibiotics_model.pkl # Trained Random Forest model (binary classifier)
├── streamlit/
│ └── config.toml # App theme configuration
├── results/ # Model results (ROC curve, confusion matrix, feature importances)
└── README.md # Project documentation

yaml
Copy
Edit

---

## ⚙️ Installation & Local Run

Clone this repo and install dependencies:

```bash
git clone https://github.com/mukulsehgal/dka_antibiotics_prediction.git
cd dka_antibiotics_prediction
pip install -r requirements.txt
Run the app locally:

bash
Copy
Edit
streamlit run app.py


**## 📊 Model Details**
Algorithm: Random Forest Classifier

Task: Binary classification

Target: Antibiotics (Yes/No)

Input Features: Labs and clinical variables (flexible to missing or renamed columns)

Performance
ROC-AUC: 0.845

Accuracy: ~83%

Precision/Recall: Balanced across classes

Key Visualizations
Confusion Matrix

ROC Curve

Feature Importances

## 📥 Usage
Input
Upload a CSV/Excel file with patient data.

The model is flexible with column names (auto-maps common variations like Glucose vs Blood Glucose).

Missing values are imputed.

Output
Predictions (Yes Antibiotics / No Antibiotics)

Probability scores

Downloadable results file

## 🔬 Research Context
Monitoring antibiotic use in DKA is crucial to:

Reduce unnecessary antibiotic exposure

Support antimicrobial stewardship

Improve patient outcomes

This project demonstrates how machine learning can help guide real-time clinical decisions using retrospective EHR data.

## 📈 Results
Metric	Score
Accuracy	0.83
Precision	0.77
Recall	0.72
ROC-AUC	0.845



##🛠️ Dependencies
See requirements.txt:

streamlit

pandas

numpy

scikit-learn

joblib

openpyxl

requests

## 📜 Citation
If you use this work in research or publications, please cite:

Sehgal M, et al.
Use of Artificial Intelligence to Predict Antibiotic Use in Pediatric DKA Patients
[In Preparation]

## 👨‍⚕️ Author
Mukul Sehgal, MD
Pediatric Intensivist | AI & Clinical Research Enthusiast
GitHub
