import streamlit as st
import joblib
import numpy as np

# Charger le modèle sauvegardé
model = joblib.load("model2_turnover.pkl")

# Fonction pour effectuer la prédiction
def predict_turnover(age, department, job_role, job_satisfaction, monthly_income, over_time, total_working_years, years_at_company):
    input_features = np.array([[age, department, job_role, job_satisfaction, monthly_income, over_time, total_working_years, years_at_company]], dtype=object)
    prediction = model.predict(input_features)
    return "Yes" if prediction[0] == 1 else "No"

# Interface utilisateur Streamlit
st.title("Prédiction de Turnover des Employés")

# Formulaire pour entrer les données
age = st.number_input("Age", min_value=18, max_value=100, step=1)
department = st.text_input("Department")
job_role = st.text_input("Job Role")
job_satisfaction = st.number_input("Job Satisfaction", min_value=1, max_value=5, step=1)
monthly_income = st.number_input("Monthly Income", min_value=0, step=100)
over_time = st.selectbox("Over Time", ["Yes", "No"])
total_working_years = st.number_input("Total Working Years", min_value=0, step=1)
years_at_company = st.number_input("Years At Company", min_value=0, step=1)

# Bouton pour prédire
if st.button("Prédire"):
    if not (department and job_role):  # Vérification des champs nécessaires
        st.error("Veuillez remplir tous les champs.")
    else:
        # Conversion de "Over Time" en format binaire
        over_time = 1 if over_time == "Yes" else 0
        result = predict_turnover(age, department, job_role, job_satisfaction, monthly_income, over_time, total_working_years, years_at_company)
        st.success(f"Résultat : {result}")
