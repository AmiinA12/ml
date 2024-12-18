from flask import Flask, request, render_template
import joblib
import numpy as np

# Charger le modèle sauvegardé
model = joblib.load("model2_turnover.pkl")

app = Flask(__name__)

# Page principale pour afficher l'interface HTML
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint pour effectuer la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer les données du formulaire HTML
    input_features = [request.form[col] for col in [
        'Age', 'Department', 'Job_Role', 'Job_Satisfaction', 
        'Monthly_Income', 'Over_Time', 
        'Total_Working_Years', 'Years_At_Company'
    ]]
    
    # Transformer les données en format adapté pour la prédiction
    features = np.array([input_features], dtype=object)
    prediction = model.predict(features)

    # Résultat
    result = "Yes" if prediction[0] == 1 else "No"
    return render_template('index.html', prediction_text=f"Résultat : {result}")

if __name__ == '__main__':
    app.run(debug=True)
