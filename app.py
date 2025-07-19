import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__,template_folder='templates')

MODEL_PATH = "artifacts/models/random_forest_model.pkl"

with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

FEATURE_NAMES = ['Age', 'Fare', 'Sex', 'Embarked', 'Familysize', 'Isalone', 'HasCabin', 'Title', 'Pclass_Fare', 'Age_Fare', 'Pclass', 'entity_id']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        Age = float(data['Age'])
        Fare = float(data['Fare'])
        Sex = int(data['Sex'])
        Embarked = int(data['Embarked'])
        Familysize = int(data['Familysize'])
        Isalone = int(data['Isalone'])
        HasCabin = int(data['HasCabin'])
        Title = int(data['Title'])
        Pclass_Fare = float(data['Pclass_Fare'])
        Age_Fare = float(data['Age_Fare'])
        Pclass = int(data['Pclass'])
        entity_id = int(data['entity_id'])

        features = pd.DataFrame([[Age, Fare, Sex, Embarked, Familysize, Isalone, HasCabin, Title, Pclass_Fare, Age_Fare, Pclass, entity_id]])
        prediction = model.predict(features)[0]

        result = "Survived" if prediction == 1 else "Did not survive"

        return render_template('index.html', prediction_text=f'Prediction: {result}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')
    
if __name__ == "__main__":
    app.run(debug=True,host = '0.0.0.0',port=5000)