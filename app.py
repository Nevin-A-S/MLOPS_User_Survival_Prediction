import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

from alibi_detect.cd import KSDrift
from src.feature_store import RedistFeatureStore
from sklearn.preprocessing import StandardScaler

from src.logger import get_logger

logger = get_logger(__name__)

app = Flask(__name__,template_folder='templates')

MODEL_PATH = "artifacts/models/random_forest_model.pkl"

with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

FEATURE_NAMES = ['Age', 'Fare', 'Sex', 'Embarked', 'Familysize', 'Isalone', 'HasCabin', 'Title', 'Pclass_Fare', 'Age_Fare', 'Pclass']

feature_store = RedistFeatureStore()
scaler = StandardScaler()

entity_id = feature_store.get_all_entity_ids()
all_features = feature_store.get_features_batch(entity_id)

all_features = pd.DataFrame.from_dict(all_features,orient='index')[FEATURE_NAMES]

historical_data = scaler.fit_transform(all_features[FEATURE_NAMES])

ksd = KSDrift(x_ref=historical_data, p_val=0.5)
print(scaler.get_feature_names_out())
    

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

        features = pd.DataFrame([[Age, Fare, Sex, Embarked, Familysize, Isalone, HasCabin, Title, Pclass_Fare, Age_Fare, Pclass]])
        
        scaled_features = scaler.transform(features)
        drift = ksd.predict(scaled_features)
        # print(f'Drift Response: {drift}')

        if drift['data']['is_drifted'] is not None:
            logger.warning("Drift detected in the input features.")

        prediction = model.predict(features)[0]

        result = "Survived" if prediction == 1 else "Did not survive"

        return render_template('index.html', prediction_text=f'Prediction: {result}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')
    
if __name__ == "__main__":
    app.run(debug=True,host = '0.0.0.0',port=5000)