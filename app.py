import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

from alibi_detect.cd import KSDrift
from src.feature_store import RedistFeatureStore
from sklearn.preprocessing import StandardScaler

from prometheus_client import start_http_server,Counter,Gauge

from src.logger import get_logger

logger = get_logger(__name__)

app = Flask(__name__,template_folder='templates')

prediction_counter = Counter('prediction_counter', 'Number of predictions made')
data_drift_counter = Counter('data_drift_counter', 'Number of data drift detections')

prediction_latency = Gauge('prediction_latency', 'Latency of predictions in seconds')

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

        features = pd.DataFrame([[Age, Fare, Sex, Embarked, Familysize, Isalone, HasCabin, Title, Pclass_Fare, Age_Fare, Pclass]], 
                               columns=FEATURE_NAMES)
    
        scaled_features = scaler.transform(features)
        
        try:
            drift = ksd.predict(scaled_features)

            is_drifted = False
            if 'data' in drift:
                if 'is_drift' in drift['data']:
                    is_drifted = drift['data']['is_drift'] == 1
            
            if is_drifted:
                data_drift_counter.inc()
                logger.warning("Drift detected in the input features.")
                
        except Exception as drift_error:
            logger.error(f"Drift detection error: {drift_error}")
            
        try:
            prediction = model.predict(features)[0]
        except Exception as model_error:
            logger.warning(f"Model prediction with unscaled features failed: {model_error}")

            prediction = model.predict(scaled_features)[0]
            
        prediction_counter.inc()

        result = "Survived" if prediction == 1 else "Did not survive"

        return render_template('index.html', prediction_text=f'Prediction: {result}')
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

@app.route('/metrics')
def metrics():
    from prometheus_client import generate_latest
    from flask import Response
    return Response(generate_latest(),content_type='text/plain') 

if __name__ == "__main__":
    start_http_server(8000) 
    app.run(debug=True,host = '0.0.0.0',port=5000)
