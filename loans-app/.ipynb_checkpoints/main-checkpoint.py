import json
import os
import pandas as pd
import joblib

from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request
from flask import url_for
# from googleapiclient import discovery
# from oauth2client.client import GoogleCredentials
from xgboost import XGBClassifier
from category_encoders import MEstimateEncoder

from custom_transformers import CrossFoldEncoder


app = Flask(__name__)


def get_prediction(features):

    trained_model = XGBClassifier()
    trained_model.load_model("xgb_model6f.json")
    feature_df = pd.DataFrame.from_dict(features,orient='index').T
    prediction = trained_model.predict(feature_df)

    return prediction[0]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():

    data = json.loads(request.data.decode())
    mandatory_items = ['term', 'dti',
                     'lti', 'num_open']
    for item in mandatory_items:
        if item not in data.keys():
            return jsonify({'result': 'Set all items.'})

    features = {}
    features['term'] = int(data['term'])
    features['dti'] = float(data['dti'])
    features['lti'] = float(data['lti'])
    features['acc_open_past_24mths'] = int(data['num_open'])

    with open('TEncoder.pkl', 'rb') as handle:
        tencoder = joblib.load(handle)
    
    print(f'opened pickle successfully')
    
    te_features = ['D1', 'RENT', 1]
    # te_features = [data['subgrade'], data['home_ownership'], 1]
    te_features = pd.DataFrame(te_features).T
    te_features.columns = ['sub_grade', 'home_ownership', 'target']
    encoded_features = tencoder.transform(te_features)

    print(f'encoded features are: {encoded_features}')
    
    features['sub_grade_encoded'] = encoded_features.sub_grade_encoded[0]
    features['home_ownership_encoded'] = encoded_features.home_ownership_encoded[0]

    print(f'features are: {features}')
    
    prediction = get_prediction(features)
    if prediction==0:
        output = 'No deafult'
    elif prediction==1:
        output = 'Default'
    else:
        output = 'Wrong output'
    return jsonify({'result': output})