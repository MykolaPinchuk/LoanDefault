import json
import os
import pandas as pd

from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request
from flask import url_for
# from googleapiclient import discovery
# from oauth2client.client import GoogleCredentials
from xgboost import XGBClassifier


# credentials = GoogleCredentials.get_application_default()
# api = discovery.build('ml', 'v1',
#         credentials=credentials, cache_discovery=False)
# project = os.environ['GOOGLE_CLOUD_PROJECT']
# model_name = os.getenv('MODEL_NAME', 'babyweight')


app = Flask(__name__)


def get_prediction(features):

    trained_model = XGBClassifier()
    trained_model.load_model("xgb_model.json")
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

    prediction = get_prediction(features)
    if prediction==0:
        output = 'No deafult'
    elif prediction==1:
        output = 'Default'
    else:
        output = 'Wrong output'
    return jsonify({'result': output})