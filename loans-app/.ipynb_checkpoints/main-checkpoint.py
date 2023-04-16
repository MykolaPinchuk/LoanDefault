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


class CrossFoldEncoder:
    def __init__(self, encoder, **kwargs):
        self.encoder_ = encoder
        self.kwargs_ = kwargs  # keyword arguments for the encoder
        self.cv_ = KFold(n_splits=4)

    # Fit an encoder on one split and transform the feature on the
    # other. Iterating over the splits in all folds gives a complete
    # transformation. We also now have one trained encoder on each
    # fold.
    def fit_transform(self, X, y, cols):
        self.fitted_encoders_ = []
        self.cols_ = cols
        X_encoded = []
        for idx_encode, idx_train in self.cv_.split(X):
            fitted_encoder = self.encoder_(cols=cols, **self.kwargs_)
            fitted_encoder.fit(
                X.iloc[idx_encode, :], y.iloc[idx_encode],
            )
            X_encoded.append(fitted_encoder.transform(X.iloc[idx_train, :])[cols])
            self.fitted_encoders_.append(fitted_encoder)
        X_encoded = pd.concat(X_encoded)
        X_encoded.columns = [name + "_encoded" for name in X_encoded.columns]
        return X_encoded

    # To transform the test data, average the encodings learned from
    # each fold.
    def transform(self, X):
        from functools import reduce

        X_encoded_list = []
        for fitted_encoder in self.fitted_encoders_:
            X_encoded = fitted_encoder.transform(X)
            X_encoded_list.append(X_encoded[self.cols_])
        X_encoded = reduce(
            lambda x, y: x.add(y, fill_value=0), X_encoded_list
        ) / len(X_encoded_list)
        X_encoded.columns = [name + "_encoded" for name in X_encoded.columns]
        return X_encoded


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

    te_features = ['C1', 'OWN', 1]
    # te_features = [data['subgrade'], data['home_ownership'], 1]
    te_features = pd.DataFrame(te_features).T
    te_features.columns = ['sub_grade', 'home_ownership', 'target']
    encoded_features = tencoder.transform(te_features)

    features['sub_grade_encoded'] = encoded_features.sub_grade_encoded
    features['home_ownership_encoded'] = encoded_features.home_ownership_encoded


    prediction = get_prediction(features)
    if prediction==0:
        output = 'No deafult'
    elif prediction==1:
        output = 'Default'
    else:
        output = 'Wrong output'
    return jsonify({'result': output})