#import Flask 
import numpy as np
import joblib, sklearn
from flask import Flask, render_template, request
from xgboost import XGBClassifier
#create an instance of Flask
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict/', methods=['GET','POST'])
def predict():
    if request.method == "POST":
        term = request.form.get('term')
        dti = request.form.get('dti')
        acc24 = request.form.get('acc24')
        lti = request.form.get('lti')
        subgrade = request.form.get('subgrade')
        employment = request.form.get('employment')
        home_ownership = request.form.get('home_ownership')
        #call preprocessDataAndPredict and pass inputs
        try:
            prediction = preprocessDataAndPredict(term, 
                                                  dti, 
                                                  acc24, 
                                                  lti, 
                                                  subgrade, 
                                                  employment, 
                                                  home_ownership)
            #pass prediction to template
            return render_template('predict.html', prediction = prediction)
        except ValueError:
            return "Please Enter valid values"
        pass        
    pass
def preprocessDataAndPredict(term, dti, acc24, lti, subgrade, employment, home_ownership):
    test_data = [term, dti, acc24, lti, subgrade, employment, home_ownership]
    print(test_data)
    test_data = np.array(test_data).astype(np.float) 
    test_data = test_data.reshape(1,-1)
    print(test_data)   
    trained_model = XGBClassifier()
    trained_model.load_model('xgb_model.bst')
    prediction = trained_model.predict(test_data)
    return prediction
    pass
if __name__ == '__main__':
    app.run(debug=True)
