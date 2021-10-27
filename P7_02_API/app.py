# url API : https://app-flask-projet7.herokuapp.com/

import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load('data/model_lgbmc_final.pkl')
df = pd.read_csv('data/test_merge_10000.csv')
# test_merge_10000.csv is obtained by keeping only the first 10000 rows (clients) of test_merge.csv

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    client_ids = df['SK_ID_CURR'].unique().tolist()
    
    id = request.form['Client ID']
    id = int(id)
    
    if id in client_ids:
        X = df[df['SK_ID_CURR'] == id]
        X = X.drop(columns=['SK_ID_CURR'])
        prediction = model.predict(X)
        if prediction == 0:
            prediction = 'Loan Approved'
        else:
            prediction = 'Loan Not Approved'
    else:
        prediction = 'Error ID'

    return render_template('index.html', prediction_text=prediction)


@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    id = data['Client ID']
    X = df[df['SK_ID_CURR'] == id]
    X = X.drop(columns=['SK_ID_CURR'])
    prediction = model.predict_proba(X)[:, 1]
    return str(prediction[0])

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)