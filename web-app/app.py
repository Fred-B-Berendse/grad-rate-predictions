from flask import Flask, render_template, request, json
import requests
import webbrowser, threading, os
import numpy as np
import pandas as pd
from pymongo import MongoClient
from datapoints import DataPoint
from model import Model

client = MongoClient()
# from model import Model

app = Flask(__name__)

def pipeline():

    dp = DataPoint()
    data_point = dp.retrieve()

    # Fake fraud results
    # fraud_prob = round(np.random.random(), 2)
    # fraud_flag = fraud_prob > 0.5

    mod = Model()
    mod.load_model('models/model_all_features_fin.pkl')
    fraud_flag, fraud_prob = mod.preds_new_data_point(data_point)    

    # merge our results with the original JSON event object
    results = {'fraud_flag': fraud_flag,
               'fraud_prob': fraud_prob,
               'event': dp.json_data} # event 
    results_copy = results.copy()
    client = MongoClient('mongodb://localhost:27017')
    db = client['fraud_db'] 
    events = db.events
    db_result = events.insert_one(results)
    print('One post: {0}'.format(db_result.inserted_id))
    print('Acknowledged: {}'.format(db_result.acknowledged))
    return results_copy


@app.route('/')
def index():
    return render_template('index.html')

@app.route("/visualize", methods=['POST'])
def visualize():   
    # response = requests.get('http://galvanize-case-study-on-fraud.herokuapp.com/data_point')
    # event = response.json()
    results = pipeline()
    return render_template('visualize.html', 
                           object_id=results['event']['object_id'],
                           event_name=results['event']['name'],
                           org_name=results['event']['org_name'],
                           description=results['event']['description'],
                           fraud_flag=results['fraud_flag'], 
                           fraud_prob=results['fraud_prob'])

@app.route("/api", methods=['GET', 'POST'])
def api():
    # response = requests.get('http://galvanize-case-study-on-fraud.herokuapp.com/data_point')
    # event = response.json()
    results = pipeline()
    response = app.response_class(
        response=json.dumps(results),
        status=200,
        mimetype='application/json'
    )

    return response

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8105, threaded=True)