import numpy as np
import pandas as pd
from flask import Flask, request, render_template
# from yourapplication import app
# from flask_mobility import Mobility
# import pkg1
# import pkg2
import pickle

app1 = Flask(__name__)
model = pickle.load(open('model1.pkl', 'rb'))

# @app1.route('/')
# def index():
#     return render_template('index.html')
# @app1.route('/work')
# def work():
#     return render_template('work.html')
# @app1.route('/breastcancer')
# def breastcancer():
#     return render_template('BreastCancer.html')

@app1.route('/heartdisease')
def heartdisease():
    return render_template('HeartDisease.html')

@app1.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['age', 'sex', 'cp', 'trestbps',
       'chol', 'fbs', 'restecg',
       'thalach', 'exang', 'oldpeak',
       'slope', 'ca', 'thal', 'target',
       ]
    
    data = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(data)
        
    if output == 0:
        res_val = "** Possible Heart Disease  **"
    else:
        res_val = "The Patient is safe."
        

    return render_template('HeartDisease.html', prediction_text='Patient has {}'.format(res_val))

if __name__ == "__main__":
    app1.run()