import pickle
from flask import Flask,jsonify,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application
## import ridge.pkl and scaler.pkl
ridge_model = pickle.load(open('models/ridge.pkl','rb'))
Standard_scaler = pickle.load(open('models/scaler.pkl','rb'))


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == "POST":
        try:
            temperature = float(request.form.get('temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            DC = float(request.form.get('DC'))
            BUI = float(request.form.get('BUI'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            new_data_scaled = Standard_scaler.transform(
                [[temperature, RH, Ws, Rain, FFMC, DMC,DC,BUI,ISI, Classes, Region]]
            )
            result = ridge_model.predict(new_data_scaled)
            return render_template('home.html', results=result[0])
        except Exception as e:
            return f"Error: {e}", 500
    else:
        return render_template('home.html')



if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000, debug=True)