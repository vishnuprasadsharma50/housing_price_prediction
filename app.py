import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app= Flask(__name__)
model= pickle.load(open('regmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data= request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    modified_data= np.array(list(data.values())).reshape(1,-1)
    output= model.predict(modified_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data= [float(x) for x in request.form.values()]
    final_input= np.array(data).reshape(1,-1)
    output= model.predict(final_input)[0]
    return render_template("home.html", prediction_text="The predicted price is {}.".format(output))
            # Here prediction_text placeholder will be replaced by given value of Right hand side.

if __name__=="__main__":
    app.run(debug=True, port=8001)

print("File Run Successfully")