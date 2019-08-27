import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    z=[]
    int_features=[]
    for x in request.form.values():
        z.append(x)
    a=[z[0],x[1],z[2],z[3],z[4]]
    b=[z[5],z[6],z[7],z[8]]
    for x in request.form.values():
        z.append(x)
    a=[z[0],x[1],z[2],z[3],z[4]]
    b=[z[5],z[6],z[7],z[8]]
    for c in request.form.values():
        if c in a:
            int_features.append(int(c))
        if c in b:
            int_features.append(float(c))
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)


    return render_template('index.html', prediction_text='patient is having {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
