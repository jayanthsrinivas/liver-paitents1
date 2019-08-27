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
    int_features=[]
    a=[request.form[0],request.form[1],request.form[2],request.form[3],request.form[4]]
    b=[request.form[5],request.form[6],request.form[7],request.form[8]]
    for x in a:
        int_features.append(int(x))
    for y in b:
        int_features.append(float(y))
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)


    return render_template('index.html', prediction_text='patient is having {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
