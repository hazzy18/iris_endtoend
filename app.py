from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
model=pickle.load(open('iris.pkl', 'rb'))

app = Flask(__name__) 
@app.route('/')
def home():
    return render_template("home.html")

@app.route("/",methods=["POST"])
def predict():
    data1= float(request.form['first'])
    data2= float(request.form['second'])
    data3= float(request.form['third'])
    data4= float(request.form['fourth'])
    pred=model.predict(np.array([[data1, data2, data3, data4]]))
    return render_template("home.html",data=pred)

if __name__ == '__main__':
  app.run(debug=True)


