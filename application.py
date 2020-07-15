from flask import Flask,render_template,redirect,url_for,request
from datetime import datetime
import keras.models
from load import *
import os
app=Flask(__name__)

def get_index(out_time):

    import pandas as pd
    df = pd.read_csv("C:/Users/Anil/Desktop/desktop/T1.csv")

    # Feature Scaling
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0, 1))

    test_set = df.iloc[40000:, 1:].values
    test_set2 = df.iloc[40000:, 0:].values

    j = 0
    for i in range(0, 10000):
        k1 = datetime.strptime(test_set2[i][0], "%d %m %Y %H:%M")
        t = datetime.strftime(k1, "%d-%m-%Y %H-%M")
        j = j + 1
        if t == out_time:
            break
    j = j-1
    return j

@app.route('/')
def json():
    return render_template("pred.html")

@app.route('/forward/',methods=["POST"])
def pdt():

        date = request.form['datetime']
        x = date[0:10]
        y = date[11:]
        date = x + " " + y
        in_time = datetime.strptime(date, "%Y-%m-%d %H:%M")
        out_time = datetime.strftime(in_time, "%d-%m-%Y %H-%M")
        print(out_time)
        ind = get_index(out_time)
        print(ind)
        return redirect(url_for("pred", out=ind))

@app.route('/<out>/')
def pred(out):
    j = int(out)

    global model, graph
    model, graph = init()



    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.read_csv("C:/Users/Anil/Desktop/desktop/T1.csv")

    # Feature Scaling
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0, 1))

    test_set = df.iloc[40000:, 1:].values

    test_set_2 = sc.fit_transform(test_set)

    X_predict = []
    y_predict = []

    X_predict.append(test_set_2[j - 432:j, 1:])
    y_predict.append(test_set_2[j, 0])
    X_predict, y_predict = np.array(X_predict), np.array(y_predict)


    a1= model.predict(X_predict)
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv("C:/Users/Anil/Desktop/desktop/T1.csv")
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0, 1))
    test_set = df.iloc[40000:, 1:].values

    test_set_2 = sc.fit_transform(test_set)
    b = np.zeros((10529, 1))
    a1 = np.concatenate((a1, b), axis=0)

    a = np.zeros((10530, 1))
    a1 = np.concatenate((a1, a, a, a), axis=1)
    a1 = sc.inverse_transform(a1)
    n = df.iloc[40000+j:40000+j+1,1:2].values


    return render_template("pred2.html",output=a1[0][0],actual=n[0][0])



if __name__ == "__main__":
    app.run(debug=False)
