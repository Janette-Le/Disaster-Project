#from winreg import REG_QWORD
##importing required resources/libraries
import sys
from grpc import Status
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

if len(sys.argv) < 2:
    print("Enter picle file address to load model")
    quit()

app = Flask(__name__)
#loading classification model already trained or saved
model = pickle.load(open(sys.argv[1], 'rb'))
##Preprocessed data is cleaned and saved as Cleaned.csv file which is loaded for tfd_idf corpus loading
data = pd.read_csv("../Cleaned.csv")
data_text = data["message"]
##TF-IDF vectorizer
vectorizer = TfidfVectorizer(analyzer = 'word' , stop_words = 'english', max_features = 500)
vectorizer.fit(data_text.values.astype('U'))
x =vectorizer.fit_transform(data_text.values.astype('U'))


def predictOutput(input):##input is taken as string form
    series = pd.Series([input])##converted to pandas core series form in order to vectorize it as well
    new = vectorizer.transform(series)
    result = model.predict(new)
    return result ##returning result

@app.route('/')
def home():
    return render_template('master.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    input_text = request.form['query']
    result = (predictOutput(input_text))
    print(result)
    
    print(f"Input Text is {input_text}")
    print("Result is ", result[0])
    sum_of_result = sum(list(result[0]))
    ##returning elements to be rendered
    return render_template('master.html', one = (result[0][0]), two = (result[0][1]),three = (result[0][2]),four = (result[0][3]),five = (result[0][4]),six = (result[0][5]),seven = (result[0][6]),eight = (result[0][7]),nine = (result[0][8]),ten = (result[0][9]),eleven = (result[0][10]),twelve = (result[0][11]),thirteen = (result[0][12]),
    forteen = (result[0][13]),fifteen = (result[0][14]),sixteen = (result[0][15]),seventeen = (result[0][16]),eighteen = (result[0][17]),nineteen = (result[0][18]),twenty = (result[0][19]),twentyOne = (result[0][20]),
    twentyTwo = (result[0][21]),twentyThree = (result[0][22]),twentyFour = (result[0][23]),twentyFive = (result[0][24]),twentySix = (result[0][25]),twentySeven = (result[0][26]),
    twentyEight = (result[0][27]),twentyNine = (result[0][28]),thirty = (result[0][29]),thirtyOne = (result[0][30]),thirtyTwo = (result[0][31]),thirtyThree = (result[0][32]),thirtyFour = (result[0][33]),thirtyFive = (result[0][34]),
    thirtySix = (result[0][35]), result_text = f"{sum_of_result} number of tags matched.")





if __name__ == "__main__":
    app.run(debug=True)

