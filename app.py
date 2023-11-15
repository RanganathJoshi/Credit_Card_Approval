from src.Credit_Card_Approval.Pipelines.prediction_pipeline import customdata,predictpipeline
from flask import Flask,render_template,request
import numpy as np

app=Flask(__name__)

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=customdata(
            A=request.form.get('A'),
            B=request.form.get('B'),
            C=request.form.get('C'),
            D=request.form.get('D'),
            E=request.form.get('E'),
            F=request.form.get('F'),
            G=request.form.get('G'),
            H=request.form.get('H'),
            I=request.form.get('I'),
            J=request.form.get('J'),
            K=request.form.get('K'),
            L=request.form.get('L'),
            M=request.form.get('M'),
            N=request.form.get('N'),
            O=request.form.get('O'),
            )
        
        final_data=data.get_custom_data()
        predict_pipeline=predictpipeline()
        pred=predict_pipeline.predict(final_data)
        if pred==1:
            result='Eligible'
        else:
            result="Not Eligible"

        return render_template('result.html',final_result=result)

        #execution begin
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080)