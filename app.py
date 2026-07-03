from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            LotFrontage=float(request.form.get("LotFrontage")),
            LotArea=float(request.form.get("LotArea")),
            OverallQual=int(request.form.get("OverallQual")),
            YearBuilt=int(request.form.get("YearBuilt")),
            YearRemodAdd=int(request.form.get("YearRemodAdd")),
            BsmtFinSF1=float(request.form.get("BsmtFinSF1")),
            TotalBsmtSF=float(request.form.get("TotalBsmtSF")),
            FirstFlrSF=float(request.form.get("FirstFlrSF")),
            SecondFlrSF=float(request.form.get("SecondFlrSF")),
            GrLivArea=float(request.form.get("GrLivArea")),
            FullBath=int(request.form.get("FullBath")),
            TotRmsAbvGrd=int(request.form.get("TotRmsAbvGrd")),
            GarageCars=float(request.form.get("GarageCars")),
            GarageArea=float(request.form.get("GarageArea")),
            GarageFinish=request.form.get("GarageFinish")
        )
        pred_df=data.get_data_as_data_frame()  
        

        predict_pipeline=PredictPipeline()
        
        results=predict_pipeline.predict(pred_df)
        
        return render_template(
            'home.html',
            results=f"${results[0]:,.2f}"
        )
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        


