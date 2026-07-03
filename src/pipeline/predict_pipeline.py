import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            feature_selector_path = os.path.join("artifacts", "feature_selector.pkl")
            default_values_path = os.path.join("artifacts", "default_values.pkl")
            
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            feature_selector = load_object(file_path=feature_selector_path)
            default_values = load_object(file_path=default_values_path)
                                                                                                                                                

            input_data = default_values.copy()

            for column in features.columns:
                input_data[column] = features.iloc[0][column]

            input_df = pd.DataFrame([input_data])
            data_scaled = preprocessor.transform(input_df)
            data_selected = feature_selector.transform(data_scaled)
            preds = model.predict(data_selected)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(
        self,
        LotFrontage,
        LotArea,
        OverallQual,
        YearBuilt,
        YearRemodAdd,
        BsmtFinSF1,
        TotalBsmtSF,
        FirstFlrSF,
        SecondFlrSF,
        GrLivArea,
        FullBath,
        TotRmsAbvGrd,
        GarageCars,
        GarageArea,
        GarageFinish
    ):

        self.LotFrontage = LotFrontage
        self.LotArea = LotArea
        self.OverallQual = OverallQual
        self.YearBuilt = YearBuilt
        self.YearRemodAdd = YearRemodAdd
        self.BsmtFinSF1 = BsmtFinSF1
        self.TotalBsmtSF = TotalBsmtSF
        self.FirstFlrSF = FirstFlrSF
        self.SecondFlrSF = SecondFlrSF
        self.GrLivArea = GrLivArea
        self.FullBath = FullBath
        self.TotRmsAbvGrd = TotRmsAbvGrd
        self.GarageCars = GarageCars
        self.GarageArea = GarageArea
        self.GarageFinish = GarageFinish

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "LotFrontage": [self.LotFrontage],
                "LotArea": [self.LotArea],
                "OverallQual": [self.OverallQual],
                "YearBuilt": [self.YearBuilt],
                "YearRemodAdd": [self.YearRemodAdd],
                "BsmtFinSF1": [self.BsmtFinSF1],
                "TotalBsmtSF": [self.TotalBsmtSF],
                "1stFlrSF": [self.FirstFlrSF],
                "2ndFlrSF": [self.SecondFlrSF],
                "GrLivArea": [self.GrLivArea],
                "FullBath": [self.FullBath],
                "TotRmsAbvGrd": [self.TotRmsAbvGrd],
                "GarageCars": [self.GarageCars],
                "GarageArea": [self.GarageArea],
                "GarageFinish": [self.GarageFinish],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

