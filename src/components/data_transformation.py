import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")
    feature_selector_file_path = os.path.join('artifacts', "feature_selector.pkl")
    default_values_file_path = os.path.join('artifacts', "default_values.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            numerical_columns = [
                "MSSubClass",
                "LotFrontage",
                "LotArea",
                "OverallQual",
                "OverallCond",
                "YearBuilt",
                "YearRemodAdd",
                "MasVnrArea",
                "BsmtFinSF1",
                "BsmtFinSF2",
                "BsmtUnfSF",
                "TotalBsmtSF",
                "1stFlrSF",
                "2ndFlrSF",
                "LowQualFinSF",
                "GrLivArea",
                "BsmtFullBath",
                "BsmtHalfBath",
                "FullBath",
                "HalfBath",
                "BedroomAbvGr",
                "KitchenAbvGr",
                "TotRmsAbvGrd",
                "Fireplaces",
                "GarageYrBlt",
                "GarageCars",
                "GarageArea",
                "WoodDeckSF",
                "OpenPorchSF",
                "EnclosedPorch",
                "3SsnPorch",
                "ScreenPorch",
                "PoolArea",
                "MiscVal",
                "MoSold",
                "YrSold"
            ]
            categorical_columns = [
                "MSZoning",
                "Street",
                "Alley",
                "LotShape",
                "LandContour",
                "Utilities",
                "LotConfig",
                "LandSlope",
                "Neighborhood",
                "Condition1",
                "Condition2",
                "BldgType",
                "HouseStyle",
                "RoofStyle",
                "RoofMatl",
                "Exterior1st",
                "Exterior2nd",
                "MasVnrType",
                "ExterQual",
                "ExterCond",
                "Foundation",
                "BsmtQual",
                "BsmtCond",
                "BsmtExposure",
                "BsmtFinType1",
                "BsmtFinType2",
                "Heating",
                "HeatingQC",
                "CentralAir",
                "Electrical",
                "KitchenQual",
                "Functional",
                "FireplaceQu",
                "GarageType",
                "GarageFinish",
                "GarageQual",
                "GarageCond",
                "PavedDrive",
                "PoolQC",
                "Fence",
                "MiscFeature",
                "SaleType",
                "SaleCondition"
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ],
                sparse_threshold=0


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)


            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name= "SalePrice"
            

            input_feature_train_df=train_df.drop(columns=[target_column_name])
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name])
            target_feature_test_df=test_df[target_column_name]
            default_values = input_feature_train_df.mode().iloc[0].to_dict()

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            print("Before Feature Selection")
            print("Train:", input_feature_train_arr.shape)
            print("Test :", input_feature_test_arr.shape)

            # Feature Selection
            selector = SelectFromModel(
                estimator=RandomForestRegressor(
                    n_estimators=100,
                    random_state=42
                ),
                max_features=15,
                threshold=-np.inf
            )

            selector.fit(input_feature_train_arr, target_feature_train_df)

            input_feature_train_arr = selector.transform(input_feature_train_arr)
            input_feature_test_arr = selector.transform(input_feature_test_arr)

            feature_names = preprocessing_obj.get_feature_names_out()
            selected_features = feature_names[selector.get_support()]

            print("\nSelected Features:")
            for feature in selected_features:
                print(feature)

            # Print shape after feature selection
            print("\nAfter Feature Selection")
            print("Train:", input_feature_train_arr.shape)
            print("Test :", input_feature_test_arr.shape)
            
            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = input_feature_train_arr.toarray()

            if hasattr(input_feature_test_arr, "toarray"):
                input_feature_test_arr = input_feature_test_arr.toarray()

            

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            save_object(
                file_path=self.data_transformation_config.feature_selector_file_path,
                obj=selector
            )

            save_object(
                file_path=self.data_transformation_config.default_values_file_path,
                obj=default_values
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
