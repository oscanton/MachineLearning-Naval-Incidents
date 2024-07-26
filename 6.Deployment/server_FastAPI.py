# -*- coding: utf-8 -*-

"""
Predictive analysis of naval incidents in the USA, 2002 - 2015:
Model deployment: Server example for VesselBalancedSample
Binomial prediction with Random Forest from a webform

@author: "Oscar Anton"
@date: "2024"
@license: "CC BY-NC-ND 4.0 DEED"
@version: "0.9"
"""



# %% LIBRARIES LOAD

import pandas as pd
import joblib

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from typing import List
from fastapi.middleware.cors import CORSMiddleware

import threading



# %% MODEL and SCALER LOAD

# Get the path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the path to model file
model_file = os.path.join(current_dir, '..', '5.DataModel', 'Models', 'rf_train.pkl')

# Load Random Forest model
rf_model = joblib.load(model_file)


# Get the path to scaler file
scaler_file = os.path.join(current_dir, '..', '5.DataModel', 'Datasets', 'scaler.pkl')

# Load scaler
scaler = joblib.load(scaler_file)



# %% INPUT DATA VALIDATION

# Define expected columns entered by the user
expected_columns = [
    'vessel_class', 'build_year', 'flag_abbr',
    'classification_society', 'solas_desc',
    'gross_ton', 'vessel_length'
]

# Columns and data types comparations
def input_validation(df):
    # Check if all expected columns are present
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")
    
    # Check if there are any extra columns
    extra_columns = set(df.columns) - set(expected_columns)
    if extra_columns:
        raise ValueError(f"Unexpected columns: {extra_columns}")
    
    # Check data types
    expected_dtypes = {
        'vessel_class': object, # object type in pandas is equivalent to str in Python
        'build_year': 'int64',
        'flag_abbr': object,
        'classification_society': object,
        'solas_desc': object,
        'gross_ton': 'float64', # Using float64 to allow both int and float
        'vessel_length': 'float64' # Using float64 to allow both int and float
    }
    
    for column, dtype in expected_dtypes.items():
        if not pd.api.types.is_dtype_equal(df[column].dtype, dtype):
            raise ValueError(f"Column '{column}' must be of type {dtype}")
    
    return True



# %% INPUT DATA FORMATING

# Define expected variables by the model
input_columns = [
    'vessel_class_Barge', 'vessel_class_Bulk Carrier',
    'vessel_class_Fishing Vessel', 'vessel_class_General Dry Cargo Ship',
    'vessel_class_Miscellaneous Vessel', 'vessel_class_Offshore',
    'vessel_class_Passenger Ship', 'vessel_class_Recreational',
    'vessel_class_Tank Ship', 'vessel_class_Towing Vessel',
    'vessel_class_other value', 'build_year_very Old', 'build_year_old',
    'build_year_average', 'build_year_new', 'build_year_very new',
    'flag_abbr_CA', 'flag_abbr_LR', 'flag_abbr_PA', 'flag_abbr_US',
    'flag_abbr_other value',
    'classification_society_AMERICAN BUREAU OF SHIPPING',
    'classification_society_DET NORSKE VERITAS',
    "classification_society_LLOYD'S REGISTER OF SHIPPING",
    'classification_society_NIPPON KAIJI KYOKAI',
    'classification_society_UNSPECIFIED',
    'classification_society_other value', 'solas_desc_Active SOLAS',
    'solas_desc_Historical SOLAS', 'solas_desc_Non SOLAS', 'gross_ton',
    'vessel_length'
]

# Transform categorical variables to one hot codifing
def value_ohe(variable_name, values):
    one_hot_df = []
    labels = list(filter(lambda x: x.startswith(variable_name), input_columns))
    for value in values:
        target_column = variable_name + '_' + value
        one_hot_row = pd.DataFrame(0, index=[0], columns=labels)
        one_hot_row[target_column] = 1
        one_hot_df.append(one_hot_row)
    # Concatenate all one-hot encoded rows into a single DataFrame
    return pd.concat(one_hot_df, ignore_index=True)

# Cutting & codifing build_year
def build_year_ohe(build_years):
    labels = ['very Old', 'old', 'average', 'new', 'very new']
    year_categories = pd.cut(build_years,
                             bins=[-float('inf'), 1940, 1960, 1980, 2000, float('inf')],
                             labels=labels,
                             include_lowest=True)
    # Return one-hot-encoded values for cutted value
    return value_ohe('build_year', year_categories)

# Transform numeric variables to scaled values
def scale_values(scaler, feature_names, **kwargs):
    # Create a DataFrame with the new entries
    new_data = pd.DataFrame(kwargs)

    # Transform the data with the scaler adjusted
    scaled_values = pd.DataFrame(scaler.transform(new_data), columns=feature_names)
    
    return scaled_values


# Define structure of data input
def input_structure(df):
    data_input = pd.concat([value_ohe('vessel_class', df['vessel_class']),
                    build_year_ohe(df['build_year']),
                    value_ohe('flag_abbr', df['flag_abbr']),
                    value_ohe('classification_society', df['classification_society']),
                    value_ohe('solas_desc', df['solas_desc']),
                    scale_values(scaler,
                                 feature_names = ['gross_ton', 'vessel_length'],
                                 gross_ton = df['gross_ton'],
                                 vessel_length = df['vessel_length'])
                    ], axis=1)
    return data_input



# %% PROBABILITIES PREDICTION

def get_prediction_probabilities(input_df):
    result = {}
    try:
        # Verify data format
        is_valid = input_validation(input_df)
        
        if not is_valid:
            raise ValueError("Invalid input data format")
        
        # Calculate probability array
        prediction_proba = rf_model.predict_proba(input_structure(input_df))
        
        # Store probability for each vessel in a dictionary
        for i, value in enumerate(prediction_proba[:, 1], start=1):
            result[f'vessel {i}'] = f'{value:.4%}'
        
    except ValueError as e:
        # Store error message in the result dictionary
        result["error"] = f"Data is invalid: {e}"
    
    return result



# %% FASTAPI REQUESTS PROCESSING

# Initiate FastAPI
app = FastAPI()

# Configure CORS to allow requests from any source
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. !!!!!!!!!!!!!! CHANGE THIS IN PRODUCTION !!!!!!!!!!!!!!
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Define the structure of the request data using Pydantic
class PredictionRequest(BaseModel):
    vessel_class: List[str]
    build_year: List[int]
    flag_abbr: List[str]
    classification_society: List[str]
    solas_desc: List[str]
    gross_ton: List[float]
    vessel_length: List[float]


@app.post('/predict')
def deploy_model(request: PredictionRequest):
    try:
        # Convert the request data to a DataFrame
        input_df = pd.DataFrame({
            "vessel_class": request.vessel_class,
            "build_year": request.build_year,
            "flag_abbr": request.flag_abbr,
            "classification_society": request.classification_society,
            "solas_desc": request.solas_desc,
            "gross_ton": request.gross_ton,
            "vessel_length": request.vessel_length
        })
        
        # Call the prediction function
        probabilities = get_prediction_probabilities(input_df)
        
        # Return the prediction probabilities
        return probabilities
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# %% START FASTAPI SERVER

# Define a function to run the Uvicorn server
def run_uvicorn():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

# Start the server in a separate thread
if __name__ == "__main__":
    thread = threading.Thread(target=run_uvicorn)
    thread.start()