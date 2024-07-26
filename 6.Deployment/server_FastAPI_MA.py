# -*- coding: utf-8 -*-

"""
Predictive analysis of naval incidents in the USA, 2002 - 2015:
Model deployment: Server example for MergedActivity
Multiclass prediction with Random Forest from a webform

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



# %% MODELS and SCALER LOAD

# Get the path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the path to Random Forest model file
rf_model_file = os.path.join(current_dir, '..', '5.DataModel', 'Models', 'rf_MA_train.pkl')

# Load Random Forest model
rf_MA_model = joblib.load(rf_model_file)

# Get the path to activity_id prediction model file
knn_model_file = os.path.join(current_dir, '..', '5.DataModel', 'Models', 'actid_knn_model.pkl')

# Load Random Forest model
actid_knn_model = joblib.load(knn_model_file)


# Get the path to scaler file
scaler_file = os.path.join(current_dir, '..', '5.DataModel', 'Datasets', 'scaler_MA.pkl')

# Load scaler
scaler = joblib.load(scaler_file)



# %% INPUT DATA VALIDATION

# Define expected columns entered by the user
expected_columns = {
    'region': str,
    'watertype': str,
    'damage_status': str,
    'vessel_class': str,
    #'activity_id': int,
    'hour': str,
    'longitude': float,
    'age': (int, float),
    'gross_ton': (int, float),
    'vessel_length': (int, float),
    'air_temp': (int, float),
    'wind_speed': (int, float)
}

# Columns and data types comparations
def input_evaluation(df):
    # Check expect columns and type
    for column, column_type in expected_columns.items():
        if column not in df.columns:
            return f"Missing expected columns: {column}"
        if not df[column].apply(lambda x: isinstance(x, column_type)).all():
            return f"Column {column} should be of type {column_type}"
    
    # Check hour column format
    try:
        pd.to_datetime(df['hour'], format='%H:%M')
    except ValueError:
        return "'hour' should be 'HH:MM' format"

    return True



# %% INPUT DATA FORMATING

# Define expected variables by the model
input_columns = [
    'region_Alaska', 'region_Canada', 'region_East Coast',
    'region_Gulf of Mexico', 'region_Mississippi', 'region_West Coast',
    'watertype_ocean', 'watertype_river', 'damage_status_Actual Total Loss',
    'damage_status_Damaged',
    'damage_status_Total Constructive Loss: Salvaged',
    'damage_status_Total Constructive Loss: Unsalvaged',
    'damage_status_Undamaged', 'vessel_class_Barge',
    'vessel_class_Bulk Carrier', 'vessel_class_Fishing Vessel',
    'vessel_class_General Dry Cargo Ship',
    'vessel_class_Miscellaneous Vessel', 'vessel_class_Offshore',
    'vessel_class_Passenger Ship', 'vessel_class_Recreational',
    'vessel_class_Tank Ship', 'vessel_class_Towing Vessel',
    'vessel_class_other value',
    #'activity_id',
    'hour', 'longitude', 'age',
    'gross_ton', 'vessel_length', 'air_temp', 'wind_speed'
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

# Transform minutes to decimal value
def hour_dec(hour):
    return (pd.to_numeric(hour.str.split(':').str[0]) + 
            pd.to_numeric(hour.str.split(':').str[1])/60).round(2)

# Transform numeric variables to scaled values
def scale_values(scaler, feature_names, **kwargs):
    # Create a DataFrame with the new entries
    new_data = pd.DataFrame(kwargs)

    # Transform the data with the scaler adjusted
    scaled_values = pd.DataFrame(scaler.transform(new_data), columns=feature_names)
    
    return scaled_values

# Define structure of data input
def input_structure(df):
    data_input = pd.concat([
        value_ohe('region', df['region']),
        value_ohe('watertype', df['watertype']),
        value_ohe('damage_status', df['damage_status']),
        value_ohe('vessel_class', df['vessel_class']),
        scale_values(scaler,
                     feature_names = ['activity_id', 'hour', 'longitude', 'age',
                                      'gross_ton', 'vessel_length', 'air_temp', 'wind_speed'],
                     activity_id = [0] * len(df),
                     hour = hour_dec(df['hour']),
                     longitude = df['longitude'],
                     age = df['age'],
                     gross_ton = df['gross_ton'],
                     vessel_length = df['vessel_length'],
                     air_temp = df['air_temp'],
                     wind_speed = df['wind_speed'],
                     )
        ], axis=1)
    return data_input



# %% PROBABILITIES PREDICTION

def get_prediction_probabilities(input_df):
    if input_evaluation(input_df) == True:
        try:
            # Format input data
            structured_input = input_structure(input_df)

            # Replace activity_id for its predictions, according to knn model
            structured_input['activity_id'] = actid_knn_model.predict(structured_input.drop(columns='activity_id'))

            # Calculate incident probabilities array (output)
            prediction_proba = rf_MA_model.predict_proba(structured_input)
            
            # Store probabilities for each vessel in a dictionary
            event_classes = ['Critical Events', 'Maritime Accidents', 'Material Issues', 'Onboard Emergencies', 'Thirdparty Damages']
            result = {
                key: [f"{prediction_proba[j][i]:.4%}" for j in range(len(prediction_proba))]
                for i, key in enumerate(event_classes)
            }
            return result
        except ValueError as e:
            return f"Calculation of prediction probabilities failed: {e}"

    else:
        return f"Data evaluation failed: {input_evaluation(input_df)}"
    


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
    region: List[str]
    watertype: List[str]
    damage_status: List[str]
    vessel_class: List[str]
    #activity_id: List[int]
    hour: List[str]
    longitude: List[float]
    age: List[float]
    gross_ton: List[float]
    vessel_length: List[float]
    air_temp: List[float]
    wind_speed: List[float]


@app.post('/predict')
def deploy_model(request: PredictionRequest):
    try:
        # Convert the request data to a DataFrame
        input_df = pd.DataFrame({
            'region': request.region,
            'watertype': request.watertype,
            'damage_status': request.damage_status,
            'vessel_class': request.vessel_class,
            #'activity_id': request.activity_id,
            'hour': request.hour,
            'longitude': request.longitude,
            'age': request.age,
            'gross_ton': request.gross_ton,
            'vessel_length': request.vessel_length,
            'air_temp': request.air_temp,
            'wind_speed': request.wind_speed
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
    uvicorn.run(app, host="127.0.0.2", port=8000, log_level="info")

# Start the server in a separate thread
if __name__ == "__main__":
    thread = threading.Thread(target=run_uvicorn)
    thread.start()