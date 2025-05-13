import joblib
import os
import pandas as pd
import pickle
import numpy as np
from pymongo import MongoClient
import datetime
import traceback

# Base directory = where utils.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# MongoDB's connection (configure your connection string and database name here)
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "fitness_coach"
COLLECTION_NAME = "predictions"

def get_db_collection():
    """Establish and return MongoDB collection."""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db[COLLECTION_NAME]

def load_model(filename):
    """Load a model from the model directory."""
    model_path = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    try:
        with open(model_path, 'rb') as f:
            try:
                model = joblib.load(f)
            except:
                f.seek(0)
                model = pickle.load(f)
        return model
    except Exception as e:
        raise Exception(f"Failed to load model '{filename}': {str(e)}")

def predict_with_model(model, input_data):
    """Make a prediction using the trained model and log it to MongoDB."""
    try:
        model_instance, label_encoder, feature_columns = model

        print(f"Input data keys: {list(input_data.keys())}")
        print(f"Expected feature columns: {feature_columns}")

        is_diet_model = 'Weight_kg' in feature_columns or any('Dietary_Restrictions' in col for col in feature_columns)
        final_input = pd.DataFrame(0, index=[0], columns=feature_columns)

        if is_diet_model:
            print("Using Diet Model mapping")
            for col in feature_columns:
                if col in input_data:
                    final_input[col] = input_data[col]
                    continue
                if col.startswith('Gender_'):
                    gender_val = col.split('_')[1]
                    if input_data.get(f'Gender_{gender_val}', 0) == 1:
                        final_input[col] = 1
                elif col.startswith('Disease_Type_'):
                    disease = col.split('_')[-1]
                    if input_data.get(f'Disease_Type_{disease}', 0) == 1:
                        final_input[col] = 1
                elif col.startswith('Severity_'):
                    severity = col.split('_')[1]
                    if input_data.get(f'Severity_{severity}', 0) == 1:
                        final_input[col] = 1
                elif col.startswith('Physical_Activity_Level_'):
                    activity = col.split('_')[-1]
                    if input_data.get(f'Physical_Activity_Level_{activity}', 0) == 1:
                        final_input[col] = 1
                elif col.startswith('Dietary_Restrictions_'):
                    restriction = col.split('_')[-1]
                    if input_data.get(f'Dietary_Restrictions_{restriction}', 0) == 1:
                        final_input[col] = 1
                elif col.startswith('Allergies_'):
                    allergy = col.split('_')[1]
                    if input_data.get(f'Allergies_{allergy}', 0) == 1:
                        final_input[col] = 1
                elif col.startswith('Preferred_Cuisine_'):
                    cuisine = col.split('_')[2]
                    if input_data.get(f'Preferred_Cuisine_{cuisine}', 0) == 1:
                        final_input[col] = 1

        else:
            print("Using Gym Model mapping")
            for basic_col in ['Age', 'BMI']:
                if basic_col in feature_columns and basic_col in input_data:
                    final_input[basic_col] = input_data[basic_col]
            if 'Height' in feature_columns:
                final_input['Height'] = input_data.get('Height', input_data.get('Height_cm', 0))
            if 'Weight' in feature_columns:
                final_input['Weight'] = input_data.get('Weight', input_data.get('Weight_kg', 0))

            for col in feature_columns:
                if col.startswith('Sex_'):
                    gender_val = col.split('_')[1]
                    if input_data.get(f'Sex_{gender_val}', 0) == 1 or input_data.get(f'Gender_{gender_val}', 0) == 1:
                        final_input[col] = 1
                elif col == 'Hypertension_Yes':
                    final_input[col] = input_data.get('Disease_Type_Hypertension', input_data.get('Hypertension_Yes', 0))
                elif col == 'Hypertension_No':
                    val = input_data.get('Disease_Type_Hypertension', None)
                    final_input[col] = 1 - val if val is not None else input_data.get('Hypertension_No', 0)
                elif col == 'Diabetes_Yes':
                    final_input[col] = input_data.get('Disease_Type_Diabetes', input_data.get('Diabetes_Yes', 0))
                elif col == 'Diabetes_No':
                    val = input_data.get('Disease_Type_Diabetes', None)
                    final_input[col] = 1 - val if val is not None else input_data.get('Diabetes_No', 0)
                elif col == 'Fitness Goal_Weight Loss' and 'Fitness Goal_Lose Weight' in input_data:
                    final_input[col] = input_data['Fitness Goal_Lose Weight']
                elif col == 'Fitness Goal_Weight Gain' and 'Fitness Goal_Gain Muscle' in input_data:
                    final_input[col] = input_data['Fitness Goal_Gain Muscle']
                elif col in input_data:
                    final_input[col] = input_data[col]

        print(f"Final input shape: {final_input.shape}")
        print(f"Final input columns: {list(final_input.columns)}")

        if final_input.isnull().values.any():
            print("Warning: Input contains missing values. Filling with zeros.")
            final_input.fillna(0, inplace=True)

        prediction = model_instance.predict(final_input)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        # Logging to MongoDB
        collection = get_db_collection()
        log_entry = {
            "model_type": "Diet" if is_diet_model else "Gym",
            "input_data": input_data,
            "formatted_input": final_input.to_dict(orient="records")[0],
            "prediction": predicted_label,
            "timestamp": datetime.datetime.now()
        }
        collection.insert_one(log_entry)

        return predicted_label

    except Exception as e:
        print(traceback.format_exc())
        raise Exception(f"Prediction failed: {str(e)}")
