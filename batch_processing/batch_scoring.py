
import os
import logging
import mlflow
import numpy as np
import pandas as pd
from typing import List
import glob

def init():
    """Initialize the scoring environment by loading the model."""
    global model
    try:
        model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model")
        model = mlflow.pyfunc.load_model(model_path)
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def run(training_data_path: str) -> List[str]:
    try:
        results = []
        # List all CSV files in the input folder
        csv_files = glob.glob(os.path.join(training_data_path, "*.csv"))
        
        if not csv_files:
            logging.error(f"No CSV files found in the folder: {training_data_path}")
            return results

        for file_path in csv_files:
            try:
                # Read the CSV file
                data = pd.read_csv(file_path)
                
                # Validate input data
                expected_columns = ['Age', 'Gender', 'Speed_of_Impact', 'Helmet_Used', 'Seatbelt_Used']
                if not all(col in data.columns for col in expected_columns):
                    raise ValueError(f"Missing required columns in {file_path}")
                
                input_data = data[expected_columns]
                
                # Make predictions
                predictions = model.predict(input_data)
                probabilities = model.predict_proba(input_data)
                
                # Validate predictions
                if len(predictions) != len(data):
                    raise ValueError("Prediction length mismatch")
                
                # Create output dataframe
                output_df = data.copy()
                output_df['prediction'] = predictions
                output_df['survival_probability'] = probabilities[:, 1]
                
                results.append(output_df.to_json(orient='records'))
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {str(e)}")
                continue
                
        return results
    except Exception as e:
        logging.error(f"Error in run: {str(e)}")
        raise
