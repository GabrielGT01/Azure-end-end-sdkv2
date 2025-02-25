import os
import sys
import warnings
import argparse
import tempfile
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
import azure.ai.ml._artifacts._artifact_utilities as artifact_utils

def get_data(ml_client, input_dir, feature_transformer_name):
    """
    Reads CSV files from input_dir, applies the feature transformer asset,
    and returns a list of transformed DataFrames.
    """
    required_columns = ['Age', 'Gender', 'Speed_of_Impact', 'Helmet_Used', 'Seatbelt_Used']
    all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')]
    missing_columns = []
    dataframes = []
    
    # Load the joblib transformer asset
    try:
        asset = ml_client.data.get(name=feature_transformer_name, version="1")
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download the artifact into the temporary directory.
            # Note: We use temp_dir as the destination directory.
            artifact_utils.download_artifact_from_aml_uri(
                uri=asset.path,
                destination=temp_dir,
                datastore_operation=ml_client.datastores
            )
            # Construct a candidate local path by joining the temp_dir and basename of the asset's path.
            candidate_path = os.path.join(temp_dir, os.path.basename(asset.path))
            print("Downloaded candidate path:", candidate_path)
            
            # Check if the candidate_path is a directory.
            if os.path.isdir(candidate_path):
                files_in_dir = os.listdir(candidate_path)
                if not files_in_dir:
                    raise Exception("Downloaded directory is empty.")
                # If only one file exists, use that; otherwise, search for a .joblib file.
                if len(files_in_dir) == 1:
                    local_file = os.path.join(candidate_path, files_in_dir[0])
                else:
                    joblib_files = [f for f in files_in_dir if f.endswith('.joblib')]
                    if not joblib_files:
                        raise Exception("No joblib file found in the downloaded directory.")
                    local_file = os.path.join(candidate_path, joblib_files[0])
            else:
                local_file = candidate_path
            
            print("Using local file:", local_file)
            with open(local_file, "rb") as f:
                features_transformer = joblib.load(f)
            print("Feature transformer loaded successfully!")
    except Exception as e:
        print(f"Error loading transformer: {e}")
        return None

    # Process each CSV file
    for file in all_files:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            continue

        # Check if all required columns are present
        for col in required_columns:
            if col not in df.columns and col not in missing_columns:
                missing_columns.append(col)

        # Apply the transformation to the data
        try:
            transformed_data = features_transformer.transform(df)
            transformed_df = pd.DataFrame(transformed_data, columns=required_columns)
            print(f"Transformation applied successfully on {file}!")
        except Exception as e:
            print(f"Error during transformation for {file}: {e}")
            continue

        dataframes.append(transformed_df)
    
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return None
    return dataframes

def main(args):
    try:
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id="cda9116f-5326-4a9b-9407-bc3a4391c27c",
            resource_group_name="gabby102",
            workspace_name="health-update"
        )
    except Exception as e:
        print(f"Default credential failed: {e}")
        print("Falling back to interactive browser login...")
        try:
            credential = InteractiveBrowserCredential()
            ml_client = MLClient(
                credential=credential,
                subscription_id="cda9116f-5326-4a9b-9407-bc3a4391c27c",
                resource_group_name="gabby102",
                workspace_name="health-update"
            )
        except Exception as e:
            print(f"Interactive login failed: {e}")
            sys.exit(1)
    
    datas = get_data(ml_client, args.input_data, args.feature_transformer_name)
    if datas is None or len(datas) == 0:
        print("Error: No data was transformed successfully")
        sys.exit(1)
    
    os.makedirs(args.output_data, exist_ok=True)
    try:
        for i, split_df in enumerate(datas):
            output_path = Path(args.output_data) / f"accident_part_{i+1}.csv"
            split_df.to_csv(output_path, index=False)
            print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Error saving transformed data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform accident data in Azure ML")
    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        default="./data",
        help="Path to the input data directory containing CSV files"
    )
    parser.add_argument(
        "--output_data",
        type=str,
        required=True,
        help="Path where the transformed data will be written"
    )
    parser.add_argument(
        "--feature_transformer_name",
        type=str,
        required=True,
        default="accident-transformer",
        help="Name of the feature transformer asset"
    )
    
    print("\n" + "*" * 60)
    args = parser.parse_args()
    main(args)
    print("*" * 60 + "\n")
