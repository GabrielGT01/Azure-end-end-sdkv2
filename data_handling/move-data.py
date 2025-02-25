# Import libraries
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import os


# Function to read and split data
def get_data(path):
    df = pd.read_csv(path)

    # Count rows
    row_count = len(df)
    print(f'Analyzing {row_count} rows of data')

    # Ensure columns exist
    feature_columns = ['Age', 'Gender', 'Speed_of_Impact', 'Helmet_Used', 'Seatbelt_Used']
    df = df[feature_columns + ['Survived']].dropna()  # Drop rows with missing values

    # Separate features and labels
    X = df[feature_columns]
    y = df['Survived']

    # Split into training & test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    return X_train, X_test, y_train, y_test

# Main function
def main(args):
    # Read and split data
    X_train, X_test, y_train, y_test = get_data(args.input_data)

    # Ensure output folders exist
    Path(args.output_datastore_train).mkdir(parents=True, exist_ok=True)
    Path(args.output_datastore_test).mkdir(parents=True, exist_ok=True)

    # Save train and test datasets
    X_train.to_csv(Path(args.output_datastore_train) / "train.csv", index=False)
    y_train.to_csv(Path(args.output_datastore_train) / "train_label.csv", index=False, header=True)
    X_test.to_csv(Path(args.output_datastore_test) / "test.csv", index=False)
    y_test.to_csv(Path(args.output_datastore_test) / "test_label.csv", index=False, header=True)

    print(f"Data saved in {args.output_datastore_train} and {args.output_datastore_test}")

# Function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--output_datastore_train", type=str, required=True)
    parser.add_argument("--output_datastore_test", type=str, required=True)
    return parser.parse_args()

# Run script
if __name__ == "__main__":
    print("\n" + "*" * 60)
    args = parse_args()
    main(args)
    print("*" * 60 + "\n")

