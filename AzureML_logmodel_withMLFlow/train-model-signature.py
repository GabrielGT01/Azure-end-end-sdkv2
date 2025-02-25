# import libraries
import mlflow
import os
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from mlflow.models import infer_signature
import mlflow.sklearn
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

# Define paths for saving model and transformer
MODEL_PATH = "model.pkl"
TRANSFORMER_PATH = "features_transformer.pkl"

def main(args):
    # read data
    df = get_data(args.training_data)

    # split data
    X_train, X_test, y_train, y_test, features_transformer = split_data(df)

    # Save the feature transformer for future use
    joblib.dump(features_transformer, TRANSFORMER_PATH)
    print(f"Feature transformer saved at: {TRANSFORMER_PATH}")

    # train model
    model = train_model(args.reg_rate, X_train, y_train)
    mlflow.log_param("regularization_rate", args.reg_rate)

    # Save trained model
    joblib.dump(model, MODEL_PATH)
    print(f"Trained model saved at: {MODEL_PATH}")

    # evaluate model
    y_hat = eval_model(model, X_test, y_test)

    # create the signature manually
    
    input_schema = Schema([
        ColSpec("double", "Age"),  # Assuming age is a float/double
        ColSpec("string", "Gender"),  # Categorical variable
        ColSpec("double", "Speed_of_Impact"),  # Assuming speed is a float/double
        ColSpec("string", "Helmet_Used"),  # Categorical variable
        ColSpec("string", "Seatbelt_Used")  # Categorical variable
    ])

    output_schema = Schema([ColSpec("integer", "Survived")])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
 

    # Log the model and feature transformer in MLflow
    mlflow.sklearn.log_model(model, artifact_path="model", signature=signature)
    mlflow.log_artifact(TRANSFORMER_PATH, artifact_path="preprocessor")

    print("Model and Feature Transformer saved in MLflow!")

# function that reads the data
def get_data(path):
    print("Reading data...")
    data = pd.read_csv(path)
    # Convert numeric columns to float
    data['Age'] = data['Age'].astype(float)
    data['Speed_of_Impact'] = data['Speed_of_Impact'].astype(float)
    df = data.copy().dropna()
    return df

# function that splits the data
def split_data(df):
    print("Splitting data...")
    
    # Numeric transformer pipeline
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
    )
    
    # Categorical transformer pipeline
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(drop='first')
    )
    
    # Define categorical and numeric columns
    cat_columns = ['Gender', 'Helmet_Used', 'Seatbelt_Used']
    num_columns = ['Age', 'Speed_of_Impact']
    
    # Combined feature transformer
    features_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, num_columns),
            ("categorical", categorical_transformer, cat_columns),
        ],
    )
    
    # Separate features and labels
    X = df[['Age', 'Gender', 'Speed_of_Impact', 'Helmet_Used', 'Seatbelt_Used']]
    y = df['Survived'].values

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    
    # Transform train and test data
    X_train = features_transformer.fit_transform(X_train)
    X_test = features_transformer.transform(X_test)

    return X_train, X_test, y_train, y_test, features_transformer

# function that trains the model
def train_model(reg_rate, X_train, y_train):
    print("Training model...")

    # Train logistic regression model
    model = LogisticRegression(C=1/reg_rate, solver="liblinear")
    model.fit(X_train, y_train)

    return model

# function that evaluates the model
def eval_model(model, X_test, y_test):
    # calculate accuracy
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)
    print('Accuracy:', acc)

    # calculate AUC
    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_scores[:,1])
    print('AUC:', auc)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("auc", auc)

    # plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
    plt.figure(figsize=(6, 4))
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal reference line
    plt.plot(fpr, tpr)  # ROC curve
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig("ROC-Curve.png")
    plt.savefig("ROC-Curve.png")
    mlflow.log_artifact("ROC-Curve.png")

    return y_hat

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data', type=str, required=True)
    parser.add_argument("--reg_rate", dest='reg_rate', type=float, default=0.01,help="Regularization rate must be positive")

    # parse args
    args = parser.parse_args()

    return args

# run script
if __name__ == "__main__":
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    print("*" * 60)
    print("\n\n")
