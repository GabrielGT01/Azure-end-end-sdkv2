# import libraries
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def main(args):
    # read data
    df = get_data(args.input_data)

    cleaned_data = clean_data(df)

    output_df = cleaned_data.to_csv((Path(args.output_data) / "accident.csv"), index = False)

# function that reads the data
def get_data(path):
    df = pd.read_csv(path)

    # Count the rows and print the result
    row_count = (len(df))
    print('Preparing {} rows of data'.format(row_count))
    
    return df

# function that removes missing values
def clean_data(df):
    df = df.dropna()
    
    return df


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--input_data", dest='input_data',
                        type=str)
    parser.add_argument("--output_data", dest='output_data',
                        type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
