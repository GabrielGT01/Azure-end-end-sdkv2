# Azure-end-end-sdkv2
# AzureML SDK v2 Demonstration

This repository contains a demonstration of the Azure Machine Learning SDK v2. It uses a CSV file from Gaggle to showcase the end-to-end process of:

- Registering data
- Creating a job
- Training a model
- Creating endpoints

> **Note:** The focus of this demonstration is on how to use the AzureML SDK v2 to perform these operations. There is no emphasis on the accuracy of the model.

## Overview

In this demonstration, you'll see how to:
- **Register Data:** Upload and register CSV data with your Azure ML workspace.
- **Create a Job:** Define and run a job that trains a model.
- **Train a Model:** Execute model training using the registered data.
- **Create Endpoints:** Deploy the trained model by creating an online endpoint for inference.

## Repository Structure

- **data/**  
  Contains the CSV file from Gaggle.

- **src/**  
  Source code files for data registration, training, and endpoint creation.

- **README.md**  
  This file.

## Prerequisites

- An active Azure subscription.
- An Azure ML workspace configured with the required resource providers.
- Python 3.8 or higher.
- Azure Machine Learning SDK v2 installed:
  ```bash
  pip install azure-ai-ml
