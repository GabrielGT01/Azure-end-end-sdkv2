
import time
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import MLClient,Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import BatchEndpoint, BatchDeployment, Model, Environment, CodeConfiguration

# Function to create batch endpoint if it doesn't exist
def create_batch_endpoint(ml_client,endpoint_name):
    try:
        # Try to get existing endpoint
        endpoint = ml_client.batch_endpoints.get(endpoint_name)
        print(f"Batch endpoint {endpoint_name} already exists.")
    except Exception:
        # Create new endpoint
        endpoint = BatchEndpoint(
            name=endpoint_name,
            description="Batch endpoint for accident prediction",
        )
        ml_client.batch_endpoints.begin_create_or_update(endpoint).result()
        print(f"Created new batch endpoint: {endpoint_name}")
    return endpoint


# Function to create and deploy the batch deployment
def create_batch_deployment(ml_client, endpoint_name,env_name, model_name, deployment_name,code_path,scoring_path,compute):
    # Get the latest version of the model
    model = ml_client.models.get(name=model_name, label="latest")
    
    # Code configuration
    code_configuration = CodeConfiguration(
        code=code_path,  # Local path to the code directory
        scoring_script=scoring_path  # The script that contains the init() and run() functions
    )
    # Create the deployment
    deployment = BatchDeployment(
        name=deployment_name,
        description="Deployment for accident prediction model",
        endpoint_name=endpoint_name,
        model=model,
        environment=env_name , # Use the registered environment ID
        code_configuration=code_configuration,
        compute=compute,  # Replace with your compute target name
        instance_count=1,
        max_concurrency_per_instance=1,
        mini_batch_size=10,
        output_action=AssetTypes.MLFLOW_MODEL,
        output_file_name="predictions.csv",
        retry_settings={"max_retries": 3, "timeout": 300},
        logging_level="info",
    )

    # Create or update the deployment
    ml_client.batch_deployments.begin_create_or_update(deployment).result()
    print("Deployment created successfully!")

    # Set the deployment as the default for the endpoint
    endpoint = ml_client.batch_endpoints.get(endpoint_name)
    endpoint.defaults.deployment_name = deployment_name
    ml_client.batch_endpoints.begin_create_or_update(endpoint).result()
    return deployment

    # Function to create and submit a batch job
def submit_batch_job(ml_client, endpoint_name,training_data_path,name_of_data):

    # Register the data in Azure ML
    data_path = training_data_path
    dataset_name = name_of_data
    dataset_in_use = Data(
        path=data_path,
        type=AssetTypes.URI_FOLDER,
        description="An unlabeled dataset for survivor classification",
        name=dataset_name,
    )
    
    try:
        registered_data = ml_client.data.create_or_update(dataset_in_use)
        print(f"Dataset registered with name: {registered_data.name}")
        print(f"Dataset version: {registered_data.version}")
        print(f"Dataset registration path: {registered_data.path}")
        return registered_data
    except Exception as e:
        print(f"Error registering dataset: {e}")
        return None
    else:
        print("Could not proceed due to missing columns or transformation errors.")
        return None
    
    #get data 
    data_to_be_processed = ml_client.data.get(name=name_of_data, label="latest")

    # Create job input
    job_input = Input(path=AssetTypes.URI_FOLDER, type=data_to_be_processed.id)
    
    # Submit the job
    job = ml_client.batch_endpoints.invoke(
        endpoint_name=endpoint_name,
        input=job_input
    )
    # Monitor job progress
    while True:
        job_status = ml_client.jobs.get(job.name)
        print(f"Job status: {job_status.status}")
        
        if job_status.status in ['Completed', 'Failed', 'Canceled']:
            print(f"Job finished with status: {job_status.status}")
            if job_status.status == 'Completed':
                print(f"Output available at: {job_status.outputs}")
            break
            
        time.sleep(30)  # Wait 30 seconds before checking again
            
        return job
    print(f"Submitted batch job with ID: {job.name}")
    return job

# Main function
def main(args):
    # Get Azure ML client
    # Get Azure ML client
    try:
        # Try DefaultAzureCredential first
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
            # Fall back to interactive login
            credential = InteractiveBrowserCredential()
            ml_client = MLClient(
                credential=credential,
                subscription_id="cda9116f-5326-4a9b-9407-bc3a4391c27c",
                resource_group_name="gabby102",
                workspace_name="health-update"
            )
        except Exception as e:
            print(f"Interactive login failed: {e}")
            return None

    # Create or get batch endpoint
    endpoint = create_batch_endpoint(ml_client, args.endpoint_name)
    # Create batch deployment if requested
    deployment = create_batch_deployment(
            ml_client, 
            args.endpoint_name, 
            args.env_name,
            args.model_name, 
            args.deployment_name,
            args.code_path,
            args.scoring_path,
            args.compute
        )
    # Submit batch job if requested
    job = submit_batch_job(
            ml_client, 
            args.endpoint_name,
            args.training_data_path,
            args.name_of_data)

    return endpoint


# Run script
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='deploy batchpipeline in Azure ML')
    
    
    parser.add_argument('--endpoint_name', type=str, default="accident-prediction-endpoint",
                       help='Name of the batch endpoint')

    parser.add_argument('--model_name', type=str, required = True,
                       help='Name of the registered model to use for batch deployment')
    
    parser.add_argument('--deployment_name', type=str, default="accident-prediction-deployment",
                       help='Name for the batch deployment')
    
    parser.add_argument('--env_name', type=str, default="accident-prediction-environment",
                       help='name of already created environment')
    
    parser.add_argument('--code_path', type=str, default="./src",
                       help='Path to the code directory')
    parser.add_argument('--scoring_path', type=str, default="batch_scoring.py",
                       help='Name of the scoring script')
    parser.add_argument('--compute', type=str, default="captgt0071",
                       help='Name of the compute target')
    parser.add_argument('--training_data_path', type=str,
                       help='the link from the output of  transforming data ')

    
    parser.add_argument('--name_of_data', type=str,default = 'accident_survival_data',
                       help='Name of the registered dataset to use for the job')
    
    
    print("\n" + "*" * 60)
    args = parser.parse_args()  
    main(args)
    print("*" * 60 + "\n")
