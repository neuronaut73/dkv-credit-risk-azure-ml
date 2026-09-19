# ML Azure Interview Task

This task is designed to evaluate your ability to build a machine learning pipeline using Azure Machine Learning and apply standard ML practices in a structured and reproducible way.

## Objective

Train and evaluate a binary classification model that predicts whether a credit card client will default on their payment next month, using an Azure Machine Learning pipeline. The pipeline should include data preprocessing, model training, evaluation, and model registration or output.

## Project Structure

A starter folder structure is provided and you may modify or extend this structure if necessary, provided the final result is well organized and reproducible.

## Dataset

Use the [UCI Credit Card Default Dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

The target variable is: "default payment next month"
where:
- `1` indicates default
- `0` indicates no default

You may rename the target variable.

## Task Requirements

1. **Data Preprocessing**  
   Implement the following in `src/preprocess.py`:
   - Load the dataset
   - Clean the data
   - Follow best practices for preprocessing

2. **Model Training**  
   Implement `src/train.py` to:
   - Load the training data
   - Train a classification model
   - Log training metrics

3. **Model Evaluation**  
   Implement `src/evaluate.py` to:
   - Load the registered model and test data
   - Evaluate the model on the test set
   - Log performance metrics and optionally plots

4. **Azure ML Pipeline**  
   Implement the pipeline in `pipeline/run_pipeline.py`, using the Azure ML SDK v2. The pipeline should run the following steps in sequence:
   - Preprocessing
   - Training
   - Evaluation

   You may implement the pipeline using either the Python SDK or YAML + CLI.

5. **Execution in Azure ML**  
   - Use Azure ML to create a compute cluster
   - Submit the pipeline job to Azure ML
   - Ensure the model is registered in the Azure ML workspace

## Deliverables

Please submit the following:
- A GitHub repository (or a zipped folder) containing:
  - All source code and pipeline definitions
  - A brief description of what was implemented and any relevant notes (really brief)
- Screenshots (or links, if applicable) showing:
  - Successful pipeline run in Azure ML
  - The model registered in the Azure ML workspace

**Additionally, prepare the execution and code for a short presentation going over your solution.**

Optional (bonus):
- Any enhancements to the code (deploying to endpoints, YAML-based component definitions, etc.)

## Environment Setup

You may use the provided `environments/conda.yaml` file or define your own.