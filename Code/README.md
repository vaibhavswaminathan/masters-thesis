# Master's Thesis Codebase
# Towards Time Series Classification of Building Automation Datapoints using Semantic Data Models

## Description

This repository contains scripts and code resources to implement the classification workflow of my Master's thesis.

## Prerequisites

### Create a virtual Python environment

The `requirements.txt` has the list of Python packages needed to run the code in this workspace. Using it, you can set up a virtual Python environment and install all required packages 

```bash
pip install virtualenv
virtualenv my_virtual_env
source my_virtual_env/bin/activate
pip install -r requirements.txt
```

### Source time series data

For our project, we sourced time series data of Air-Handling Unit datapoints using the aedifion API. Refer [here](https://api.ercebc.aedifion.io/ui/#!/) for the API documentation.

[get_timeseries.py](get_timeseries.py) downloads time series data using the aedifion API and save it as a .CSV file. Make sure to change the parameters (datapoints, start date, end date, sampling rate) within the script for your use-case.

```bash
python3 get_timeseries.py
```

## Time Series Classification

### Data Pre-processing

1. Run [format_csv_data.ipynb](format_csv_data.ipynb) to transform the CSV data into a format accepted by a neural network architecture. The data is split into train, test, and validation sets and stored.

2. Run [create_dataset.py](SimTSC/create_dataset.py) to convert the data from .CSV to numpy arrays.

    Pass the data set name to the `dataset` parameter.

    ```bash
    create_dataset.py --dataset AHU_principal_SUMMER_2023_stanscaler
    ```
3. After the data has been processed and saved, run [compute_distances.ipynb](compute_distances.ipynb) to compute pairwise distances between data samples and save it as a Numpy array.

### Training the neural network

The implementation in this section is based on  https://github.com/daochenzha/SimTSC/tree/main and additional features for the training procedure, evaluation, and logging have been added.

4. Train the classifier

    For ResNet:
    ```bash
    python3 train_resnet.py --dataset AHU_principal_2023
    ```

    For GNN (SimTSC):
    ```bash
    python3 train_simtsc.py --dataset AHU_principal_2023 --distance euclidean
    ```
    The neural network architectures are described in `model.py` for ResNet and GNN (SimTSC). These can be changed if needed.

5. The log files created during each training run are stored in /home/vaibhavs/Master_Thesis/ma-vaibhav/Code/SimTSC/logs. 

### Rule-based classification

6. The data model for the Air-Handling Unit is created using the Brick Schema. Rules are defined in a triple-style (subject-predicate-object) using terms from the Brick ontology. See [data_model.ttl](data_model.ttl) for an example.

    Use this as a template to define new rules or modify existing rules.

7. [rule_loader.py](Utils/rule_loader.py) has example SPARQL queries to extract rule triples from the data model.

8. [run_rules_summer.py](Utils/run_rules_summer.py) and [run_rules_winter.py](Utils/run_rules_winter.py) define the rule logic in Python for each rule. This script uses the GNN predictions and the data set as input to evaluate rules and generate rule predictions.


