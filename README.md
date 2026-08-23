# Shipment-Price-Prediction-System-ML-AWS

Developed an end-to-end Machine Learning project for shipment price prediction using Python, Scikit-learn, Fast-api, Docker, and AWS deployment with real-time prediction capability.

![AI](Data/shipment-demo.png)

## Shipment-Price-Prediction-ML-Project

## How to run

Before you run this project make sure you have MongoDB Atlas account and you have the shipping dataset into it.

## Workflow

0. Config yaml
1. Constants
2. config_entity
3. artifacts_entity
4. Components
5. pipeline
6. main.py

## 1️⃣ Clone Repository

```bash
git clone https://github.com/Ahmed2797/Shipment-Price-Prediction-System-ML-AWS-.git
```

### 2️⃣ Create Environment

```bash
conda create -n ship python=3.10 -y
conda activate ship
```

### Install pip packages from requirements.txt

``` bash
pip install -r requirements.txt

## .env
AWS_ACCESS_KEY_ID = your_access_key_here
AWS_SECRET_ACCESS_KEY = your_secret_key_here
AWS_DEFAULT_REGION = us-east-1

export AWS_ACCESS_KEY_ID = "YOUR_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY = "YOUR_SECRET_ACCESS_KEY"


## Load-Data
python push_data_mongo.py
## Train The model-pipeline
python main.py

## web app
python app.py
## Now open up you local host and port
```

## AWS-CICD-Deployment-with-Github-Actions

### 1. Login to AWS console

### 2. Create IAM user for deployment

``` bash
# with specific access
1. EC2 access : It is virtual machine

2. ECR: Elastic Container registry to save your docker image in aws

#Description: About the deployment

1. Build docker image of the source code

2. Push your docker image to ECR

3. Launch Your EC2 

4. Pull Your image from ECR in EC2

5. Lauch your docker image in EC2

#Policy:

1. AmazonEC2ContainerRegistryFullAccess

2. AmazonEC2FullAccess
```

### 3. Create ECR repo to store/save docker image

- Save the URI: 520551197421.dkr.ecr.us-east-1.amazonaws.com/shipment-cost

### 4. Create EC2 machine (Ubuntu)

### 5. Open EC2 and Install docker in EC2 Machine

``` bash
    #optinal

    sudo apt-get update -y

    sudo apt-get upgrade

    #required

    curl -fsSL https://get.docker.com -o get-docker.sh

    sudo sh get-docker.sh

    sudo usermod -aG docker ubuntu

    newgrp docker
```

### 6. Setup github secrets

```bash

AWS_ACCESS_KEY_ID =

AWS_SECRET_ACCESS_KEY =

AWS_REGION = us-east-1

AWS_ECR_LOGIN_URI = 520551197421.dkr.ecr.us-east-1.amazonaws.com

ECR_REPOSITORY_NAME = Shipment-Cost
```
