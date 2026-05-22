#!/bin/bash

# Project Name
PROJECT="src"

# =========================
# Create Main Directories
# =========================

mkdir -p $PROJECT
mkdir -p $PROJECT/cloud
mkdir -p $PROJECT/components
mkdir -p $PROJECT/configuration
mkdir -p $PROJECT/constants
mkdir -p $PROJECT/entity
mkdir -p $PROJECT/exception
mkdir -p $PROJECT/logger
mkdir -p $PROJECT/pipeline
mkdir -p $PROJECT/utils

mkdir -p config
mkdir -p notebook

# =========================
# Create __init__.py Files
# =========================

touch $PROJECT/__init__.py

touch $PROJECT/cloud/__init__.py

touch $PROJECT/components/__init__.py
touch $PROJECT/components/data_ingestion.py
touch $PROJECT/components/data_validation.py
touch $PROJECT/components/data_transformation.py
touch $PROJECT/components/model_trainer.py
touch $PROJECT/components/model_evaluation.py
touch $PROJECT/components/model_pusher.py

touch $PROJECT/configuration/__init__.py

touch $PROJECT/constants/__init__.py

touch $PROJECT/entity/__init__.py
touch $PROJECT/entity/config_entity.py
touch $PROJECT/entity/artifact_entity.py

touch $PROJECT/exception/__init__.py

touch $PROJECT/logger/__init__.py

touch $PROJECT/pipeline/__init__.py
touch $PROJECT/pipeline/training_pipeline.py
touch $PROJECT/pipeline/prediction_pipeline.py

touch $PROJECT/utils/__init__.py

# =========================
# Create Config Files
# =========================
touch config/model.yaml
touch config/schema.yaml

# =========================
# Create Notebook
# =========================
touch notebook/experiment.ipynb

# =========================
# Create Main Project Files
# =========================

touch app.py
touch requirements.txt
touch requirement-dev.txt
touch Dockerfile
touch .dockerignore
touch demo.py
touch setup.py
touch README.md

echo "Project structure created successfully!"

## bash template.sh