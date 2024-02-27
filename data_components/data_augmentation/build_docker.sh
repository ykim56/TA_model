#!/bin/bash

PROJECT_ID=$(gcloud config list project --format "value(core.project)")
REPO_NAME='ta-model-pipelines'
IMAGE_NAME='data_augmentation_test_5'
IMAGE_TAG='test'
IMAGE_URI=us-east1-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}
#export IMAGE_URI=us-central1-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}

docker build -f Dockerfile_1 -t ${IMAGE_URI} ./
