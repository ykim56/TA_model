#!/bin/bash

PROJECT_ID=$(gcloud config list project --format "value(core.project)")
REPO_NAME='resnet'
IMAGE_NAME='resnet-test'
IMAGE_TAG='latest'
IMAGE_URI=us-central1-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}
#export IMAGE_URI=us-central1-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}

docker build -f Dockerfile -t ${IMAGE_URI} ./
