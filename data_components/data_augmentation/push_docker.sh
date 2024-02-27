#!/bin/bash

PROJECT_ID=$(gcloud config list project --format "value(core.project)")
REPO_NAME='ta-model-pipelines'
IMAGE_NAME='data_augmentation_test_5'
IMAGE_TAG='test'
IMAGE_URI=us-east1-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}
# refer to https://www.googlecloudcommunity.com/gc/Developer-Tools/Permission-quot-artifactregistry-repositories-uploadArtifacts/m-p/541769
gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin https://us-east1-docker.pkg.dev
docker push ${IMAGE_URI}


