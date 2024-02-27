#!/bin/bash

PROJECT_ID=$(gcloud config list project --format "value(core.project)")
REPO_NAME='resnet'
IMAGE_NAME='resnet-test-1'
IMAGE_TAG='latest'
IMAGE_URI=us-central1-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}
# refer to https://www.googlecloudcommunity.com/gc/Developer-Tools/Permission-quot-artifactregistry-repositories-uploadArtifacts/m-p/541769
gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin https://us-central1-docker.pkg.dev
docker push ${IMAGE_URI}


