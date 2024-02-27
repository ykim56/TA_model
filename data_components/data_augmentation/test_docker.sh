#!/bin/bash

IMAGE_URI=us-east1-docker.pkg.dev/ta-model-data-preprocess/ta-model-pipelines/data_preprocess_test_1@sha256:402727ef4038e3c50006775edb673afea2c95c20e28073308b9cd5a06ed86118


docker run -it ${IMAGE_URI}
