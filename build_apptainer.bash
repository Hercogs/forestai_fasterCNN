#! /bin/bash

docker compose build --no-cache
apptainer build fasterCNN.sif docker-daemon://faster_cnn:latest


