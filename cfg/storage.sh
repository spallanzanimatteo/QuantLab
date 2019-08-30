#!/bin/bash

# data folder
HARD_STORAGE_DATA=$(python -c "import sys; import json; fp = open('hard_storage.json', 'r'); d = json.load(fp); fp.close(); print(d['data'])")/QuantLab
mkdir $HARD_STORAGE_DATA

# logs folder
HARD_STORAGE_LOGS=$(python -c "import sys; import json; fp = open('hard_storage.json', 'r'); d = json.load(fp); fp.close(); print(d['logs'])")/QuantLab
mkdir $HARD_STORAGE_LOGS
