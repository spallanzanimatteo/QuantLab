#!/bin/bash

PROBLEM=$1

# local folder
mkdir ../$PROBLEM
touch ../$PROBLEM/config.json

# data folder
HARD_STORAGE_DATA=$(python -c "import sys; import json; fp = open('hard_storage.json', 'r'); d = json.load(fp); fp.close(); print(d['data'])")/QuantLab
mkdir $HARD_STORAGE_DATA/$PROBLEM
mkdir $HARD_STORAGE_DATA/$PROBLEM/data

# logs folder
HARD_STORAGE_LOGS=$(python -c "import sys; import json; fp = open('hard_storage.json', 'r'); d = json.load(fp); fp.close(); print(d['logs'])")/QuantLab
mkdir $HARD_STORAGE_LOGS/$PROBLEM
mkdir $HARD_STORAGE_LOGS/$PROBLEM/logs

# QuantLab package
PACKAGE=../quantlab
mkdir $PACKAGE/$PROBLEM
touch $PACKAGE/$PROBLEM/__init__.py
mkdir $PACKAGE/$PROBLEM/utils
touch $PACKAGE/$PROBLEM/utils/__init__.py
touch $PACKAGE/$PROBLEM/utils/meter.py
touch $PACKAGE/$PROBLEM/utils/inference.py
