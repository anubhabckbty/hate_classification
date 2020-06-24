#!/bin/bash

pip install venv
python -m venv $1 
source $1/bin/activate
pip install --upgrade pip
pip install pandas
pip install tensorflow==1.15.0
pip install tensorflow-hub==0.6.0
pip install keras==1.1.0
pip install tweet-preprocessor
