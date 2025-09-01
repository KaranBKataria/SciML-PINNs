#!/usr/bin/env bash

# Create virtual enviroment to prevent clashing dependencies
python3 -m venv venv

# Activate the virtual enviroment via running the 'activate' shell script
source venv/bin/activate

# Install all required dependencies within the virtual env.
pip3 install -r requirements.txt