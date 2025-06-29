#!/bin/bash

# Activate virtual environment
source ~/venvs/torch/bin/activate

# Install requirements if not already installed
pip install -r requirements.txt

# Run the script using the virtual environment's Python
python process_videos.py

# Deactivate virtual environment
deactivate