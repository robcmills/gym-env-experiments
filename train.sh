#!/bin/bash

# Create a virtual environment
python3 -m venv gym_venv

# Activate the virtual environment
source gym_venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train the agent
python3 train2.py
