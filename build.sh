#!/bin/bash

# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Force reinstall numpy properly
pip install --force-reinstall --no-cache-dir numpy==1.24.3

# Install the rest of the dependencies
pip install -r requirements.txt
