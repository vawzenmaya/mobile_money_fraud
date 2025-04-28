#!/usr/bin/env bash
# exit on error
set -o errexit

# Update pip
pip install --upgrade pip

# Install dependencies
pip install --no-cache-dir -r requirements.txt
