#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -o errexit

# Install CPU-only PyTorch first to avoid 2.5GB NVIDIA CUDA libraries
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install -r requirements.txt

# Collect static files
python manage.py collectstatic --no-input

# Apply database migrations
python manage.py migrate
