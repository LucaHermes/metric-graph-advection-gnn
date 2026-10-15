#!/bin/bash

set -e

ENV_NAME=quality
PYTHON_VERSION=3.9

# Activate conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Check if environment exists
if conda info --envs | grep -q "^$ENV_NAME "; then
  echo "Conda environment '$ENV_NAME' already exists."
  read -p "Do you want to remove it and recreate it? (y/[n]): " CONFIRM
  if [[ "$CONFIRM" == "y" ]]; then
    conda env remove -n $ENV_NAME
  else
    echo "Exiting without changes."
    exit 0
  fi
fi

echo "Creating conda environment '$ENV_NAME'..."
conda env create -f environment.yml

# Activate the environment
conda activate $ENV_NAME

# Upgrade pip
pip install --upgrade pip

# Detect OS
OS_NAME="$(uname -s)"

if [[ "$OS_NAME" == "Darwin" ]]; then
  echo "Detected macOS — installing CPU-only PyTorch."
  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
  pip install torch-geometric
  pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
else
  echo "Detected Linux — installing PyTorch + CUDA 11.8"
  pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

  # Install PyTorch Geometric and friends
  pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
    -f https://data.pyg.org/whl/torch-2.5.1+cu118.html
fi

echo "Environment '$ENV_NAME' setup complete."
