#!/bin/bash
# Create a clean environment for RL experiments on MIT Supercloud

echo "🆕 Creating clean RL environment..."

# Remove old environment if it exists
conda env remove -n rl_clean -y

# Create new environment with Python 3.10
conda create -n rl_clean python=3.10 -y

# Activate the environment
conda activate rl_clean

# Install packages via conda first (better dependency resolution)
conda install -c conda-forge -y \
    numpy=1.24.3 \
    pandas=2.0.3 \
    scikit-learn=1.3.0 \
    pyarrow=12.0.1 \
    protobuf=3.20.3

# Install PyTorch for CPU/CUDA (adjust for your system)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install ML packages via pip
pip install \
    datasets==2.14.0 \
    transformers==4.36.0 \
    accelerate==0.24.0 \
    peft==0.6.0 \
    trl==0.7.4

# Test the installation
echo "🧪 Testing installation..."
python -c "
import torch
import datasets
import transformers
import pyarrow
print('✅ All packages imported successfully!')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'Datasets: {datasets.__version__}')
"

echo "✅ Environment 'rl_clean' created successfully!"
echo "To use: conda activate rl_clean" 