#!/bin/bash
# Fix for PyArrow/Protobuf conflict on MIT Supercloud
# Run this in your conda environment

echo "🔧 Fixing PyArrow/Protobuf conflict..."

# Activate your environment (replace 'new_rl_code' with your env name if different)
# conda activate new_rl_code

# Uninstall conflicting packages
pip uninstall -y pyarrow datasets protobuf

# Reinstall with compatible versions
pip install protobuf==3.20.3
pip install pyarrow==12.0.1
pip install datasets==2.14.0

# Alternative: Use conda instead of pip for these packages
# conda install -c conda-forge pyarrow=12.0.1 datasets=2.14.0 protobuf=3.20.3

echo "✅ Fixed! Try running your script again." 