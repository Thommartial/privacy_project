#!/bin/bash
# setup_env.sh - Phase 1: Minimal Environment Setup

echo "🚀 Starting Phase 1: Environment Setup"
echo "--------------------------------------"

# 1. Create Conda environment
conda create -n pii-jax python=3.10 -y

# 2. Activate
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pii-jax

# 3. Install ONLY essential packages
pip install --upgrade pip
pip install jax[cpu]
pip install flax optax
pip install transformers datasets
pip install numpy scikit-learn pandas
pip install matplotlib tqdm seaborn

# 4. Verify installation
echo "✅ Environment created: pii-jax"
echo "✅ JAX version: $(python -c 'import jax; print(jax.__version__)')"
echo "✅ Python version: $(python --version)"

# 5. Create requirements.txt
echo "jax[cpu]" > requirements.txt
echo "flax" >> requirements.txt
echo "optax" >> requirements.txt
echo "transformers" >> requirements.txt
echo "datasets" >> requirements.txt

echo "--------------------------------------"
echo "🎯 Phase 1 Complete: Ready for Phase 2"
echo "Next: mkdir -p src/{data,models,baseline,training,eval}"