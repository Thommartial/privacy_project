#!/bin/bash
# run_preprocess.sh - Run data preprocessing

echo "=" * 50
echo "📊 DATA PREPROCESSING"
echo "=" * 50
echo "Dataset: /home/thom/Desktop/dpjax/data/raw/PII-Masking-43K.csv"
echo "Samples: 5000"
echo "Split: 70%/15%/15% (3500/750/750)"
echo "=" * 50

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pii-jax

# Run preprocessing
python src/data/preprocess.py

echo ""
echo "✅ Data ready in data/processed/"
ls -la data/processed/
