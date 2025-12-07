#!/bin/bash
# create_structure.sh - Phase 2: Project Structure

echo "🚀 Starting Phase 2: Project Structure"
echo "--------------------------------------"

# Create ONLY essential directories
mkdir -p src/{data,models,baseline,training,eval}
mkdir -p configs
mkdir -p outputs/{models,plots,logs,results}
mkdir -p data/raw

# Create empty __init__.py files
for dir in src src/data src/models src/baseline src/training src/eval; do
    touch $dir/__init__.py
done

# Create minimal config file
cat > configs/project.yaml << 'EOF'
# Project Configuration
project:
  name: "pii-detection-dp"
  author: "Lanre Atoye, Ekwelle Epalle Thomas Martial"
  course: "CIS*6550"

data:
  dataset: "PII-Masking-43K"
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1

model:
  base: "distilbert-base-uncased"
  num_labels: 3  # BIO tagging

training:
  epsilons: [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
  batch_size: 8
  epochs: 3
  learning_rate: 5e-5
EOF

# Create minimal README
cat > README.md << 'EOF'
# PII Detection with Differential Privacy

## Quick Start
1. Setup environment: `./setup_env.sh`
2. Create structure: `./create_structure.sh`
3. Run baseline: `python src/baseline/regex_detector.py`
4. Train models: `python src/training/train.py`

## Structure
- `src/` - Source code
- `data/` - Dataset files
- `outputs/` - Results, models, plots
- `configs/` - Configuration files
EOF

echo "✅ Project structure created:"
echo "   src/       - Source code"
echo "   data/      - Dataset storage"
echo "   outputs/   - Results and models"
echo "   configs/   - Configuration"
echo ""
echo "--------------------------------------"
echo "🎯 Phase 2 Complete: Ready for Phase 3"
echo "Next: Implement Data Preprocessing"