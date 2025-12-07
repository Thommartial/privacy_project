#!/bin/bash
echo "Verifying ε-aware fixes..."
echo ""

echo "1. Checking distilbert_dp.py..."
grep -n "epsilon: float = " src/models/distilbert_dp.py
echo ""

echo "2. Checking train_dp.py argument parsing..."
grep -n "parser.add_argument.*epsilon" src/training/train_dp.py
echo ""

echo "3. Checking plot_utils.py ε handling..."
grep -n "epsilon" src/evaluation/plot_utils.py | head -5
echo ""

echo "✅ Verification complete!"
