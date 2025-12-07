#!/bin/bash
# test_evaluation.sh - Test evaluation on minimal model

echo "🧪 Testing evaluation on minimal model..."
echo "="*60

# Create evaluation directory
mkdir -p outputs/evaluation

# Run evaluation
python src/evaluation/evaluate_dp_model.py \
    --model_dir outputs/models/minimal_epsilon_8.0 \
    --output_dir outputs/evaluation

echo ""
echo "📁 Checking output files:"
ls -la outputs/evaluation/minimal_epsilon_8.0/ 2>/dev/null || echo "No evaluation output found"

echo ""
echo "🎯 If evaluation works, you should see:"
echo "   - evaluation_report.md"
echo "   - training_curves_epsilon_8.0.png"
echo "   - privacy_tradeoff_epsilon_8.0.png"
echo "="*60
