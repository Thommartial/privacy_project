#!/bin/bash
# train_all_epsilons_proper.sh - Train proper models for all ε values

set -e

echo "="*70
echo "🎯 TRAIN ALL ε-VALUES (PROPER MODELS)"
echo "="*70

EPSILONS=(8.0 5.0 3.0 2.0 1.0 0.5)
SUMMARY_FILE="outputs/training_summary_proper_$(date +%Y%m%d_%H%M%S).txt"

echo "Training ε values: ${EPSILONS[*]}" > "$SUMMARY_FILE"
echo "Timestamp: $(date)" >> "$SUMMARY_FILE"
echo "="*60 >> "$SUMMARY_FILE"
echo "ε | Best Val Acc | Privacy ε | Time" >> "$SUMMARY_FILE"
echo "--|-------------|-----------|------" >> "$SUMMARY_FILE"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate pii-jax

for epsilon in "${EPSILONS[@]}"; do
    echo -e "\n\033[1;34m"  # Blue
    echo "="*60
    echo "   TRAINING ε = $epsilon"
    echo "="*60
    echo -e "\033[0m"
    
    START_TIME=$(date +%s)
    
    # Adjust parameters based on epsilon
    if (( $(echo "$epsilon >= 5.0" | bc -l) )); then
        epochs=5
        batch_size=16
        max_samples=1000
    elif (( $(echo "$epsilon >= 2.0" | bc -l) )); then
        epochs=4
        batch_size=12
        max_samples=800
    else
        epochs=3
        batch_size=8
        max_samples=500
    fi
    
    echo "  Configuration:"
    echo "    Epochs: $epochs"
    echo "    Batch size: $batch_size"
    echo "    Max samples: $max_samples"
    
    python src/training/train_dp_proper.py \
        --epsilon $epsilon \
        --epochs $epochs \
        --batch_size $batch_size \
        --max_samples $max_samples \
        --max_seq_length 64 \
        --hidden_size 256
    
    END_TIME=$(date +%s)
    TRAINING_TIME=$((END_TIME - START_TIME))
    
    # Extract results
    MODEL_DIR="outputs/models/proper_epsilon_$epsilon"
    if [[ -f "$MODEL_DIR/history.json" ]]; then
        BEST_VAL_ACC=$(python -c "
import json
with open('$MODEL_DIR/history.json') as f:
    data = json.load(f)
print(f\"{max(data['val_acc']):.4f}\")" 2>/dev/null || echo "N/A")
        
        PRIVACY_EPSILON=$(python -c "
import json
with open('$MODEL_DIR/history.json') as f:
    data = json.load(f)
if 'privacy_cost' in data:
    print(f\"{data['privacy_cost']['epsilon']:.3f}\")
else:
    print('N/A')" 2>/dev/null || echo "N/A")
    else
        BEST_VAL_ACC="N/A"
        PRIVACY_EPSILON="N/A"
    fi
    
    # Add to summary
    printf "%-4s | %-12s | %-10s | %dm %ds\n" \
        "$epsilon" "$BEST_VAL_ACC" "$PRIVACY_EPSILON" \
        $((TRAINING_TIME / 60)) $((TRAINING_TIME % 60)) >> "$SUMMARY_FILE"
    
    echo -e "\033[1;32m"  # Green
    echo "✅ Completed ε=$epsilon in $((TRAINING_TIME / 60))m $((TRAINING_TIME % 60))s"
    echo -e "\033[0m"
    
    # Wait before next training
    if [[ "$epsilon" != "0.5" ]]; then
        echo "⏳ Waiting 10 seconds..."
        sleep 10
    fi
done

# Generate comparison plot
echo -e "\n\033[1;36m"  # Cyan
echo "="*70
echo "   GENERATING COMPARISON PLOTS"
echo "="*70
echo -e "\033[0m"

python src/evaluation/evaluate_dp_model.py --compare_all --output_dir outputs/evaluation

echo -e "\n\033[1;32m"  # Green
echo "="*70
echo "   ✅ ALL MODELS TRAINED!"
echo "="*70
echo -e "\033[0m"

echo "📋 Summary:"
cat "$SUMMARY_FILE"

echo ""
echo "📁 Models saved in: outputs/models/proper_epsilon_*/"
echo "📊 Evaluation plots: outputs/evaluation/"
echo "📄 Summary file: $SUMMARY_FILE"
echo ""
echo "📈 To evaluate a specific model:"
echo "   python src/evaluation/evaluate_dp_model.py --model_dir outputs/models/proper_epsilon_8.0"
echo "="*70
