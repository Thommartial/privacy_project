#!/bin/bash
# train_all_epsilons.sh - Train models for all epsilon values

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}"
echo "="*70
echo "   🚀 TRAIN ALL ε-VALUES"
echo "="*70
echo -e "${NC}"

# Epsilon values to train
EPSILONS=(8.0 5.0 3.0 2.0 1.0 0.5)
EPOCHS=5
BATCH_SIZE=32

# Create results summary file
SUMMARY_FILE="outputs/training_summary_$(date +%Y%m%d_%H%M%S).txt"
echo "Training Summary - $(date)" > "$SUMMARY_FILE"
echo "Epsilon | Best F1 | Training Time | Model Directory" >> "$SUMMARY_FILE"
echo "--------|---------|---------------|----------------" >> "$SUMMARY_FILE"

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pii-jax

# Train each epsilon
for epsilon in "${EPSILONS[@]}"; do
    echo -e "\n${BLUE}="*60
    echo "   TRAINING: ε = $epsilon"
    echo "="*60
    echo -e "${NC}"
    
    START_TIME=$(date +%s)
    
    # Generate model name
    MODEL_NAME="distilbert_epsilon_${epsilon}"
    
    echo -e "${YELLOW}Configuration:${NC}"
    echo "  ε (epsilon): $epsilon"
    echo "  Model name:  $MODEL_NAME"
    echo "  Epochs:      $EPOCHS"
    echo "  Batch size:  $BATCH_SIZE"
    
    # Run training
    python src/training/train_dp.py \
        --epsilon $epsilon \
        --model_name "$MODEL_NAME" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate 5e-5 \
        --max_seq_length 128 \
        --output_dir "outputs/models" \
        --early_stopping_patience 2
    
    # Generate plots
    python src/evaluation/plot_utils.py \
        --model_dir "outputs/models/$MODEL_NAME" \
        --epsilon $epsilon \
        --output_dir "outputs/plots"
    
    END_TIME=$(date +%s)
    TRAINING_TIME=$((END_TIME - START_TIME))
    
    # Extract best F1 score
    HISTORY_FILE="outputs/models/$MODEL_NAME/training_history.json"
    if [[ -f "$HISTORY_FILE" ]]; then
        BEST_F1=$(python -c "
import json
try:
    with open('$HISTORY_FILE') as f:
        data = json.load(f)
    if 'val_f1' in data and data['val_f1']:
        print(f'{max(data[\"val_f1\"]):.4f}')
    else:
        print('N/A')
except:
    print('N/A')
")
    else
        BEST_F1="N/A"
    fi
    
    # Add to summary
    printf "%-8s | %-7s | %-13s | outputs/models/%s\n" \
        "$epsilon" "$BEST_F1" "$((TRAINING_TIME / 60))m $((TRAINING_TIME % 60))s" "$MODEL_NAME" >> "$SUMMARY_FILE"
    
    echo -e "${GREEN}✅ Completed ε=$epsilon in $((TRAINING_TIME / 60))m $((TRAINING_TIME % 60))s${NC}"
    
    # Wait before next training
    if [[ "$epsilon" != "0.5" ]]; then
        echo -e "${YELLOW}Waiting 15 seconds before next training...${NC}"
        sleep 15
    fi
done

# Generate combined comparison plot
echo -e "\n${BLUE}Generating combined comparison plot...${NC}"
python src/evaluation/plot_utils.py --compare_all --model_dir "outputs/models"

echo -e "\n${GREEN}"
echo "="*70
echo "   ✅ ALL MODELS TRAINED SUCCESSFULLY!"
echo "="*70
echo -e "${NC}"

echo -e "${YELLOW}📋 Training Summary:${NC}"
cat "$SUMMARY_FILE"

echo -e "\n${YELLOW}📁 Output Structure:${NC}"
echo "outputs/models/distilbert_epsilon_*/  - Model checkpoints & configs"
echo "outputs/plots/distilbert_epsilon_*/   - Training plots"
echo "outputs/privacy_tradeoff_all_models.png - Combined comparison"
echo ""
echo -e "${YELLOW}📊 To train a single epsilon value:${NC}"
echo "  python src/training/train_dp.py --epsilon 2.0"
echo ""
echo -e "${YELLOW}📈 To generate plots for existing model:${NC}"
echo "  python src/evaluation/plot_utils.py --model_dir outputs/models/distilbert_epsilon_8.0"
