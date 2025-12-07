#!/bin/bash
# run_all_dp.sh - Master script to train DP models for any epsilon

set -e  # Exit on error

# Default values
DEFAULT_EPSILON=8.0
DEFAULT_EPOCHS=5
DEFAULT_BATCH_SIZE=32

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}"
    echo "="*70
    echo "   🚀 DP MODEL TRAINING CONTROL PANEL"
    echo "="*70
    echo -e "${NC}"
}

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --epsilon VALUE    Privacy budget ε (default: $DEFAULT_EPSILON)"
    echo "  -n, --epochs VALUE     Number of epochs (default: $DEFAULT_EPOCHS)"
    echo "  -b, --batch_size VALUE Batch size (default: $DEFAULT_BATCH_SIZE)"
    echo "  -a, --all              Train all epsilon values: 0.5, 1.0, 2.0, 3.0, 5.0, 8.0"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -e 1.0              Train model with ε=1.0"
    echo "  $0 -e 2.0 -n 10        Train with ε=2.0 for 10 epochs"
    echo "  $0 -a                  Train all epsilon values sequentially"
    echo ""
}

# Parse command line arguments
EPSILON=$DEFAULT_EPSILON
EPOCHS=$DEFAULT_EPOCHS
BATCH_SIZE=$DEFAULT_BATCH_SIZE
TRAIN_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--epsilon)
            EPSILON="$2"
            shift 2
            ;;
        -n|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -b|--batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -a|--all)
            TRAIN_ALL=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Function to update epsilon in Python files
update_epsilon_in_files() {
    local epsilon=$1
    
    echo -e "${YELLOW}Updating files for ε=$epsilon...${NC}"
    
    # 1. Update distilbert_dp.py default epsilon
    sed -i "s/epsilon: float = [0-9]*\.[0-9]*/epsilon: float = $epsilon/" src/models/distilbert_dp.py
    
    # 2. Update train_dp.py default epsilon
    sed -i "s/parser.add_argument(\"--epsilon\", type=float, default=[0-9]*\.[0-9]*/parser.add_argument(\"--epsilon\", type=float, default=$epsilon/" src/training/train_dp.py
    
    # 3. Update noise multiplier based on epsilon (inverse relationship)
    # Higher epsilon = less privacy = less noise
    local noise_multiplier=$(echo "scale=2; 2.0 / $epsilon" | bc)
    sed -i "s/noise_multiplier: float = [0-9]*\.[0-9]*/noise_multiplier: float = $noise_multiplier/" src/models/distilbert_dp.py
    
    echo -e "${GREEN}✅ Files updated for ε=$epsilon (noise_multiplier=$noise_multiplier)${NC}"
}

# Function to train a single model
train_single_model() {
    local epsilon=$1
    local epochs=$2
    local batch_size=$3
    
    echo -e "\n${BLUE}="*70
    echo "   TRAINING MODEL: ε = $epsilon"
    echo "="*70
    echo -e "${NC}"
    
    # Update files with current epsilon
    update_epsilon_in_files $epsilon
    
    # Generate model name
    MODEL_NAME="distilbert_epsilon_${epsilon}"
    
    # Create directories
    mkdir -p "outputs/models/$MODEL_NAME"
    mkdir -p "outputs/plots/$MODEL_NAME"
    mkdir -p "outputs/logs/$MODEL_NAME"
    
    # Set parameters
    LEARNING_RATE=5e-5
    MAX_SEQ_LENGTH=128
    OUTPUT_DIR="outputs/models/$MODEL_NAME"
    
    echo -e "${YELLOW}Configuration:${NC}"
    echo "  ε (epsilon):        $epsilon"
    echo "  Model name:         $MODEL_NAME"
    echo "  Epochs:             $epochs"
    echo "  Batch size:         $batch_size"
    echo "  Learning rate:      $LEARNING_RATE"
    echo "  Output directory:   $OUTPUT_DIR"
    
    # Activate environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate pii-jax
    
    # Check hardware
    echo -e "\n${YELLOW}Hardware check:${NC}"
    python -c "
import jax
devices = jax.device_count()
device_type = jax.devices()[0].device_kind if devices > 0 else 'CPU'
print(f'  JAX devices: {devices}')
print(f'  Device type: {device_type}')
print(f'  Platform: {jax.default_backend()}')
"
    
    # Start timer
    START_TIME=$(date +%s)
    
    # Run training
    echo -e "\n${YELLOW}Starting training...${NC}"
    python src/training/train_dp.py \
        --epsilon $epsilon \
        --model_name $MODEL_NAME \
        --epochs $epochs \
        --batch_size $batch_size \
        --learning_rate $LEARNING_RATE \
        --max_seq_length $MAX_SEQ_LENGTH \
        --output_dir "outputs/models" \
        --early_stopping_patience 2
    
    # Generate plots
    echo -e "\n${YELLOW}Generating plots...${NC}"
    python src/evaluation/plot_utils.py \
        --model_dir "$OUTPUT_DIR" \
        --epsilon $epsilon \
        --output_dir "outputs/plots/$MODEL_NAME"
    
    # Calculate training time
    END_TIME=$(date +%s)
    TRAINING_TIME=$((END_TIME - START_TIME))
    
    # Display summary
    echo -e "\n${GREEN}✅ TRAINING COMPLETE!${NC}"
    echo -e "${BLUE}Summary for ε=$epsilon:${NC}"
    echo "  Training time:      $((TRAINING_TIME / 60)) minutes $((TRAINING_TIME % 60)) seconds"
    echo "  Model directory:    $OUTPUT_DIR"
    echo "  Plots directory:    outputs/plots/$MODEL_NAME"
    
    # Check if results file exists and show metrics
    RESULTS_FILE="$OUTPUT_DIR/training_history.json"
    if [[ -f "$RESULTS_FILE" ]]; then
        BEST_F1=$(python -c "
import json
with open('$RESULTS_FILE') as f:
    data = json.load(f)
if 'val_f1' in data and data['val_f1']:
    print(f\"{max(data['val_f1']):.4f}\")
else:
    print('N/A')
")
        echo "  Best validation F1:  $BEST_F1"
    fi
    
    # Save run info
    RUN_INFO="$OUTPUT_DIR/run_info.txt"
    cat > "$RUN_INFO" << EOF
Training run information:
=======================
Timestamp:         $(date)
Epsilon (ε):       $epsilon
Epochs:            $epochs
Batch size:        $batch_size
Learning rate:     $LEARNING_RATE
Training time:     $((TRAINING_TIME / 60))m $((TRAINING_TIME % 60))s
Model name:        $MODEL_NAME
Output directory:  $OUTPUT_DIR
Command:           $0 -e $epsilon -n $epochs -b $batch_size
EOF
    
    echo -e "${BLUE}Run info saved to: $RUN_INFO${NC}"
}

# Function to train all epsilon values
train_all_models() {
    local epochs=$1
    local batch_size=$2
    
    local epsilons=(0.5 1.0 2.0 3.0 5.0 8.0)
    
    echo -e "${BLUE}"
    echo "="*70
    echo "   TRAINING ALL MODELS (ε = ${epsilons[*]})"
    echo "="*70
    echo -e "${NC}"
    
    # Create summary file
    SUMMARY_FILE="outputs/dp_training_summary_$(date +%Y%m%d_%H%M%S).txt"
    echo "DP Training Summary - $(date)" > "$SUMMARY_FILE"
    echo "="*60 >> "$SUMMARY_FILE"
    
    for epsilon in "${epsilons[@]}"; do
        echo -e "\n${BLUE}--- Training ε=$epsilon ---${NC}"
        
        # Start timer for this epsilon
        EPSILON_START=$(date +%s)
        
        # Train model
        train_single_model $epsilon $epochs $batch_size
        
        # Calculate time for this epsilon
        EPSILON_END=$(date +%s)
        EPSILON_TIME=$((EPSILON_END - EPSILON_START))
        
        # Add to summary
        echo "ε=$epsilon: $((EPSILON_TIME / 60))m $((EPSILON_TIME % 60))s" >> "$SUMMARY_FILE"
        
        # Wait a moment before next training
        echo -e "${YELLOW}Waiting 10 seconds before next training...${NC}"
        sleep 10
    done
    
    # Generate combined privacy-utility plot
    echo -e "\n${BLUE}Generating combined privacy-utility plot...${NC}"
    python -c "
import json
import glob
import matplotlib.pyplot as plt
import os

epsilons = []
f1_scores = []

# Find all training_history.json files
for eps in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
    history_file = f'outputs/models/distilbert_epsilon_{eps}/training_history.json'
    if os.path.exists(history_file):
        with open(history_file) as f:
            data = json.load(f)
        if 'val_f1' in data and data['val_f1']:
            best_f1 = max(data['val_f1'])
            epsilons.append(eps)
            f1_scores.append(best_f1)
            print(f'ε={eps}: F1={best_f1:.4f}')

# Create plot
if epsilons:
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, f1_scores, 'o-', linewidth=3, markersize=10)
    plt.xlabel('Privacy Budget (ε)')
    plt.ylabel('F1 Score')
    plt.title('Privacy-Utility Tradeoff (All Models)')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for eps, f1 in zip(epsilons, f1_scores):
        plt.annotate(f'{f1:.3f}', (eps, f1), xytext=(0, 10), 
                    textcoords='offset points', ha='center')
    
    plt.tight_layout()
    plt.savefig('outputs/privacy_tradeoff_all.png', dpi=150)
    print('✅ Combined plot saved: outputs/privacy_tradeoff_all.png')
"
    
    echo -e "\n${GREEN}"
    echo "="*70
    echo "   ALL MODELS TRAINING COMPLETE!"
    echo "="*70
    echo -e "${NC}"
    echo "📊 Summary saved to: $SUMMARY_FILE"
    echo "📈 Combined plot:    outputs/privacy_tradeoff_all.png"
    echo ""
    echo "Individual models:"
    for epsilon in "${epsilons[@]}"; do
        echo "  ε=$epsilon: outputs/models/distilbert_epsilon_$epsilon/"
    done
}

# Main execution
print_header

# Create outputs directory
mkdir -p outputs/{models,plots,logs,results}

if [[ "$TRAIN_ALL" == true ]]; then
    train_all_models $EPOCHS $BATCH_SIZE
else
    train_single_model $EPSILON $EPOCHS $BATCH_SIZE
fi

# Create final summary
echo -e "\n${BLUE}Final output structure:${NC}"
tree outputs/ --dirsfirst -L 2 2>/dev/null || find outputs/ -type f | head -20

echo -e "\n${GREEN}✅ Script completed successfully!${NC}"
