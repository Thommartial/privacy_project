---

## **Phase 1: Environment Setup** (30 min)
**Task:** Create Conda environment with JAX  
**Script:** `setup_env.sh`
```bash
conda create -n pii-jax python=3.10 -y
conda activate pii-jax
pip install jax[cpu] flax optax transformers datasets numpy scikit-learn matplotlib
```

**Deliverable:** ✅ Working environment

---

## **Phase 2: Project Structure** (30 min)
**Task:** Create minimal directory structure  
```
src/
├── data/          # Dataset loading
├── models/        # JAX model code
├── baseline/      # Regex model
├── training/      # Training loops
└── eval/          # Evaluation scripts
configs/           # YAML configs
outputs/
├── models/        # Saved checkpoints
├── plots/         # All figures
└── logs/          # Training logs
```

**Deliverable:** ✅ Organized project layout

---

## **Phase 3: Regex Baseline** (1 hour)
**Task:** Implement and test regex PII detector
```python
# src/baseline/regex_detector.py
class RegexPIIDetector:
    patterns = {'EMAIL': r'...', 'PHONE': r'...'}
    def detect(text): ...
    def evaluate(): ...  # Get baseline metrics
```

**Deliverable:** ✅ Baseline F1 score (save to `outputs/results/baseline.json`)

---

## **Phase 4: DP Model Training** (2 hours)
**Task:** Train 6 DP models (ε=0.5,1,2,3,5,8)

1. **Create training script** `src/training/train.py`:
   - Load DistilBERT tokenizer
   - Implement DP-SGD in JAX
   - Train with different noise multipliers

2. **Log training curves** (loss, accuracy) for each ε
3. **Save models** to `outputs/models/epsilon_{eps}.pkl`

**Deliverable:** ✅ 6 trained models + training curves

---

## **Phase 5: Evaluation & Visualization** (1.5 hours)
**Tasks:**

1. **Privacy-Accuracy Plot** (`src/eval/tradeoff.py`):
   ```python
   # X-axis: ε values [0.5,1,2,3,5,8]
   # Y-axis: F1 scores for each model
   # Plot: line chart showing trade-off
   ```

2. **Training Curves Figure** (6 subplots):
   - One subplot per ε value
   - Show loss vs. epochs
   - Show accuracy vs. epochs

3. **Entity-Level Performance** (bar chart):
   - X: PII types (NAME, EMAIL, PHONE...)
   - Y: F1 score
   - Two bars per type: Baseline vs. Best ML model

4. **Model Architecture Diagram** (simple figure):
   - Input → DistilBERT layers → Token classification
   - DP-SGD noise injection point

**Deliverable:** ✅ 4 figures in `outputs/plots/`

---

## **Phase 6: Report Generation** (30 min)
**Task:** Auto-generate report from results
```python
# src/report/generate.py
template = """
# Results
## Privacy Trade-off
![trade-off](plots/tradeoff.png)

## Training Curves  
![curves](plots/training_curves.png)

## Entity Performance
![entities](plots/entity_f1.png)
"""
```

**Deliverable:** ✅ `outputs/report.md` with embedded figures

---

## **Timeline: 6 Hours Total**
- **Phase 1:** 0:00-0:30  
- **Phase 2:** 0:30-1:00  
- **Phase 3:** 1:00-2:00  
- **Phase 4:** 2:00-4:00  
- **Phase 5:** 4:00-5:30  
- **Phase 6:** 5:30-6:00  

---

**Want me to start with Phase 1 script?** I'll make it minimal - just environment setup, nothing extra.