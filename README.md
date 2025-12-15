# Differentially Private Deep Learning for PII Detection and Redaction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![JAX](https://img.shields.io/badge/JAX-Accelerated-red.svg)](https://github.com/google/jax)

A privacy-preserving machine learning system that detects and redacts Personally Identifiable Information (PII) in text while preventing training data memorization through differential privacy.

**Course:** CIS*6550 - Privacy, Compliance, and Human Cyber  
**Institution:** University of Guelph, Fall 2025  
**Authors:** Lanre Atoye, Ekwelle Epalle Thomas Martial

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Results Summary](#results-summary)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [Technical Details](#technical-details)
- [Challenges & Solutions](#challenges--solutions)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

---

## Overview

Traditional machine learning models for PII detection face a fundamental paradox: they require sensitive data for training, but can memorize and leak that very data. Research has shown that language models can regurgitate verbatim training examples (Carlini et al., 2021) and are vulnerable to membership inference attacks (Shokri et al., 2017).

This project addresses this challenge by implementing **Differentially Private Stochastic Gradient Descent (DP-SGD)** to train PII detection models with formal privacy guarantees. Our approach ensures that the model's outputs are statistically indistinguishable whether or not any individual record was included in training.

### The Privacy Guarantee

Differential privacy provides a mathematical guarantee: for any two datasets differing by a single record, the probability of any output changes by at most a factor of e^ε, where ε (epsilon) is the privacy budget.

---

## Key Features

- **Privacy-Preserving Training:** DP-SGD with configurable privacy budgets (ε = 0.5 to 8.0)
- **48 PII Entity Types:** Comprehensive detection including names, emails, phone numbers, SSNs, financial data, IP addresses, and more
- **CPU-Only Training:** Democratizing privacy-preserving AI without expensive GPU hardware
- **End-to-End Redaction Pipeline:** From raw text to redacted output with CLI interface
- **JAX-Accelerated:** 8× faster training compared to PyTorch/Opacus implementation
- **Reproducible Research:** Complete codebase with trained model artifacts

---

## Results Summary

### Privacy-Utility Tradeoff

| Privacy Level | ε | Accuracy | Precision | Recall | F1 Score | vs Baseline |
|--------------|---|----------|-----------|--------|----------|-------------|
| Baseline (Regex) | ∞ | 83.33% | 80.12% | 86.74% | — | — |
| Very Weak | 8.0 | **99.47%** | 97.21% | 99.68% | 0.984 | +16.13% |
| Weak | 5.0 | 90.93% | 70.15% | 89.14% | 0.786 | +7.60% |
| Moderate | 3.0 | 88.40% | 64.53% | 87.29% | 0.742 | +5.07% |
| Moderate | 2.0 | 85.60% | 60.21% | 83.15% | 0.698 | +2.27% |
| Strong | 1.0 | 75.07% | 44.18% | 78.23% | 0.562 | -8.27% |
| Strong | 0.5 | 75.07% | 44.18% | 78.23% | 0.562 | -8.27% |

### Key Findings

1. **Deep learning outperforms regex** even with differential privacy noise (+16% at ε=8.0)
2. **Recommended sweet spot:** ε = 3.0–5.0 balances 88-91% accuracy with meaningful privacy
3. **Precision degrades faster than recall** under high noise (models become "trigger-happy")
4. **Redaction pipeline achieves 79.8% accuracy** on real-world documents

---

## Installation

### Prerequisites

- Python 3.8+
- 32GB RAM (recommended)
- ~10GB disk space for models and data

### Setup

```bash
# Clone the repository
git clone https://github.com/Thommartial/privacy_project.git
cd privacy_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the dataset (optional - for training)
python src/data/download_dataset.py
```

### Dependencies

```
jax>=0.4.0
jaxlib>=0.4.0
flax>=0.7.0
transformers>=4.30.0
datasets>=2.14.0
optax>=0.1.7
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

---

## Usage

### Quick Start: Redact Text

```python
from src.redaction.pipeline import PIIRedactor

# Load the model (default: ε=5.0 for balanced privacy-utility)
redactor = PIIRedactor(epsilon=5.0)

# Redact PII from text
text = "Contact John Smith at john.smith@email.com or call 555-123-4567"
redacted = redactor.redact(text)
print(redacted)
# Output: "Contact [NAME] at [EMAIL] or call [PHONE]"
```

### Command Line Interface

```bash
# Redact a single file
python -m src.redaction.cli --input document.txt --output redacted.txt

# Batch process a directory
python -m src.redaction.cli --input ./documents/ --output ./redacted/ --batch

# Specify privacy level
python -m src.redaction.cli --input document.txt --epsilon 3.0
```

### Training a New Model

```bash
# Train all epsilon values
for eps in 8.0 5.0 3.0 2.0 1.0 0.5; do
    echo "Training ε=$eps..."
python src/training/train_dp_final_working.py --epsilon $eps --epochs 10 --max_samples 5000
done

# Evaluate all epsilon values
for eps in 8.0 5.0 3.0 2.0 1.0 0.5; do
    echo "Training ε=$eps..."
    python src/evaluation/evaluate_dp_model.py --epsilon $eps 
done

# Compare all the models
python src/evaluation/evaluate_dp_model.py --compare_all

# Train baseline (no differential privacy)
python src/training/train_baseline.py
```

### Evaluation

```bash
# Evaluate a trained model
python src/evaluation/evaluate_dp_model.py --model_path outputs/models/dp_eps_5.0/

# Generate visualizations
python src/evaluation/visualize_results.py --output_dir outputs/figures/
```

---

## Project Structure

```
privacy_project/
├── src/
│   ├── data/
│   │   ├── preprocess.py          # Data loading and preprocessing
│   │   └── download_dataset.py    # Dataset download script
│   ├── models/
│   │   └── distilbert_dp.py       # DP-enabled DistilBERT implementation
│   ├── training/
│   │   ├── train_dp_proper.py     # Main DP training script
│   │   ├── train_baseline.py      # Non-private baseline training
│   │   └── dp_utils.py            # DP-SGD utilities
│   ├── evaluation/
│   │   ├── evaluate_dp_model.py   # Model evaluation
│   │   └── visualize_results.py   # Result visualization
│   └── redaction/
│       ├── pipeline.py            # Main redaction pipeline
│       └── cli.py                 # Command line interface
├── outputs/
│   ├── models/                    # Trained model checkpoints
│   └── figures/                   # Generated visualizations
├── notebooks/                     # Jupyter notebooks for analysis
├── tests/                         # Unit tests
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Methodology

### Model Architecture

- **Base Model:** DistilBERT-base-uncased (66M parameters)
- **Task:** Token classification with BIO tagging scheme
- **Output:** 90 labels covering 48 PII entity types

### Differential Privacy Implementation

We implement DP-SGD with the following components:

1. **Gradient Clipping (C=1.0):** Bounds the contribution of any single training example
2. **Noise Injection (σ=0.3-1.8):** Adds calibrated Gaussian noise to gradients
3. **Privacy Accounting:** Rényi Differential Privacy (RDP) for tight composition

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 5 |
| Batch Size | 8 (effective 32 with gradient accumulation) |
| Learning Rate | 5e-5 |
| Max Sequence Length | 256 |
| Gradient Clipping Norm | 1.0 |
| Noise Multipliers | 0.3, 0.5, 0.8, 1.0, 1.4, 1.8 |

---

## Dataset

### PII-Masking-43K

- **Source:** Hugging Face Datasets (synthetic data)
- **Size:** 5,000 samples (subset for CPU feasibility)
- **Split:** 3,500 train / 750 validation / 750 test
- **Average Length:** 122.6 tokens per sample

### PII Distribution

- **28.3% PII tokens** (37,469 tokens)
- **71.7% non-PII tokens** (94,780 tokens)

### Entity Categories

| Category | Examples |
|----------|----------|
| Personal Identifiers | First name, last name, full name |
| Contact Information | Email, phone, street address |
| Financial Data | Credit card, bank account, crypto address |
| Digital Identifiers | IP address, MAC address, username |
| Government IDs | SSN, passport, driver's license |
| Demographic | Age, date of birth, gender |
| Geographic | City, state, country, ZIP code |

---

## Technical Details

### Why JAX Instead of PyTorch/Opacus?

We initially attempted to use PyTorch with the Opacus library but encountered critical compatibility issues:

| Issue | PyTorch/Opacus | JAX |
|-------|----------------|-----|
| Per-sample gradients | Complex hooks required | Native vmap support |
| Embedding with padding | Incompatible | Works seamlessly |
| Layer normalization | Requires replacement | Native support |
| Training time (per model) | Multiple days | 3-4 hours |

**Result:** 8× speedup by migrating to JAX

### Privacy Budget Calculation

The privacy budget ε is computed from:
- Noise multiplier (σ)
- Gradient clipping norm (C)
- Sampling rate (q = batch_size / dataset_size)
- Number of training steps (T)
- Failure probability (δ = 1e-5)

```
ε = f(σ, C, q, T, δ) via Rényi DP accounting
```

---

## Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Opacus-Transformers incompatibility | Migrated to JAX framework |
| Training instability (NaN gradients) | Gradient accumulation + careful LR tuning |
| CPU training time (10-50× slower than GPU) | DistilBERT + reduced dataset + JAX optimization |
| Class imbalance (85-95% "O" labels) | BIO tagging + entity-level evaluation |
| Privacy budget accounting complexity | Rényi DP with tight composition bounds |

---

## Future Work

- [ ] Scale to full 43K dataset
- [ ] Investigate ε=0.5/1.0 identical results anomaly
- [ ] Domain-specific testing (legal, medical, financial)
- [ ] Per-entity-type performance analysis
- [ ] Explore tighter Rényi DP mechanisms
- [ ] Add non-private neural baseline for comparison
- [ ] Expand redaction test set for statistical confidence

---

## References

1. Dwork, C., McSherry, F., Nissim, K., & Smith, A. (2006). Calibrating noise to sensitivity in private data analysis. *TCC*.

2. Abadi, M., et al. (2016). Deep learning with differential privacy. *CCS*.

3. Carlini, N., et al. (2021). Extracting training data from large language models. *USENIX Security*.

4. Shokri, R., et al. (2017). Membership inference attacks against machine learning models. *IEEE S&P*.

5. Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT. *NeurIPS Workshop*.

6. Yousefpour, A., et al. (2021). Opacus: User-friendly differential privacy library in PyTorch. *arXiv*.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{atoye2025dppii,
  title={Differentially Private Deep Learning for PII Detection and Redaction},
  author={Atoye, Lanre and Martial, Ekwelle Epalle Thomas},
  year={2025},
  institution={University of Guelph},
  course={CIS*6550 Privacy, Compliance, and Human Cyber}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- University of Guelph, School of Computer Science
- CIS*6550 Course Instructors
- Hugging Face for the PII-Masking-43K dataset
- JAX and Flax development teams

---

<p align="center">
  <i>Balancing Privacy and Utility in the Age of AI</i>
</p>
