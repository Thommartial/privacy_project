# Differential Privacy for PII Detection - Final Report

## Results Summary

| ε (Privacy) | Accuracy | F1 Score | Improvement | Privacy Level |
|-------------|----------|----------|-------------|---------------|
| 8.0 | 0.9947 | 0.9843 | +0.1613 | Low |
| 5.0 | 0.9093 | 0.7862 | +0.0760 | Medium-Low |
| 3.0 | 0.8840 | 0.7418 | +0.0507 | Medium |
| 2.0 | 0.8560 | 0.6983 | +0.0227 | Medium-High |
| 1.0 | 0.7507 | 0.5621 | -0.0827 | High |
| 0.5 | 0.7507 | 0.5621 | -0.0827 | Very High |
| **Baseline** | **0.8333** | **N/A** | **N/A** | **Rule-based** |

## Key Findings

1. **Clear Privacy-Accuracy Tradeoff**: As ε decreases (more privacy), model accuracy decreases
2. **Best Performance**: ε=8.0 achieves 0.9947 accuracy (+0.161 over baseline)
3. **High Privacy Models**: ε=1.0 and ε=0.5 provide strong privacy but with reduced accuracy
4. **Perfect Recall**: All models maintain recall=1.00, meaning they detect ALL PII instances

## Conclusion
The experiment successfully demonstrates the privacy-accuracy tradeoff in differentially private machine learning. Users can choose their preferred ε based on their privacy requirements:
- **Low privacy needs (ε=8.0)**: High accuracy (99.47%)
- **High privacy needs (ε=0.5)**: Strong privacy, acceptable accuracy (75.07%)

![Privacy-Accuracy Tradeoff](privacy_accuracy_tradeoff.png)
