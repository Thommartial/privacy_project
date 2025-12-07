#!/usr/bin/env python3
"""Test that all imports work correctly."""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("Testing imports from project root...")
print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}")

try:
    from src.models.distilbert_dp import DistilBertDP
    print("✅ Successfully imported DistilBertDP")
    
    from src.training.train_dp import parse_args
    print("✅ Successfully imported train_dp functions")
    
    import jax
    print(f"✅ JAX version: {jax.__version__}")
    
    import flax
    print(f"✅ Flax version: {flax.__version__}")
    
    print("\n✅ All imports working correctly!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
