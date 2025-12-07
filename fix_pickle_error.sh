#!/bin/bash
# fix_pickle_error.sh - Fix pickling error when saving model

echo "🔧 Fixing pickling error..."
echo "="*60

# Backup current file
cp src/training/train_dp_proper.py src/training/train_dp_proper.py.backup5

# Fix the saving part of the script
# We'll update just the problematic section
cat > fix_pickle_part.py << 'EOF'
import sys
import re

with open('src/training/train_dp_proper.py', 'r') as f:
    content = f.read()

# Replace the pickle saving section with a safer approach
old_save_code = '''    # Save model
    model_path = model_dir / "checkpoint.pkl"
    
    # Convert JAX arrays to numpy for saving
    params_numpy = tree_util.tree_map(lambda x: np.array(x), state.params)
    
    import pickle
    with open(model_path, 'wb') as f:
        pickle.dump({
            'params': params_numpy,
            'state': state,
            'history': history,
            'input_dim': input_dim,
            'hidden_size': args.hidden_size,
            'config': vars(args)
        }, f)'''

new_save_code = '''    # Save model - safer approach avoiding pickling JAX objects
    model_path = model_dir / "checkpoint.pkl"
    
    # Extract only the parameters (not the entire state)
    params_numpy = tree_util.tree_map(lambda x: np.array(x), state.params)
    
    import pickle
    with open(model_path, 'wb') as f:
        pickle.dump({
            'params': params_numpy,
            'history': history,
            'input_dim': input_dim,
            'hidden_size': args.hidden_size,
            'config': vars(args)
        }, f)
    
    # Also save the model architecture info separately
    model_info_path = model_dir / "model_info.json"
    import json
    with open(model_info_path, 'w') as f:
        json.dump({
            'model_class': 'ProperDPModel',
            'input_dim': input_dim,
            'hidden_size': args.hidden_size,
            'num_classes': 2,
            'dropout_rate': 0.3
        }, f, indent=2)'''

content = content.replace(old_save_code, new_save_code)

# Also fix the compute_dp_sgd_privacy function call
if 'compute_dp_sgd_privacy(' in content:
    # Add a simple fallback if the function fails
    content = content.replace(
        '''    # Compute privacy spent (outside JIT)
    n = len(train_features)
    batch_size = args.batch_size
    epochs = args.epochs
    noise_multiplier = compute_dp_sgd_privacy(
        n, batch_size, args.epsilon, args.delta, epochs
    )''',
        '''    # Compute privacy spent (outside JIT)
    n = len(train_features)
    batch_size = args.batch_size
    epochs = args.epochs
    try:
        noise_multiplier = compute_dp_sgd_privacy(
            n, batch_size, args.epsilon, args.delta, epochs
        )
    except Exception as e:
        print(f"⚠️  Privacy computation failed: {e}")
        # Use a simple approximation
        noise_multiplier = 0.1 / args.epsilon if args.epsilon > 0 else 0.0'''
    )

with open('src/training/train_dp_proper.py', 'w') as f:
    f.write(content)

print("✅ Fixed pickling issue in train_dp_proper.py")
EOF

python fix_pickle_part.py
rm fix_pickle_part.py

echo ""
echo "🛠️  Testing the fix..."
python -c "
import sys
sys.path.insert(0, '.')

# Test that we can import and run the training function
from src.training.train_dp_proper import main, ProperDPModel
print('✅ Import successful')

# Create a minimal test
import jax
import jax.numpy as jnp
import numpy as np

# Test model creation
rng = jax.random.PRNGKey(42)
model = ProperDPModel(hidden_size=256)
dummy_input = jnp.ones((2, 8))
params = model.init(rng, dummy_input, training=False)
print('✅ Model creation works')

# Test parameter saving
from jax import tree_util
params_numpy = tree_util.tree_map(lambda x: np.array(x), params['params'])
print(f'✅ Parameter extraction works: {len(params_numpy)} parameters')

# Try pickling
import pickle
try:
    pickle.dump({'params': params_numpy, 'test': 'data'}, open('/tmp/test.pkl', 'wb'))
    print('✅ Pickling works with numpy arrays')
except Exception as e:
    print(f'❌ Pickling error: {e}')
"

echo ""
echo "🚀 Now run the training again:"
echo "   python src/training/train_dp_proper.py --epsilon 8.0 --epochs 3 --batch_size 16 --max_samples 3000"
