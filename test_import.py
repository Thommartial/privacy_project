import sys
sys.path.insert(0, '.')
try:
    from src.models.minimal_dp_model import MinimalPIIModel
    print('✅ Minimal model imports OK')
    
    import jax
    import jax.numpy as jnp
    
    # Test initialization
    model = MinimalPIIModel()
    rng = jax.random.PRNGKey(0)
    dummy = jnp.ones((2, 32), dtype=jnp.int32)
    
    params = model.init(rng, dummy, dummy, training=False)
    print(f'✅ Model initialized: {sum(x.size for x in jax.tree_util.tree_leaves(params)):,} params')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
