import jax
import jax.numpy as jnp

# Test that jax.tree.map works
x = {'a': jnp.array([1, 2, 3]), 'b': jnp.array([4, 5, 6])}
y = jax.tree.map(lambda v: v * 2, x)
print(f'✅ jax.tree.map works: {y}')
