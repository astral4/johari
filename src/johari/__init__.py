"""Personal character ranking via adaptive pairwise choice.

First, a bucket stage (one ordinal probit per item) warm-starts the model.
Then, a pairwise comparison stage (Rao-Kupper threshold pairs) refines it.
The posterior is a Laplace approximation around a damped-Newton MAP.
"""

import jax

jax.config.update("jax_enable_x64", val=True)
