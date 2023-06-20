import jax.numpy as jnp
import numpy as onp
from .featurize import Featurizer
from .featurize import _keeps, _needs_nan
from .mu_sinkhorn import clouds_to_dual_sinkhorn, WeightedPointCloud


def generate_mu(mu_size, stats, skip_feats, seed):
  """Generate a Mu distribution for Sinkhorn.

  Args:
    mu_size: int, number of points in the Mu distribution.
    stats: dict, statistics of the dataset.
    skip_feats: list of str, names of features to skip.

  Returns:
    a WeightedPointCloud reference distribution Mu.
  """
  onp.random.seed(seed)
  
  num_real = len(stats['real_feats'])

  # standard normal for real features
  mu_real = onp.zeros((mu_size, num_real))

  # sparse discrete for one-hot features
  mu_discrete = []

  for k, vc in stats["value_counts"].items():
    if k in skip_feats:
        continue

    needs_nan = _needs_nan(k, stats)
    n_codes = len(vc) + int(needs_nan)

    discrete = onp.zeros((mu_size, n_codes))
    idx = onp.random.randint(shape=(mu_size,), minval=0, maxval=n_codes)
    discrete[onp.arange(mu_size), idx] = 1

    mu_discrete.append(discrete)

  # full Mu distribution
  mu_discrete = onp.concatenate(mu_discrete, axis=1)
  mu = onp.concatenate([mu_real, mu_discrete], axis=0)

  # uniform weights
  mu_uniform_weight = onp.ones(mu_size) / mu_size

  # move to GPU
  mu = jnp.array(mu)
  mu_uniform_weight = jnp.array(mu_uniform_weight)

  return WeightedPointCloud(mu, mu_uniform_weight)


class MuSinkhornFeaturizer(Featurizer):
  """Featurizer for Mu Sinkhorn.

  Args:
    stats: dict, statistics of the dataset.
    mu_size: int, number of points in the Mu distribution.
    skip_feats: list of str, names of features to skip.
    sinkhorn_kwargs: dict, kwargs for the Sinkhorn algorithm.
    **kwargs: kwargs for the Featurizer.
  """

  def __init__(self, stats, mu_size, skip_feats, sinkhorn_kwargs, seed, **kwargs):
      super().__init__(stats, **kwargs)
      self.out_size = self.n_feats
      self.skip_feats = skip_feats
      self.sinkhorn_kwargs = sinkhorn_kwargs
      self.mu = generate_mu(mu_size, stats, skip_feats, seed=seed)


  def __call__(self, feats, wts, out=None):
      """Compute the Mu Sinkhorn features.

      Args:
        feats: a jnp.array of shape (n, n_feats) where n is the number of points.
        wts: a jnp.array of shape (n, 1) of weights for each point.
        out: a jnp.array of shape (n, out_size) to write the output to (optional).
      
      Returns:
        a jnp.array of shape (n, out_size).
      """
      return clouds_to_dual_sinkhorn(feats, self.mu, **self.sinkhorn_kwargs)

  def set_feat_name_ids(self, names, ids):
      self.feat_names = names
      self.feat_ids = ids
      self.keep_multilevels = _keeps(self.feat_ids)
