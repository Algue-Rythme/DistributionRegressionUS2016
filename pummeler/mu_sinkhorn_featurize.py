import jax
import jax.numpy as jnp
import numpy as onp
from .featurize import Featurizer
from .featurize import _keeps, _needs_nan
from .mu_sinkhorn import embed_single_cloud, to_simplex, WeightedPointCloud, pad_point_cloud


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
  
  num_real = len(stats['real_means'])

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
    idx = onp.random.randint(low=0, high=n_codes, size=(mu_size,))
    discrete[onp.arange(mu_size), idx] = 1

    mu_discrete.append(discrete)

  # full Mu distribution
  mu_discrete = onp.concatenate(mu_discrete, axis=1)
  mu = onp.concatenate([mu_real, mu_discrete], axis=1)  # concatenate along features.

  # uniform weights
  mu_uniform_weight = onp.ones(mu_size) / mu_size

  # move to GPU
  mu = jnp.array(mu)
  mu_uniform_weight = jnp.array(mu_uniform_weight)

  mu = WeightedPointCloud(mu, mu_uniform_weight)
  mu = to_simplex(mu)
  return mu


class MuSinkhornFeaturizer(Featurizer):
  """Featurizer for Mu Sinkhorn.

  Args:
    stats: dict, statistics of the dataset.
    mu_size: int, number of points in the Mu distribution.
    skip_feats: list of str, names of features to skip.
    sinkhorn_kwargs: dict, kwargs for the Sinkhorn algorithm.
    seed: int, random seed for Mu generation.
    pad_clouds: bool, whether to pad the point clouds to the next power of 2.
    **kwargs: kwargs for the Featurizer.
  """

  def __init__(self, stats, mu_size,
               skip_feats,
               sinkhorn_kwargs,
               seed,
               pad_clouds,
               **kwargs):
      super().__init__(stats, **kwargs)
      self.out_size = mu_size
      skip_feats = frozenset() if skip_feats is None else frozenset(skip_feats)
      self.skip_feats = skip_feats
      self.sinkhorn_kwargs = sinkhorn_kwargs
      self.pad_clouds = pad_clouds
      self.mu = generate_mu(mu_size, stats, skip_feats, seed=seed)
      if self.pad_clouds is not None:
        self.embed_single_cloud = jax.jit(embed_single_cloud)
      else:
        self.embed_single_cloud = embed_single_cloud

  def __call__(self, feats, wts, out=None):
      """Compute the Mu Sinkhorn features.

      Args:
        feats: a jnp.array of shape (n, n_feats) where n is the number of points.
        wts: a jnp.array of shape (n, 1) of weights for each point.
        out: a jnp.array of shape (1, out_size) to write the output to (optional).
      
      Returns:
        a jnp.array of shape (1, out_size).
      """
      assert out is None, "in place modification is not supported"
      weights = jnp.squeeze(wts, axis=0)
      weights = weights / jnp.sum(weights)
      cloud = WeightedPointCloud(feats, weights)
      if self.pad_clouds is not None:
        # pad the point cloud to the next power of 2
        max_cloud_size = 2 ** (cloud.cloud.shape[0] - 1).bit_length()
        # this ensures that every problem is solved with a compiled jit function,
        # while ensuring that the number of re-compilations is bounded by log2(n)
        # this limits the number of re-compilations to 15 for a 2^15 sized cloud.
        cloud = pad_point_cloud(cloud, max_cloud_size=max_cloud_size, fail_on_too_big=False)
      g = self.embed_single_cloud(cloud, self.mu,
                                  sinkhorn_solver_kwargs=self.sinkhorn_kwargs)
      g = onp.array(g)[:,onp.newaxis]  # shape (mu_size, 1) 
      return g

  def set_feat_name_ids(self, names, ids):
      self.feat_names = names
      self.feat_ids = ids
      self.keep_multilevels = _keeps(self.feat_ids)
