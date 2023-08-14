import numpy as np
from .featurize import Featurizer
from .featurize import _keeps, _needs_nan
from .mu_sinkhorn import embed_single_cloud, to_simplex, WeightedPointCloud, pad_point_cloud


def random_projections(n_projections, dim, seed=None):
  """Generate random projection vectors.
  
  Args:
    n_projections: integer, number of projection vectors
    dim: integer, dimension of the ambient space
    seed: integer, random seed
  
  Returns:
    (n_projections, dim) array of projection vectors
  """
  np.random.seed(seed)
  theta = np.random.randn(n_projections, dim)
  theta = theta / np.linalg.norm(theta, axis=1, keepdims=True)
  return theta


def deterministic_projections(n_projections, dim, seed=None):
  """Generate random projection vectors.
  
  Args:
    n_projections: integer, number of projection vectors
    dim: integer, dimension of the ambient space
    seed: integer, random seed
  
  Returns:
    (n_projections, dim) array of projection vectors
  """
  assert dim == 2, "Only implemented for dim=2"
  angles = np.linspace(-np.pi, np.pi, n_projections)
  theta = np.array([np.cos(angles), np.sin(angles)]).T
  return theta, angles


def sliced_wasserstein_embeddings(point_cloud, projections, t):
  """Sliced Wasserstein features from projections.
  
  Args:
    point_cloud: pair of (cloud of shape (n, dim), weights of shape (n,))
    projections: (n_projections, dim) array of projection vectors
    t: integer, number of points used in integration of CDF.

  Returns:
    (n_projections, T) array of Sliced Wasserstein features
  """
  # shape (n, k), (n, 1)
  point, weights = point_cloud
  points = point @ projections.T  # shape (n, n_projections)
  cum_t = np.linspace(0, 1, t)  # shape (T,)
  weights = weights / np.sum(weights, axis=0, keepdims=True)  # re-normalize weights
  inv_cdf = []
  for proj in range(projections.shape[0]):  # for each projection
    points_slice  = np.array(points[:, proj])  # shape (n,)
    points_idx    = np.argsort(points_slice, axis=0, kind='mergesort')
    points_slice  = np.take_along_axis(points_slice, points_idx, axis=0)  # sort points
    weights_slice = np.take_along_axis(weights, points_idx, axis=0)  # sort weights
    cum_cdf_slice = np.cumsum(weights_slice)  # shape (n,)
    inv_cdf_slice = np.interp(cum_t, cum_cdf_slice, points_slice)  # shape (T,)
    inv_cdf.append(inv_cdf_slice)
  inv_cdf = np.array(inv_cdf)  # shape (n_projections, T)
  return inv_cdf


def projections_from_feats(stats, skip_feats, n_projections, seed=None):
  """Generate random projection vectors from statistics.

  Args:
    stats: dict, statistics of the dataset.
    skip_feats: list of str, names of features to skip.
    n_projections: int, number of points in the n_projections.
    seed: int, random seed for Mu generation.

  Returns:
    (n_projections, dim) array of projection vectors
  """
  real_feats = len(stats['real_means'])
  dim = real_feats
  
  for k, vc in stats["value_counts"].items():
    if k in skip_feats:
        continue

    # length induced by discrete encoding with one-hot and nan
    needs_nan = _needs_nan(k, stats)
    n_codes = len(vc) + int(needs_nan)  
    dim += n_codes

  return random_projections(n_projections, dim, seed=seed)


class SlicedWassersteinFeaturizer(Featurizer):
  """Featurizer for Slice Wasserstein.

  Args:
    stats: dict, statistics of the dataset.
    n_projections: int, number of points in the n_projections.
    n_discrete: int, number of discrete points to use in the CDF.
    skip_feats: list of str, names of features to skip.
    seed: int, random seed for Mu generation.
    **kwargs: kwargs for the Featurizer.
  """

  def __init__(self, stats,
               n_projections,
               n_discrete,
               skip_feats,
               seed,
               **kwargs):
      super().__init__(stats, **kwargs)
      self.n_projections = n_projections
      self.n_discrete = n_discrete
      self.out_size = n_projections * n_discrete
      skip_feats = frozenset() if skip_feats is None else frozenset(skip_feats)
      self.skip_feats = skip_feats
      self.projections = projections_from_feats(stats, skip_feats, n_projections, seed=seed)

  def __call__(self, feats, wts, out=None):
      """Compute the Mu Sinkhorn features.

      Args:
        feats: a jnp.array of shape (n, n_feats) where n is the number of points.
        wts: a jnp.array of shape (1, n) of weights for each point.
        out: a jnp.array of shape (1, out_size) to write the output to (optional).
      
      Returns:
        a jnp.array of shape (n_projections * T, 1).
      """
      assert out is None, "in place modification is not supported"
      point_cloud = feats, wts.flatten()
      embeddings = sliced_wasserstein_embeddings(point_cloud, self.projections, self.n_discrete)
      embeddings = embeddings.flatten()[:,np.newaxis]  # shape (n_projections * T, 1)
      # |phi(P) - phi(Q)|^2 = mean((x(P) - x(Q))^2)
      # factor (1/dim)**0.5 is necessary.
      embeddings = embeddings / np.sqrt(self.n_projections * self.n_discrete)
      return embeddings

  def set_feat_name_ids(self, names, ids):
      self.feat_names = names
      self.feat_ids = ids
      self.keep_multilevels = _keeps(self.feat_ids)
