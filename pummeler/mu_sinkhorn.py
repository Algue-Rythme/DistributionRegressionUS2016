from dataclasses import dataclass
from dataclasses import replace

from functools import partial

from typing import Optional

import jax
import jax.numpy as jnp
from flax import struct
import optax as ox
import gpjax as gpx

from ott.geometry.pointcloud import PointCloud
from ott.problems.linear.linear_problem import LinearProblem
from ott.solvers.linear.sinkhorn import Sinkhorn

from sklearn.metrics import root_mean_squared_error, mean_absolute_error, explained_variance_score


def gp_gaussian_posterior(dataset: gpx.Dataset):
  """Returns a GP posterior with a Gaussian likelihood.
  
  Args:
    dataset: a gpx.Dataset.
  """
  kernel = gpx.kernels.RBF()
  meanf = gpx.mean_functions.Zero()
  prior = gpx.Prior(mean_function=meanf, kernel=kernel)
  likelihood = gpx.Gaussian(num_datapoints=dataset.n)
  posterior = prior * likelihood
  negative_mll = gpx.objectives.ConjugateMLL(negative=True)
  return posterior, negative_mll


def gp_map_posterior(dataset: gpx.Dataset):
  """Returns a GP posterior with a Bernoulli likelihood.
  
  Args:
    dataset: a gpx.Dataset.
  """
  kernel = gpx.RBF()
  meanf = gpx.Constant()
  prior = gpx.Prior(mean_function=meanf, kernel=kernel)
  likelihood = gpx.Bernoulli(num_datapoints=dataset.n)
  posterior = prior * likelihood
  negative_lpd = gpx.LogPosteriorDensity(negative=True)
  return posterior, negative_lpd


def fit_posterior(task: str, x_train: jax.Array, y_train: jax.Array, key: jax.Array):
  """Fits a GP posterior.

  Args:
    task: 'classification' or 'regression'.
    sinkhorn_dual: Array of shape (n, m) where n is the number of points in the
                    dataset and m the number of points in mu.
    labels: Array of shape (n,) where n is the number of points in the dataset.
    key: jax.random.PRNGKey.
  
  Returns:
    A GP posterior.
  """
  dataset = gpx.Dataset(X=x_train, y=y_train)

  if task == 'classification':
    posterior, objective = gp_map_posterior(dataset)
  elif task == 'regression':
    posterior, objective = gp_gaussian_posterior(dataset)

  objective = jax.jit(objective)
  optimiser = ox.adamw(learning_rate=0.01)
  opt_posterior, history = gpx.fit(
      model=posterior,
      objective=objective,
      train_data=dataset,
      optim=optimiser,
      num_iters=1000,
      key=key,
  )

  return opt_posterior, history


@struct.dataclass
class WeightedPointCloud:
  """A weighted point cloud.
  
  Attributes:
    cloud: Array of shape (n, d) where n is the number of points and d the dimension.
    weights: Array of shape (n,) where n is the number of points.
  """
  cloud: jnp.array
  weights: jnp.array

  def __len__(self):
    return self.cloud.shape[0]


@struct.dataclass
class VectorizedWeightedPointCloud:
  """Vectorized version of WeightedPointCloud.

  Assume that b clouds are all of size n and dimension d.
  
  Attributes:
    _private_cloud: Array of shape (b, n, d) where n is the number of points and d the dimension.
    _private_weights: Array of shape (b, n) where n is the number of points.
  
  Methods:
    unpack: returns the cloud and weights.
  """
  _private_cloud: jnp.array
  _private_weights: jnp.array

  def __getitem__(self, idx: int):
    return WeightedPointCloud(self._private_cloud[idx], self._private_cloud[idx])
  
  def __len__(self):
    return self._private_cloud.shape[0]
  
  def __iter__(self):
    for i in range(len(self)):
      yield self[i]

  def unpack(self):
    return self._private_cloud, self._private_weights


def pad_point_cloud(point_cloud: WeightedPointCloud, max_cloud_size: int, fail_on_too_big: bool = True):
  """Pad a single point cloud with zeros to have the same size.
  
  Args:
    point_cloud: a weighted point cloud.
    max_cloud_size: the size of the biggest point cloud.
    fail_on_too_big: if True, raise an error if the cloud is too big for padding.
  
  Returns:
    a WeightedPointCloud with padded cloud and weights.
  """
  cloud, weights = point_cloud.cloud, point_cloud.weights
  delta = max_cloud_size - cloud.shape[0]
  if delta <= 0:
    if fail_on_too_big:
      assert False, 'Cloud is too big for padding.'
    return point_cloud

  ratio = 1e-3  # less than 0.1% of the total mass.
  smallest_weight = jnp.min(weights) / delta * ratio
  small_weights = jnp.ones(delta) * smallest_weight

  weights = weights * (1 - ratio)  # keep 99.9% of the mass.
  weights = jnp.concatenate([weights, small_weights], axis=0)

  cloud = jnp.pad(cloud, pad_width=((0, delta), (0,0)), mode='mean')

  point_cloud = WeightedPointCloud(cloud, weights)

  return point_cloud


def pad_point_clouds(cloud_list: list[WeightedPointCloud]):
  """Pad the point clouds with zeros to have the same size.

  Note: this function should be used outside of jax.jit because the computation graph
        is huge. O(len(cloud_list)) nodes are generated.

  Args:
    cloud_list: a list of WeightedPointCloud.
  
  Returns:
    a VectrorizedWeightedPointCloud with padded clouds and weights.
  """
  # sentinel for unified processing of all clouds, including biggest one.
  max_cloud_size = max([len(cloud) for cloud in cloud_list]) + 1
  sentinel_padder = partial(pad_point_cloud, max_cloud_size=max_cloud_size)

  cloud_list = list(map(sentinel_padder, cloud_list))
  coordinates = jnp.stack([cloud.cloud for cloud in cloud_list])
  weights = jnp.stack([cloud.weights for cloud in cloud_list])
  return VectorizedWeightedPointCloud(coordinates, weights)


def clouds_barycenter(points: VectorizedWeightedPointCloud):
  """Compute the barycenter of a set of clouds.
  
  Args:
    points: a VectorizedWeightedPointCloud.
    
  Returns:
    a barycenter of the clouds of points, of shape (1, d) where d is the dimension.
  """
  clouds, weights = points.unpack()
  barycenter = jnp.sum(clouds * weights[:,:,jnp.newaxis], axis=1)
  barycenter = jnp.mean(barycenter, axis=0, keepdims=True)
  return barycenter


def to_simplex(mu):
  """Project weights to the simplex.
  
  Args: 
    mu: a WeightedPointCloud.
    
  Returns:
    a WeightedPointCloud with weights projected to the simplex."""
  if mu.weights is None:
    mu_weights = None
  else:
    mu_weights = jax.nn.softmax(mu.weights)
  return replace(mu, weights=mu_weights)


def reparametrize_mu(mu: jax.Array, cloud_barycenter: jax.Array, scale: float):
  """Re-parametrize mu to be invariant by translation and scaling.

  Args:
    mu: a WeightedPointCloud.
    cloud_barycenter: Array of shape (1, d) where d is the dimension.
    scale: float, scaling parameter for the re-parametrization of mu.
  
  Returns:
    a WeightedPointCloud with re-parametrized weights and cloud.
  """
  # invariance by translation : recenter mu around its mean
  mu_cloud = mu.cloud - jnp.mean(mu.cloud, axis=0, keepdims=True)  # center.
  mu_cloud = scale * jnp.tanh(mu_cloud)  # re-parametrization of the domain.
  mu_cloud = mu_cloud + cloud_barycenter  # re-center toward barycenter of all clouds.
  return replace(mu, cloud=mu_cloud)


def embed_single_cloud(weighted_cloud: WeightedPointCloud, mu: jax.Array,
                       sinkhorn_solver_kwargs: dict,
                       has_aux: bool = False):
  """Compute the embedding of a single cloud with regularized OT towards mu.

  Args:
    weighted_cloud: a WeightedPointCloud.
    mu: a WeightedPointCloud.
    has_aux: bool, whether to return the whole output vector.
    sinkhorn_solver_kwargs: kwargs for the Sinkhorn solver.

  Returns:
    a vector of shape (n,) where n is the number of points in Mu.
  """
  sinkhorn_solver_kwargs = dict(**sinkhorn_solver_kwargs)  # copy to avoid modifying the function argument.
  sinkhorn_epsilon = sinkhorn_solver_kwargs.pop('epsilon')

  geom = PointCloud(weighted_cloud.cloud, mu.cloud,
                    epsilon=sinkhorn_epsilon)
  
  ot_prob = LinearProblem(geom,
                          weighted_cloud.weights,
                          mu.weights)
  
  solver = Sinkhorn(**sinkhorn_solver_kwargs)

  outs = solver(ot_prob)

  if has_aux:
    return outs.g, outs
  return outs.g


def clouds_to_dual_sinkhorn(points: VectorizedWeightedPointCloud,
                            mu: jax.Array, 
                            init_dual: tuple = (None, None),
                            scale: float = 1.,
                            has_aux: bool = False,
                            sinkhorn_solver_kwargs = Optional[None]):
  """Compute the embeddings of the clouds with regularized OT towards mu.
  
  Args:
    points: a VectorizedWeightedPointCloud.
    init_dual: tuple of two arrays of shape (b, n) and (b, m) where b is the number of clouds,
               n is the number of points in each cloud, and m the number of points in mu.
    scale: float, scaling parameter for the re-parametrization of mu.
    has_aux: bool, whether to return the full Sinkhorn output or only the dual variables.
    sinkhorn_solver_kwargs: dict, kwargs for the Sinkhorn solver.
      Must contain the key 'epsilon' for the regularization parameter.

  Returns:
    a tuple (dual, init_dual) with dual variables of shape (n, m) where n is the number of points
    and m the number of points in mu, and init_dual a tuple (init_dual_cloud, init_dual_mu) 
  """
  sinkhorn_epsilon = sinkhorn_solver_kwargs.pop('epsilon')
  
  # weight projection
  barycenter = clouds_barycenter(points)
  mu = to_simplex(mu)

  # cloud projection
  mu = reparametrize_mu(mu, barycenter, scale)

  def sinkhorn_single_cloud(cloud, weights, init_dual):
    geom = PointCloud(cloud, mu.cloud,
                      epsilon=sinkhorn_epsilon)
    ot_prob = LinearProblem(geom,
                            weights,
                            mu.weights)
    solver = Sinkhorn(**sinkhorn_solver_kwargs)
    ot = solver(ot_prob, init=init_dual)
    return ot

  parallel_sinkhorn = jax.vmap(sinkhorn_single_cloud,
                               in_axes=(0, 0, (0, 0)),
                               out_axes=0)
  
  outs = parallel_sinkhorn(*points.unpack(), init_dual)

  if has_aux:
    return outs.g, outs
  return outs.g


def evaluate_regression(opt_posterior,
                        mu: jax.Array,
                        train_data: list[jax.Array],
                        cloud_test: list[jax.Array],
                        y_test: jax.Array,
                        sinkhorn_solver_kwargs: dict) -> tuple[float, float, float, float]:
  """Evaluate the quality of the GP regressor.
  
  Args:
    opt_posterior: optimal GP posterior
    mu: reference measure of shape (n, d_mu)
    train_data: list of arrays of shape (n, di)
    cloud_test: list of arrays of shape (n, di)
    y_test: array of shape (n,)
    sinkhorn_solver_kwargs: dict of kwargs for Sinkhorn
  
  Return:
    a tuple of 4 metrics: the explained variance score, the root mean square error,
    mean abolsute error, and the log likelihood.
  """
  cloud_test = pad_point_clouds(cloud_test)
  x_test = clouds_to_dual_sinkhorn(cloud_test, mu, sinkhorn_solver_kwargs)

  latent_dist = opt_posterior.predict(x_test, train_data=train_data)
  predictive_dist = opt_posterior.likelihood(latent_dist)

  predictive_mean = predictive_dist.mean()
  predictive_std = predictive_dist.stddev()

  log_likelihood = float('inf')

  try:
    evs = explained_variance_score(y_test, predictive_mean)
    rmse = root_mean_squared_error(y_test, predictive_mean)
    mae = mean_absolute_error(y_test, predictive_mean)
  except Exception as e:
    evs = float('nan')
    rmse = float('nan')
    mae = float('nan')
  
  msg = f"[GPJAX] TrainSetSize={len(train_data.n)} mae={mae:.5f} rmse={rmse:.5f} evs={evs:.5f} log-likelihood={log_likelihood:.3f}"
  print(msg)
  return evs, rmse, mae, log_likelihood


def mu_uniform(sample_train: VectorizedWeightedPointCloud,
               key: jax.Array,
               mu_size: int,
               domain: str = 'ball',
               radius: float = 1.):
  """Sample mu from a uniform ball of radius radius around the barycenter of the clouds.
  
  Args:
    sample_train: a VectorizedWeightedPointCloud.
    key: a jax.random.PRNGKey.
    mu_size: int, number of points in mu.
    domain: str, domain of the uniform distribution. Can be 'ball' or 'sphere'.
    radius: float, radius of the uniform distribution.
    with_weight: bool, whether to return weights or not.
  
  Returns:
    a WeightedPointCloud.
  """
  dim = sample_train[0].shape[-1]
  key_theta, key_r = jax.random.split(key)
  mu_cloud = jax.random.normal(key_theta, shape=(mu_size, dim))
  norms = jnp.sqrt(jnp.sum(mu_cloud**2, axis=1, keepdims=True))
  mu_cloud = mu_cloud / norms
  if domain == 'ball':
    radii = jax.random.uniform(key_r, shape=(mu_size, 1))
  else:
    radii = jnp.ones((mu_size, 1))
  mu_cloud = mu_cloud * radius * radii
  centroids = jnp.mean(sample_train[0], axis=1)
  centroids_center = jnp.mean(centroids, axis=0, keepdims=True)
  mu_cloud = mu_cloud + centroids_center  # OT is invariant by translation
  mu_weight = None
  mu_weight = jnp.zeros(len(mu_cloud))
  return WeightedPointCloud(mu_cloud, mu_weight)

