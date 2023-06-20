from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jaxtyping import Array
import optax as ox
import gpjax as gpx

from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score


def gp_gaussian_posterior(dataset):
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


def gp_map_posterior(dataset):
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


def fit_posterior(task, x_train, y_train, key):
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


@dataclass
class WeightedPointCloud:
  """A weighted point cloud.
  
  Attributes:
    cloud: Array of shape (n, d) where n is the number of points and d the dimension.
    weights: Array of shape (n,) where n is the number of points.
  """
  cloud: Array
  weights: Array


@dataclass
class VectorizedWeightedPointCloud:
  """Vectorized version of WeightedPointCloud.
  
  Attributes:
    _private_cloud: Array of shape (n, d) where n is the number of points and d the dimension.
    _private_weights: Array of shape (n,) where n is the number of points.
  
  Methods:
    unpack: returns the cloud and weights.
  """
  _private_cloud: Array
  _private_weights: Array

  def __getitem__(self, idx):
    return WeightedPointCloud(self._private_cloud[idx], self._private_cloud[idx])
  
  def __len__(self):
    return self._private_cloud.shape[0]
  
  def __iter__(self):
    for i in range(len(self)):
      yield self[i]

  def unpack(self):
    return self._private_cloud, self._private_weights


def pad_point_clouds(cloud_list):
  """Pad point clouds with zeros to have the same size.

  Note: this function should be used outside of jax.jit because the computation graph
        is huge. O(len(cloud_list)) nodes are generated).

  Args:
    cloud_list: a list of point clouds, each of shape (n, d) where n is the number of
                points and d the dimension.
  
  Returns:
    a WeightedPointCloud with padded clouds and weights.
  """
  # sentinel for unified processing of all clouds, including biggest one.
  max_cloud_size = max([cloud.shape[0] for cloud in data]) + 1

  def pad_cloud(cloud):
    delta = max_cloud_size - cloud.shape[0]
    uniform = jnp.ones(cloud.shape[0]) / cloud.shape[0]
    zeros = jnp.zeros(delta)
    weights = jnp.concatenate([uniform, zeros], axis=0)
    cloud = jnp.pad(cloud, pad_width=((0, delta), (0,0)), mode='mean')
    return cloud, weights

  data = list(map(pad_cloud, data))
  coordinates = jnp.stack([t[0] for t in data])
  weights = jnp.stack([t[1] for t in data])
  return VectorizedWeightedPointCloud(coordinates, weights)


def clouds_barycenter(points):
  """Compute the barycenter of a set of clouds.
  
  Args:
    points: a VectorizedWeightedPointCloud.
    
  Returns:
    a barycenter of the clouds of points, of shape (1, d) where d is the dimension.
  """
  coordinates, weights = points  # unpack distribution
  barycenter = jnp.sum(coordinates * weights[:,:,jnp.newaxis], axis=1)
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
  return mu._replace(weights=mu_weights)


def reparametrize_mu(mu, cloud_barycenter, scale):
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
  return mu._replace(cloud=mu_cloud)


def clouds_to_dual_sinkhorn(points, mu, 
                            init_dual=(None, None),
                            scale=1.,
                            has_aux=False,
                            sinkhorn_solver_kwargs=None):
  """Compute the embeddings of the clouds with regularized OT towards mu.
  
  Args:
    points: a VectorizedWeightedPointCloud.
    init_dual: tuple of two arrays of shape (n, m) where n is the number of points
               and m the number of points in mu.
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
  clouds_barycenter = clouds_barycenter(points)
  mu = to_simplex(mu)

  # cloud projection
  mu = reparametrize_mu(mu, clouds_barycenter, scale)

  def sinkhorn_single_cloud(cloud, weights, init_dual):
    geom = pointcloud.PointCloud(cloud, mu.cloud,
                                 epsilon=sinkhorn_epsilon)
    ot_prob = linear_problem.LinearProblem(geom,
                                           weights,
                                           mu.weights)
    solver = sinkhorn.Sinkhorn(**sinkhorn_solver_kwargs)
    ot = solver(ot_prob, init=init_dual)
    return ot

  parallel_sinkhorn = jax.vmap(sinkhorn_single_cloud,
                               in_axes=(0, 0, 0, 0),
                               out_axes=0)
  
  outs = parallel_sinkhorn(*points.unpack(), init_dual)

  if has_aux:
    return outs.g, outs
  return outs.g


def evaluate_regression(opt_posterior, mu, train_data, cloud_test, y_test, sinkhorn_solver_kwargs):
  cloud_test = pad_point_clouds(cloud_test)
  x_test = clouds_to_dual_sinkhorn(cloud_test, mu, sinkhorn_solver_kwargs)

  latent_dist = opt_posterior.predict(x_test, train_data=train_data)
  predictive_dist = opt_posterior.likelihood(latent_dist)

  predictive_mean = predictive_dist.mean()
  predictive_std = predictive_dist.stddev()

  log_likelihood = float('inf')

  try:
    evs = explained_variance_score(y_test, predictive_mean)
    rmse = mean_squared_error(y_test, predictive_mean, squared=False)
    mae = mean_absolute_error(y_test, predictive_mean)
  except Exception as e:
    evs = float('nan')
    rmse = float('nan')
    mae = float('nan')
  
  msg = f"[GPJAX] TrainSetSize={len(train_data.n)} mae={mae:.5f} rmse={rmse:.5f} evs={evs:.5f} log-likelihood={log_likelihood:.3f}"
  print(msg)
  return evs, rmse, mae, log_likelihood


def mu_uniform(sample_train,
               key,
               mu_size,
               domain='ball',
               radius=1.):
  """Sample mu from a uniform ball of radius radius around the barycenter of the clouds.
  
  Args:
    sample_train: a VectorizedWeightedPointCloud.
    key: a jax.random.PRNGKey.
    mu_size: int, number of points in mu.
    domain: str, domain of the uniform distribution. Can be 'ball' or 'sphere'.
    radius: float, radius of the uniform distribution.
    with_weight: bool, whether to return weights or not.
  
  Returns:
    a WeightedPointCloud."""
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

