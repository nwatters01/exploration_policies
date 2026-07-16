import numpy as np
from scipy.linalg import cholesky

from .base import Task


def _rbf_kernel(x, y, temperature):
    """Squared-exponential kernel matrix; `temperature` is the lengthscale."""
    diff = x[:, None] - y[None, :]
    return np.exp(-0.5 * (diff / temperature) ** 2)


class GaussianProcessTask(Task):
    """Analog vector in [0, 1] driven by a Gaussian process over space and time.

    A zero-mean latent Gaussian field is built with a separable covariance:
    an RBF kernel over spatial position (lengthscale `spatial_temperature`)
    and an RBF kernel over time (lengthscale `temporal_temperature`). Each
    new timestep's latent vector is sampled conditional on the full history
    using the standard Gaussian-process regression update, then squashed
    elementwise through a logistic sigmoid to land in [0, 1].

    Note: _advance() recomputes a (t x t) solve against the full time history,
    so this is O(num_steps^3) overall -- fine for toy-scale horizons.

    Test mode: at the midpoint the spatial and temporal frequencies **double**
    (both lengthscales are halved), so the field starts wiggling faster in space
    and time.
    """

    def __init__(self, vector_length, spatial_temperature=1.0, temporal_temperature=1.0, jitter=1e-6):
        super().__init__(vector_length)
        self.spatial_temperature = spatial_temperature
        self.temporal_temperature = temporal_temperature
        self.jitter = jitter

        self._positions = np.arange(vector_length, dtype=np.float64)
        # Spatial Choleskys for the normal and the (halved-lengthscale) test regime.
        self._spatial_chol = self._build_spatial_chol(spatial_temperature)
        self._spatial_chol_test = self._build_spatial_chol(spatial_temperature / 2.0)

        self._t = 0
        self._times = None
        self._history = None

    def _build_spatial_chol(self, temperature):
        cov = _rbf_kernel(self._positions, self._positions, temperature)
        cov += self.jitter * np.eye(self.vector_length)
        return cholesky(cov, lower=True)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self._t = 0
        self._times = np.zeros((0,), dtype=np.float64)
        self._history = np.zeros((0, self.vector_length), dtype=np.float64)

    def _sample_fill(self):
        # A fresh draw from the marginal: sigmoid of a standard normal, in [0, 1].
        return float(1.0 / (1.0 + np.exp(-self._rng.standard_normal())))

    def _advance(self):
        if self._history is None:
            self.reset()

        test = self._test_active()
        temporal_temp = self.temporal_temperature / 2.0 if test else self.temporal_temperature
        spatial_chol = self._spatial_chol_test if test else self._spatial_chol

        t_new = np.array([float(self._t)])
        n_hist = self._times.shape[0]

        if n_hist == 0:
            mean = np.zeros(self.vector_length)
            var = 1.0
        else:
            k_hist = _rbf_kernel(self._times, self._times, temporal_temp)
            k_hist += self.jitter * np.eye(n_hist)
            k_row = _rbf_kernel(t_new, self._times, temporal_temp)[0]

            weights = np.linalg.solve(k_hist, k_row)
            mean = weights @ self._history
            var = max(1.0 - k_row @ weights, self.jitter)

        z = self._rng.standard_normal(self.vector_length)
        latent = mean + np.sqrt(var) * (spatial_chol @ z)

        self._times = np.concatenate([self._times, t_new])
        self._history = np.concatenate([self._history, latent[None, :]], axis=0)
        self._t += 1

        self._current = 1.0 / (1.0 + np.exp(-latent))
        return self._current
