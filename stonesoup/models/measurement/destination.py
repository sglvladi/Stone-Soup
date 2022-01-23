# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import multivariate_normal as mvn

from stonesoup.types.state import ParticleState2
from ...base import Property
from ...types.array import CovarianceMatrix
from ...custom.graph import get_xy_from_range_edge, get_xy_from_sv
from ...types.numeric import Probability
from .nonlinear import NonLinearGaussianMeasurement


# def logpdf(x, mean, cov):
#     # `eigh` assumes the matrix is Hermitian.
#     vals, vecs = np.linalg.eigh(cov)
#     logdet = np.sum(np.log(vals))
#     valsinv = np.array([1. / v for v in vals])
#     # `vecs` is R times D while `vals` is a R-vector where R is the matrix
#     # rank. The asterisk performs element-wise multiplication.
#     U = vecs * np.sqrt(valsinv)
#     rank = len(vals)
#     dev = x - mean
#     # "maha" for "Mahalanobis distance".
#     maha = np.square(np.dot(dev, U)).sum()
#     log2pi = np.log(2 * np.pi)
#     return -0.5 * (rank * log2pi + maha + logdet)


class DestinationMeasurementModel(NonLinearGaussianMeasurement):

    graph = Property(dict, doc="")
    spaths = Property(dict, doc="The short paths")

    @property
    def ndim_meas(self):
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions
        """

        return 2

    def function(self, state, noise=None, **kwargs):

        if isinstance(state, ParticleState2):
            state_vectors = state.particles
        else:
            state_vectors = state.state_vector

        if noise is None:
            noise = self.rvs(state_vectors.shape[1])

        r = state_vectors[0, :]
        e = state_vectors[2, :]
        # Transform range and edge to xy
        xy = get_xy_from_range_edge(r, e, self.graph)
        # xy = get_xy_from_sv(state_vectors, self.spaths, self.graph)
        return xy + noise

    def pdf(self, state1, state2, **kwargs):
        sv = self.function(state2, noise=0, **kwargs)
        num_particles = sv.shape[1]
        likelihood = mvn.logpdf(
            sv.T,
            mean=state1.state_vector.ravel(),
            cov=self.covar(**kwargs)
        )
        # likelihood = np.zeros((num_particles,))
        # for i in range(num_particles):
        #     # Evaluate standard likelihood
        #     likelihood[i] = mvn.logpdf(
        #         sv.T,
        #         mean=state1.state_vector,
        #         cov=self.covar(**kwargs)
        #     )
        # If edge is not in the path, set likelihood to 0 (log=1)
        if isinstance(state2, ParticleState2):
            state_vectors = state2.particles
        else:
            state_vectors = state2.state_vector
        e = state_vectors[2, :]
        d = state_vectors[3, :]
        s = state_vectors[4, :]
        for i in range(num_particles):
            path = self.spaths[(s[i], d[i])]
            idx = np.where(path == e[i])[0]
            if len(idx) == 0:
                likelihood[i] = -np.inf

        return [Probability(likelihood[i], log_value=True) for i in range(num_particles)]

