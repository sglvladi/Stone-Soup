# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import multivariate_normal as mvn

from stonesoup.types.state import ParticleState2
from ...base import Property
from ...types.array import CovarianceMatrix
from ...custom.graph import get_xy_from_range_edge
from ...types.numeric import Probability
from .nonlinear import NonLinearGaussianMeasurement


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
        r = state_vectors[0, :]
        e = state_vectors[2, :]
        # Transform range and edge to xy
        xy = get_xy_from_range_edge(r, e, self.graph)
        return xy

    def pdf(self, state1, state2, **kwargs):
        sv = self.function(state2, noise=0, **kwargs)
        num_particles = sv.shape[1]
        likelihood = np.zeros((num_particles,))
        for i in range(num_particles):
            # Evaluate standard likelihood
            likelihood[i] = mvn.logpdf(
                state1.state_vector.T,
                mean=sv[:, i].ravel(),
                cov=self.covar(**kwargs)
            )
        # If edge is not in the path, set likelihood to 0 (log=1)
        if isinstance(state2, ParticleState2):
            state_vectors = state2.particles
        else:
            state_vectors = state2.state_vector
        e = state_vectors[2, :]
        d = state_vectors[3, :]
        for i in range(num_particles):
            path = self.spaths[d[i]]
            idx = np.where(path == e[i])[0]
            if len(idx) == 0:
                likelihood[i] = 1

        return [Probability(likelihood[i], log_value=True) for i in range(num_particles)]

