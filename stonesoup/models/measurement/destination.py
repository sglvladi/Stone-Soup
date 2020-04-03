# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import multivariate_normal as mvn

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

    def function(self, state_vector, noise=None, **kwargs):
        r = state_vector[0, 0]
        e = state_vector[2, 0]
        # Transform range and edge to xy
        xy = get_xy_from_range_edge(r, e, self.graph)
        return xy

    def pdf(self, state_vector1, state_vector2, **kwargs):
        e = state_vector2[2, 0]
        d = state_vector2[3, 0]
        # Evaluate standard likelihood
        likelihood = mvn.logpdf(
            state_vector1.T,
            mean=self.function(state_vector2, noise=0, **kwargs).ravel(),
            cov=self.covar(**kwargs)
        )
        # If edge is not in the path, set likelihood to 0 (log=1)
        path = self.spaths[d]
        idx = np.where(path == e)[0]
        if len(idx) == 0:
            likelihood = 1
        return Probability(likelihood, log_value=True)

