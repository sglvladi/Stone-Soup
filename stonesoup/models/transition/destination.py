import networkx as nx
import numpy as np
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal as mn

from stonesoup.types.particle import Particle
from stonesoup.types.state import ParticleState2
from ...base import Property
from ..base import LinearModel, NonLinearModel
from .base import TransitionModel
from .linear import ConstantVelocity, LinearGaussianTransitionModel
from ...custom.graph import CustomDiGraph, normalise_re, normalise_re2


class DestinationTransitionModel(LinearGaussianTransitionModel):
    noise_diff_coeff = Property(
        float, doc="The position noise diffusion coefficient :math:`q`")
    graph = Property(CustomDiGraph, doc="The graph")
    use_smc = Property(bool, default=False)

    @property
    def ndim_state(self):
        return 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cv_model = ConstantVelocity(self.noise_diff_coeff)

    def matrix(self, time_interval, **kwargs):
        transition_matrices = [self.cv_model.matrix(time_interval, **kwargs), np.eye(3)]
        return block_diag(*transition_matrices)

    def covar(self, time_interval, **kwargs):
        covar_list = [self.cv_model.covar(time_interval, **kwargs), np.zeros((3,3))]
        return block_diag(*covar_list)

    def function(self, state, time_interval, noise=None, **kwargs):

        if isinstance(state, ParticleState2):
            num_particles = len(state)
        else:
            num_particles = 1

        if noise is None:
            noise = self.rvs(num_particles, time_interval=time_interval, **kwargs)

        # 1) CV (Position-Velocity) Propagation
        if num_particles > 1:
            n_state_vectors = self.matrix(time_interval) @ state.particles + noise
        else:
            n_state_vectors = self.matrix(time_interval)@state.state_vector + noise

        # 2) SMC destination sampling
        # Get all valid destinations given the current edge
        v_dest = dict()
        e = n_state_vectors[2, :]
        u_edges = np.unique(e)
        for edge in u_edges:
            for dest, path in self.graph.short_paths_e.items():
                # If the path contains the edge
                if len(np.where(path == edge)[0]) > 0:
                    if edge in v_dest:
                        v_dest[edge].append(dest[1])
                    else:
                        v_dest[edge] = [dest[1]]
        # Perform the sampling
        for i in range(num_particles):
            if np.random.rand() > 0.9:
                edge = e[i]
                n_state_vectors[3, i] = np.random.choice(v_dest[edge])

        # 3) Process edge change
        # The CV propagation may lead to r's which are either less that zero or more than the
        # length of the edge. This means that the range and edge identifier needs to be adjusted
        # to correctly place the particle.

        # Get shortcuts for faster accessing
        r = n_state_vectors[0, :]
        e = n_state_vectors[2, :]
        d = n_state_vectors[3, :]
        s = n_state_vectors[4, :]

        for i in range(num_particles):
            r_i, e_i, d_i, s_i = normalise_re(r[i], e[i], d[i], s[i], self.graph)
            n_state_vectors[0, i] = r_i
            n_state_vectors[2, i] = e_i

        return n_state_vectors