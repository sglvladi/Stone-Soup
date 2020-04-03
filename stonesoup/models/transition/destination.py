import networkx as nx
import numpy as np
from scipy.linalg import block_diag

from stonesoup.types.particle import Particle
from stonesoup.types.state import ParticleState2
from ...base import Property
from ..base import LinearModel, NonLinearModel
from .base import TransitionModel
from .linear import ConstantVelocity, LinearGaussianTransitionModel
from ...custom.graph import CustomDiGraph

class DestinationTransitionModel(LinearGaussianTransitionModel):
    noise_diff_coeff = Property(
        float, doc="The position noise diffusion coefficient :math:`q`")
    graph = Property(CustomDiGraph, doc="The graph")
    spaths = Property(dict, doc="The short paths")
    use_smc = Property(bool, default=False)

    @property
    def ndim_state(self):
        return 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cv_model = ConstantVelocity(self.noise_diff_coeff)

    def matrix(self, time_interval, **kwargs):
        transition_matrices = [self.cv_model.matrix(time_interval, **kwargs), np.eye(2)]
        return block_diag(*transition_matrices)

    def covar(self, time_interval, **kwargs):
        covar_list = [self.cv_model.covar(time_interval, **kwargs), np.zeros((2,2))]
        return block_diag(*covar_list)

    def function(self, state, time_interval, noise=None, **kwargs):

        if isinstance(state, ParticleState2):
            num_particles = len(state)
        else:
            num_particles = 1

        if noise is None:
            noise = self.rvs(num_particles, time_interval=time_interval, **kwargs)

        # CV (Position-Velocity) Propagation
        if num_particles > 1:
            n_state_vectors = self.matrix(time_interval) @ state.particles + noise
        else:
            n_state_vectors = self.matrix(time_interval)@state.state_vector + noise

        r = n_state_vectors[0, :]
        e = n_state_vectors[2, :]
        d = n_state_vectors[3, :]

        # Get all valid destinations given the current edge
        v_dest = []
        u_edges = np.unique(e)
        for edge in u_edges:
            for dest, path in self.spaths.items():
                # If the path contains the edge
                if len(np.where(path == edge)[0]) > 0:
                    v_dest.append(dest)

        for i in range(num_particles):
            r_i = r[i]
            e_i = e[i]
            d_i = d[i]
            edge_len = self.graph['Edges']['Weight'][int(e_i)]
            path = self.spaths[d_i]
            idx = np.where(path == e_i)[0]  # Find path index of current edge
            if len(idx) > 0 and len(path) > idx[0] + 1:
                idx = idx[0]
                # If particle has NOT reached the end of the path
                while r_i > edge_len:
                    r_i = r_i - edge_len
                    e_i = path[idx + 1]
                    edge_len = self.graph['Edges']['Weight'][int(e_i)]
                    idx = idx + 1
                    if len(path) == idx + 1:
                        # If particle has reached the end of the path
                        if r_i > edge_len:
                            # Cap r_i to edge_length
                            r_i = edge_len
                        break
            elif len(idx) > 0 and len(path) == idx[0] + 1:
                # If particle has reached the end of the path
                if r_i > edge_len:
                    # Cap r_i to edge_length
                    r_i = edge_len

            n_state_vectors[0, i] = r_i
            n_state_vectors[2, i] = e_i

            if np.random.rand() > 0.98:
                n_state_vectors[3, i] = np.random.choice(v_dest)

        return n_state_vectors