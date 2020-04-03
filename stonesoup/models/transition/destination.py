import networkx as nx
import numpy as np
from scipy.linalg import block_diag

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

    def function(self, state_vector, time_interval, noise=None, **kwargs):

        if noise is None:
            noise = self.rvs(time_interval=time_interval, **kwargs)

        n_state_vector = self.matrix(time_interval)@state_vector + noise

        r = n_state_vector[0,0]
        e = n_state_vector[2,0]
        d = n_state_vector[3,0]

        # Get all valid destinations given the current edge
        v_dest = []
        for dest, path in self.spaths.items():
            # If the path contains the edge
            if len(np.where(path == e)[0]) > 0:
                v_dest.append(dest)

        range = r
        edge = e
        edge_len = self.graph.weights_by_idx(edge)[0]
        path = self.spaths[d]
        idx = np.where(path == e)[0]  # Find path index of current edge
        if len(idx) > 0 and len(path) > idx[0]+1:
            idx = idx[0]
            # If particle has NOT reached the end of the path
            while range > edge_len:
                range = range-edge_len
                edge = path[idx+1]
                edge_len = self.graph.weights_by_idx(edge)[0]
                idx = idx+1
                if len(path) == idx+1:
                    # If particle has reached the end of the path
                    if range > edge_len:
                        # Cap range to edge_length
                        range = edge_len
                    break
        elif len(idx) > 0 and len(path) == idx[0]+1:
            # If particle has reached the end of the path
            if range > edge_len:
                # Cap range to edge_length
                range = edge_len

        n_state_vector[0, 0] = range
        n_state_vector[2, 0] = edge

        if np.random.rand() > 0.98:
            n_state_vector[3, 0] = np.random.choice(v_dest)

        return n_state_vector