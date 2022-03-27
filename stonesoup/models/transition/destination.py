import networkx as nx
import numpy as np
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal as mn

from stonesoup.types.state import ParticleState2
from ..base import TimeVariantModel
from ...base import Property
from .linear import ConstantVelocity, LinearGaussianTransitionModel
from ...custom.graph import CustomDiGraph, normalise_re, normalise_re2
from pybsp.bsp import BSP
# from pybsp.geometry import Point
from bsppy import Point, BSPTree


class DestinationTransitionModel(LinearGaussianTransitionModel, TimeVariantModel):
    noise_diff_coeff: float = Property(doc="The position noise diffusion coefficient :math:`q`")
    graph: CustomDiGraph = Property(doc="The graph")
    use_smc: bool = Property(default=False)

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


class AimpointTransitionModel(LinearGaussianTransitionModel):
    noise_diff_coeff: float = Property(doc="The position noise diffusion coefficient :math:`q`")
    graph: CustomDiGraph = Property(doc="The graph")
    bsptree: BSP = Property(doc="The bsp tree")
    use_smc: bool = Property(default=False)

    @property
    def ndim_state(self):
        return 9

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cv_model = ConstantVelocity(self.noise_diff_coeff)

    def matrix(self, time_interval, **kwargs):
        transition_matrices = [self.cv_model.matrix(time_interval, **kwargs), np.eye(7)]
        return block_diag(*transition_matrices)

    def covar(self, time_interval, **kwargs):
        covar_list = [self.cv_model.covar(time_interval, **kwargs), np.zeros((7, 7))]
        return block_diag(*covar_list)

    def function(self, state, time_interval=None, noise=None, **kwargs):

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
            if np.random.rand() > 0.98:
                edge = e[i]
                n_state_vectors[3, i] = np.random.choice(v_dest[edge])

        # 3) Perform aimpoint sampling
        aimpoint_locs = n_state_vectors[[5, 6], :]

        resample_inds = np.flatnonzero(np.random.binomial(1, 0.1, num_particles))
        aimpoint_locs_res = aimpoint_locs[:, resample_inds]
        num_samples = len(resample_inds)
        samples = mn.rvs(np.zeros((2,)), np.diag([1e4, 1e4]), size=num_samples).T
        new_locs = aimpoint_locs_res + samples
        while True:
            ps = [Point(*new_locs[:, i]) for i in range(num_samples)]
            valid = self.bsptree.are_empty_leaves(ps) # [self.bsptree.is_empty_leaf(p) for p in ps]
            if np.alltrue(valid):
                try:
                    aimpoint_locs[:, resample_inds] = new_locs
                except:
                    a = 2
                break
            else:
                valid_inds = np.flatnonzero(valid)
                aimpoint_locs[:, resample_inds[valid_inds]] = new_locs[:, valid_inds]
                resample_inds = resample_inds[np.logical_not(valid)]
                num_samples = len(resample_inds)
                aimpoint_locs_res = aimpoint_locs[:, resample_inds]
                samples = np.atleast_2d(mn.rvs(np.zeros((2,)), np.diag([1e4, 1e4]), size=num_samples)).T
                new_locs = aimpoint_locs_res + samples
        #
        # n_state_vectors[[5, 6], :] = aimpoint_locs
        # for i in range(num_particles):
        #     if np.random.rand() > 0.9:
        #         aimpoint_loc = aimpoint_locs[:, i]
        #         new_loc = mn.rvs(aimpoint_loc.ravel(), np.diag([1e4, 1e4]))
        #         p = Point(new_loc[0], new_loc[1])
        #         while self.bsptree.find_leaf(p).is_solid:
        #             new_loc = mn.rvs(aimpoint_loc.ravel(), np.diag([1e4, 1e4]))
        #             p = Point(new_loc[0], new_loc[1])
        #         n_state_vectors[[5, 6], i] = np.array([p.x, p.y])

        # 4) Process edge change
        # The CV propagation may lead to r's which are either less that zero or more than the
        # length of the edge. This means that the range and edge identifier needs to be adjusted
        # to correctly place the particle.

        # Get shortcuts for faster accessing
        r = n_state_vectors[0, :]
        e = n_state_vectors[2, :]
        d = n_state_vectors[3, :]
        s = n_state_vectors[4, :]
        a = n_state_vectors[[5, 6], :]
        am1 = n_state_vectors[[7, 8], :]

        for i in range(num_particles):
            r_i, e_i, d_i, s_i, a_i, am1_i = normalise_re2(r[i], e[i], d[i], s[i], a[:, i], am1[:, i], self.graph)
            n_state_vectors[0, i] = r_i
            n_state_vectors[2, i] = e_i
            n_state_vectors[[5, 6], i] = a_i
            n_state_vectors[[7, 8], i] = am1_i

        return n_state_vectors