import networkx as nx
import numpy as np
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal as mn

from stonesoup.types.state import ParticleState2
from ..base import TimeVariantModel
from ...base import Property
from .linear import ConstantVelocity, LinearGaussianTransitionModel, OrnsteinUhlenbeck
from ...custom.graph import CustomDiGraph, normalise_re, normalise_re2, get_xy_from_range_endnodes
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
    check_los: bool = Property(default=False)
    prior_on_endnodes: bool = Property(default=True)
    aimpoint_sample_covar: np.ndarray = Property(default=None)

    @property
    def ndim_state(self):
        return 9

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.cv_model = ConstantVelocity(self.noise_diff_coeff)
        self.cv_model = OrnsteinUhlenbeck(self.noise_diff_coeff, 7e-4)
        if self.aimpoint_sample_covar is None:
            self.aimpoint_sample_covar = np.diag([1e9, 1e9])

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

        # SMC edge resampling
        # data = state.particles
        # a = data[[5, 6], :]
        # am1 = data[[7, 8], :]
        # xy = get_xy_from_range_endnodes(data[0, :], am1, a)
        # mu = np.mean(np.mean(xy, axis=1)).ravel()
        #
        # gdf = self.graph.gdf  # construct a GeoDataFrame object
        # rtree = self.graph.rtree  # generate a spatial index (R-tree)
        # center = state.state_vector.ravel()
        # radius = 3 * np.sqrt(np.max(state.covar()[0, 0], state.covar)
        # query_point = ShapelyPoint(center).buffer(20 * radius)
        # # Get rough result of edges that intersect query point envelope
        # result_rough = rtree.query(query_point)
        # # The exact edges are those in the rough result that intersect the query point
        # gdf_rough = gdf.iloc[result_rough]
        # return result_rough[np.flatnonzero(gdf_rough.intersects(query_point))]

        # 2) SMC destination sampling
        # Get all valid destinations given the current edge
        # v_dest = dict()
        # e = n_state_vectors[2, :]
        # u_edges = np.unique(e)
        # for edge in u_edges:
        #     for dest, path in self.graph.short_paths_e.items():
        #         # If the path contains the edge
        #         if len(np.where(path == edge)[0]) > 0:
        #             if edge in v_dest:
        #                 v_dest[edge].append(dest[1])
        #             else:
        #                 v_dest[edge] = [dest[1]]
        # # Perform the sampling
        # for i in range(num_particles):
        #     if np.random.rand() > 0.98:
        #         edge = e[i]
        #         n_state_vectors[3, i] = np.random.choice(v_dest[edge])

        # 3) Perform aimpoint sampling
        resample_inds = np.flatnonzero(np.random.binomial(1, 0.5, num_particles))
        aimpoint_locs = n_state_vectors[[5, 6], :]
        if self.check_los:
            aimpointm1_locs = n_state_vectors[[7, 8], :]
        if self.prior_on_endnodes:
            e = n_state_vectors[2, :].astype(int)
            endnodes = self.graph.dict['Edges']['EndNodes'][e, 1]
            endnode_locs = np.array([self.graph.dict['Nodes']['Longitude'][endnodes],
                                     self.graph.dict['Nodes']['Latitude'][endnodes]])

        while True:
            num_samples = len(resample_inds)
            if self.prior_on_endnodes:
                aimpoint_locs_res = endnode_locs[:, resample_inds]
            else:
                aimpoint_locs_res = aimpoint_locs[:, resample_inds]
            samples = np.atleast_2d(mn.rvs(np.zeros((2,)),
                                           self.aimpoint_sample_covar,
                                           size=num_samples)).T
            new_locs = aimpoint_locs_res + samples
            ps = [Point(*new_locs[:, i]) for i in range(num_samples)]
            if self.check_los:
                aimpointm1_locs_res = aimpointm1_locs[:, resample_inds]
                ps_m1 = [Point(*aimpointm1_locs_res[:, i]) for i in range(num_samples)]
                valid = self.bsptree.check_los(ps_m1, ps)
            else:
                valid = self.bsptree.are_empty_leaves(ps)

            if np.alltrue(valid):
                aimpoint_locs[:, resample_inds] = new_locs
                break
            else:
                valid_inds = np.flatnonzero(valid)
                aimpoint_locs[:, resample_inds[valid_inds]] = new_locs[:, valid_inds]
                resample_inds = resample_inds[np.logical_not(valid)]

        # 4) Process edge change
        # The CV propagation may lead to r's which are either less that zero or more than the
        # length of the edge. This means that the range and edge identifier needs to be adjusted
        # to correctly place the particle.

        # Get shortcuts for faster accessing
        r = n_state_vectors[0, :]
        e = n_state_vectors[2, :]
        d = n_state_vectors[3, :]
        s = n_state_vectors[4, :]
        a = aimpoint_locs # n_state_vectors[[5, 6], :]
        am1 = n_state_vectors[[7, 8], :]

        for i in range(num_particles):
            r_i, e_i, d_i, s_i, a_i, am1_i = normalise_re2(r[i], e[i], d[i], s[i], a[:, i], am1[:, i], self.graph)
            n_state_vectors[0, i] = r_i
            n_state_vectors[2, i] = e_i
            n_state_vectors[[5, 6], i] = a_i
            n_state_vectors[[7, 8], i] = am1_i

        return n_state_vectors