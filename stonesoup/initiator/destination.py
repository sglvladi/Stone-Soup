import numpy as np
import geopandas
from pybsp.bsp import BSP
# from pybsp.geometry import Point
from bsppy import Point, BSPTree
from scipy.stats import multivariate_normal as mvn
from shapely.geometry import LineString, Point as ShapelyPoint

from stonesoup.base import Property
from stonesoup.custom.graph import normalise_re, line_circle_test, calculate_r, CustomDiGraph, \
    normalise_re2
from stonesoup.initiator import Initiator
from stonesoup.models.measurement import MeasurementModel
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track
from stonesoup.types.update import ParticleStateUpdate2


class DestinationBasedInitiator(Initiator):
    measurement_model: MeasurementModel = Property(doc="The measurement model")
    num_particles: float = Property(doc="Number of particles to use for initial state")
    speed_std: float = Property(doc="Std. of initial speed")
    graph: CustomDiGraph = Property(doc="A dictionary representation of the road network")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.next_track_id = 0         # generate a spatial index (R-tree)

    def initiate(self, detections, **kwargs):
        init_tracks = set()
        for detection in detections:

            # Get valid edges for detection
            v_edges = self._get_v_edges(detection)

            # Get valid destinations from valid edges
            v_dest = self._get_v_dest(v_edges)

            # Get subset of edges that match the destinations
            v_edges2 = np.unique([key[1] for key in v_dest])

            if not len(v_edges2):
                continue

            track = self._init_track(detection, v_edges2, v_dest)

            self.next_track_id += 1
            init_tracks.add(track)
        return init_tracks

    def _get_v_edges(self, detection):

        gdf = self.graph.gdf  # construct a GeoDataFrame object
        rtree = self.graph.rtree  # generate a spatial index (R-tree)

        # Get valid edges for detection
        # ==============================
        # Get query point
        center = detection.state_vector.ravel()
        radius = 3 * np.sqrt(self.measurement_model.covar()[0, 0])
        query_point = ShapelyPoint(center).buffer(radius)
        # Get rough result of edges that intersect query point envelope
        result_rough = rtree.query(query_point)
        # The exact edges are those in the rough result that intersect the query point
        gdf_rough = gdf.iloc[result_rough]
        return result_rough[np.flatnonzero(gdf_rough.intersects(query_point))]

    def _get_v_dest(self, v_edges):
        # Get valid destinations for valid edges
        # ======================================
        v_dest = dict()
        for (source, dest), path in self.graph.short_paths_e.items():
            edges_in_path = set(v_edges).intersection(set(path))
            for edge in edges_in_path:
                try:
                    v_dest[(source, edge)].append(dest)
                except KeyError:
                    v_dest[(source, edge)] = [dest]
        return v_dest

    def _init_track(self, detection,  v_edges2, v_dest):
        S = self.graph.as_dict()
        prior_e = np.random.choice(v_edges2, (self.num_particles,))
        prior_speed = mvn.rvs(0, self.speed_std ** 2, (self.num_particles,))
        prior_destinations = []
        prior_source = []
        prior_r = []
        for e in prior_e:
            endnodes = S['Edges']['EndNodes'][e, :]
            p1 = np.array(
                [S['Nodes']['Longitude'][endnodes[0]],
                 S['Nodes']['Latitude'][endnodes[0]]])
            p2 = np.array(
                [S['Nodes']['Longitude'][endnodes[1]],
                 S['Nodes']['Latitude'][endnodes[1]]])
            r = calculate_r((p1, p2), detection.state_vector.ravel()) + mvn.rvs(cov=10)
            prior_r.append(r)
            sources = [key[0] for key in v_dest if key[1] == e]
            source = np.random.choice(sources)
            v_d = v_dest[(source, e)]
            dest = np.random.choice(v_d)
            prior_source.append(source)
            prior_destinations.append(dest)

        prior_particle_sv = np.zeros((5, self.num_particles))
        for i, sv in enumerate(
                zip(prior_r, prior_speed, prior_e, prior_destinations, prior_source)):
            r_i = sv[0]
            e_i = sv[2]
            d_i = sv[3]
            s_i = sv[4]
            r_i, e_i, d_i, s_i = normalise_re(r_i, e_i, d_i, s_i, self.graph)
            sv = (r_i, sv[1], e_i, d_i, s_i)
            prior_particle_sv[:, i] = np.array(sv)

        prior_state = ParticleStateUpdate2(particles=prior_particle_sv,
                                           hypothesis=SingleHypothesis(None, detection),
                                           timestamp=detection.timestamp)
        return Track([prior_state], id=detection.metadata['gnd_id'])


class DestinationBasedInitiatorAimpoint(DestinationBasedInitiator):

    bsptree: BSP = Property(doc="The bsp tree")

    def _init_track(self, detection,  v_edges2, v_dest):
        S = self.graph.as_dict()
        prior_e = np.random.choice(v_edges2, (self.num_particles,))
        prior_speed = mvn.rvs(0, self.speed_std ** 2, (self.num_particles,))
        prior_destinations = []
        prior_source = []
        prior_r = []
        prior_a = []
        prior_am1 = []
        for e in prior_e:
            endnodes = S['Edges']['EndNodes'][e, :]
            p1 = Point(S['Nodes']['Longitude'][endnodes[0]], S['Nodes']['Latitude'][endnodes[0]])
            p2 = Point(S['Nodes']['Longitude'][endnodes[1]], S['Nodes']['Latitude'][endnodes[1]])
            if np.random.rand() > 0.98:
                new_loc = mvn.rvs(p2.to_array(), np.diag([1e3, 1e3]))
                p2 = Point(new_loc[0], new_loc[1])
                while self.bsptree.find_leaf(p2).is_solid:
                    new_loc = mvn.rvs(p2.to_array(), np.diag([1e3, 1e3]))
                    p2 = Point(new_loc[0], new_loc[1])
            r = calculate_r((p1.to_array(), p2.to_array()), detection.state_vector.ravel()) + mvn.rvs(cov=10)
            prior_r.append(r)
            prior_a.append(p2.to_array())
            prior_am1.append(p1.to_array())
            sources = [key[0] for key in v_dest if key[1] == e]
            source = np.random.choice(sources)
            v_d = v_dest[(source, e)]
            dest = np.random.choice(v_d)
            prior_source.append(source)
            prior_destinations.append(dest)

        prior_particle_sv = np.zeros((9, self.num_particles))
        for i, sv in enumerate(
                zip(prior_r, prior_speed, prior_e, prior_destinations, prior_source, prior_a, prior_am1)):
            r_i = sv[0]
            e_i = sv[2]
            d_i = sv[3]
            s_i = sv[4]
            a_i = sv[5]
            am1_i = sv[6]
            r_i, e_i, d_i, s_i, a_i, am1_i = normalise_re2(r_i, e_i, d_i, s_i, a_i, am1_i, self.graph)
            sv = (r_i, sv[1], e_i, d_i, s_i, *a_i, *am1_i)
            prior_particle_sv[:, i] = np.array(sv)

        prior_state = ParticleStateUpdate2(particles=prior_particle_sv,
                                           hypothesis=SingleHypothesis(None, detection),
                                           timestamp=detection.timestamp)
        return Track([prior_state], id=detection.metadata['gnd_id'])
