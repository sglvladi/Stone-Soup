import numpy as np
import geopandas
from scipy.stats import multivariate_normal as mvn
from shapely.geometry import LineString, Point

from stonesoup.base import Property
from stonesoup.custom.graph import normalise_re, line_circle_test, calculate_r, CustomDiGraph
from stonesoup.initiator import Initiator
from stonesoup.models.measurement import MeasurementModel
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track
from stonesoup.types.update import ParticleStateUpdate2


class DestinationBasedInitiator(Initiator):
    measurement_model = Property(MeasurementModel, doc="The measurement model")
    num_particles = Property(float, doc="Number of particles to use for initial state")
    speed_std = Property(float, doc="Std. of initial speed")
    graph = Property(CustomDiGraph, doc="A dictionary representation of the road network")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.next_track_id = 0         # generate a spatial index (R-tree)
        a = 2

    def initiate(self, detections, **kwargs):
        gdf = self.graph.gdf  # construct a GeoDataFrame object
        rtree = self.graph.rtree  # generate a spatial index (R-tree)
        S = self.graph.as_dict()
        edges = [i for i in range(len(S['Edges']['Weight']))]
        init_tracks = set()
        for detection in detections:
            # query = Point(detection.state_vector.ravel()).buffer(3 * np.sqrt(self.measurement_model.covar()[0, 0]))
            # bounds = query.bounds  # minimum bounding region
            # result_rough = list(rtree.intersection(bounds))
            # result_precise = []
            # for i in result_rough:
            #     if query.intersects(gdf.geometry[i]):
            #         result_precise.append(i)
            # v_edges = result_precise

            # Get valid edges for detection
            # ==============================
            # Get query point
            center = detection.state_vector.ravel()
            radius = 3 * np.sqrt(self.measurement_model.covar()[0, 0])
            query_point = Point(center).buffer(radius)
            # Get rough result of edges that intersect query point envelope
            result_rough = rtree.query(query_point)
            # The exact edges are those in the rough result that intersect the query point
            gdf_rough = gdf.iloc[result_rough]
            v_edges = result_rough[np.flatnonzero(gdf_rough.intersects(query_point))]

            # if not len(v_edges):
            #     continue

            # Get valid edges for detection
            # v_edges = []
            # for edge in edges:
            #     endnodes = S['Edges']['EndNodes'][edge, :]
            #     # Get endnode coordinates
            #     p1 = np.array(
            #         [S['Nodes']['Longitude'][endnodes[0]],
            #          S['Nodes']['Latitude'][endnodes[0]]])
            #     p2 = np.array(
            #         [S['Nodes']['Longitude'][endnodes[1]],
            #          S['Nodes']['Latitude'][endnodes[1]]])
            #
            #     line = (p1, p2)
            #     circle = (detection.state_vector.ravel(), 3 * np.sqrt(self.measurement_model.covar()[0, 0]))
            #     if line_circle_test(line, circle):
            #         v_edges.append(edge)
            # v_edges = np.unique(v_edges)

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

            # v_dest = dict()
            # for edge in v_edges:
            #     for sd, path in self.graph.short_paths_e.items():
            #         s = sd[0]
            #         d = sd[1]
            #         # If the path contains the edge
            #         if len(np.flatnonzero(np.array(path) == edge)) > 0:
            #             if (s, edge) in v_dest:
            #                 v_dest[(s, edge)].append(d)
            #             else:
            #                 v_dest[(s, edge)] = [d]

            v_edges2 = np.unique([key[1] for key in v_dest])
            if not len(v_edges2):
                continue
            prior_e = np.random.choice(v_edges2, (self.num_particles,))
            # prior_r = np.zeros((self.num_particles,)) + mvn.rvs(cov=0.001, size=self.num_particles)
            prior_speed = mvn.rvs(0, self.speed_std**2, (self.num_particles,))
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
                r = calculate_r((p1, p2), detection.state_vector.ravel()) + mvn.rvs(cov=0.001)
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
            track = Track(prior_state, id=detection.metadata['gnd_id'])
            self.next_track_id += 1
            init_tracks.add(track)
        return init_tracks

    @staticmethod
    def _dict2gdf(S):
        """Convert dictionary into a GeoDataFrame object"""

        d = {'col1': [], 'geometry': []}  # Initialise a new dictionary
        last_index = S['Edges']['EndNodes'].shape[0]  # Count the road segments

        for i in list(range(0, last_index)):
            d['col1'].append('Edge ' + str(i))
            node1, node2 = S['Edges']['EndNodes'][i]
            lat1 = S['Nodes']['Latitude'][node1]
            lon1 = S['Nodes']['Longitude'][node1]
            lat2 = S['Nodes']['Latitude'][node2]
            lon2 = S['Nodes']['Longitude'][node2]
            segment = LineString([(lon1, lat1), (lon2, lat2)])
            d['geometry'].append(segment)

        gdf = geopandas.GeoDataFrame(d)  # GeoDataFrame from dict
        return gdf