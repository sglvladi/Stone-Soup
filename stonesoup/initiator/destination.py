import numpy as np
from scipy.stats import multivariate_normal as mvn

from stonesoup.base import Property
from stonesoup.custom.graph import normalise_re, line_circle_test
from stonesoup.initiator import Initiator
from stonesoup.models.measurement import MeasurementModel
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track
from stonesoup.types.update import ParticleStateUpdate2


class DestinationBasedInitiator(Initiator):
    measurement_model = Property(MeasurementModel, doc="The measurement model")
    num_particles = Property(float, doc="Number of particles to use for initial state")
    speed_std = Property(float, doc="Std. of initial speed")
    short_paths = Property(dict, doc="The shortet paths container")
    graph = Property(dict, doc="A dictionary representation of the road network")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.next_track_id = 0

    def initiate(self, detections, **kwargs):

        edges = [i for i in range(len(self.graph['Edges']['Weight']))]
        init_tracks = set()
        for detection in detections:
            # Get valid edges for detection
            v_edges = []
            for edge in edges:
                endnodes = self.graph['Edges']['EndNodes'][edge, :]
                # Get endnode coordinates
                p1 = np.array(
                    [self.graph['Nodes']['Longitude'][endnodes[0]],
                     self.graph['Nodes']['Latitude'][endnodes[0]]])
                p2 = np.array(
                    [self.graph['Nodes']['Longitude'][endnodes[1]],
                     self.graph['Nodes']['Latitude'][endnodes[1]]])

                line = (p1, p2)
                circle = (detection.state_vector.ravel(), 3 * np.sqrt(self.measurement_model.covar()[0, 0]))
                if line_circle_test(line, circle):
                    v_edges.append(edge)
            v_edges = np.unique(v_edges)

            # Get valid destinations for valid edges
            v_dest = dict()
            for edge in v_edges:
                for sd, path in self.short_paths.items():
                    s = sd[0]
                    d = sd[1]
                    # If the path contains the edge
                    if len(np.where(path == edge)[0]) > 0:
                        if (s, edge) in v_dest:
                            v_dest[(s, edge)].append(d)
                        else:
                            v_dest[(s, edge)] = [d]

            v_edges = [key[1] for key in v_dest]
            prior_e = np.random.choice(v_edges, (self.num_particles,))
            prior_r = np.zeros((self.num_particles,)) + mvn.rvs(cov=0.001, size=self.num_particles)
            prior_speed = mvn.rvs(0, self.speed_std, (self.num_particles,))
            prior_destinations = []
            prior_source = []
            for e in prior_e:
                # endnodes = S['Edges']['EndNodes'][e, :]
                endnodes = [key[0] for key in v_dest if key[1] == e]
                source = np.random.choice(endnodes)
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
                r_i, e_i, d_i, s_i = normalise_re(r_i, e_i, d_i, s_i, self.short_paths, self.graph)
                sv = (r_i, sv[1], e_i, d_i, s_i)
                prior_particle_sv[:, i] = np.array(sv)
            prior_state = ParticleStateUpdate2(particles=prior_particle_sv,
                                               hypothesis=SingleHypothesis(None, detection),
                                               timestamp=detection.timestamp)
            track = Track(prior_state, id=self.next_track_id)
            self.next_track_id += 1
            init_tracks.add(track)
        return init_tracks