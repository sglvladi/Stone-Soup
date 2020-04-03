import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.stats import multivariate_normal as mvn
import cProfile as profile

from stonesoup.custom.graph import load_graph_dict, dict_to_graph, shortest_path, get_xy_from_range_edge
from stonesoup.custom.plotting import plot_network
from stonesoup.custom.simulation import simulate
from stonesoup.models.transition.destination import DestinationTransitionModel
from stonesoup.models.measurement.destination import DestinationMeasurementModel
from stonesoup.predictor.particle import ParticlePredictor, ParticlePredictor2
from stonesoup.resampler.particle import SystematicResampler, SystematicResampler2
from stonesoup.types.array import StateVector
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.state import ParticleState, ParticleState2
from stonesoup.types.detection import Detection
from stonesoup.types.particle import Particle

# Load Graph
from stonesoup.types.track import Track
from stonesoup.updater.particle import ParticleUpdater, ParticleUpdater2

path = r'C:\Users\sglvladi\OneDrive\Workspace\PostDoc\CADMURI\MATLAB\simple\data\minn_2.mat'
S = load_graph_dict(path)
G = dict_to_graph(S)
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

# Global vars
USE_NEW_PF = True
num_destinations = 100
num_particles = 1000
source = 519
destination = 115
speed = 0.1


# Simulate ground-truth
gnd_path, gnd_route_n, gnd_route_e = simulate(G, source, destination, speed)

# Pre-compute short_paths
feed = [destination]
feed_tmp = set([i for i in range(num_nodes)])-set(feed)
destinations = feed + list(np.random.choice(list(feed_tmp),(num_destinations-len(feed),),False))
short_paths_n = dict()
short_paths_e = dict()
for i in range(num_destinations):
    dest = destinations[i]
    short_paths_n[dest], short_paths_e[dest] = shortest_path(G, source, dest)

# Transition model
transition_model = DestinationTransitionModel(0.001, S, short_paths_e)

# Measurement model
mapping = [0,1]
R = np.eye(2)*0.0002
measurement_model = DestinationMeasurementModel(ndim_state=4, mapping=mapping, noise_covar=R,
                                                spaths=short_paths_e, graph=S)

# Simulate detections
scans = []
for gnd_state in gnd_path:
    gnd_sv = gnd_state.state_vector
    det_sv = gnd_sv + measurement_model.rvs()
    timestamp = gnd_state.timestamp
    detection = Detection(state_vector=det_sv, timestamp=timestamp)
    scans.append((timestamp, set([detection])))

# Prior
timestamp_init = scans[0][0]
prior_positions = list(scans[0][1])[0].state_vector + measurement_model.rvs(num_particles)
prior_r = np.zeros((num_particles,))
prior_speed = mvn.rvs(0, speed, (num_particles,))
prior_e = np.ones((num_particles,))*gnd_route_e[0]
prior_destinations = np.random.choice(destinations, (num_particles,))
prior_particles = []
prior_particle_sv = np.zeros((4,num_particles))
for i, sv in enumerate(zip(prior_r, prior_speed, prior_e, prior_destinations)):
    particle = Particle(StateVector(list(sv)), 1.0/num_particles)
    prior_particles.append(particle)
    prior_particle_sv[:,i] = np.array(sv)
prior_state = ParticleState(particles=prior_particles, timestamp=timestamp_init)
prior_state2 = ParticleState2(particles=prior_particle_sv, timestamp=timestamp_init)

if USE_NEW_PF:
    # Predictor
    predictor = ParticlePredictor2(transition_model)
    # Updater
    resampler = SystematicResampler2()
    updater = ParticleUpdater2(measurement_model, resampler)
    # Initiate track
    track = Track(prior_state2)
else:
    # Predictor
    predictor = ParticlePredictor(transition_model)
    # Updater
    resampler = SystematicResampler()
    updater = ParticleUpdater(measurement_model, resampler)
    # Initiate track
    track = Track(prior_state)


fig, ax = plt.subplots()
pr = profile.Profile()
pr.disable()
for timestamp, detections in scans:
    print(timestamp)
    detection = list(detections)[0]
    pr.enable()
    # Run PF
    prediction = predictor.predict(track.state, timestamp=timestamp)
    hypothesis = SingleHypothesis(prediction, detection)
    posterior = updater.update(hypothesis)
    track.append(posterior)

    pr.disable()

    # Compute statistics
    if USE_NEW_PF:
        data = posterior.particles
    else:
        data = np.array([p.state_vector for p in posterior.particles])[:, :, 0].T

    # Compute counts of destinations and current positions
    v_dest, vd_counts = np.unique(data[3,:], return_counts=True)
    id = np.argmax(vd_counts)
    v_edges, ve_counts = np.unique(data[2,:], return_counts=True)
    ie = np.argmax(ve_counts)
    print('Estimated edge: {} - Estimated destination: {}'.format(v_edges[ie], v_dest[id]))
    xy = get_xy_from_range_edge(data[0, :], data[2, :], S)

    # Plot
    ax.cla()
    plot_network(G, ax)
    plt.plot(xy[0,:],xy[1,:],'.r')
    plt.pause(0.0001)

pf = 'new' if USE_NEW_PF else 'old'
pr.dump_stats('profile_{}.pstat'.format(pf))