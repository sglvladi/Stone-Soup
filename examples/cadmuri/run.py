import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from matplotlib.patches import Circle
from scipy.stats import multivariate_normal as mvn
import cProfile as profile

from stonesoup.custom.graph import load_graph_dict, dict_to_graph, shortest_path, \
    get_xy_from_range_edge, CustomDiGraph
from stonesoup.custom.plotting import plot_network, highlight_edges, highlight_nodes, \
    plot_cov_ellipse
from stonesoup.custom.simulation import simulate
from stonesoup.initiator.destination import DestinationBasedInitiator
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


seed = 500
np.random.seed(seed)

path = r'C:\Users\sglvladi\OneDrive\Workspace\PostDoc\CADMURI\MATLAB\simple\data\minn_2.mat'

S = load_graph_dict(path)
# for i in range(len(S['Edges']['Weight'])):
#     S['Edges']['Weight'][i] += np.random.normal(0, 200)
#     if S['Edges']['Weight'][i] <= 0:
#         S['Edges']['Weight'][i] = 0.01
G = CustomDiGraph.from_dict(S)
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

# Global vars
USE_NEW_PF = True
num_destinations = 100
num_particles = 10000
source = 519
destination = 115
speed = 0.1
zoom = 0.2
LOAD = False


if LOAD:
    gnd_path, gnd_route_n, gnd_route_e = \
        pickle.load(open(f'./data/single_track_{source}_{destination}.pickle', 'rb'))
else:
    # Simulate ground-truth
    gnd_path, gnd_route_n, gnd_route_e = simulate(G, source, destination, speed)
    pickle.dump([gnd_path, gnd_route_n, gnd_route_e],
                open(f'./data/single_track_{source}_{destination}.pickle', 'wb'))

# Pre-compute short_paths
feed = [destination]
feed_tmp = set([i for i in range(num_nodes)])-set(feed)
destinations = feed + list(np.random.choice(list(feed_tmp),(num_destinations-len(feed),),False))
short_paths_n = dict()
short_paths_e = dict()
for i in range(num_destinations):
    dest = destinations[i]
    short_paths_n_tmp, short_paths_e_tmp = G.shortest_path(source, dest)
    short_paths_n.update(short_paths_n_tmp)
    short_paths_e.update(short_paths_e_tmp)

# Transition model
transition_model = DestinationTransitionModel(0.01, G)

# Measurement model
mapping = [0,1]
R = np.eye(2)*0.0002
measurement_model = DestinationMeasurementModel(ndim_state=4, mapping=mapping, noise_covar=R,
                                                graph=G)

# Simulate detections
scans = []
for gnd_state in gnd_path:
    gnd_sv = gnd_state.state_vector
    det_sv = gnd_sv + measurement_model.rvs()
    timestamp = gnd_state.timestamp
    metadata = {"gnd_id": gnd_path.id}
    detection = Detection(state_vector=det_sv, timestamp=timestamp, metadata=metadata)
    scans.append((timestamp, set([detection])))

initiator = DestinationBasedInitiator(measurement_model, num_particles, speed, G)
tracks = initiator.initiate(scans[0][1])

# Prior
timestamp_init = scans[0][0]
prior_positions = list(scans[0][1])[0].state_vector + measurement_model.rvs(num_particles)
prior_r = np.zeros((num_particles,))
prior_speed = mvn.rvs(0, speed, (num_particles,))
prior_e = np.ones((num_particles,))*gnd_route_e[0]
prior_destinations = np.random.choice(destinations, (num_particles,))
prior_source = np.ones((num_particles,))*gnd_route_n[0]
prior_particles = []
prior_particle_sv = np.zeros((5,num_particles))
for i, sv in enumerate(zip(prior_r, prior_speed, prior_e, prior_destinations, prior_source)):
    particle = Particle(StateVector(list(sv)), 1.0/num_particles)
    prior_particles.append(particle)
    prior_particle_sv[:, i] = np.array(sv)
prior_state = ParticleState(particles=prior_particles, timestamp=timestamp_init)
prior_state2 = ParticleState2(particles=prior_particle_sv, timestamp=timestamp_init)

if USE_NEW_PF:
    # Predictor
    predictor = ParticlePredictor2(transition_model)
    # Updater
    resampler = SystematicResampler2()
    updater = ParticleUpdater2(measurement_model, resampler)
    # Initiate track
    # track = Track(prior_state2)
    track = next(t for t in tracks)
else:
    # Predictor
    predictor = ParticlePredictor(transition_model)
    # Updater
    resampler = SystematicResampler()
    updater = ParticleUpdater(measurement_model, resampler)
    # Initiate track
    track = Track(prior_state)


pos = nx.get_node_attributes(G, 'pos')

fig = plt.figure(figsize=(17, 10))
gs = fig.add_gridspec(2, 3)
ax1 = fig.add_subplot(gs[:, :2])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, 2])
plot_network(G, ax1)
plot_network(G, ax2)
for key, value in short_paths_e.items():
    highlight_edges(G, ax1, value, edge_color='y')
    highlight_edges(G, ax2, value, edge_color='y')
highlight_edges(G, ax1, gnd_route_e, edge_color='g')
highlight_edges(G, ax2, gnd_route_e, edge_color='g')
highlight_nodes(G, ax1, destinations, node_color='m', node_size=10)
highlight_nodes(G, ax2, destinations, node_color='m', node_size=10)
highlight_nodes(G, ax1, [destination], node_color='r', node_size=10)
highlight_nodes(G, ax2, [destination], node_color='r', node_size=10)
ax1.plot([], [], '-g', label='True path')
ax1.plot([], [], 'sr', label='True destination')
ax1.plot([], [], '-y', label='Confuser paths')
ax1.plot([], [], 'sm', label='Confuser destinations')
ax1.legend(loc='upper right')
ax1.set_title('Global view')
ax2.plot([], [], 'r.', label='Particles')
ax2.plot([], [], 'b-', label='Trajectory')
ax2.plot([], [], 'cx', label='Measurements')
ax2.set_title('Zoomed view')
ax2.legend(loc='upper right')
arts1 = []
arts2 = []

pr = profile.Profile()
pr.disable()
est = np.array([[],[]])
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
    est = np.append(est, [[track.state.mean[0, 0]], [v_edges[ie]]], axis=1)
    print('Estimated edge: {} - Estimated destination: {}'.format(v_edges[ie], v_dest[id]))
    xy = get_xy_from_range_edge(data[0, :], data[2, :], G)

    est_dest_pos = np.array([list(pos[node]) for node in data[3, :]]).T
    mu = np.average(est_dest_pos, axis=1, weights=posterior.weights)
    cov = np.cov(est_dest_pos, ddof=0, aweights=posterior.weights)

    # Plot
    # plot_network(G, ax)
    for art in arts1:
        art.remove()
    for art in arts2:
        art.remove()
    arts1 = []
    arts2 = []

    ind1 = np.flatnonzero(v_dest == destination)
    xy = get_xy_from_range_edge(data[0, :], data[2, :], G)
    arts1.append(ax1.plot(xy[0, :], xy[1, :], '.r')[0])
    arts2.append(ax2.plot(xy[0, :], xy[1, :], '.r')[0])
    xy1 = get_xy_from_range_edge(est[0, :], est[1, :], G)
    arts1.append(ax1.plot(xy1[0, :], xy1[1, :], '-b')[0])
    arts2.append(ax2.plot(xy1[0, :], xy1[1, :], '-b')[0])
    detection_data = np.array([detection.state_vector for detection in detections])
    arts1.append(ax1.plot(detection_data[:, 0], detection_data[:, 1], 'xc', label="Detections")[0])
    arts2.append(ax2.plot(detection_data[:, 0], detection_data[:, 1], 'xc', label="Detections")[0])
    if np.trace(cov) > 1e-10:
        arts1.append(plot_cov_ellipse(cov, mu, ax=ax1, nstd=3, fill=None, edgecolor='r'))
    else:
        circ = Circle(mu, 0.2, fill=None, edgecolor='r')
        ax1.add_artist(circ)
        arts1.append(circ)
    ax3.cla()
    barlist = ax3.bar([str(int(d)) for d in v_dest], vd_counts / np.sum(vd_counts))
    # barlist = ax3.hist(data[3,:])
    try:
        idx = v_dest.tolist().index(destination)
        barlist[idx].set_color('m')
    except:
        pass
    ax3.set_title('Destination Distribution')
    plt.xticks(rotation=90, fontsize=5)

    mu = np.mean(xy, axis=1)
    arts1.append(ax1.plot([mu[0] - zoom, mu[0] + zoom, mu[0] + zoom, mu[0] - zoom, mu[0] - zoom],
                          [mu[1] - zoom, mu[1] - zoom, mu[1] + zoom, mu[1] + zoom, mu[1] - zoom],
                          '-k')[0])
    ax2.set_xlim((mu[0] - zoom, mu[0] + zoom))
    ax2.set_ylim((mu[1] - zoom, mu[1] + zoom))

    plt.pause(0.01)

pf = 'new' if USE_NEW_PF else 'old'
pr.dump_stats('profile_{}.pstat'.format(pf))