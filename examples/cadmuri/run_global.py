import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from datetime import datetime, timedelta
from scipy.stats import multivariate_normal as mvn
import cProfile as profile

from moviepy.video.io.bindings import mplfig_to_npimage

from stonesoup.custom.graph import graph_to_dict, shortest_path, get_xy_from_range_edge, \
    CustomDiGraph
from stonesoup.custom.plotting import plot_network, highlight_edges, highlight_nodes, plot_cov_ellipse
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

path =r'C:\Users\sglvladi\OneDrive\Workspace\PostDoc\CADMURI\Python\PyBSP\data\graphs\custom_digraph_v4.1.pickle'
G = pickle.load(open(path, 'rb'))
G = CustomDiGraph.fix(G)

path2 =r'C:\Users\sglvladi\OneDrive\Workspace\PostDoc\CADMURI\Python\PyBSP\data\ports_polygons.pickle'
ports, new_polygons = pickle.load(open(path2, 'rb'))
# G._rtree = None
S = G.as_dict()
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

# Global vars
USE_NEW_PF = True
num_destinations = 51
num_particles = 1000
source = 6327
destination = 2431
speed = 10000
zoom = 50000
LOAD = True

if LOAD:
    gnd_path, gnd_route_n, gnd_route_e, short_paths_n, short_paths_e, destinations = \
        pickle.load(open(f'./data/bsp_single_track_{source}_{destination}.pickle', 'rb'))
    G.short_paths_n = short_paths_n
    G.short_paths_e = short_paths_e
else:
    # Simulate ground-truth
    gnd_path, gnd_route_n, gnd_route_e = simulate(G, source, destination, speed)

    # Pre-compute short_paths
    feed = [destination]
    feed_tmp = set(ports['Node'].to_list()) - set(feed)
    destinations = feed + list(
        np.random.choice(list(feed_tmp), (num_destinations - len(feed),), False))
    short_paths_n = dict()
    short_paths_e = dict()
    for i in range(num_destinations):
        dest = destinations[i]
        short_paths_n_tmp, short_paths_e_tmp = G.shortest_path(source, dest)
        short_paths_n.update(short_paths_n_tmp)
        short_paths_e.update(short_paths_e_tmp)
    # G.short_paths_n = short_paths_n
    # G.short_paths_e = short_paths_e
    pickle.dump([gnd_path, gnd_route_n, gnd_route_e, short_paths_n, short_paths_e, destinations],
                open(f'./data/bsp_single_track_{source}_{destination}.pickle', 'wb'))

# Transition model
transition_model = DestinationTransitionModel(10000, G)

# Measurement model
mapping = [0,1]
R = np.eye(2)*500000
measurement_model = DestinationMeasurementModel(ndim_state=4, mapping=mapping, noise_covar=R,
                                                graph=G)

# Simulate detections
scans = []
for gnd_state in gnd_path:
    gnd_sv = gnd_state.state_vector
    det_sv = gnd_sv + measurement_model.rvs()
    timestamp = gnd_state.timestamp
    detection = Detection(state_vector=det_sv, timestamp=timestamp)
    scans.append((timestamp, set([detection])))

# initiator = DestinationBasedInitiator(measurement_model, num_particles, speed, G)
# tracks = initiator.initiate(scans[0][1])
# Prior
timestamp_init = scans[0][0]
prior_positions = list(scans[0][1])[0].state_vector + measurement_model.rvs(num_particles)
prior_r = np.ones((num_particles,))
prior_speed = mvn.rvs(0, speed**2, (num_particles,))
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
    prior = prior_state2  # predictor.predict(prior_state2, timestamp=timestamp_init+timedelta(seconds=3))
    prior.timestamp = timestamp_init
    # Initiate track
    track = Track(prior)
    # track = next(t for t in tracks)
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

# Plot the map
for polygon in new_polygons:
    x, y = polygon.exterior.xy
    ax1.plot(x, y, 'k-')
    ax2.plot(x, y, 'k-')
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
frames = []
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
    v_dest, vd_ind, vd_counts = np.unique(data[3,:], return_counts=True, return_index=True)
    id = np.argmax(vd_counts)
    v_edges, ve_counts = np.unique(data[2,:], return_counts=True)
    ie = np.argmax(ve_counts)
    est = np.append(est, [[track.state.mean[0, 0]], [v_edges[ie]]], axis=1)
    # print(f'Estimated edge: {v_edges[ie]} - Estimated destination node: {v_dest[id]}')
    port_name = ports.loc[ports['Node']==v_dest[id]]['NAME'].to_list()[0]
    print(f'Estimated edge: {v_edges[ie]} - Estimated destination node: {v_dest[id]} - Estimated destination port: {port_name}')

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
    arts1.append(ax1.plot(xy[0, :],xy[1, :], '.r')[0])
    arts2.append(ax2.plot(xy[0, :],xy[1, :], '.r')[0])
    xy1 = get_xy_from_range_edge(est[0, :], est[1, :], G)
    arts1.append(ax1.plot(xy1[0, :], xy1[1, :], '-b')[0])
    arts2.append(ax2.plot(xy1[0, :], xy1[1, :], '-b')[0])
    detection_data = np.array([detection.state_vector for detection in detections])
    arts1.append(ax1.plot(detection_data[:, 0], detection_data[:, 1], 'xc', label="Detections")[0])
    arts2.append(ax2.plot(detection_data[:, 0], detection_data[:, 1], 'xc', label="Detections")[0])
    port_names = ports.loc[ports['Node'].isin(v_dest)].sort_values('Node').drop_duplicates(subset='Node')['NAME'].to_list()
    if np.trace(cov) > 1e-10:
        arts1.append(plot_cov_ellipse(cov, mu, ax=ax1, nstd=3, fill=None, edgecolor='r'))
    else:
        circ = Circle(mu, 100000)
        ax1.add_artist(circ)
        arts1.append(circ)
    ax3.cla()
    barlist = ax3.bar(port_names, vd_counts/np.sum(vd_counts))
    try:
        idx = port_names.index('TOKYO KO')
        barlist[idx].set_color('m')
    except:
        pass
    ax3.set_title('Destination Distribution')
    plt.xticks(rotation=90, fontsize=5)


    mu = np.mean(xy, axis=1)
    arts1.append(ax1.plot([mu[0]-zoom, mu[0]+zoom, mu[0]+zoom, mu[0]-zoom, mu[0]-zoom],
                          [mu[1]-zoom, mu[1]-zoom, mu[1]+zoom, mu[1]+zoom, mu[1]-zoom], '-k')[0])
    ax2.set_xlim((mu[0]-zoom, mu[0]+zoom))
    ax2.set_ylim((mu[1]-zoom, mu[1]+zoom))

    plt.pause(0.01)
    # frame = mplfig_to_npimage(fig)
    # frames.append(frame)
    a=2

pf = 'new' if USE_NEW_PF else 'old'
pr.dump_stats('profile_{}.pstat'.format(pf))

# fig, ax = plt.subplots()
#
# for k in range(1, len(gnd_path)):
#     ax.cla()
#     for polygon in new_polygons:
#         x, y = polygon.exterior.xy
#         ax.plot(x, y, 'k-')
#     for key, value in short_paths_e.items():
#         nx.draw_networkx_edges(G, pos, value, edge_color='y', ax=ax)
#     highlight_edges(G, ax, gnd_route_e, edge_color='g')
#     highlight_nodes(G, ax, destinations, node_color='m')
#     highlight_nodes(G, ax, [destination], node_color='r')
#     data = np.array([s.state_vector for s in gnd_path.states[:k]])
#     plt.plot(data[:, 0], data[:, 1], '-r')
#     plt.pause(0.01)
# a = 2