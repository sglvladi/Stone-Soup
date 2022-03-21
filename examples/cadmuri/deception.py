import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy

from matplotlib.patches import Circle

from stonesoup.custom.graph import load_graph_dict, get_xy_from_range_edge, CustomDiGraph
from stonesoup.custom.plotting import plot_network, highlight_edges, highlight_nodes, \
    plot_cov_ellipse
from stonesoup.custom.simulation import simulate
from stonesoup.initiator.destination import DestinationBasedInitiator
from stonesoup.models.transition.destination import DestinationTransitionModel
from stonesoup.models.measurement.destination import DestinationMeasurementModel
from stonesoup.predictor.particle import ParticlePredictor2
from stonesoup.updater.particle import ParticleUpdater2
from stonesoup.resampler.particle import SystematicResampler2
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.detection import Detection


seed = 163321 #900 #649312# 900
np.random.seed(seed)

# Global vars
PLOT = True
num_particles = 1000
source = 402 #519
destination = 115
speed = 0.1
zoom = 0.2
path = r'C:\Users\sglvladi\OneDrive\Workspace\PostDoc\CADMURI\MATLAB\simple\data\minn_2.mat'
S = load_graph_dict(path)
S2 = deepcopy(S)
for i in range(len(S2['Edges']['Weight'])):
    S2['Edges']['Weight'][i] += np.random.normal(0, 200)
    if S2['Edges']['Weight'][i] <= 0:
        S2['Edges']['Weight'][i] = 0.01
G = CustomDiGraph.from_dict(S2)
G2 = CustomDiGraph.from_dict(S)
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
num_destinations = 100 # num_nodes


def prepare_plot(G, short_paths_e, gnd_route_e, destination, destinations):
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
    return dict(axes=[ax1, ax2, ax3], arts=[[],[]])


def plot(cfg, est, track, detections, v_dest, vd_counts, G):
    posterior = track.state

    # Compute statistics
    data = track.state.particles

    pos = nx.get_node_attributes(G, 'pos')

    est_dest_pos = np.array([list(pos[node]) for node in data[3, :]]).T


    # Plot
    # plot_network(G, ax)
    arts1, arts2 = cfg['arts']
    ax1, ax2, ax3 = cfg['axes']
    for art in arts1:
        art.remove()
    for art in arts2:
        art.remove()
    arts1.clear()
    arts2.clear()

    xy = get_xy_from_range_edge(data[0, :], data[2, :], G)
    arts1.append(ax1.plot(xy[0, :], xy[1, :], '.r')[0])
    arts2.append(ax2.plot(xy[0, :], xy[1, :], '.r')[0])
    xy1 = get_xy_from_range_edge(est[0, :], est[1, :], G)
    arts1.append(ax1.plot(xy1[0, :], xy1[1, :], '-b')[0])
    arts2.append(ax2.plot(xy1[0, :], xy1[1, :], '-b')[0])
    detection_data = np.array([detection.state_vector for detection in detections])
    arts1.append(ax1.plot(detection_data[:, 0], detection_data[:, 1], 'xc', label="Detections")[0])
    arts2.append(ax2.plot(detection_data[:, 0], detection_data[:, 1], 'xc', label="Detections")[0])
    try:
        mu = np.average(est_dest_pos, axis=1, weights=posterior.weights)
        cov = np.cov(est_dest_pos, ddof=0, aweights=posterior.weights)
    except ZeroDivisionError:
        print('zzz')
        a = 2
        weights = np.ones((len(posterior),)) / len(posterior)
        mu = np.average(est_dest_pos, axis=1, weights=weights)
        cov = np.cov(est_dest_pos, ddof=0, aweights=weights)

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

    return cfg


def run_filter(G, scans, do_plot=False):
    # Transition model
    transition_model = DestinationTransitionModel(0.001, G)

    # Measurement model
    mapping = [0, 1]
    R = np.eye(2) * 0.0002
    measurement_model = DestinationMeasurementModel(ndim_state=4, mapping=mapping, noise_covar=R,
                                                    graph=G)

    # Initiator track
    initiator = DestinationBasedInitiator(measurement_model, num_particles, speed, G)

    # Predictor
    predictor = ParticlePredictor2(transition_model)

    # Updater
    updater = ParticleUpdater2(measurement_model, SystematicResampler2())

    # Initiate track
    tracks = initiator.initiate(scans[0][1])
    track = next(t for t in tracks)

    if do_plot:
        cfg = prepare_plot(G, short_paths_e, gnd_route_e, destination, destinations)

    est = np.array([[], []])
    for timestamp, detections in scans:
        print(timestamp)
        detection = list(detections)[0]

        # Run PF
        prediction = predictor.predict(track.state, timestamp=timestamp)
        hypothesis = SingleHypothesis(prediction, detection)
        posterior = updater.update(hypothesis)
        track.append(posterior)

        # Compute counts of destinations and current positions
        data = track.state.particles
        v_dest, vd_counts = np.unique(data[3, :], return_counts=True)
        id = np.argmax(vd_counts)
        v_edges, ve_counts = np.unique(data[2, :], return_counts=True)
        ie = np.argmax(ve_counts)
        est = np.append(est, [[track.state.mean[0, 0]], [v_edges[ie]]], axis=1)
        print('Estimated edge: {} - Estimated destination: {}'.format(v_edges[ie], v_dest[id]))

        if do_plot:
            cfg = plot(cfg, est, track, detections, v_dest, vd_counts, G)

    return track


# Simulate ground-truth
gnd_path, gnd_route_n, gnd_route_e = simulate(G, source, destination, speed)
pickle.dump([gnd_path, gnd_route_n, gnd_route_e],
            open(f'./data/single_track_{source}_{destination}.pickle', 'wb'))

# Pre-compute short_paths
feed = [destination]
feed_tmp = set([i for i in range(num_nodes)])-set(feed)
destinations = feed + list(np.random.choice(list(feed_tmp),(num_destinations-len(feed),), False)) # [i for i in range(num_nodes)]
targets = None if len(destinations) == num_nodes else destinations
short_paths_n, short_paths_e = G2.shortest_path(source, targets)
G.shortest_path(source, targets)

# Measurement model
mapping = [0, 1]
R = np.eye(2) * 0.0002
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


track1 = run_filter(G, scans, PLOT)
track2 = run_filter(G2, scans, PLOT)

lik1 = []
lik2 = []
ratio = []
for state1, state2 in zip(track1, track2):
    lik1.append(np.sum(state1.weights))
    lik2.append(np.sum(state2.weights))
    # ratio_m1 = 1 if not len(ratio) else ratio[-1]
    ratio.append(lik1[-1]/(lik1[-1]+lik2[-1]))

ratio2 = []
for i in range(len(lik1)):
    prod1 = np.prod(np.array(lik1[:i+1]))
    prod2 = np.prod(np.array(lik2[:i+1]))
    ratio2.append(prod1/(prod1+prod2))

plt.plot(ratio2)
