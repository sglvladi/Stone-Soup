import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import pickle

from stonesoup.custom.graph import load_graph_dict, dict_to_graph, shortest_path, \
    get_xy_from_range_edge, CustomDiGraph
from stonesoup.custom.plotting import plot_network, highlight_nodes, highlight_edges
from stonesoup.custom.simulation import simulate_gnd, simulate_detections
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.initiator.destination import DestinationBasedInitiator
from stonesoup.models.transition.destination import DestinationTransitionModel
from stonesoup.models.measurement.destination import DestinationMeasurementModel
from stonesoup.predictor.particle import ParticlePredictor2
from stonesoup.resampler.particle import SystematicResampler2
from stonesoup.types.state import ParticleState2
from stonesoup.types.update import Update
from stonesoup.updater.particle import ParticleUpdater2
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.measures import Mahalanobis

# Load Graph


def del_tracks(tracks, meas_model, thresh=0.005):
    d_tracks = set()
    for track in tracks:
        t_sv = meas_model.function(track.state, noise=0)
        st = ParticleState2(t_sv, track.state.weights)
        if (np.trace(st.covar) > thresh) \
                or not any(isinstance(state, Update) and state.hypothesis for state in track[-4:]):
            d_tracks.add(track)
    return d_tracks


def plot(meta, tracks, detections, G, t_colors, destinations, gnd_paths, gnd_routes_e, VBOUNDS):
    track_ids = [track.id for track in tracks]

    # Plot
    ax = meta['base']['ax']
    ax.cla()
    plot_network(G, ax)
    highlight_nodes(G, ax, destinations, node_size=10, label="Possible Destinations")
    detection_data = np.array([detection.state_vector for detection in detections])
    for gnd_path in gnd_paths:
        gnd_route_e = gnd_routes_e[gnd_path]
        highlight_edges(G, ax, gnd_route_e, edge_color=t_colors[gnd_path.id])
    for id, val in meta['sub'].items():
        ax2 = val['ax']
        ax2.cla()
        ax2.tick_params(
            axis='both',
            which='both',
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False)
        if id < len(tracks):
            plot_network(G, ax2)
            highlight_nodes(G, ax2, destinations, node_size=10)
            ax2.plot(detection_data[:, 0], detection_data[:, 1], 'xc')
            for gnd_path in gnd_paths:
                gnd_route_e = gnd_routes_e[gnd_path]
                highlight_edges(G, ax2, gnd_route_e, edge_color=t_colors[gnd_path.id])
        # if gnd_path.id in track_ids:
        #     plot_network(G, ax2)
        #     highlight_nodes(G, ax2, destinations, node_size=10)
        #     ax2.plot(detection_data[:, 0], detection_data[:, 1], 'xc')
        #     highlight_edges(G, ax2, gnd_route_e, edge_color=t_colors[gnd_path.id])

    for i, track in enumerate(tracks):
        data = track.state.particles
        xy = get_xy_from_range_edge(data[0, :], data[2, :], S)
        x_mean = np.mean(xy[0, :])
        y_mean = np.mean(xy[1, :])
        # xy2 = get_xy_from_sv(data2, short_paths_e, S)

        ax.plot(xy[0, :], xy[1, :], '.', label="Track {}".format(track.id))
        ax.text(x_mean, y_mean, '{}'.format(track.id))

        for i2, val in meta['sub'].items():
            if i2 < len(tracks):
                ax2 = meta['sub'][i2]['ax']
                # plot_network(G, ax2)
                # highlight_nodes(G, ax2, destinations, node_size=10)
                # ax2.plot(detection_data[:, 0], detection_data[:, 1], 'xc')
                # if id in track_ids:
                ax2.plot(xy[0, :], xy[1, :], '.')
                ax2.text(x_mean, y_mean, '{}'.format(track.id)).set_clip_on(True)
                if i == i2:
                    ax2.set_xlim(x_mean - 0.1, x_mean + 0.1)
                    ax2.set_ylim(y_mean - 0.1, y_mean + 0.1)
                    ax2.set_title('Track {}'.format(track.id))
        # for id, val in meta['sub'].items():
        #     ax2 = val['ax']
        #     if id == 'base':
        #         continue
        #     if id in track_ids:
        #         ax2.plot(xy[0, :], xy[1, :], '.')
        #         ax2.text(x_mean, y_mean, '{}'.format(track.id)).set_clip_on(True)
        #     if id == track.id:
        #         ax2.set_xlim(x_mean - 0.1, x_mean + 0.1)
        #         ax2.set_ylim(y_mean - 0.1, y_mean + 0.1)
        #         ax2.set_title('Track {}'.format(track.id))

    # for detection in detections:
    ax.plot(detection_data[:, 0], detection_data[:, 1], 'xc', label="Detections")

    ax.set_xlim(VBOUNDS[0][0], VBOUNDS[0][1])
    ax.set_ylim(VBOUNDS[1][0], VBOUNDS[1][1])
    ax.legend(loc='lower right')

def plot_scratch(detection, meas_model):
    ax = plot_data['base']['ax']
    plot_network(G, ax)
    detection_data = np.array([detection.state_vector for detection in detections])
    ax.plot(detection_data[:, 0], detection_data[:, 1], 'xc', label="Detections")
    c1 = plt.Circle(detection.state_vector.ravel(), np.sqrt(meas_model.covar()[0, 0]),
                    fill=False)
    ax.add_artist(c1)
    plt.show()

def plot_line_circle(line, circle):
    ax = plot_data['base']['ax']
    plot_network(G, ax)
    x = [line[0][0], line[1][0]]
    y = [line[0][1], line[1][1]]
    ax.plot(x,y, 'r-')
    detection_data = np.array([detection.state_vector for detection in detections])
    ax.plot(detection_data[:, 0], detection_data[:, 1], 'xc', label="Detections")
    c1 = plt.Circle(circle[0].ravel(), circle[1], color='r', fill=False)
    ax.add_artist(c1)
    plt.show()


# HIGH-LEVEL CONFIG
path = './data/minn_2.mat'  # Path to mat file containing the road network
num_tracks = 1             # Number of simulated targets
num_destinations = 50       # Number of possible destinations
num_particles = 1000        # Number of particles to use in SMC sampler
speed = 0.01                # Target speed
P_D = 0.95                  # Probability of detection
lambda_FA = 0               # Clutter density
PLOT = True                 # Set True/False to enable/disable plotting
RECORD = PLOT and False     # Set True/False to enable/disable recording
LOAD = False                 # Set True/False to enable/disable loading data from file

S = load_graph_dict(path)
G = CustomDiGraph.from_dict(S)
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

VBOUNDS = ((-97.62183187288396, -89.14054378521554), (43.19878867683182, 49.27774394589634))

t_sources = []
t_destinations = []
t_colors = []

if LOAD:
    with open('./data/tracks_{}_dest_{}_t.pkl'.format(num_tracks, num_destinations), 'rb') as f:
        gnd_paths, gnd_routes_n, gnd_routes_e, t_sources, t_destinations, t_colors, destinations, \
        short_paths_e, short_paths_n, scans = pickle.load(f)

    # Transition model
    transition_model = DestinationTransitionModel(0.00001, S, short_paths_e)

    # Measurement model
    mapping = [0, 1]
    R = np.eye(2) * 0.00005
    measurement_model = DestinationMeasurementModel(ndim_state=4, mapping=mapping,
                                                    noise_covar=R,
                                                    spaths=short_paths_e, graph=S)
else:
    gnd_paths, gnd_routes_n, gnd_routes_e, t_sources, t_destinations, t_colors, destinations = \
        simulate_gnd(G, num_tracks, num_destinations, speed)

    short_paths_n, short_paths_e = shortest_path(G, t_sources, destinations)

    # Transition model
    transition_model = DestinationTransitionModel(0.00001, S, short_paths_e)

    # Measurement model
    mapping = [0, 1]
    R = np.eye(2) * 0.000005
    measurement_model = DestinationMeasurementModel(ndim_state=4, mapping=mapping, noise_covar=R,
                                                    spaths=short_paths_e, graph=S)

    scans = simulate_detections(gnd_paths, measurement_model, P_D, lambda_FA, VBOUNDS)

    with open('./data/tracks_{}_dest_{}_t.pkl'.format(num_tracks, num_destinations), 'wb') as f:
        pickle.dump([gnd_paths, gnd_routes_n, gnd_routes_e, t_sources, t_destinations, t_colors,
                     destinations, short_paths_e, short_paths_n, scans ], f)


# Predictor
predictor = ParticlePredictor2(transition_model)

# Updater
resampler = SystematicResampler2()
updater = ParticleUpdater2(measurement_model, resampler)

# Hypothesiser and Data-Associator
hypothesiser = DistanceHypothesiser(predictor, updater, Mahalanobis(), 20)
associator = GNNWith2DAssignment(hypothesiser)

deleter = UpdateTimeStepsDeleter(4)
initiator = DestinationBasedInitiator(measurement_model, num_particles, speed, short_paths_e, G)

# Initiate tracks
tracks = set()

# Initiate plotting grid
fig = plt.figure(figsize=(20, 15))
gs = fig.add_gridspec(5, 7)

plot_data = dict()
plot_data['base'] = {
    'ax': fig.add_subplot(gs[1:-1, 1:-1])
}
plot_data['sub'] = dict()
for i in range(20):
    id = i
    if 0 <= id < 7:
        i1 = 0
        i2 = id
    elif 7 <= id < 11:
        i1 = id - 6
        i2 = -1
    elif 11 <= id < 17:
        i1 = -1
        i2 = 16 - id
    else:
        i1 = 20 - id
        i2 = 0
    plot_data['sub'][id] = {
        'ax': fig.add_subplot(gs[i1, i2])
    }

if RECORD:
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=1, metadata=metadata)
    writer.setup(fig, "writer_test_20.mp4", 400)

for timestamp, detections in scans:

    print(timestamp)

    # Perform data association
    # pr.enable()
    associations = associator.associate(tracks, detections, timestamp)

    # Update tracks based on association hypotheses
    associated_detections = set()
    for track, hypothesis in associations.items():
        if hypothesis:
            state_post = updater.update(hypothesis)
            track.append(state_post)
            associated_detections.add(hypothesis.measurement)
        else:
            track.append(hypothesis.prediction)

    unassociated_detections = detections - associated_detections

    tracks |= initiator.initiate(unassociated_detections)
    tracks -= del_tracks(tracks, measurement_model)
    print('Tracked')

    if PLOT:
        plot(plot_data, tracks, detections, G, t_colors,
             destinations, gnd_paths, gnd_routes_e, VBOUNDS)
        plt.pause(0.0001)
    if RECORD:
        writer.grab_frame()

    # pr.disable()

if RECORD:
    writer.finish()