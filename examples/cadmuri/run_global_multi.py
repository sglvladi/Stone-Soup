import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import pickle

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from stonesoup.custom.graph import load_graph_dict, dict_to_graph, shortest_path, \
    get_xy_from_range_edge, graph_to_dict, CustomDiGraph
from stonesoup.custom.plotting import plot_network, highlight_nodes, highlight_edges, \
    remove_artists, plot_polygons, plot_short_paths_e
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


def del_tracks(tracks, meas_model, thresh=100000000):
    d_tracks = set()
    for track in tracks:
        t_sv = meas_model.function(track.state, noise=0)
        st = ParticleState2(t_sv, track.state.weights)
        if (np.trace(st.covar) > thresh) \
                or not any(isinstance(state, Update) and state.hypothesis for state in track[-4:]):
            d_tracks.add(track)
    return d_tracks


def plot(meta, tracks, detections, G, t_colors, destinations, gnd_paths, gnd_routes_e, VBOUNDS):

    # Plot
    ax = meta['base']['ax']
    remove_artists(meta['base']['dynamic_arts'])
    meta['base']['dynamic_arts'] = []
    detection_data = np.array([detection.state_vector for detection in detections])
    for gnd_path in gnd_paths:
        if gnd_path not in meta['base']['gnd_arts']:
            gnd_route_e = gnd_routes_e[gnd_path]
            meta['base']['gnd_arts'][gnd_path] = highlight_edges(G, ax, gnd_route_e, edge_color=t_colors[gnd_path.id])

    for id, val in meta['sub'].items():
        ax2 = val['ax']
        remove_artists(val['dynamic_arts'])
        val['dynamic_arts'] = []
        if id < len(tracks):
            if not val['net_plotted']:
                plot_polygons(new_polygons, ax2, zorder=0)
                highlight_nodes(G, ax2, destinations, node_size=10)
                ax2.tick_params(
                    axis='both',
                    which='both',
                    bottom=False,
                    left=False,
                    labelbottom=False,
                    labelleft=False)
                val['net_plotted'] = True
                plot_short_paths_e(short_paths_e, G, ax=ax2, edge_color='y', width=1.0)
                # for key, value in short_paths_e.items():
                #     highlight_edges(G, ax2, value, edge_color='y')
                print('plotting net')
            val['dynamic_arts'] += ax2.plot(detection_data[:, 0], detection_data[:, 1], 'xc')
            # for gnd_path in gnd_paths:
            #     if gnd_path not in val['gnd_arts']:
            #         gnd_route_e = gnd_routes_e[gnd_path]
            #         val['gnd_arts'][gnd_path] = highlight_edges(G, ax2, gnd_route_e, edge_color=t_colors[gnd_path.id])
        # if gnd_path.id in track_ids:
        #     plot_network(G, ax2)
        #     highlight_nodes(G, ax2, destinations, node_size=10)
        #     ax2.plot(detection_data[:, 0], detection_data[:, 1], 'xc')
        #     highlight_edges(G, ax2, gnd_route_e, edge_color=t_colors[gnd_path.id])

    for i, track in enumerate(tracks):
        data = track.state.particles
        xy = get_xy_from_range_edge(data[0, :], data[2, :], G)
        x_mean = np.mean(xy[0, :])
        y_mean = np.mean(xy[1, :])
        # xy2 = get_xy_from_sv(data2, short_paths_e, S)

        meta['base']['dynamic_arts'] += ax.plot(xy[0, :], xy[1, :], '.', label="Track {}".format(track.id), color=t_colors[track.id])
        meta['base']['dynamic_arts'].append(ax.text(x_mean, y_mean, f'{track.id}', color='r'))

        for i2, val in meta['sub'].items():
            if i2 < len(tracks):
                ax2 = meta['sub'][i2]['ax']
                meta['sub'][i2]['dynamic_arts'] += ax2.plot(xy[0, :], xy[1, :], '.', color=t_colors[track.id])
                text_art = ax2.text(x_mean, y_mean, f'{track.id}', color='r')
                text_art.set_clip_on(True)
                meta['sub'][i2]['dynamic_arts'].append(text_art)
                if i == i2:
                    ax2.set_xlim(x_mean - zoom, x_mean + zoom)
                    ax2.set_ylim(y_mean - zoom, y_mean + zoom)
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
    meta['base']['dynamic_arts'] += ax.plot(detection_data[:, 0], detection_data[:, 1], 'xc', label="Detections")
    ax.legend(loc='lower right')

# HIGH-LEVEL CONFIG
num_tracks = 20             # Number of simulated targets
num_destinations = 50       # Number of possible destinations
num_particles = 1000        # Number of particles to use in SMC sampler
speed = 10000                # Target speed
P_D = 0.95                  # Probability of detection
lambda_FA = 0               # Clutter density
PLOT = True                # Set True/False to enable/disable plotting
RECORD = PLOT and False     # Set True/False to enable/disable recording
zoom = 50000
LOAD = True                 # Set True/False to enable/disable loading data from file

# Load the network
path =r'C:\Users\sglvladi\OneDrive\Workspace\PostDoc\CADMURI\Python\PyBSP\data\graphs\custom_digraph_v4.1.pickle'
G = pickle.load(open(path, 'rb'))
G = CustomDiGraph.fix(G)
S = G.as_dict()

# Load the polygons
path2 =r'C:\Users\sglvladi\OneDrive\Workspace\PostDoc\CADMURI\Python\PyBSP\data\ports_polygons.pickle'
ports, new_polygons = pickle.load(open(path2, 'rb'))

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

VBOUNDS = ((-22041259.177068166, 22041259.177068166), (-11818985.537814114, 19893557.410806347))

t_sources = []
t_destinations = []
t_colors = []

if LOAD:
    with open('./data/bsp_tracks_{}_dest_{}_t.pkl'.format(num_tracks, num_destinations), 'rb') as f:
        gnd_paths, gnd_routes_n, gnd_routes_e, t_sources, t_destinations, t_colors, destinations, \
        short_paths_e, short_paths_n, scans = pickle.load(f)

    G.short_paths_n = short_paths_n
    G.short_paths_e = short_paths_e

    # Transition model
    transition_model = DestinationTransitionModel(10000, G)

    # Measurement model
    mapping = [0, 1]
    R = np.eye(2)*500000
    measurement_model = DestinationMeasurementModel(ndim_state=4, mapping=mapping,
                                                    noise_covar=R, graph=G)
else:
    gnd_paths, gnd_routes_n, gnd_routes_e, t_sources, t_destinations, t_colors, destinations = \
        simulate_gnd(G, num_tracks, num_destinations, speed)

    print('Generating short paths...', end='')
    short_paths_n, short_paths_e = shortest_path(G, t_sources, destinations)
    print('Done!')
    G.short_paths_n = short_paths_n
    G.short_paths_e = short_paths_e

    # Transition model
    transition_model = DestinationTransitionModel(10000, G)

    # Measurement model
    mapping = [0, 1]
    R = np.eye(2)*500000
    measurement_model = DestinationMeasurementModel(ndim_state=4, mapping=mapping, noise_covar=R,
                                                    graph=G)

    scans = simulate_detections(gnd_paths, measurement_model, P_D, lambda_FA, VBOUNDS)

    with open('./data/bsp_tracks_{}_dest_{}_t.pkl'.format(num_tracks, num_destinations), 'wb') as f:
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
initiator = DestinationBasedInitiator(measurement_model, num_particles, speed, G)

# Initiate tracks
tracks = set()

if PLOT:
    # Initiate plotting grid
    fig = plt.figure(figsize=(17, 12))
    gs = fig.add_gridspec(5, 7)

    plot_data = dict()
    plot_data['base'] = {
        'ax': fig.add_subplot(gs[1:-1, 1:-1]),
        'dynamic_arts': [],
        'gnd_arts': dict(),
    }
    # plot_data['arts']['tracks'] =
    # plot_network(G, plot_data['base']['ax'])
    plot_polygons(new_polygons, plot_data['base']['ax'], zorder=0)
    highlight_nodes(G, plot_data['base']['ax'], destinations, node_size=10, label="Possible Destinations")
    plot_short_paths_e(short_paths_e, G, ax=plot_data['base']['ax'], edge_color='y')
    # for key, value in short_paths_e.items():
    #     highlight_edges(G, plot_data['base']['ax'], value, edge_color='y')
    plot_data['base']['ax'].set_xlim(VBOUNDS[0][0], VBOUNDS[0][1])
    plot_data['base']['ax'].set_ylim(VBOUNDS[1][0], VBOUNDS[1][1])
    plot_data['base']['ax'].legend(loc='lower right')

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
            'ax': fig.add_subplot(gs[i1, i2]),
            'dynamic_arts': [],
            'gnd_arts': dict(),
            'net_plotted': False,
        }
    plt.tight_layout()
    plt.pause(0.01)

if RECORD:
    frames = []
    # FFMpegWriter = manimation.writers['ffmpeg']
    # metadata = dict(title='Movie Test', artist='Matplotlib',
    #                 comment='Movie support!')
    # writer = FFMpegWriter(fps=1, metadata=metadata)
    # writer.setup(fig, "writer_test_20.mp4", 400)

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

    new_tracks = initiator.initiate(unassociated_detections)
    tracks |= new_tracks
    bad_tracks = del_tracks(tracks, measurement_model)
    tracks -= bad_tracks
    print(f'Tracked | Initiated {len(new_tracks)} tracks | Deleted {len(bad_tracks)} tracks')

    if PLOT:
        plot(plot_data, tracks, detections, G, t_colors,
             destinations, gnd_paths, gnd_routes_e, VBOUNDS)
        plt.pause(0.0001)
    if RECORD:
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        # writer.grab_frame()

    # pr.disable()

if RECORD:
    clip = ImageSequenceClip(frames, fps=10)
    clip.write_videofile('test.mp4', codec='libx264')