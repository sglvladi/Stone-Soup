from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import pickle

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from stonesoup.custom.graph import load_graph_dict, dict_to_graph, shortest_path, \
    get_xy_from_range_edge, graph_to_dict, CustomDiGraph, edge_resample, get_xy_from_range_endnodes
from stonesoup.custom.plotting import plot_network, highlight_nodes, highlight_edges, \
    remove_artists, plot_polygons, plot_short_paths_e
from stonesoup.custom.simulation import simulate_gnd, simulate_detections
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.gater.filtered import FilteredDetectionsGater
from stonesoup.initiator.destination import DestinationBasedInitiator, \
    DestinationBasedInitiatorAimpoint
from stonesoup.models.transition.destination import DestinationTransitionModel, \
    AimpointTransitionModel
from stonesoup.models.measurement.destination import DestinationMeasurementModel, \
    AimpointMeasurementModel
from stonesoup.predictor.particle import ParticlePredictor2
from stonesoup.resampler.particle import SystematicResampler2, ESSResampler2
from stonesoup.types.array import StateVector
from stonesoup.types.detection import Detection
from stonesoup.types.state import ParticleState2
from stonesoup.types.update import Update
from stonesoup.updater.particle import ParticleUpdater2
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.measures import Mahalanobis

# from pybsp.bsp import BSP
from bsppy import BSPTree

# Load Graph


def del_tracks(tracks, meas_model, thresh=10000000000):
    d_tracks = set()
    for track in tracks:
        t_sv = meas_model.function(track.state, noise=0)
        st = ParticleState2(t_sv, track.state.weights)
        if not any(isinstance(state, Update) and state.hypothesis for state in track[-120:]):
            d_tracks.add(track)
        # if (np.trace(st.covar) > thresh) \
        #         or not any(isinstance(state, Update) and state.hypothesis for state in track[-4:]):

    return d_tracks


def plot(meta, tracks, detections, G, t_colors, destinations, VBOUNDS):

    # Plot
    ax = meta['base']['ax']
    remove_artists(meta['base']['dynamic_arts'])
    meta['base']['dynamic_arts'] = []
    detection_data = np.array([detection.state_vector for detection in detections])

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

    for i, track in enumerate(tracks | deleted_tracks):
        data = track.state.particles
        # xy = get_xy_from_range_edge(data[0, :], data[2, :], G)
        a = data[[5, 6], :]
        am1 = data[[7, 8], :]
        # xy = get_xy_from_range_endnodes(data[0, :], am1, a)
        # xy2 = get_xy_from_sv(data2, short_paths_e, S)
        est = np.array([[], []])
        for state in track:
            if not isinstance(state, Update):
                continue
            data = state.particles
            a = data[[5, 6], :]
            am1 = data[[7, 8], :]
            xy = get_xy_from_range_endnodes(data[0, :], am1, a)
            est = np.append(est, np.atleast_2d(np.mean(np.array(xy), axis=1)).T, axis=1)

        try:
            x_mean = np.mean(np.array(xy[0, :]))
        except:
            a = 2
        y_mean = np.mean(np.array(xy[1, :]))

        meta['base']['dynamic_arts'] += ax.plot(est[0,:], est[1,:], '-', color=t_colors[track.id])
        meta['base']['dynamic_arts'] += ax.plot(xy[0, :], xy[1, :], '.', label="Track {}".format(track.id), color=t_colors[track.id])
        meta['base']['dynamic_arts'].append(ax.text(x_mean, y_mean, f'{track.id}', color='r'))
        id = track.id
        for i2, val in meta['sub'].items():
            if i2 < 5:
                ax2 = meta['sub'][i2]['ax']
                meta['sub'][i2]['dynamic_arts'] += ax2.plot(xy[0, :], xy[1, :], '.', color=t_colors[track.id])
                meta['sub'][i2]['dynamic_arts'] += ax2.plot(est[0, :], est[1, :], '-',
                                                        color=t_colors[track.id])
                meta['sub'][i2]['dynamic_arts'] += ax2.plot(detection_data[:, 0], detection_data[:, 1], 'xc')
                text_art = ax2.text(x_mean, y_mean, f'{track.id}', color='r')
                text_art.set_clip_on(True)
                meta['sub'][i2]['dynamic_arts'].append(text_art)
                if id == i2:
                    ax2.set_xlim(x_mean - zoom, x_mean + zoom)
                    ax2.set_ylim(y_mean - zoom, y_mean + zoom)
                    ax2.set_title('Track {}'.format(track.id))

        i2 = 5 + id
        ax2 = meta['sub'][i2]['ax']
        ax2.cla()
        v_dest, vd_ind, vd_counts = np.unique(data[3, :], return_counts=True, return_index=True)
        port_names = \
        ports.loc[ports['Node'].isin(v_dest)].sort_values('Node').drop_duplicates(subset='Node')[
            'NAME'].to_list()
        barlist = ax2.bar(port_names, vd_counts / np.sum(vd_counts))
        try:
            idx = port_names.index(destination_ports[track.id][0])
            barlist[idx].set_color('m')
        except:
            pass
        ax2.set_title(f'Destination Distribution')
        ax2.tick_params(axis='x', labelrotation=90, labelsize=5)
        # plt.xticks(rotation=90, fontsize=5)
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
num_destinations = 20       # Number of possible destinations
num_particles = 1000        # Number of particles to use in SMC sampler
speed = 10                  # Target speed
P_D = 0.95                  # Probability of detection
lambda_FA = 0               # Clutter density
PLOT = True                # Set True/False to enable/disable plotting
RECORD = PLOT and True     # Set True/False to enable/disable recording
record_counter = 0          # Counter for recording
record_interval = 1         # Record every n-th frame
zoom = 500000
LOAD = True                # Set True/False to enable/disable loading data from file

data_path = Path(r'C:\Users\sglvladi\OneDrive\Documents\GitHub\StoneSoup-sglvladi\examples\cadmuri\data\ais\exact-earth\all_data.pickle')
all_data_dict = pickle.load(open(data_path, 'rb'))
del all_data_dict[(('BISSAU', 'GW'), ('TANGER', 'MA'))]
del all_data_dict[(('PUERTO PRODECO', 'CO'), ('VLISSINGEN', 'NL'))]
timestamp_init = datetime.now()
all_data = [data for data in all_data_dict.values()]
for i, data in enumerate(all_data):
    dt = data[:, 2] - data[0, 2]
    data[:, 2] = timestamp_init + dt
    current_time = data[0, 2].replace(second=0, microsecond=0)
    data_tmp = data[0, :]
    for row in data:
        timestamp = row[2].replace(second=0, microsecond=0)
        if timestamp - current_time > timedelta(minutes=1):
            data_tmp = np.vstack((data_tmp, row))
            current_time = timestamp

    all_data[i] = np.hstack((data_tmp, np.full((data_tmp.shape[0], 1), i)))

all_data_stacked = np.vstack(all_data)
all_data_stacked_sorted = all_data_stacked[all_data_stacked[:, 2].argsort()]
timestamps = np.unique(all_data_stacked_sorted[:, 2])

scans = []
for timestamp in timestamps:
    data = all_data_stacked_sorted[all_data_stacked_sorted[:, 2] == timestamp]
    detections = set()
    for row in data:
        det_sv = StateVector([row[0], row[1]])
        metadata = {"gnd_id": row[3]}
        detection = Detection(state_vector=det_sv, timestamp=timestamp, metadata=metadata)
        detections.add(detection)
    scans.append((timestamp, detections))

# Load the network
path = r'C:\Users\sglvladi\OneDrive\Workspace\PostDoc\CADMURI\Python\PyBSP\data\graphs\custom_digraph_v4.1.4'
G = CustomDiGraph.load(path)
# G.save(path)
# G._rtree = None
S = G.as_dict()
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()


# Load the polygons
path2 =r'C:\Users\sglvladi\OneDrive\Workspace\PostDoc\CADMURI\Python\PyBSP\data\ports_polygons_v4.1.2.pickle'
ports, new_polygons = pickle.load(open(path2, 'rb'))

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

VBOUNDS = ((-22041259.177068166, 22041259.177068166), (-11818985.537814114, 19893557.410806347))

source_ports = [data[0] for data in iter(all_data_dict)]
destination_ports = [data[1] for data in iter(all_data_dict)]
print(destination_ports)

if not LOAD:
    t_sources = []
    t_destinations = []
    t_colors = []
    for source_port, destination_port in zip(source_ports, destination_ports):
        source = ports.loc[(ports['NAME'] == source_port[0]) & (ports['COUNTRY'] == source_port[1])]['Node'].to_list()[0]
        destination = \
        ports.loc[(ports['NAME'] == destination_port[0]) & (ports['COUNTRY'] == destination_port[1])]['Node'].to_list()[0]
        t_sources.append(source)
        t_destinations.append(destination)
        t_colors.append(np.random.rand(3, ))
    feed = t_destinations
    feed_tmp = set(ports['Node'].to_list()) - set(feed)
    destinations = feed + list(
        np.random.choice(list(feed_tmp), (num_destinations - len(feed),), False))
    short_paths_n, short_paths_e = G.shortest_path(t_sources, destinations)

    pickle.dump([short_paths_n, short_paths_e, t_sources, t_destinations, t_colors, destinations],
                open(f'./data/ais_exact_earth_multi_track.pickle', 'wb'))
else:
    short_paths_n, short_paths_e, t_sources, t_destinations, t_colors, destinations = \
        pickle.load(open(f'./data/ais_exact_earth_multi_track.pickle', 'rb'))
    G.short_paths_n = short_paths_n
    G.short_paths_e = short_paths_e

# Measurement model
mapping = [0, 1]
R = np.eye(2)*(2e3**2)
measurement_model = AimpointMeasurementModel(ndim_state=4, mapping=mapping, noise_covar=R, graph=G)


bsptree = BSPTree.load(r'C:\Users\sglvladi\source\repos\bsp\main\data\trees\global_even',
                       'Stage2', 'final')

# Transition model
transition_model = AimpointTransitionModel(speed, G, bsptree, aimpoint_sample_covar=np.diag([3e3**2, 3e3**2]))


# Predictor
predictor = ParticlePredictor2(transition_model)

# Updater
resampler = SystematicResampler2()
updater = ParticleUpdater2(measurement_model, resampler)

# Hypothesiser and Data-Associator
hypothesiser = DistanceHypothesiser(predictor, updater, Mahalanobis(), 20)
hypothesiser = FilteredDetectionsGater(hypothesiser, metadata_filter='gnd_id')
associator = GNNWith2DAssignment(hypothesiser)

deleter = UpdateTimeStepsDeleter(120)
initiator = DestinationBasedInitiatorAimpoint(measurement_model, num_particles, speed, G, bsptree)


# Initiate tracks
tracks = set()
deleted_tracks = set()

if PLOT:
    # Initiate plotting grid
    fig = plt.figure(figsize=(27, 12))
    gs = fig.add_gridspec(2, 9)

    plot_data = dict()
    plot_data['base'] = {
        'ax': fig.add_subplot(gs[:, 0:4]),
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
    for i in range(10):
        id = i
        if 0 <= id < 5:
            i1 = 0
            i2 = 4+id
        else:
            i1 = 1
            i2 = 4 + id - 5
        plot_data['sub'][id] = {
            'ax': fig.add_subplot(gs[i1, i2]),
            'dynamic_arts': [],
            'gnd_arts': dict(),
            'net_plotted': False,
        }
    plt.tight_layout()
    fig.show()

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
            # Edge resampling
            hypothesis.prediction, resample_inds = edge_resample(hypothesis.prediction, hypothesis.measurement,
                                                      measurement_model, G,
                                                      bsptree,
                                                      transition_model.aimpoint_sample_covar)
            state_post = updater.update(hypothesis)
            track.append(state_post)
            associated_detections.add(hypothesis.measurement)
        else:
            track.append(hypothesis.prediction)

    unassociated_detections = detections - associated_detections

    new_tracks = initiator.initiate(unassociated_detections)
    tracks |= new_tracks
    bad_tracks = del_tracks(tracks, measurement_model)
    deleted_tracks |= bad_tracks
    tracks -= bad_tracks
    print(f'Tracked | Initiated {len(new_tracks)} tracks | Deleted {len(bad_tracks)} tracks')

    if PLOT:
        plot(plot_data, tracks, detections, G, t_colors,
             destinations, VBOUNDS)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        # plt.pause(0.0001)
    if RECORD:
        record_interval += 1
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        if record_interval >= 5000:
            clip = ImageSequenceClip(frames, fps=10)
            clip.write_videofile(f'exact_earth_multi_3_{record_counter}.mp4', codec='libx264')
            clip = None
            record_counter += 1
            record_interval = 0
            frames = []
        # writer.grab_frame()

    # pr.disable()

if RECORD:
    clip = ImageSequenceClip(frames, fps=10)
    clip.write_videofile(f'exact_earth_multi_3_{record_counter}.mp4', codec='libx264')