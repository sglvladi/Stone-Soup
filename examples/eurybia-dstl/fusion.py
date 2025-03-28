# Bias tracker for sensors that feed detections straight to the Fusion Engine
import datetime
from copy import deepcopy
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from pyehm.plugins.stonesoup import JPDAWithEHM2

from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.deleter.error import MeasurementCovarianceBasedDeleter
from stonesoup.deleter.multi import CompositeDeleter
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.gater.distance import DistanceGater
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.hypothesiser.mfa import MFAHypothesiser
from stonesoup.hypothesiser.probability import PDAHypothesiser, PDAHypothesiserNoPrediction
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.initiator.twostate import TwoStateInitiator, TwoStateMeasurementInitiator, \
    TwoStateMeasurementInitiatorMixture
from stonesoup.measures import Mahalanobis
from stonesoup.metricgenerator.metrictables import SIAPTableGenerator
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, OrnsteinUhlenbeck, \
    NthDerivativeDecay, ConstantVelocity
from stonesoup.predictor.kalman import ExtendedKalmanPredictor, UnscentedKalmanPredictor
from stonesoup.predictor.twostate import TwoStatePredictor
from stonesoup.reader.niag import STANAGContactReader
from stonesoup.reader.track import TrackReader
from stonesoup.reader.tracklet import TrackletExtractor, PseudoMeasExtractor
from stonesoup.tracker.fuse import FuseTracker
from stonesoup.tracker.simple import MultiTargetMixtureTracker
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.numeric import Probability
from stonesoup.types.state import GaussianState
from stonesoup.types.update import Update
from stonesoup.updater.kalman import UnscentedKalmanUpdater, ExtendedKalmanUpdater
from stonesoup.updater.twostate import TwoStateKalmanUpdater

from plotting_utils import plot_gnd, plot_platform, plot_ospa, plot_gospa
from metrics import gen_metric_fuse

# Parameters
DATASET = 'Real'    # 'Sim1', 'Sim2', 'Real'
plot_coord = 'xyz'
ref_lat=49.725
ref_lon=-4.85
# stanag_msg_directory = Path(r'C:\Users\sglvladi\OneDrive\Documents\University of Liverpool\PostDoc\EURYBIA - Dstl\Data\Drop 2 - 13Feb2025\20250213_UoLExample')
# stanag_msg_directory = Path(r'C:\Users\sglvladi\OneDrive\Documents\University of Liverpool\PostDoc\EURYBIA - Dstl\Data\Drop 3 - 05Mar2025\20250305_UoL_Sim_Two_O')

stanag_config = 'NIAGSparse'
stanag_meta_header_name = 'LatencyHeader'

stanag_msg_directory = Path(r'C:\Users\sglvladi\OneDrive\Documents\University of Liverpool\PostDoc\EURYBIA - Dstl\Data\Drop 4 - 11Mar2025\20250305_UoL_Real_Three_OS')
q_factor = 0.1
decay_factor = 0.0001
# rerr = 850**2
rerr = 100**2
berr = np.radians(3)**2
snr_threshold = 14
fuse_interval = datetime.timedelta(seconds=40)
target_plat_unit_id=[(4,1)]# [(3,1), (4,1), (91,1), (92,1), (93,1)]
update_rate=datetime.timedelta(seconds=20)
init_threshold = 10
use_prior = False
use_ukf = True
bias_prior = GaussianState(StateVector([0., 0., 0., 0., 0., 0.]),
                           CovarianceMatrix(np.diag([0, 10., 0, 10., np.pi / 6, 50.]) ** 2))

if DATASET == 'Sim1':
    rx_plat_id_selects = [2, 1]
    q_factor = 0.01
    r_bias_q_factor = 1e-1
    b_bias_q_factor = np.radians(1e-4)
    decay_factor = 0.0001
    rerr = 50 ** 2
    berr = np.radians(.1) ** 2
    snr_threshold = 10
    update_rate = None
    fuse_interval = datetime.timedelta(minutes=10)
    prob_detect = 0.9
    clutter_rate = 1                                      # Mean number of clutter points per scan
    max_range = 10000                                     # Max range of sensor (meters)
    surveillance_area = np.pi*max_range**2                # Surveillance region area
    clutter_density = clutter_rate/surveillance_area      # Mean number of clutter points per unit area
    # clutter_density = 1e-15
    time_steps_since_update = 5
    target_plat_unit_id = [(3, 1)]  # [(3,1), (4,1), (91,1), (92,1), (93,1)]
    stanag_msg_directory = Path(
        r'C:\Users\sglvladi\OneDrive\Documents\University of Liverpool\PostDoc\EURYBIA - Dstl\Data\Drop 2 - 13Feb2025\20250213_UoLExample')
    xlim = [0, 25000]
    ylim = [0, 75000]
elif DATASET == 'Sim2':
    rx_plat_id_selects = [1, 2]
    q_factor = 0.01
    r_bias_q_factor = 1e-6
    b_bias_q_factor = np.radians(1e-4)
    decay_factor = 0.0001
    rerr = 200 ** 2
    berr = np.radians(1) ** 2
    snr_threshold = 10
    update_rate = None
    fuse_interval = datetime.timedelta(minutes=2)
    prob_detect = 0.9
    clutter_rate = 20  # Mean number of clutter points per scan
    max_range = 10000  # Max range of sensor (meters)
    surveillance_area = np.pi * max_range ** 2  # Surveillance region area
    clutter_density = clutter_rate / surveillance_area  # Mean number of clutter points per unit area
    # clutter_density = 1e-7
    time_steps_since_update = 5
    target_plat_unit_id = [(3,1), (4,1), (91,1), (92,1), (93,1)]
    stanag_msg_directory = Path(r'C:\Users\sglvladi\OneDrive\Documents\University of Liverpool\PostDoc\EURYBIA - Dstl\Data\Drop 3 - 05Mar2025\20250305_UoL_Sim_Two_O')
    xlim = [-15000, 15000]
    ylim = [-15000, 15000]
elif DATASET == 'Real':
    rx_plat_id_selects = [1, 2]
    q_factor = 0.01
    r_bias_q_factor = 1e-4
    b_bias_q_factor = np.radians(1e-4)
    decay_factor = 0.0001
    rerr = 200 ** 2
    berr = np.radians(3) ** 2
    snr_threshold = 14
    update_rate = datetime.timedelta(seconds=20)
    fuse_interval = datetime.timedelta(seconds=40)
    prob_detect = 0.9
    clutter_rate = 20  # Mean number of clutter points per scan
    max_range = 10000  # Max range of sensor (meters)
    surveillance_area = np.pi * max_range ** 2  # Surveillance region area
    clutter_density = clutter_rate / surveillance_area  # Mean number of clutter points per unit area
    # clutter_density = 1e-7
    time_steps_since_update = 5
    target_plat_unit_id = [(4, 1)]  # [(3,1), (4,1), (91,1), (92,1), (93,1)]
    stanag_msg_directory = Path(r'C:\Users\sglvladi\OneDrive\Documents\University of Liverpool\PostDoc\EURYBIA - Dstl\Data\Drop 4 - 11Mar2025\20250305_UoL_Real_Three_OS')
    xlim = [-15000, 15000]
    ylim = [-15000, 15000]

# Sensor trackers
readers = []
trackers = []
for i, rx_plat_id_select in enumerate(rx_plat_id_selects):
    # Detector/Reader
    contacts_reader = STANAGContactReader(stanag_msg_directory,
                                          state_vector_fields=("RelBearing", "RX2contact_range"),
                                          time_field = None,
                                          snr_threshold=snr_threshold,
                                          rerr=rerr,
                                          berr=berr,
                                          endianness = 0,
                                          stanag_msg_directory=stanag_msg_directory,
                                          reference_lat = ref_lat,
                                          reference_lon = ref_lon,
                                          with_bias=True,
                                          update_rate=update_rate
                                          )
    contacts_reader.read_stanag_files(rx_plat_id_select=rx_plat_id_select, config_subfolder=stanag_config, meta_header_name =stanag_meta_header_name)
    contacts_reader.get_stanag_ground_truth_from_SM01(target_plat_unit_id=target_plat_unit_id)
    readers.append(contacts_reader)

    # Transition model
    # bias_transition_model = CombinedLinearGaussianTransitionModel([OrnsteinUhlenbeck(q_factor, decay_factor),
    #                                                                OrnsteinUhlenbeck(q_factor, decay_factor),
    #                                                                NthDerivativeDecay(0, np.radians(.0001), decay_factor),
    #                                                                NthDerivativeDecay(0, 1e-2, decay_factor)])
    bias_transition_model = CombinedLinearGaussianTransitionModel([OrnsteinUhlenbeck(q_factor, decay_factor),
                                                                   OrnsteinUhlenbeck(q_factor, decay_factor),
                                                                   NthDerivativeDecay(0, b_bias_q_factor, decay_factor),
                                                                   NthDerivativeDecay(0, r_bias_q_factor, decay_factor)])
    # Predictor and Updater
    if not use_ukf:
        predictor = ExtendedKalmanPredictor(bias_transition_model)
        updater = ExtendedKalmanUpdater(None, True)
    else:
        predictor = UnscentedKalmanPredictor(bias_transition_model)
        updater = UnscentedKalmanUpdater(None, True)

    # Initiator components
    # hypothesiser_init = DistanceHypothesiser(predictor, updater, Mahalanobis(), 10)
    hypothesiser_init = PDAHypothesiser(predictor, updater, 1e-3, prob_detect)
    hypothesiser_init = DistanceGater(hypothesiser_init, Mahalanobis(), 10)
    data_associator_init = GNNWith2DAssignment(hypothesiser_init)
    deleter_init = UpdateTimeStepsDeleter(time_steps_since_update=time_steps_since_update)
    initiator = MultiMeasurementInitiator(bias_prior, None, deleter_init,
                                          data_associator_init, updater, init_threshold)

    # Tracker components
    deleter1 = UpdateTimeStepsDeleter(10)
    deleter2 = MeasurementCovarianceBasedDeleter([np.pi / 4, 5e6])
    deleter = CompositeDeleter([deleter1, deleter2], intersect=False)
    hypothesiser = PDAHypothesiser(predictor, updater, clutter_density, prob_detect)
    hypothesiser = DistanceGater(hypothesiser, Mahalanobis(), 10)
    data_associator = JPDAWithEHM2(hypothesiser)

    # Tracker
    bias_tracker = MultiTargetMixtureTracker(initiator, deleter, contacts_reader, data_associator, updater)
    trackers.append(TrackReader(bias_tracker, run_async=False,
                                transition_model=bias_transition_model,
                                sensor_id=i))

# Fusion Tracker
# ==============
# Transition model
transition_model = CombinedLinearGaussianTransitionModel([OrnsteinUhlenbeck(q_factor, decay_factor),
                                                          OrnsteinUhlenbeck(q_factor, decay_factor)])
# Tracklet extractor & Pseudo measurement extractor
tracklet_extractor = TrackletExtractor(trackers=trackers,
                                       transition_model=transition_model,
                                       fuse_interval=fuse_interval)
detector = PseudoMeasExtractor(tracklet_extractor, state_idx_to_use=[0,1,2,3], use_prior=use_prior)

# Predictor and Updater
two_state_predictor = TwoStatePredictor(transition_model)
two_state_updater = TwoStateKalmanUpdater(None, True)

# Hypothesiser and Data Associator
hypothesiser1 = PDAHypothesiserNoPrediction(predictor=None,
                                            updater=two_state_updater,
                                            clutter_spatial_density=Probability(-80, log_value=True),
                                            prob_detect=Probability(.9),
                                            prob_gate=Probability(0.99))
# hypothesiser1 = DistanceGater(hypothesiser1, Mahalanobis(), 100)
fuse_associator = JPDAWithEHM2(hypothesiser1)

# Initiator
prior = GaussianState(StateVector([0., 0., 0., 0.]),
                      CovarianceMatrix(np.diag([0., 10., 0., 10.]))**2)
initiator1 = TwoStateMeasurementInitiator(prior, transition_model, two_state_updater)

# Tracker
fuse_tracker = FuseTracker(initiator=initiator1, predictor=two_state_predictor,
                           updater=two_state_updater, associator=fuse_associator,
                           detector=detector, death_rate=1e-4,
                           prob_detect=Probability(.9),
                           delete_thresh=Probability(0.1))

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
all_tracks = set()
all_detections = set()
test_ground_truths = set([track for track in readers[0].ground_truth.values()])
all_tracklets = {i: set() for i in range(len(trackers)) }

timestamps = []
for time, ctracks in fuse_tracker:
    timestamps.append(time)

    all_tracks.update(ctracks)
    for i, tracker in enumerate(trackers):
        all_tracklets[i].update(tracker.current[1])

    for reader in readers:
        all_detections.update(reader.detections)

    colors = ['r', 'g', 'b']
    ax.cla()
    ax.set_xlabel('East')
    ax.set_ylabel('North')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    print(f'Time: {time} | Number of tracks: {len(ctracks)}')

    ax.plot([], [], 'bo', label='TX')
    ax.plot([], [], 'mo', label='RX')
    for reader in readers:
        plot_platform(reader.truth_TX, ref_lat, ref_lon, ax, plot_coord, 'b', )
        plot_platform(reader.truth_RX, ref_lat, ref_lon, ax, plot_coord, 'm', )

    # Pl
    plot_gnd(test_ground_truths, ref_lat, ref_lon, ax, plot_coord)
    ax.plot([], [], 'bx', label='Detections')
    for detection in all_detections:
        x, y = detection.measurement_model.inverse_function(detection)[[0, 2]]
        ax.plot(x, y, 'bx')
    for i, (tracklets, color) in enumerate(zip(tracklet_extractor.current[1], colors)):
        ax.plot([], [], f':.{color}', label=f'Sensor {i} Tracklets')
        for tracklet in tracklets:
            data = np.array([s.mean for s in tracklet.states if isinstance(s, Update)])
            if data.shape[1] > 8:
                idx = [6, 8]
            else:
                idx = [4, 6]
            plt.plot(data[:, idx[0]], data[:, idx[1]], f':.{color}')

    plt.plot([], [], '-*m', label='Fused Tracks')
    for track in ctracks:
        data = np.array([state.state_vector for state in track])
        plt.plot(data[:, 4], data[:, 6], '-*m')

    # ax2.cla()
    # for track in ctracks:
    #     data = np.array([state.state_vector for state in track.states])
    #     num_steps = len(data)
    #     ax2.plot([i for i in range(num_steps)], data[:, -2], 'r-')
    #     ax2.plot([i for i in range(num_steps)], data[:, -1], 'c-')
    ax.legend()
    plt.pause(.1)

if DATASET in ['Sim1', 'Real']:
    tracks ={sorted(all_tracks, key=lambda x: len(x))[-1]}
    filtered_tracklets = {key: {sorted(t, key=lambda x: len(x))[-1]} for key, t in all_tracklets.items()}
else:
    tracks = all_tracks
    filtered_tracklets = all_tracklets
ospa_metrics = gen_metric_fuse('OSPA', timestamps, test_ground_truths, tracks, filtered_tracklets)
gospa_metrics = gen_metric_fuse('GOSPA', timestamps, test_ground_truths, tracks, filtered_tracklets)
siap_metrics = gen_metric_fuse('SIAP', timestamps, test_ground_truths, tracks, filtered_tracklets)

for key, siap_metric in siap_metrics.items():
    siap_averages = {metric for metric in siap_metric
                     if metric.title.startswith("SIAP") and not metric.title.endswith(" at times")}
    siap_time_based = {metric for metric in siap_metric if metric.title.endswith(' at times')}
    _ = SIAPTableGenerator(siap_averages).compute_metric()
    plt.title(f'{key} SIAP Averages')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# for key, metric in ospa_metrics.items():
#     plot_ospa(metric, f'OSPA {key}', ax=ax)
for key, metric in gospa_metrics.items():
    plot_gospa(metric, f'GOSPA {key}', ax=ax)
ax.set_ylabel("(G)OSPA distance")
ax.tick_params(labelbottom=False)
_ = ax.set_xlabel("Time")
plt.legend()
# pickle.dump({'ospa': ospa, 'gospa': gospa}, open('./output/jpda_metrics.pickle', 'wb'))

plt.show(block=True)