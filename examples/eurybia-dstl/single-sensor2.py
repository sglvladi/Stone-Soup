# Bias tracker for sensors that feed detections straight to the Fusion Engine
import datetime
from copy import deepcopy
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from pyehm.plugins.stonesoup import JPDAWithEHM2

import pymap3d as pm

from stonesoup.dataassociator.mfa import MFADataAssociator
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.deleter.error import MeasurementCovarianceBasedDeleter
from stonesoup.deleter.multi import CompositeDeleter
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.gater.distance import DistanceGater
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.hypothesiser.mfa import MFAHypothesiser
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.initiator.simple import MultiMeasurementInitiator, MultiMeasurementInitiatorMixture
from stonesoup.measures import Mahalanobis
from stonesoup.metricgenerator.manager import SimpleManager
from stonesoup.metricgenerator.metrictables import SIAPTableGenerator
from stonesoup.metricgenerator.ospametric import GOSPAMetric
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, OrnsteinUhlenbeck, \
    NthDerivativeDecay, ConstantVelocity
from stonesoup.predictor.kalman import ExtendedKalmanPredictor, UnscentedKalmanPredictor
from stonesoup.reader.niag import STANAGContactReader
from stonesoup.tracker.simple import MultiTargetMixtureTracker, MultiTargetMultiMixtureTracker
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.state import GaussianState
from stonesoup.updater.kalman import ExtendedKalmanUpdater, UnscentedKalmanUpdater
from plotting_utils import plot_tracks, plot_gnd, plot_platform, plot_gospa
from metrics import prepare_tracks, gen_metric

DATASET = 'Sim1'
plot_coord = 'xyz'
ref_lat=49.725
ref_lon=-4.85
# stanag_msg_directory = Path(r'C:\Users\sglvladi\OneDrive\Documents\University of Liverpool\PostDoc\EURYBIA - Dstl\Data\Drop 2 - 13Feb2025\20250213_UoLExample')
# stanag_msg_directory = Path(r'C:\Users\sglvladi\OneDrive\Documents\University of Liverpool\PostDoc\EURYBIA - Dstl\Data\Drop 3 - 05Mar2025\20250305_UoL_Sim_Two_O')
stanag_msg_directory = Path(r'C:\Users\sglvladi\OneDrive\Documents\University of Liverpool\PostDoc\EURYBIA - Dstl\Data\Drop 4 - 11Mar2025\20250305_UoL_Real_Three_OS')
stanag_config = 'NIAGSparse'
stanag_meta_header_name = 'LatencyHeader'
rx_plat_id_select = 2

q_factor = 0.01
decay_factor = 0.0001
# rerr = 100**2
rerr = 100**2
berr = np.radians(3)**2
snr_threshold = 14
target_plat_unit_id=[(4,1)]# [(3,1), (4,1), (91,1), (92,1), (93,1)]
update_rate=datetime.timedelta(seconds=20)
init_threshold = 10
prob_detect = 0.9
clutter_density = 5e-3
time_steps_since_update = 5
use_ukf = True
use_mfa = True
slide_window = 2
bias_prior = GaussianState(StateVector([0., 0., 0., 0., 0., 0.]),
                           CovarianceMatrix(np.diag([0, 10., 0, 10., np.pi/6, 50.])**2))
max_range_std = 1e3
max_bearing_std = np.radians(15)
num_detections = []
if DATASET == 'Sim1':
    rx_plat_id_selects = [1, 2]
    q = 0.01
    q_bias_range = 1e-1
    q_bias_bearing = np.radians(1e-7)
    decay_factor = 0.0001
    sigma_r = 200.
    sigma_b = np.radians(3.)
    snr_threshold = 10
    update_rate = None
    fuse_interval = datetime.timedelta(minutes=10)
    prob_detect = 1.
    clutter_rate = 0.0001                                      # Mean number of clutter points per scan
    max_range = 10000                                     # Max range of sensor (meters)
    surveillance_area = np.pi*max_range**2                # Surveillance region area
    clutter_density = clutter_rate/surveillance_area      # Mean number of clutter points per unit area
    init_clutter_density = 1e-15
    time_steps_since_update = 5
    max_range_std = 3e3
    init_threshold = 2
    target_plat_unit_id = [(3, 1)]  # [(3,1), (4,1), (91,1), (92,1), (93,1)]
    stanag_msg_directory = Path(
        r'C:\Users\sglvladi\OneDrive\Documents\University of Liverpool\PostDoc\EURYBIA - Dstl\Data\Drop 2 - 13Feb2025\20250213_UoLExample')
    xlim = [0, 25000]
    ylim = [0, 75000]
elif DATASET == 'Sim2':
    rx_plat_id_selects = [1, 2]
    q = 0.01                                 # Process noise (q)
    q_bias_range = 1e-1                      # Bias process noise for range (q_b^r)
    q_bias_bearing = np.radians(1e-7)        # Bias process noise for bearing (q_b^b)
    decay_factor = 0.0001                    # Decay factor (K)
    sigma_r = 200.                            # Range measurement noise (sigma_r)
    sigma_b = np.radians(1.)                  # Bearing measurement noise (sigma_b)
    snr_threshold = 10
    update_rate = None
    fuse_interval = datetime.timedelta(minutes=2)
    prob_detect = 0.9
    clutter_rate = 10  # Mean number of clutter points per scan
    max_range = 35000  # Max range of sensor (meters)
    surveillance_area = np.pi * max_range ** 2  # Surveillance region area
    clutter_density = clutter_rate / surveillance_area  # Mean number of clutter points per unit area
    init_clutter_density = 1e-3
    time_steps_since_update = 5
    target_plat_unit_id = [(3,1), (4,1), (91,1), (92,1), (93,1)]
    stanag_msg_directory = Path(r'C:\Users\sglvladi\OneDrive\Documents\University of Liverpool\PostDoc\EURYBIA - Dstl\Data\Drop 3 - 05Mar2025\20250305_UoL_Sim_Two_O')
    xlim = [-15000, 15000]
    ylim = [-15000, 15000]
elif DATASET == 'Real':
    rx_plat_id_selects = [1, 2]
    q = 0.01
    q_bias_range = 1e-1
    q_bias_bearing = np.radians(1e-7)
    decay_factor = 0.0001
    sigma_r = 200.
    sigma_b = np.radians(1.)
    snr_threshold = 14
    update_rate = datetime.timedelta(seconds=20)
    fuse_interval = datetime.timedelta(seconds=40)
    prob_detect = 0.9
    clutter_rate = 20  # Mean number of clutter points per scan
    max_range = 35000  # Max range of sensor (meters)
    surveillance_area = np.pi * max_range ** 2  # Surveillance region area
    clutter_density = clutter_rate / surveillance_area  # Mean number of clutter points per unit area
    init_clutter_density = 1e-3
    time_steps_since_update = 5
    max_range_std = 1e3
    max_bearing_std = np.radians(45)
    target_plat_unit_id = [(4, 1)]  # [(3,1), (4,1), (91,1), (92,1), (93,1)]
    stanag_msg_directory = Path(r'C:\Users\sglvladi\OneDrive\Documents\University of Liverpool\PostDoc\EURYBIA - Dstl\Data\Drop 4 - 11Mar2025\20250305_UoL_Real_Three_OS')
    xlim = [-15000, 15000]
    ylim = [-15000, 15000]

# Transition model
bias_transition_model = CombinedLinearGaussianTransitionModel([OrnsteinUhlenbeck(q_factor, decay_factor),
                                                               OrnsteinUhlenbeck(q_factor, decay_factor),
                                                               NthDerivativeDecay(0, q_bias_bearing, decay_factor),
                                                               NthDerivativeDecay(0, q_bias_range, decay_factor)])
# Predictor and Updater
# Predictor and Updater
if not use_ukf:
    predictor = ExtendedKalmanPredictor(bias_transition_model)
    updater = ExtendedKalmanUpdater(None, True)
else:
    predictor = UnscentedKalmanPredictor(bias_transition_model)
    updater = UnscentedKalmanUpdater(None, True)

# Initiator components
hypothesiser_init = PDAHypothesiser(predictor, updater, init_clutter_density, prob_detect)
hypothesiser_init = DistanceGater(hypothesiser_init, Mahalanobis(), 10)
data_associator_init = GNNWith2DAssignment(hypothesiser_init)
deleter_init1 = UpdateTimeStepsDeleter(time_steps_since_update=time_steps_since_update)
deleter_init2 = MeasurementCovarianceBasedDeleter([max_bearing_std**2, max_range_std**2])
deleter_init = CompositeDeleter([deleter_init1, deleter_init2], intersect=False)
if use_mfa:
    initiator = MultiMeasurementInitiatorMixture(bias_prior, None, deleter_init,
                                                 data_associator_init, updater, init_threshold)
else:
    initiator = MultiMeasurementInitiator(bias_prior, None, deleter_init,
                                          data_associator_init, updater, init_threshold)

# Tracker components
deleter1 = UpdateTimeStepsDeleter(10)
deleter2 = MeasurementCovarianceBasedDeleter([max_bearing_std**2, max_range_std**2])
deleter = CompositeDeleter([deleter1, deleter2], intersect=False)
hypothesiser = PDAHypothesiser(predictor, updater, clutter_density, prob_detect)
hypothesiser = DistanceGater(hypothesiser, Mahalanobis(), 10)
if use_mfa:
    hypothesiser = MFAHypothesiser(hypothesiser)
    data_associator = MFADataAssociator(hypothesiser, slide_window=slide_window)
else:
    data_associator = JPDAWithEHM2(hypothesiser)

# Detector/Reader
contacts_reader = STANAGContactReader(stanag_msg_directory,
                                      state_vector_fields=("RelBearing", "RX2contact_range"),
                                      time_field = None,
                                      snr_threshold=snr_threshold,
                                      rerr=sigma_r**2,
                                      berr=sigma_b**2,
                                      endianness = 0,
                                      stanag_msg_directory=stanag_msg_directory,
                                      reference_lat = ref_lat,
                                      reference_lon = ref_lon,
                                      with_bias=True,
                                      update_rate=update_rate
                                      )
contacts_reader.read_stanag_files(rx_plat_id_select=rx_plat_id_select, config_subfolder=stanag_config, meta_header_name =stanag_meta_header_name)
contacts_reader.get_stanag_ground_truth_from_SM01(target_plat_unit_id=target_plat_unit_id)

# Tracker
if use_mfa:
    bias_tracker = MultiTargetMultiMixtureTracker(initiator, deleter, contacts_reader, data_associator, updater)
else:
    bias_tracker = MultiTargetMixtureTracker(initiator, deleter, contacts_reader, data_associator, updater)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
all_tracks = set()
all_detections = set()
test_ground_truths = set([track for track in contacts_reader.ground_truth.values()])

metric = GOSPAMetric(p=1, c=100)
metric_manager = SimpleManager([metric])

fig2 = plt.figure(figsize=(10, 10))
ax2, ax3 = fig2.subplots(2, 1)
ax.set_xlim(xlim)
ax.set_ylim(ylim)

timestamps = []
for time, ctracks in bias_tracker:
    timestamps.append(time)

    num_detections.append(len(contacts_reader.detections))
    all_tracks.update(ctracks)
    all_detections.update(contacts_reader.detections)
    detections = contacts_reader.detections

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.cla()
    ax.set_xlabel('East')
    ax.set_ylabel('North')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # ax.set_xlim([0, 25000])
    # ax.set_ylim([0, 75000])
    print(f'Time: {time} | Number of tracks: {len(ctracks)} | Number of detections: {len(detections)}')
    plot_gnd(test_ground_truths, ref_lat, ref_lon, ax, plot_coord)
    plot_platform(contacts_reader.truth_TX, ref_lat, ref_lon, ax, plot_coord, 'b', 'TX')
    plot_platform(contacts_reader.truth_RX, ref_lat, ref_lon, ax, plot_coord, 'm', 'RX')
    for detection in all_detections:
        inv_det = detection.measurement_model.inverse_function(detection)
        ax.plot(inv_det[0], inv_det[2], 'bx')
    plot_tracks(ctracks, ax=ax)
    plot_tracks(bias_tracker.initiator.holding_tracks, ax=ax, color='y')

    ax2.cla()
    for track in ctracks:
        data = np.array([state.state_vector for state in track.states])
        num_steps = len(data)
        ax2.plot([i for i in range(num_steps)], data[:, -2], 'r-')
        sd = np.sqrt(np.squeeze([state.covar[-2, -2] for state in track.states]))
        ax2.fill_between([i for i in range(num_steps)], data[:, -2].ravel() - sd, data[:, -2].ravel() + sd, facecolor='g', alpha=0.5)
        ax3.plot([i for i in range(num_steps)], data[:, -1], 'c-')
        sd = np.sqrt(np.squeeze([state.covar[-1, -1] for state in track.states]))
        ax3.fill_between([i for i in range(num_steps)], data[:, -1].ravel() - sd, data[:, -1].ravel() + sd, facecolor='g', alpha=0.5)


    plt.pause(.1)

print(f'Mean number of detections: {np.mean(np.array(num_detections)-5)}')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

all_gnd = deepcopy(test_ground_truths)
for gnd in all_gnd:
    gnd.states = [state for state in gnd.states if state.timestamp in timestamps]
filtered_tracks = prepare_tracks(DATASET, all_gnd, all_tracks)
gospa_metric = gen_metric('GOSPA', filtered_tracks, all_gnd)
siap_metric = gen_metric('SIAP', filtered_tracks, all_gnd)
siap_averages = {metric for metric in siap_metric
                 if metric.title.startswith("SIAP") and not metric.title.endswith(" at times")}
siap_time_based = {metric for metric in siap_metric if metric.title.endswith(' at times')}
_ = SIAPTableGenerator(siap_averages).compute_metric()
plt.title(f'SIAP Averages')
plot_gospa(gospa_metric, f'GOSPA', ax=ax)

# ax.plot(timestamps, ospa, label='OSPA')
# ax.plot(timestamps, gospa['distance'], label='GOSPA')
ax.set_ylabel("(G)OSPA distance")
ax.tick_params(labelbottom=False)
_ = ax.set_xlabel("Time")
plt.legend()
# pickle.dump({'ospa': ospa, 'gospa': gospa}, open('./output/jpda_metrics.pickle', 'wb'))
plt.show()

