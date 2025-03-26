# Bias tracker for sensors that feed detections straight to the Fusion Engine
import datetime
from copy import deepcopy
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from pyehm.plugins.stonesoup import JPDAWithEHM2

import pymap3d as pm

from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.deleter.error import MeasurementCovarianceBasedDeleter
from stonesoup.deleter.multi import CompositeDeleter
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.gater.distance import DistanceGater
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.measures import Mahalanobis
from stonesoup.metricgenerator.manager import SimpleManager
from stonesoup.metricgenerator.ospametric import GOSPAMetric
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, OrnsteinUhlenbeck, \
    NthDerivativeDecay, ConstantVelocity
from stonesoup.predictor.kalman import ExtendedKalmanPredictor, UnscentedKalmanPredictor
from stonesoup.reader.niag import STANAGContactReader
from stonesoup.tracker.simple import MultiTargetMixtureTracker
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.state import GaussianState
from stonesoup.updater.kalman import ExtendedKalmanUpdater, UnscentedKalmanUpdater
from plotting_utils import plot_tracks, plot_gnd, plot_platform

plot_coord = 'xyz'
ref_lat=49.725
ref_lon=-4.85
# stanag_msg_directory = Path(r'C:\Users\sglvladi\OneDrive\Documents\University of Liverpool\PostDoc\EURYBIA - Dstl\Data\Drop 2 - 13Feb2025\20250213_UoLExample')
# stanag_msg_directory = Path(r'C:\Users\sglvladi\OneDrive\Documents\University of Liverpool\PostDoc\EURYBIA - Dstl\Data\Drop 3 - 05Mar2025\20250305_UoL_Sim_Two_O')
stanag_msg_directory = Path(r'C:\Users\sglvladi\OneDrive\Documents\University of Liverpool\PostDoc\EURYBIA - Dstl\Data\Drop 4 - 11Mar2025\20250305_UoL_Real_Three_OS')
stanag_config = 'NIAGSparse'
stanag_meta_header_name = 'LatencyHeader'
rx_plat_id_select = 1

q_factor = 0.001
decay_factor = 0.0001
# rerr = 100**2
rerr = 200**2
berr = np.radians(3)**2
prob_detect = 0.9
clutter_density = 1e-3
time_steps_since_update = 5
use_ukf = True
bias_prior = GaussianState(StateVector([0., 0., 0., 0., 0., 0.]),
                           CovarianceMatrix(np.diag([0, 10., 0, 10., np.pi/6, 50.])**2))

# Transition model
bias_transition_model = CombinedLinearGaussianTransitionModel([OrnsteinUhlenbeck(q_factor, decay_factor),
                                                               OrnsteinUhlenbeck(q_factor, decay_factor),
                                                               NthDerivativeDecay(0, np.radians(.0001), decay_factor),
                                                               NthDerivativeDecay(0, 1e-1, decay_factor)])
# Predictor and Updater
# Predictor and Updater
if not use_ukf:
    predictor = ExtendedKalmanPredictor(bias_transition_model)
    updater = ExtendedKalmanUpdater(None, True)
else:
    predictor = UnscentedKalmanPredictor(bias_transition_model)
    updater = UnscentedKalmanUpdater(None, True)

# Initiator components
hypothesiser_init = PDAHypothesiser(predictor, updater, clutter_density, prob_detect)
hypothesiser_init = DistanceGater(hypothesiser_init, Mahalanobis(), 10)
data_associator_init = GNNWith2DAssignment(hypothesiser_init)
deleter_init = UpdateTimeStepsDeleter(time_steps_since_update=time_steps_since_update)
initiator = MultiMeasurementInitiator(bias_prior, None, deleter_init,
                                      data_associator_init, updater, 10)

# Tracker components
deleter1 = UpdateTimeStepsDeleter(10)
deleter2 = MeasurementCovarianceBasedDeleter([np.pi/4, 5e6])
deleter = CompositeDeleter([deleter1, deleter2], intersect=False)
hypothesiser = PDAHypothesiser(predictor, updater, clutter_density, prob_detect)
hypothesiser = DistanceGater(hypothesiser, Mahalanobis(), 10)
data_associator = JPDAWithEHM2(hypothesiser)

# Detector/Reader
contacts_reader = STANAGContactReader(stanag_msg_directory,
                                      state_vector_fields=("RelBearing", "RX2contact_range"),
                                      time_field = None,
                                      snr_threshold=14,
                                      rerr=rerr,
                                      berr=berr,
                                      endianness = 0,
                                      stanag_msg_directory=stanag_msg_directory,
                                      reference_lat = ref_lat,
                                      reference_lon = ref_lon,
                                      with_bias=True,
                                      update_rate=datetime.timedelta(seconds=20)
                                      )
contacts_reader.read_stanag_files(rx_plat_id_select=rx_plat_id_select, config_subfolder=stanag_config, meta_header_name =stanag_meta_header_name)
contacts_reader.get_stanag_ground_truth_from_SM01(target_plat_unit_id=[(4,1)])

# Tracker
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

timestamps = []
for time, ctracks in bias_tracker:
    timestamps.append(time)

    all_tracks.update(ctracks)
    all_detections.update(contacts_reader.detections)
    detections = contacts_reader.detections

    ax.cla()
    ax.set_xlabel('East')
    ax.set_ylabel('North')
    # ax.set_xlim([0, 25000])
    # ax.set_ylim([0, 75000])
    ax.set_xlim([-25000, 25000])
    ax.set_ylim([-25000, 25000])
    print(f'Time: {time} | Number of tracks: {len(ctracks)}')
    plot_gnd(test_ground_truths, ref_lat, ref_lon, ax, plot_coord)
    plot_platform(contacts_reader.truth_TX, ref_lat, ref_lon, ax, plot_coord, 'b', 'TX')
    plot_platform(contacts_reader.truth_RX, ref_lat, ref_lon, ax, plot_coord, 'm', 'RX')
    for detection in detections:
        inv_det = detection.measurement_model.inverse_function(detection)
        ax.plot(inv_det[0], inv_det[2], 'bx')
    plot_tracks(ctracks, ax=ax)
    # plot_tracks(bias_tracker.initiator.holding_tracks, ax=ax)

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


all_gnd = deepcopy(test_ground_truths)
for gnd in all_gnd:
    gnd.states = [state for state in gnd.states if state.timestamp in timestamps]

from stonesoup.metricgenerator.ospametric import OSPAMetric
from stonesoup.measures import Euclidean

ospa_generator = OSPAMetric(c=1000, p=1, measure=Euclidean([0, 2]))
ospa_metric = ospa_generator.compute_over_time(ospa_generator.extract_states(all_tracks),
                                               ospa_generator.extract_states(all_gnd))
gospa_generator = GOSPAMetric(c=1000, p=1, measure=Euclidean([0, 2]))
gospa_metric = gospa_generator.compute_over_time(gospa_generator.extract_states(all_tracks),
                                                 gospa_generator.extract_states(all_gnd))

ospa = np.array([i.value for i in ospa_metric.value])
gospa = {'distance': 0.0,
             'localisation': 0.0,
             'missed': 0,
             'false': 0}
for key in gospa:
    metric_mat = np.array(
        [i.value[key] for i in gospa_metric.value])
    gospa[key] = metric_mat
timestamps = [i.timestamp for i in ospa_metric.value]
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(timestamps, ospa, label='OSPA')
ax.plot(timestamps, gospa['distance'], label='GOSPA')
ax.set_ylabel("(G)OSPA distance")
ax.tick_params(labelbottom=False)
_ = ax.set_xlabel("Time")
plt.legend()
# pickle.dump({'ospa': ospa, 'gospa': gospa}, open('./output/jpda_metrics.pickle', 'wb'))
plt.show()



plt.show(block=True)