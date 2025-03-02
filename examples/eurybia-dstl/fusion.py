# Bias tracker for sensors that feed detections straight to the Fusion Engine
import datetime
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
from stonesoup.updater.kalman import UnscentedKalmanUpdater
from stonesoup.updater.twostate import TwoStateKalmanUpdater

from plotting_utils import plot_gnd, plot_platform

# Parameters
plot_coord = 'xyz'
ref_lat=49.725
ref_lon=-4.85
stanag_msg_directory = Path(r'C:\Users\sglvladi\OneDrive\Documents\University of Liverpool\PostDoc\EURYBIA - Dstl\Data\Drop 2 - 13Feb2025\20250213_UoLExample')
stanag_config = 'NIAGSparse'
stanag_meta_header_name = 'LatencyHeader'
rx_plat_id_selects = [1, 2]

q_factor = 0.1
rerr = 100**2
time_steps_since_update = 10
init_threshold = 5
bias_prior = GaussianState(StateVector([0., 0., 0., 0., 0., 0.]),
                           CovarianceMatrix(np.diag([0, 10., 0, 10., np.pi / 6, 50.]) ** 2))

# Sensor trackers
readers = []
trackers = []
for rx_plat_id_select in rx_plat_id_selects:
    # Detector/Reader
    contacts_reader = STANAGContactReader(stanag_msg_directory,
                                          state_vector_fields=("RelBearing", "RX2contact_range"),
                                          time_field = None,
                                          snr_threshold=10,
                                          rerr=rerr,
                                          endianness = 0,
                                          stanag_msg_directory=stanag_msg_directory,
                                          reference_lat = ref_lat,
                                          reference_lon = ref_lon,
                                          with_bias=True)
    contacts_reader.read_stanag_files(rx_plat_id_select=rx_plat_id_select, config_subfolder=stanag_config, meta_header_name =stanag_meta_header_name)
    contacts_reader.get_stanag_ground_truth_from_SM01(target_plat_unit_id=(3,1))
    readers.append(contacts_reader)

    # Transition model
    bias_transition_model = CombinedLinearGaussianTransitionModel([OrnsteinUhlenbeck(q_factor, 0.0001),
                                                                   OrnsteinUhlenbeck(q_factor, 0.0001),
                                                                   NthDerivativeDecay(0, 1e-6, 5),
                                                                   NthDerivativeDecay(0, 1e-4, 5)])
    # Predictor and Updater
    predictor = UnscentedKalmanPredictor(bias_transition_model)
    updater = UnscentedKalmanUpdater(None, True)

    # Initiator components
    hypothesiser_init = DistanceHypothesiser(predictor, updater, Mahalanobis(), 10)
    data_associator_init = GNNWith2DAssignment(hypothesiser_init)
    time_steps_since_update = 10
    deleter_init = UpdateTimeStepsDeleter(time_steps_since_update=time_steps_since_update)
    initiator = MultiMeasurementInitiator(bias_prior, None, deleter_init,
                                          data_associator_init, updater, init_threshold)

    # Tracker components
    deleter1 = UpdateTimeStepsDeleter(10)
    deleter2 = MeasurementCovarianceBasedDeleter([np.pi / 4, 5e6])
    deleter = CompositeDeleter([deleter1, deleter2], intersect=False)
    hypothesiser = PDAHypothesiser(predictor, updater, 1e-4, .9)
    hypothesiser = DistanceGater(hypothesiser, Mahalanobis(), 10)
    data_associator = JPDAWithEHM2(hypothesiser)

    # Tracker
    bias_tracker = MultiTargetMixtureTracker(initiator, deleter, contacts_reader, data_associator, updater)
    trackers.append(TrackReader(bias_tracker, run_async=False,
                                transition_model=bias_transition_model,
                                sensor_id=rx_plat_id_select))

# Fusion Tracker
# ==============
# Transition model
transition_model = CombinedLinearGaussianTransitionModel([OrnsteinUhlenbeck(q_factor, 0.0001),
                                                          OrnsteinUhlenbeck(q_factor, 0.0001)])
# Tracklet extractor & Pseudo measurement extractor
tracklet_extractor = TrackletExtractor(trackers=trackers,
                                       transition_model=transition_model,
                                       fuse_interval=datetime.timedelta(minutes=7))
detector = PseudoMeasExtractor(tracklet_extractor, state_idx_to_use=[0,1,2,3], use_prior=False)

# Predictor and Updater
two_state_predictor = TwoStatePredictor(transition_model)
two_state_updater = TwoStateKalmanUpdater(None, True)

# Hypothesiser and Data Associator
hypothesiser1 = PDAHypothesiserNoPrediction(predictor=None,
                                            updater=two_state_updater,
                                            clutter_spatial_density=1e-10,
                                            prob_detect=Probability(.7),
                                            prob_gate=Probability(0.99))
hypothesiser1 = DistanceGater(hypothesiser1, Mahalanobis(), 10)
fuse_associator = JPDAWithEHM2(hypothesiser1)

# Initiator
prior = GaussianState(StateVector([0., 0., 0., 0.]),
                      CovarianceMatrix(np.diag([0., 1000., 0., 1000.]))**2)
initiator1 = TwoStateMeasurementInitiator(prior, transition_model, two_state_updater)

# Tracker
fuse_tracker = FuseTracker(initiator=initiator1, predictor=two_state_predictor,
                           updater=two_state_updater, associator=fuse_associator,
                           detector=detector, death_rate=1e-4,
                           prob_detect=Probability(.7),
                           delete_thresh=Probability(0.1))

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
all_tracks = set()
all_detections = set()
test_ground_truths = set()

for time, ctracks in fuse_tracker:

    all_tracks.update(ctracks)
    for reader in readers:
        all_detections.update(reader.detections)
    test_ground_truths.update(set([gt for gt in readers[0].ground_truth if
                                   time - datetime.timedelta(seconds=300) <= gt.timestamp <= time]))

    colors = ['r', 'g', 'b']
    ax.cla()
    ax.set_xlabel('East')
    ax.set_ylabel('North')
    ax.set_xlim([0, 25000])
    ax.set_ylim([0, 75000])
    print(f'Time: {time} | Number of tracks: {len(ctracks)}')
    plot_gnd(test_ground_truths, ref_lat, ref_lon, ax, plot_coord)
    for reader in readers:
        plot_platform(reader.truth_TX, ref_lat, ref_lon, ax, plot_coord, 'b', 'TX')
        plot_platform(reader.truth_RX, ref_lat, ref_lon, ax, plot_coord, 'm', 'RX')
    for detection in all_detections:
        x, y = detection.measurement_model.inverse_function(detection)[[0, 2]]
        ax.plot(x, y, 'bx')
    # plot_tracks(ctracks, ax=ax)
    for i, (tracklets, color) in enumerate(zip(tracklet_extractor.current[1], colors)):
        for tracklet in tracklets:
            data = np.array([s.mean for s in tracklet.states if isinstance(s, Update)])
            if data.shape[1] > 8:
                idx = [6, 8]
            else:
                idx = [4, 6]
            plt.plot(data[:, idx[0]], data[:, idx[1]], f':{color}')
    for track in ctracks:
        data = np.array([state.state_vector for state in track])
        plt.plot(data[:, 4], data[:, 6], '-*m')

    # ax2.cla()
    # for track in ctracks:
    #     data = np.array([state.state_vector for state in track.states])
    #     num_steps = len(data)
    #     ax2.plot([i for i in range(num_steps)], data[:, -2], 'r-')
    #     ax2.plot([i for i in range(num_steps)], data[:, -1], 'c-')

    plt.pause(.1)

plt.show(block=True)