"""
multi-sonar-ehm-fuse.py

This example script simulates 3 moving platforms, each equipped with a single active sonar sensor 
(StoneSoup does not have an implementation of an active sonar so a radar is used instead), and 1 
target. Each sensor generates detections of all other objects (excluding itself).

The tracking configuration is as follows:
- For each sensor whose index in the 'all_detectors' list is not in 'bias_tracker_idx', a
  local tracker is configured that acts like a contact follower and generates Track objects. The
  outputs of these trackers are the fed into the Fusion engine.
- For all other sensors, their data is fed directly into the Fusion engine. Note that the
  TrackletExtractorWithTracker is used here, meaning that a (local) bias estimation tracker is run
  on the data read from each sensor, before it is fed into the main Fuse Tracker (i.e. the
  component of the Fusion Engine that produces the fused tracks).
- The data association algorithm used for both the local and fuse trackers is JPDA with EHM.

"""
import numpy as np
import pickle
from datetime import datetime, timedelta
from copy import deepcopy, copy
import matplotlib.pyplot as plt
import tqdm
import multiprocessing as mpp
from matplotlib.patches import Ellipse

from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.initiator.twostate import TwoStateInitiator
from stonesoup.metricgenerator.ospametric import GOSPAMetric
from stonesoup.reader.generic import DetectionReplayer
from stonesoup.types.numeric import Probability
from stonesoup.types.state import State, GaussianState
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.platform.base import MovingPlatform
from stonesoup.models.transition.linear import (CombinedLinearGaussianTransitionModel,
                                                ConstantVelocity, ConstantTurn, NthDerivativeDecay)
from stonesoup.sensor.radar import RadarBearingRangeWithClutter
from stonesoup.platform.base import MultiTransitionMovingPlatform
from stonesoup.simulator.simple import DummyGroundTruthSimulator
from stonesoup.simulator.platform import PlatformTargetDetectionSimulator
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.types.update import Update
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.hypothesiser.probability import PDAHypothesiser, PDAHypothesiserNoPrediction
from stonesoup.gater.distance import DistanceGater
from stonesoup.plugins.pyehm import JPDAWithEHM2
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.deleter.error import CovarianceBasedDeleter, MeasurementCovarianceBasedDeleter
from stonesoup.deleter.multi import CompositeDeleter
from stonesoup.measures import Mahalanobis
from stonesoup.tracker.simple import MultiTargetMixtureTracker
from stonesoup.predictor.twostate import TwoStatePredictor
from stonesoup.updater.twostate import TwoStateKalmanUpdater
from stonesoup.reader.track import TrackReader
from stonesoup.reader.tracklet import TrackletExtractor, PseudoMeasExtractor, \
    TrackletExtractorWithTracker
from stonesoup.tracker.fuse import FuseTracker


np.random.seed(1000)
PLOT = False
num_sims = 1

# Simulation start time
start_time = datetime.now()
ospa_metrics = []

def imap_tqdm(pool, f, inputs, chunksize=None, **tqdm_kwargs):
    # Calculation of chunksize taken from pool._map_async
    if not chunksize:
        chunksize, extra = divmod(len(inputs), len(pool._pool) * 4)
        if extra:
            chunksize += 1
    results = list(tqdm.tqdm(pool.imap_unordered(f, inputs, chunksize=chunksize), total=len(inputs), **tqdm_kwargs))
    return results

def run_sim(sim_iter):
    # Load data
    sim_path = f'./data/sim{sim_iter}.pickle'
    sim_data = pickle.load(open(sim_path, 'rb'))
    all_scans = sim_data['scans']
    all_gnd = sim_data['groundtruth']
    for gnd in all_gnd:
        for state in gnd:
            state.timestamp = start_time + state.timestamp

    # Parameters
    clutter_rate = sim_data['clutter_rate']  # Mean number of clutter points per scan
    max_range = 100  # Max range of sensor (meters)
    surveillance_area = np.pi * max_range ** 2  # Surveillance region area
    clutter_density = clutter_rate / surveillance_area  # Mean number of clutter points per unit area
    prob_detect = sim_data['prob_detect']  # Probability of Detection
    bias_tracker_idx = sim_data['bias_tracker_idx']  # Indices of trackers that run with bias model

    # Simulation components

    # Detection simulators (1 for each platform)
    detector1 = DetectionReplayer(all_scans[0], start_time=start_time)
    detector2 = DetectionReplayer(all_scans[1], start_time=start_time)
    detector3 = DetectionReplayer(all_scans[2], start_time=start_time)

    all_detectors = [detector1, detector2, detector3]
    non_bias_detectors = [detector for i, detector in enumerate(all_detectors) if
                          i not in bias_tracker_idx]
    bias_detectors = [detector for i, detector in enumerate(all_detectors) if
                      i in bias_tracker_idx]

    # Multi-Target Trackers (1 per platform)
    non_bias_trackers = []
    non_bias_track_readers = []
    for i, detector in enumerate(non_bias_detectors):
        transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05, seed=sim_iter),
                                                                  ConstantVelocity(0.05, seed=sim_iter)])
        prior = GaussianState(StateVector([0, 0, 0, 0]),
                              CovarianceMatrix(np.diag([50, 5, 50, 5])))
        predictor = ExtendedKalmanPredictor(transition_model)
        updater = ExtendedKalmanUpdater(None, True)

        # Initiator components
        hypothesiser_init = DistanceHypothesiser(predictor, updater, Mahalanobis(), 10)
        data_associator_init = GNNWith2DAssignment(hypothesiser_init)
        deleter_init1 = UpdateTimeStepsDeleter(2)
        # deleter_init2 = CovarianceBasedDeleter(20, mapping=[0, 2])
        deleter_init2 = MeasurementCovarianceBasedDeleter([np.pi / 4, 20])
        deleter_init = CompositeDeleter([deleter_init1, deleter_init2], intersect=False)

        # Tracker components
        # MofN initiator
        initiator = MultiMeasurementInitiator(prior, None, deleter_init,
                                              data_associator_init, updater, 10)
        deleter1 = UpdateTimeStepsDeleter(10)
        # deleter2 = CovarianceBasedDeleter(200, mapping=[0,2])
        deleter2 = MeasurementCovarianceBasedDeleter([np.pi / 4, 200])
        deleter = CompositeDeleter([deleter1, deleter2], intersect=False)
        hypothesiser = PDAHypothesiser(predictor, updater, clutter_density, prob_detect, 0.95)
        hypothesiser = DistanceGater(hypothesiser, Mahalanobis(), 10)
        data_associator = JPDAWithEHM2(hypothesiser)
        tracker = MultiTargetMixtureTracker(initiator, deleter, detector, data_associator, updater)
        non_bias_trackers.append(tracker)
        non_bias_track_readers.append(TrackReader(tracker, run_async=False,
                                                  transition_model=transition_model,
                                                  sensor_id=i))

    # Bias tracker for sensors that feed detections straight to the Fusion Engine
    bias_transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05, seed=sim_iter),
                                                                   ConstantVelocity(0.05, seed=sim_iter),
                                                                   NthDerivativeDecay(0, 1e-6, 5, seed=sim_iter),
                                                                   NthDerivativeDecay(0, 1e-4, 5, seed=sim_iter)])
    bias_prior = GaussianState(StateVector([0, 0, 0, 0, 0, 0]),
                               CovarianceMatrix(np.diag([50, 5, 50, 5, np.pi / 6, 5])))
    predictor = ExtendedKalmanPredictor(bias_transition_model)
    updater = ExtendedKalmanUpdater(None, True)
    # Initiator components
    hypothesiser_init = DistanceHypothesiser(predictor, updater, Mahalanobis(), 10)
    data_associator_init = GNNWith2DAssignment(hypothesiser_init)
    deleter_init1 = UpdateTimeStepsDeleter(2)
    # deleter_init2 = CovarianceBasedDeleter(20, mapping=[0, 2])
    deleter_init2 = MeasurementCovarianceBasedDeleter([np.pi / 4, 20])
    deleter_init = CompositeDeleter([deleter_init1, deleter_init2], intersect=False)
    initiator = MultiMeasurementInitiator(bias_prior, None, deleter_init,
                                          data_associator_init, updater, 10)
    deleter1 = UpdateTimeStepsDeleter(10)
    # deleter2 = CovarianceBasedDeleter(200, mapping=[0,2])
    deleter2 = MeasurementCovarianceBasedDeleter([np.pi / 4, 200])
    deleter = CompositeDeleter([deleter1, deleter2], intersect=False)
    hypothesiser = PDAHypothesiser(predictor, updater, clutter_density, prob_detect, 0.95)
    hypothesiser = DistanceGater(hypothesiser, Mahalanobis(), 10)
    data_associator = JPDAWithEHM2(hypothesiser)
    bias_tracker = MultiTargetMixtureTracker(initiator, deleter, None, data_associator, updater)

    # Fusion Tracker
    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05, seed=sim_iter),
                                                              ConstantVelocity(0.05, seed=sim_iter)])
    prior = GaussianState(StateVector([0, 0, 0, 0]),
                          CovarianceMatrix(np.diag([50, 5, 50, 5])))
    tracklet_extractor = TrackletExtractorWithTracker(trackers=non_bias_track_readers,
                                                      transition_model=transition_model,
                                                      detectors=bias_detectors,
                                                      core_tracker=bias_tracker,
                                                      fuse_interval=timedelta(seconds=3))
    detector = PseudoMeasExtractor(tracklet_extractor, state_idx_to_use=[0, 1, 2, 3])
    two_state_predictor = TwoStatePredictor(transition_model)
    two_state_updater = TwoStateKalmanUpdater(None, True)
    hypothesiser1 = PDAHypothesiserNoPrediction(predictor=None,
                                                updater=two_state_updater,
                                                clutter_spatial_density=Probability(-80,
                                                                                    log_value=True),
                                                prob_detect=Probability(prob_detect),
                                                prob_gate=Probability(0.99))
    hypothesiser1 = DistanceGater(hypothesiser1, Mahalanobis(), 10)  # Uncomment to use JPDA+EHM2
    fuse_associator = JPDAWithEHM2(hypothesiser1)  # in Fuse tracker
    # fuse_associator = GNNWith2DAssignment(hypothesiser1)          # Uncomment for GNN in Fuse Tracker
    initiator1 = TwoStateInitiator(prior, transition_model, two_state_updater)
    fuse_tracker = FuseTracker(initiator=initiator1, predictor=two_state_predictor,
                               updater=two_state_updater, associator=fuse_associator,
                               detector=detector, death_rate=1e-4,
                               prob_detect=Probability(prob_detect),
                               delete_thresh=Probability(0.1))

    tracks = set()
    timestamps = []
    for i, (timestamp, ctracks) in enumerate(fuse_tracker):
        timestamps.append(timestamp)
        # print(f'{timestamp - start_time} - No. Tracks: {len(ctracks)}')
        tracks.update(ctracks)
        # Plot
        if PLOT:
            plt.clf()
            all_detections = [detector.detections for detector in all_detectors]
            colors = ['r', 'g', 'b']
            # data = np.array([state.state_vector for state in target])
            # plt.plot(data[:, 0], data[:, 2], '--k', label='Groundtruth (Target)')
            # for i, (platform, color) in enumerate(zip(platforms, colors)):
            #     data = np.array([state.state_vector for state in platform])
            #     plt.plot(data[:, 0], data[:, 2], f'--{color}')

            for i, (detections, color) in enumerate(zip(all_detections, colors)):
                for detection in detections:
                    model = detection.measurement_model
                    x, y = detection.measurement_model.inverse_function(detection)[[0, 2]]
                    plt.plot(x, y, f'{color}x')

            for i, (tracklets, color) in enumerate(zip(tracklet_extractor.current[1], colors)):
                for tracklet in tracklets:
                    data = np.array([s.mean for s in tracklet.states if isinstance(s, Update)])
                    if data.shape[1] > 8:
                        idx = [6, 8]
                    else:
                        idx = [4, 6]
                    plt.plot(data[:, idx[0]], data[:, idx[1]], f':{color}')

            for track in tracks:
                data = np.array([state.state_vector for state in track])
                plt.plot(data[:, 4], data[:, 6], '-*m')

            # Add legend info
            for i, color in enumerate(colors):
                plt.plot([], [], f'--{color}', label=f'Groundtruth (Sensor {i + 1})')
                plt.plot([], [], f':{color}', label=f'Tracklets (Sensor {i + 1})')
                plt.plot([], [], f'x{color}', label=f'Detections (Sensor {i + 1})')
            plt.plot([], [], f'-*m', label=f'Fused Tracks')

            plt.legend(loc='upper right')
            plt.xlim((-100, 100))
            plt.ylim((-100, 100))
            plt.pause(0.01)

    for gnd in all_gnd:
        gnd.states = [state for state in gnd.states if state.timestamp in timestamps]

    from stonesoup.metricgenerator.ospametric import OSPAMetric
    from stonesoup.measures import Euclidean

    ospa_generator = OSPAMetric(c=10, p=1, measure=Euclidean([0, 2]))
    ospa_metric = ospa_generator.compute_over_time(ospa_generator.extract_states(tracks),
                                                   ospa_generator.extract_states(all_gnd))
    gospa_generator = GOSPAMetric(c=10, p=1, measure=Euclidean([0, 2]))
    gospa_metric = gospa_generator.compute_over_time(ospa_generator.extract_states(tracks),
                                                   ospa_generator.extract_states(all_gnd))
    return ospa_metric, gospa_metric

def main():
    run_sim(0)
    pool = mpp.Pool(mpp.cpu_count())
    inputs = [sim_iter for sim_iter in range(num_sims)]
    results = imap_tqdm(pool, run_sim, inputs, desc='Sim')
    metrics = list(results)
    ospa_metrics = [metric[0] for metric in metrics]
    gospa_metrics = [metric[1] for metric in metrics]
    gospa = {'distance': 0.0,
             'localisation': 0.0,
             'missed': 0,
             'false': 0}

    for key in gospa:
        metric_mat = np.array(
            [[i.value[key] for i in gospa_metric.value] for gospa_metric in gospa_metrics])
        gospa[key] = np.mean(metric_mat, axis=0)

    metric_mat = np.array([[i.value for i in ospa_metric.value] for ospa_metric in ospa_metrics])
    metric = np.mean(metric_mat, axis=0)
    timestamps = [i.timestamp for i in ospa_metrics[0].value]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(timestamps, metric)
    ax.set_ylabel("OSPA distance")
    ax.tick_params(labelbottom=False)
    _ = ax.set_xlabel("Time")
    pickle.dump({'ospa': metric}, open('./output/jpda_ospa.pickle', 'wb'))
    plt.show()
    

if __name__ == '__main__':
    main()
    