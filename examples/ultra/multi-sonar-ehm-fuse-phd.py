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
from datetime import datetime, timedelta
from copy import deepcopy, copy
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

from stonesoup.custom.initiator import TwoStateSMCPHDInitiator, SMCPHDInitiator
from stonesoup.custom.smcphd import TwoStateSMCPHDFilter2, SMCPHDFilter
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.initiator.twostate import TwoStateInitiator
from stonesoup.resampler.particle import SystematicResampler
from stonesoup.types.numeric import Probability
from stonesoup.types.particle import Particle
from stonesoup.types.state import State, GaussianState, TwoStateParticleState, ParticleState
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


# Parameters
np.random.seed(1001)
clutter_rate = 5                                    # Mean number of clutter points per scan
max_range = 100                                     # Max range of sensor (meters)
surveillance_area = np.pi*max_range**2              # Surveillance region area
clutter_density = clutter_rate/surveillance_area    # Mean number of clutter points per unit area
prob_detect = 0.9                                   # Probability of Detection
num_timesteps = 101                                 # Number of simulation timesteps
bias_tracker_idx = [0, 2]                           # Indices of trackers that run with bias model
num_particles_phd = 2**15
PLOT = True

# Simulation start time
start_time = datetime.now()

# Define transition model and position for 3D platform
platform_transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.),
                                                                   ConstantVelocity(0.),
                                                                   ConstantVelocity(0.)])

# Create platforms
init_states = [State(StateVector([-50, 0, -25, 1, 0, 0]), start_time),
               State(StateVector([50, 0, -25, 1, 0, 0]), start_time),
               State(StateVector([-25, 1, 50, 0, 0, 0]), start_time)]
platforms = []
for i, init_state in enumerate(init_states):
    # Platform
    platform = MovingPlatform(states=init_state,
                                position_mapping=(0, 2, 4),
                                velocity_mapping=(1, 3, 5),
                                transition_model=platform_transition_model)

    # Sensor
    with_bias = i in bias_tracker_idx
    radar_noise_covar = CovarianceMatrix(np.diag([np.deg2rad(.1), .5]))
    sensor = RadarBearingRangeWithClutter(ndim_state=6,
                                          position_mapping=(0, 2, 4),
                                          noise_covar=radar_noise_covar,
                                          mounting_offset=StateVector([0, 0, 0]),
                                          rotation_offset=StateVector([0, 0, 0]),
                                          clutter_rate=clutter_rate,
                                          max_range=max_range,
                                          prob_detect=prob_detect,
                                          model_with_bias=with_bias)
    platform.add_sensor(sensor)
    platforms.append(platform)


# Simulation components

# The target
cv_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.), ConstantVelocity(0.)])
ct_model = ConstantTurn(np.array([0., 0.]), np.pi/32)
manoeuvres = [cv_model, ct_model]
manoeuvre_times = [timedelta(seconds=4), timedelta(seconds=4)]
init_state_gnd = State(StateVector([25, -2, 25, 0]), start_time)
target = MultiTransitionMovingPlatform(transition_models=manoeuvres,
                                       transition_times=manoeuvre_times,
                                       states=init_state_gnd,
                                       position_mapping=(0, 2, 4),
                                       velocity_mapping=(1, 3, 5),
                                       sensors=None)

times = np.arange(0, num_timesteps, 1)
timestamps = [start_time + timedelta(seconds=float(elapsed_time)) for elapsed_time in times]

gnd_simulator = DummyGroundTruthSimulator(times=timestamps)

# Detection simulators (1 for each platform)
detector1 = PlatformTargetDetectionSimulator(groundtruth=gnd_simulator, platforms=[platforms[0]],
                                             targets=[platforms[1], platforms[2], target])
detector2 = PlatformTargetDetectionSimulator(groundtruth=gnd_simulator, platforms=[platforms[1]],
                                             targets=[platforms[0], platforms[2], target])
detector3 = PlatformTargetDetectionSimulator(groundtruth=gnd_simulator, platforms=[platforms[2]],
                                             targets=[platforms[0], platforms[1], target])

all_detectors = [detector1, detector2, detector3]
non_bias_detectors = [detector for i, detector in enumerate(all_detectors) if i not in bias_tracker_idx]
bias_detectors = [detector for i, detector in enumerate(all_detectors) if i in bias_tracker_idx]

# Multi-Target Trackers (1 per platform)
non_bias_trackers = []
non_bias_track_readers = []
for i, detector in enumerate(non_bias_detectors):
    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05),
                                                              ConstantVelocity(0.05)])
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
    # num_p = 2**12
    # resampler = SystematicResampler()
    # filter = SMCPHDFilter(prior=prior, transition_model=transition_model,
    #                       measurement_model=None, prob_detect=prob_detect,
    #                       prob_death=Probability(0.01), prob_birth=Probability(0.1),
    #                       birth_rate=0.1, clutter_density=clutter_density,
    #                       num_samples=num_p, resampler=resampler,
    #                       birth_scheme='expansion')
    # samples = multivariate_normal.rvs(prior.mean.ravel(),
    #                                   prior.covar,
    #                                   size=num_p)
    # weight = Probability(1 / num_p)
    # particles = [Particle(sample.reshape(-1, 1), weight=weight) for sample in samples]
    # state = ParticleState(particles=particles, timestamp=start_time)
    # initiator = SMCPHDInitiator(filter=filter, prior=state)
    initiator = MultiMeasurementInitiator(prior, None, deleter_init,
                                          data_associator_init, updater, 10)
    deleter1 = UpdateTimeStepsDeleter(10)
    # deleter2 = CovarianceBasedDeleter(200, mapping=[0,2])
    deleter2 = MeasurementCovarianceBasedDeleter([np.pi/4, 200])
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
bias_transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05),
                                                               ConstantVelocity(0.05),
                                                               NthDerivativeDecay(0, 1e-6, 5),
                                                               NthDerivativeDecay(0, 1e-4, 5)])
bias_prior = GaussianState(StateVector([0, 0, 0, 0, 0, 0]),
                           CovarianceMatrix(np.diag([50, 5, 50, 5, np.pi/6, 5])))
predictor = ExtendedKalmanPredictor(bias_transition_model)
updater = ExtendedKalmanUpdater(None, True)
# Initiator components
hypothesiser_init = DistanceHypothesiser(predictor, updater, Mahalanobis(), 10)
data_associator_init = GNNWith2DAssignment(hypothesiser_init)
deleter_init1 = UpdateTimeStepsDeleter(2)
# deleter_init2 = CovarianceBasedDeleter(20, mapping=[0, 2])
deleter_init2 = MeasurementCovarianceBasedDeleter([np.pi/4, 20])
deleter_init = CompositeDeleter([deleter_init1, deleter_init2], intersect=False)
initiator = MultiMeasurementInitiator(bias_prior, None, deleter_init,
                                      data_associator_init, updater, 10)
deleter1 = UpdateTimeStepsDeleter(10)
# deleter2 = CovarianceBasedDeleter(200, mapping=[0,2])
deleter2 = MeasurementCovarianceBasedDeleter([np.pi/4, 200])
deleter = CompositeDeleter([deleter1, deleter2], intersect=False)
hypothesiser = PDAHypothesiser(predictor, updater, clutter_density, prob_detect, 0.95)
hypothesiser = DistanceGater(hypothesiser, Mahalanobis(), 10)
data_associator = JPDAWithEHM2(hypothesiser)
bias_tracker = MultiTargetMixtureTracker(initiator, deleter, None, data_associator, updater)

# Fusion Tracker
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05),
                                                          ConstantVelocity(0.05)])
prior = GaussianState(StateVector([0, 0, 0, 0]),
                      CovarianceMatrix(np.diag([50, 5, 50, 5])))
tracklet_extractor = TrackletExtractorWithTracker(trackers=non_bias_track_readers,
                                                  transition_model=transition_model,
                                                  detectors=bias_detectors,
                                                  core_tracker=bias_tracker,
                                                  fuse_interval=timedelta(seconds=3))
detector = PseudoMeasExtractor(tracklet_extractor, state_idx_to_use=[0,1,2,3], use_prior=True)
two_state_predictor = TwoStatePredictor(transition_model)
two_state_updater = TwoStateKalmanUpdater(None, True)
fuse_prior = GaussianState(StateVector([0., 0., 0., 0., 0., 0., 0., 0.]),
                           CovarianceMatrix(np.diag([50.**2, 2.**2, 50.**2, 2.**2, 50.**2, 2.**2, 50.**2, 2.**2])))
hypothesiser1 = PDAHypothesiserNoPrediction(predictor=None,
                                            updater=two_state_updater,
                                            clutter_spatial_density=Probability(-80, log_value=True),
                                            prob_detect=Probability(prob_detect),
                                            prob_gate=Probability(0.99))
hypothesiser1 = DistanceGater(hypothesiser1, Mahalanobis(), 10)   # Uncomment to use JPDA+EHM2
fuse_associator = JPDAWithEHM2(hypothesiser1)                     # in Fuse tracker
# fuse_associator = GNNWith2DAssignment(hypothesiser1)          # Uncomment for GNN in Fuse Tracker
resampler = SystematicResampler()
phd_filter = TwoStateSMCPHDFilter2(prior=prior, transition_model=transition_model,
                                   measurement_model=None, prob_detect=prob_detect,
                                   prob_death=Probability(0.01), prob_birth=Probability(0.2),
                                   birth_rate=Probability(0.1), clutter_density=clutter_density,
                                   num_samples=num_particles_phd, resampler=resampler,
                                   birth_scheme='expansion')
samples = multivariate_normal.rvs(fuse_prior.mean.ravel(),
                                  fuse_prior.covar,
                                  size=num_particles_phd)
weight = Probability(1 / num_particles_phd)
particles = [Particle(sample.reshape(-1, 1), weight=weight) for sample in samples]
state = TwoStateParticleState(particles=particles, start_time=start_time, end_time=start_time)
initiator1 = TwoStateSMCPHDInitiator(phd_filter, state)
fuse_tracker = FuseTracker(initiator=initiator1, predictor=two_state_predictor,
                           updater=two_state_updater, associator=fuse_associator,
                           detector=detector, death_rate=1e-4,
                           prob_detect=Probability(prob_detect),
                           delete_thresh=Probability(0.1))

tracks = set()
for i, (timestamp, ctracks) in enumerate(fuse_tracker):
    print(f'{timestamp-start_time} - No. Tracks: {len(ctracks)}')
    tracks.update(ctracks)
    # Plot
    if PLOT:
        plt.clf()
        all_detections = [detector.detections for detector in all_detectors]
        colors = ['r', 'g', 'b']
        data = np.array([state.state_vector for state in target])
        plt.plot(data[:, 0], data[:, 2], '--k', label='Groundtruth (Target)')
        for i, (platform, color) in enumerate(zip(platforms, colors)):
            data = np.array([state.state_vector for state in platform])
            plt.plot(data[:, 0], data[:, 2], f'--{color}')

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

