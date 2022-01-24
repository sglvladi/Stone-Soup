import numpy as np
from datetime import datetime, timedelta
from copy import deepcopy, copy
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.initiator.twostate import TwoStateInitiator
from stonesoup.types.numeric import Probability
from stonesoup.types.state import State, GaussianState
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.platform.base import MovingPlatform
from stonesoup.models.transition.linear import (CombinedLinearGaussianTransitionModel,
                                                ConstantVelocity, ConstantTurn)
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
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.deleter.multi import CompositeDeleter
from stonesoup.measures import Mahalanobis
from stonesoup.tracker.simple import MultiTargetMixtureTracker
from stonesoup.predictor.twostate import TwoStatePredictor
from stonesoup.updater.twostate import TwoStateKalmanUpdater
from stonesoup.reader.track import TrackReader
from stonesoup.reader.tracklet import TrackletExtractor, PseudoMeasExtractor, \
    TrackletExtractorWithTracker
from stonesoup.tracker.fuse import FuseTracker


def plot_covar(cov, pos, nstd=1, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta,
                    alpha=0.4, **kwargs)

    ax.add_artist(ellip)
    return ellip


# Parameters
np.random.seed(1000)
clutter_rate = 5                                    # Mean number of clutter points per scan
max_range = 100                                     # Max range of sensor (meters)
surveillance_area = np.pi*max_range**2              # Surveillance region area
clutter_density = clutter_rate/surveillance_area    # Mean number of clutter points per unit area
prob_detect = 0.9                                   # Probability of Detection
num_timesteps = 101                                 # Number of simulation timesteps
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
for init_state in init_states:
    # Platform
    platform = MovingPlatform(states=init_state,
                                position_mapping=(0, 2, 4),
                                velocity_mapping=(1, 3, 5),
                                transition_model=platform_transition_model)

    # Sensor
    radar_noise_covar = CovarianceMatrix(np.diag([np.deg2rad(.1), .5]))
    sensor = RadarBearingRangeWithClutter(ndim_state=6,
                                          position_mapping=(0, 2, 4),
                                          noise_covar=radar_noise_covar,
                                          mounting_offset= StateVector([0, 0, 0]),
                                          rotation_offset= StateVector([0, 0, 0]),
                                          clutter_rate=clutter_rate,
                                          max_range=max_range,
                                          prob_detect=prob_detect)
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
detectors = [detector1, detector2, detector3]

# Multi-Target Trackers (1 per platform)
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05),
                                                              ConstantVelocity(0.05)])
predictor = ExtendedKalmanPredictor(transition_model)
updater = ExtendedKalmanUpdater(None, True)

# Initiator components
hypothesiser_init = DistanceHypothesiser(predictor, updater, Mahalanobis(), 10)
data_associator_init = GNNWith2DAssignment(hypothesiser_init)
prior = GaussianState(StateVector([0, 0, 0, 0]),
                      CovarianceMatrix(np.diag([50, 5, 50, 5])))
deleter_init1 = UpdateTimeStepsDeleter(2)
deleter_init2 = CovarianceBasedDeleter(20, mapping=[0, 2])
deleter_init = CompositeDeleter([deleter_init1, deleter_init2], intersect=False)

# Tracker components
# MofN initiator
initiator = MultiMeasurementInitiator(prior, None, deleter_init,
                                      data_associator_init, updater, 10)
deleter1 = UpdateTimeStepsDeleter(10)
deleter2 = CovarianceBasedDeleter(200, mapping=[0,2])
deleter = CompositeDeleter([deleter1, deleter2], intersect=False)
hypothesiser = PDAHypothesiser(predictor, updater, clutter_density, prob_detect, 0.95)
hypothesiser = DistanceGater(hypothesiser, Mahalanobis(), 10)
data_associator = JPDAWithEHM2(hypothesiser)
core_tracker = MultiTargetMixtureTracker(initiator, deleter, None, data_associator, updater)
trackers = []
for detector in detectors:
    tracker = deepcopy(core_tracker)
    tracker.detector = detector
    trackers.append(tracker)


# Fusion Tracker
track_readers = [TrackReader(t, run_async=False, transition_model=transition_model, sensor_id=i)
                 for i, t in enumerate(trackers)]
tracklet_extractor = TrackletExtractor(track_readers, transition_model, fuse_interval=timedelta(seconds=3))
# tracklet_extractor = TrackletExtractorWithTracker([], transition_model, fuse_interval=timedelta(seconds=3),
#                                                   detectors=detectors, core_tracker=core_tracker)
detector = PseudoMeasExtractor(tracklet_extractor)
two_state_predictor = TwoStatePredictor(transition_model)
two_state_updater = TwoStateKalmanUpdater(None, True)
hypothesiser1 = PDAHypothesiserNoPrediction(predictor=None,
                                            updater=two_state_updater,
                                            clutter_spatial_density=Probability(-80, log_value=True),
                                            prob_detect=Probability(prob_detect),
                                            prob_gate=Probability(0.99))
hypothesiser1 = DistanceGater(hypothesiser1, Mahalanobis(), 10)   # Uncomment to use JPDA+EHM2
fuse_associator = JPDAWithEHM2(hypothesiser1)                     # in Fuse tracker
# fuse_associator = GNNWith2DAssignment(hypothesiser1)          # Uncomment for GNN in Fuse Tracker
initiator1 = TwoStateInitiator(prior, transition_model, two_state_updater)
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
        all_detections = [tracker.detector.detections for tracker in trackers]
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
            idx = [4, 6]
            for tracklet in tracklets:
                data = np.array([s.mean for s in tracklet.states if isinstance(s, Update)])
                plt.plot(data[:, 4], data[:, 6], f':{color}')

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

