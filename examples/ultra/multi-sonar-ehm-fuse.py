import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pickle

from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.types.numeric import Probability
from stonesoup.types.state import State
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.platform.base import MovingPlatform
from stonesoup.models.transition.linear import (CombinedLinearGaussianTransitionModel,
                                                ConstantVelocity, ConstantTurn)
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.sensor.radar import RadarBearingRangeWithClutter
from stonesoup.platform.base import MultiTransitionMovingPlatform
from stonesoup.simulator.simple import DummyGroundTruthSimulator
from stonesoup.simulator.platform import PlatformTargetDetectionSimulator
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater, KalmanUpdater
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.gater.distance import DistanceGater
from stonesoup.plugins.pyehm import JPDAWithEHM2
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.deleter.multi import CompositeDeleter
from stonesoup.measures import Mahalanobis
from stonesoup.types.state import GaussianState
from stonesoup.tracker.simple import MultiTargetMixtureTracker

from tracklet import TrackletExtractor, TwoStatePredictor, CustomPDAHypothesiser, FuseTracker


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

def plot_tracklets(all_tracklets):
    colors = ['r', 'g', 'b']
    idx = [4, 6]
    for i, (tracklets, color) in enumerate(zip(all_tracklets, colors)):
        for tracklet in tracklets:
            data = tracklet.twoStatePostMeans[[0, 2], :]
            plt.plot(data[0, :], data[1, :], f'-*{color}')
            for j in range(tracklet.twoStatePostMeans.shape[1]):
                plot_covar(tracklet.twoStatePostCovs[idx, :, j][:, idx],
                           tracklet.twoStatePostMeans[idx, j],
                           ax=plt.gca())

# Parameters
np.random.seed(1000)
clutter_rate = 5                                    # Mean number of clutter points per scan
max_range = 100                                     # Max range of sensor (meters)
surveillance_area = np.pi*max_range**2              # Surveillance region area
clutter_density = clutter_rate/surveillance_area    # Mean number of clutter points per unit area
prob_detect = 0.99                                  # Probability of Detection
num_timesteps = 50                                 # Number of simulation timesteps
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
trackers = []
for detector in detectors:
    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05),
                                                              ConstantVelocity(0.05)])
    # This is a dummy measurement model and will not be used because detections carry their own
    measurement_model = LinearGaussian(ndim_state=4, mapping=[0, 2],
                                       noise_covar=CovarianceMatrix(np.diag([1., 1.])))
    predictor = ExtendedKalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model, True)

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
    initiator = MultiMeasurementInitiator(prior, measurement_model, deleter_init,
                                          data_associator_init, updater, 10)
    deleter1 = UpdateTimeStepsDeleter(10)
    deleter2 = CovarianceBasedDeleter(200, mapping=[0,2])
    deleter = CompositeDeleter([deleter1, deleter2], intersect=False)
    hypothesiser = PDAHypothesiser(predictor, updater, clutter_density, prob_detect, 0.95)
    hypothesiser = DistanceGater(hypothesiser, Mahalanobis(), 10)
    data_associator = JPDAWithEHM2(hypothesiser)
    tracker = MultiTargetMixtureTracker(initiator, deleter, detector, data_associator, updater)
    trackers.append(tracker)


# Generate tracks and plot
tracks1 = set()
tracks2 = set()
tracks3 = set()
for (timestamp, ctracks1), (_, ctracks2), (_, ctracks3) in zip(*trackers):
    tracks1.update(ctracks1)
    tracks2.update(ctracks2)
    tracks3.update(ctracks3)

    # Plot
    if PLOT:
        plt.clf()
        all_tracks = [tracks1, tracks2, tracks3]
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

        for i, (tracks, color) in enumerate(zip(all_tracks, colors)):
            for track in tracks:
                data = np.array([state.state_vector for state in track])
                plt.plot(data[:, 0], data[:, 2], f'-{color}')
                plot_covar(track.state.covar[[0, 2], :][:, [0, 2]],
                           track.state.mean[[0, 2], :],
                           ax=plt.gca())

        # Add legend info
        for i, color in enumerate(colors):
            plt.plot([], [], f'--{color}', label=f'Groundtruth (Sensor {i + 1})')
            plt.plot([], [], f'-{color}', label=f'Tracks (Sensor {i+1})')
            plt.plot([], [], f'x{color}', label=f'Detections (Sensor {i + 1})')

        plt.legend(loc='upper right')
        plt.xlim((-100, 100))
        plt.ylim((-100, 100))
        plt.pause(0.01)

all_tracks = [tracks1, tracks2, tracks3]
# pickle.dump([timestamps, all_tracks], open('tracks-ehm.pickle', 'wb'))
fuse_times = np.array([t+timedelta(seconds=0.2) for t in timestamps])

predictor1 = TwoStatePredictor(transition_model)
updater1 = KalmanUpdater(None, True)
hypothesiser1 = CustomPDAHypothesiser(predictor1, updater1, Probability(-70, log_value=True),
                                      Probability(0.9),
                                      Probability(0.99))
associator1 = GNNWith2DAssignment(hypothesiser1)
tracker = FuseTracker(prior, predictor1, updater1, associator1, 1e-4, 0.9, 0.1)
fused_tracks = tracker.run_batch(all_tracks, fuse_times)
for track in fused_tracks:
    data = np.array([state.state_vector for state in track])
    plt.plot(data[:, 0], data[:, 2], '*-')
plt.show()

