import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pickle

from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.hypothesiser.distance import DistanceHypothesiser
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
from stonesoup.updater.kalman import ExtendedKalmanUpdater
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
from stonesoup.reader.generic import DetectionReplayer

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

# Simulation start time
start_time = datetime.now()

# Load data
for sim_iter in range(10):
    sim_path = f'./data/sim{sim_iter}.pickle'
    sim_data = pickle.load(open(sim_path, 'rb'))
    all_scans = sim_data['scans']
    all_gnd = sim_data['gnd']

    # Parameters
    clutter_rate = sim_data['clutter_rate']  # Mean number of clutter points per scan
    max_range = 100  # Max range of sensor (meters)
    surveillance_area = np.pi * max_range ** 2  # Surveillance region area
    clutter_density = clutter_rate / surveillance_area  # Mean number of clutter points per unit area
    prob_detect = sim_data['prob_detect']  # Probability of Detection
    num_timesteps = 100  # Number of simulation timesteps

    # Simulation components

    # Detection simulators (1 for each platform)
    detector1 = DetectionReplayer(all_scans[0], start_time=start_time)
    detector2 = DetectionReplayer(all_scans[1], start_time=start_time)
    detector3 = DetectionReplayer(all_scans[2], start_time=start_time)
    detectors = [detector1, detector2, detector3]

    # Multi-Target Trackers (1 per platform)
    trackers = []
    for detector in detectors:
        transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05),
                                                                  ConstantVelocity(0.05)])
        predictor = ExtendedKalmanPredictor(transition_model)
        updater = ExtendedKalmanUpdater(None)

        # Initiator components
        hypothesiser_init = DistanceHypothesiser(predictor, updater, Mahalanobis(), 10)
        data_associator_init = GNNWith2DAssignment(hypothesiser_init)
        prior = GaussianState(StateVector([0, 0, 0, 0]),
                              CovarianceMatrix(np.diag([50, 5, 50, 5])))
        deleter_init1 = UpdateTimeStepsDeleter(2)
        deleter_init2 = CovarianceBasedDeleter(20, mapping=[0,2])
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
        tracker = MultiTargetMixtureTracker(initiator, deleter, detector, data_associator, updater)
        trackers.append(tracker)


    # Generate tracks and plot
    tracks1 = set()
    tracks2 = set()
    tracks3 = set()
    timestamps = []
    for (timestamp, ctracks1), (_, ctracks2), (_, ctracks3) in zip(*trackers):
        tracks1.update(ctracks1)
        tracks2.update(ctracks2)
        tracks3.update(ctracks3)
        timestamps.append(timestamp)

        # Plot
        plt.clf()
        all_tracks = [tracks1, tracks2, tracks3]
        all_detections = [tracker.detector.detections for tracker in trackers]
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
plt.show()

