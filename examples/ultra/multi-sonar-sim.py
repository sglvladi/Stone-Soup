import os
import pickle
import numpy as np
from datetime import datetime, timedelta
from copy import deepcopy, copy
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import tqdm
import multiprocessing as mpp

from stonesoup.dataassociator.mfa import MFADataAssociator
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.hypothesiser.mfa import MFAHypothesiser
from stonesoup.initiator.twostate import TwoStateInitiatorMixture
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
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
from stonesoup.initiator.simple import MultiMeasurementInitiator, MultiMeasurementInitiatorMixture
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.deleter.error import CovarianceBasedDeleter, MeasurementCovarianceBasedDeleter
from stonesoup.deleter.multi import CompositeDeleter
from stonesoup.measures import Mahalanobis
from stonesoup.tracker.simple import MultiTargetMixtureTracker, MultiTargetMultiMixtureTracker
from stonesoup.predictor.twostate import TwoStatePredictor
from stonesoup.updater.twostate import TwoStateKalmanUpdater
from stonesoup.reader.track import TrackReader
from stonesoup.reader.tracklet import TrackletExtractor, PseudoMeasExtractor, \
    TrackletExtractorWithTracker
from stonesoup.tracker.fuse import FuseTracker


# Parameters
np.random.seed(1000)
clutter_rate = 5                                    # Mean number of clutter points per scan
max_range = 100                                     # Max range of sensor (meters)
surveillance_area = np.pi*max_range**2              # Surveillance region area
clutter_density = clutter_rate/surveillance_area    # Mean number of clutter points per unit area
prob_detect = 0.9                                   # Probability of Detection
num_timesteps = 101                                 # Number of simulation timesteps
bias_tracker_idx = [0, 2]                           # Indices of trackers that run with bias model
slide_window = 2
PLOT = True
num_sims = 1000

# Simulation start time
start_time = datetime.now()

def imap_tqdm(pool, f, inputs, chunksize=None, **tqdm_kwargs):
    # Calculation of chunksize taken from pool._map_async
    if not chunksize:
        chunksize, extra = divmod(len(inputs), len(pool._pool) * 4)
        if extra:
            chunksize += 1
    results = list(tqdm.tqdm(pool.imap_unordered(f, inputs, chunksize=chunksize), total=len(inputs), **tqdm_kwargs))
    return results

def run_sim(sim_iter):
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
                                              model_with_bias=with_bias,
                                              seed=sim_iter)
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

    sim_path = f'./data/sim{sim_iter}.pickle'
    all_scans = [[] for _ in range(3)]
    all_gnd = [GroundTruthPath() for _ in range(4)]
    for data in zip(*all_detectors):
        for i, scan in enumerate(data):
            all_scans[i].append((scan[0]-start_time, scan[1]))

    for i, platform in enumerate(platforms):
        states = [GroundTruthState(state.state_vector[:5], state.timestamp-start_time) for state in platform]
        all_gnd[i] = GroundTruthPath(states, id=i)

    states = [GroundTruthState(state.state_vector[:5], state.timestamp-start_time) for state in target]
    all_gnd[-1] = GroundTruthPath(states, id=3)

    pickle.dump({'clutter_rate': clutter_rate,
                 'prob_detect': prob_detect,
                 'bias_tracker_idx': bias_tracker_idx,
                 'scans': all_scans,
                 'groundtruth': all_gnd},
                open(sim_path, 'wb'))

def main():
    pool = mpp.Pool(mpp.cpu_count())
    inputs = [sim_iter for sim_iter in range(num_sims)]
    imap_tqdm(pool, run_sim, inputs, desc='Sim')

if __name__ == '__main__':
    main()