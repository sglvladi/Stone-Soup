import copy
import datetime
from collections.abc import Callable
from copy import deepcopy
from typing import Mapping, Sequence, Set, List, Any
import itertools as it

import numpy as np
from matplotlib.path import Path
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from stonesoup.base import Property
from stonesoup.custom.tracker import SMCPHD_JIPDA
from stonesoup.functions import gm_reduce_single
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.sensormanager.action import Action
from stonesoup.sensor.sensor import Sensor
from stonesoup.sensormanager.reward import RewardFunction
from stonesoup.tracker import Tracker
from stonesoup.types.array import StateVectors
from stonesoup.types.detection import TrueDetection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.numeric import Probability
from stonesoup.types.state import ParticleState
from stonesoup.types.track import Track
from stonesoup.types.update import GaussianStateUpdate
from stonesoup.updater.kalman import ExtendedKalmanUpdater


class RolloutUncertaintyRewardFunction(RewardFunction):
    """A reward function which calculates the potential reduction in the uncertainty of track estimates
    if a particular action is taken by a sensor or group of sensors.

    Given a configuration of sensors and actions, a metric is calculated for the potential
    reduction in the uncertainty of the tracks that would occur if the sensing configuration
    were used to make an observation. A larger value indicates a greater reduction in
    uncertainty.
    """

    predictor: KalmanPredictor = Property(doc="Predictor used to predict the track to a new state")
    updater: ExtendedKalmanUpdater = Property(doc="Updater used to update "
                                                  "the track to the new state.")
    timesteps: int = Property(doc="Number of timesteps to rollout")
    num_samples: int = Property(doc="Number of samples to take for each timestep", default=30)
    interval: datetime.timedelta = Property(doc="Interval between timesteps",
                                            default=datetime.timedelta(seconds=1))

    def __call__(self, config: Mapping[Sensor, Sequence[Action]], tracks: Set[Track],
                 metric_time: datetime.datetime, *args, **kwargs):
        """
        For a given configuration of sensors and actions this reward function calculates the
        potential uncertainty reduction of each track by
        computing the difference between the covariance matrix norms of the prediction
        and the posterior assuming a predicted measurement corresponding to that prediction.

        This requires a mapping of sensors to action(s)
        to be evaluated by reward function, a set of tracks at given time and the time at which
        the actions would be carried out until.

        The metric returned is the total potential reduction in uncertainty across all tracks.

        Returns
        -------
        : float
            Metric of uncertainty for given configuration

        """

        # Reward value
        end_time = metric_time + datetime.timedelta(seconds=self.timesteps)
        config_metric = self._rollout(config, tracks, metric_time, end_time)

        # Return value of configuration metric
        return config_metric

    def _rollout(self, config: Mapping[Sensor, Sequence[Action]], tracks: Set[Track],
                 timestamp: datetime.datetime, end_time: datetime.datetime):
        """
        For a given configuration of sensors and actions this reward function calculates the
        potential uncertainty reduction of each track by
        computing the difference between the covariance matrix norms of the prediction
        and the posterior assuming a predicted measurement corresponding to that prediction.

        This requires a mapping of sensors to action(s)
        to be evaluated by reward function, a set of tracks at given time and the time at which
        the actions would be carried out until.

        The metric returned is the total potential reduction in uncertainty across all tracks.

        Returns
        -------
        : float
            Metric of uncertainty for given configuration

        """

        # Reward value
        config_metric = 0

        predicted_sensors = list()
        memo = {}

        # For each sensor in the configuration
        for sensor, actions in config.items():
            predicted_sensor = copy.deepcopy(sensor, memo)
            predicted_sensor.add_actions(actions)
            predicted_sensor.act(timestamp)
            if isinstance(sensor, Sensor):
                predicted_sensors.append(predicted_sensor)  # checks if its a sensor

        # Create dictionary of predictions for the tracks in the configuration
        predicted_tracks = set()
        for track in tracks:
            predicted_track = copy.copy(track)
            predicted_track.append(self.predictor.predict(predicted_track, timestamp=timestamp))
            predicted_tracks.add(predicted_track)

        for sensor in predicted_sensors:

            # Assumes one detection per track
            detections = {detection.groundtruth_path: detection
                          for detection in sensor.measure(predicted_tracks, noise=False)
                          if isinstance(detection, TrueDetection)}

            for predicted_track, detection in detections.items():
                # Generate hypothesis based on prediction/previous update and detection
                hypothesis = SingleHypothesis(predicted_track.state, detection)

                # Do the update based on this hypothesis and store covariance matrix
                update = self.updater.update(hypothesis)

                previous_cov_norm = np.linalg.norm(predicted_track.covar)
                update_cov_norm = np.linalg.norm(update.covar)

                # Replace prediction with update
                predicted_track.append(update)

                # Calculate metric for the track observation and add to the metric
                # for the configuration
                metric = previous_cov_norm - update_cov_norm
                config_metric += metric

        if timestamp == end_time:
            return config_metric

        timestamp = timestamp + datetime.timedelta(seconds=1)

        all_action_choices = dict()
        for sensor in predicted_sensors:
            # get action 'generator(s)'
            action_generators = sensor.actions(timestamp)
            # list possible action combinations for the sensor
            action_choices = list(it.product(*action_generators))
            # dictionary of sensors: list(action combinations)
            all_action_choices[sensor] = action_choices

        configs = list({sensor: action
                        for sensor, action in zip(all_action_choices.keys(), actionconfig)}
                       for actionconfig in it.product(*all_action_choices.values()))

        idx = np.random.choice(len(configs), self.num_samples)
        configs = [configs[i] for i in idx]

        rewards = [self._rollout(config, tracks, timestamp, end_time) for config in configs]
        config_metric += np.max(rewards)

        return config_metric


class RolloutPriorityRewardFunction(RewardFunction):
    """A reward function which calculates the potential reduction in the uncertainty of track estimates
    if a particular action is taken by a sensor or group of sensors.

    Given a configuration of sensors and actions, a metric is calculated for the potential
    reduction in the uncertainty of the tracks that would occur if the sensing configuration
    were used to make an observation. A larger value indicates a greater reduction in
    uncertainty.
    """

    tracker: Tracker = Property(doc="Tracker used to track the tracks")
    timesteps: int = Property(doc="Number of timesteps to rollout")
    num_samples: int = Property(doc="Number of samples to take for each timestep", default=30)
    interval: datetime.timedelta = Property(doc="Interval between timesteps",
                                            default=datetime.timedelta(seconds=1))
    rfis: List[Any] = Property(doc="List of reward functions to use for prioritisation",
                               default=None)
    prob_survive: Probability = Property(doc="Probability of survival", default=Probability(0.99))
    use_variance: bool = Property(doc="Use variance in prioritisation", default=False)
    eval_irs: Callable[[Any, ...], float] = Property(doc="Function to evaluate rfis", default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.rfis is None:
            self.rfis = []

    def __call__(self, config: Mapping[Sensor, Sequence[Action]], tracks: Set[Track],
                 metric_time: datetime.datetime, phd_density=None, *args, **kwargs):
        """
        For a given configuration of sensors and actions this reward function calculates the
        potential uncertainty reduction of each track by
        computing the difference between the covariance matrix norms of the prediction
        and the posterior assuming a predicted measurement corresponding to that prediction.

        This requires a mapping of sensors to action(s)
        to be evaluated by reward function, a set of tracks at given time and the time at which
        the actions would be carried out until.

        The metric returned is the total potential reduction in uncertainty across all tracks.

        Returns
        -------
        : float
            Metric of uncertainty for given configuration

        """

        if not len(self.rfis):
            return 0

        # Reward value
        end_time = metric_time + self.timesteps * self.interval

        # Reward value
        config_metric, updated_tracks, predicted_sensors = \
            self._compute_metric(config, tracks, metric_time, self.tracker)

        if metric_time == end_time:
            return config_metric, [config]

        timestamp = metric_time + self.interval

        all_action_choices = dict()
        for sensor in predicted_sensors:
            # get action 'generator(s)'
            action_generators = sensor.actions(timestamp)
            # list possible action combinations for the sensor
            action_choices = list(it.product(*action_generators))
            # dictionary of sensors: list(action combinations)
            all_action_choices[sensor] = action_choices

        configs = list({sensor: action
                        for sensor, action in zip(all_action_choices.keys(), actionconfig)}
                       for actionconfig in it.product(*all_action_choices.values()))

        if len(configs) > self.num_samples:
            idx = np.random.choice(len(configs), self.num_samples, replace=False)
            configs = [configs[i] for i in idx]

        rewards = []
        full_configs = []
        for cfg in configs:
            tracker = deepcopy(self.tracker)
            sim_reward, sim_config = self._rollout(cfg, updated_tracks, timestamp, end_time, tracker)
            cfg_reward = config_metric + sim_reward
            rewards.append(cfg_reward)
            full_cfg_tmp = [config, cfg] + sim_config
            full_configs.append(full_cfg_tmp)

        max_idx = np.argmax(rewards)
        # Return value of configuration metric
        return rewards[max_idx], full_configs[max_idx]

    def _compute_metric(self, config: Mapping[Sensor, Sequence[Action]], tracks: Set[Track],
                        timestamp: datetime.datetime, tracker: Tracker=None):

        if tracker is None:
            tracker = self.tracker

        # Reward value
        config_metric = 0

        predicted_sensors = list()
        memo = {}

        # For each sensor in the configuration
        for sensor, actions in config.items():
            predicted_sensor = copy.deepcopy(sensor, memo)
            predicted_sensor.add_actions(actions)
            predicted_sensor.act(timestamp)
            if isinstance(sensor, Sensor):
                predicted_sensors.append(predicted_sensor)  # checks if its a sensor

        # Create dictionary of predictions for the tracks in the configuration
        predicted_tracks = set()
        for track in tracks:
            predicted_track = copy.copy(track)
            predicted_track.append(
                tracker._predictor.predict(predicted_track, timestamp=timestamp))
            predicted_tracks.add(predicted_track)

        tracks_copy = [copy.copy(track) for track in tracks]

        for sensor in predicted_sensors:

            # Assumes one detection per track
            detections = {detection
                          for detection in sensor.measure(predicted_tracks, noise=False)
                          if isinstance(detection, TrueDetection)}

            # center = (sensor.position[1], sensor.position[0])
            # radius = sensor.fov_radius
            # p = geodesic_point_buffer(*center, radius)
            p = sensor.footprint
            tracker.prob_detect = _prob_detect_func([p])

            tracks_copy = tracker.track(detections, timestamp)

        for rfi in self.rfis:
            config_metric += self.eval_irs(rfi, tracks_copy, predicted_sensors[0], tracker._initiator._state,
                                           use_variance=self.use_variance, timestamp=timestamp)

        return config_metric, tracks_copy, predicted_sensors

    def _rollout(self, config: Mapping[Sensor, Sequence[Action]], tracks: Set[Track],
                 timestamp: datetime.datetime, end_time: datetime.datetime, tracker: Tracker=None):
        """
        For a given configuration of sensors and actions this reward function calculates the
        potential uncertainty reduction of each track by
        computing the difference between the covariance matrix norms of the prediction
        and the posterior assuming a predicted measurement corresponding to that prediction.

        This requires a mapping of sensors to action(s)
        to be evaluated by reward function, a set of tracks at given time and the time at which
        the actions would be carried out until.

        The metric returned is the total potential reduction in uncertainty across all tracks.

        Returns
        -------
        : float
            Metric of uncertainty for given configuration

        """

        if not len(self.rfis):
            return 0

        # Reward value
        config_metric, updated_tracks, predicted_sensors = self._compute_metric(config, tracks,
                                                                                timestamp, tracker=tracker)

        if timestamp == end_time:
            return config_metric, []

        timestamp += self.interval

        all_action_choices = dict()
        for sensor in predicted_sensors:
            # get action 'generator(s)'
            action_generators = sensor.actions(timestamp)
            # list possible action combinations for the sensor
            action_choices = list(it.product(*action_generators))
            # dictionary of sensors: list(action combinations)
            all_action_choices[sensor] = action_choices

        configs = list({sensor: action
                        for sensor, action in zip(all_action_choices.keys(), actionconfig)}
                        for actionconfig in it.product(*all_action_choices.values()))

        idx = np.random.choice(len(configs), 1, replace=False)
        next_config = configs[idx[0]]

        sim_config_metric, sim_configs = self._rollout(next_config, updated_tracks, timestamp, end_time, tracker)

        return config_metric + sim_config_metric, [next_config] + sim_configs


def _prob_detect_func(fovs):
    """Closure to return the probability of detection function for a given environment scan"""
    prob_detect = Probability(0.9)
    # Get the union of all field of views
    fovs_union = unary_union(fovs)
    if fovs_union.geom_type == 'MultiPolygon':
        fovs = [poly for poly in fovs_union]
    else:
        fovs = [fovs_union]

    paths = [Path(poly.boundary.coords) for poly in fovs]

    # Probability of detection nested function
    def prob_detect_func(state):
        for path_p in paths:
            if isinstance(state, ParticleState):
                prob_detect_arr = np.full((len(state),), Probability(0.1))
                points = state.state_vector[[0, 2], :].T
                inside_points = path_p.contains_points(points)
                prob_detect_arr[inside_points] = prob_detect
                return prob_detect_arr
            else:
                points = state.state_vector[[0, 2], :].T
                return prob_detect if np.all(path_p.contains_points(points)) \
                    else Probability(0)

    return prob_detect_func