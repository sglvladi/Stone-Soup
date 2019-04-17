# -*- coding: utf-8 -*-
import datetime

import numpy as np

from ..gmphd import GMPHDTargetTracker
from ...types import TaggedWeightedGaussianState, Detection


def test_gmphd_multi_target_tracker_init(
        predictor, updater):
    association_threshold = 10
    merge_threshold = 4
    prune_threshold = 1e-6
    GMPHDTargetTracker(
        predictor=predictor,
        updater=updater,
        association_threshold=association_threshold,
        merge_threshold=merge_threshold,
        prune_threshold=prune_threshold)


def test_gmphd_multi_target_tracker_init_no_kwargs(
        predictor, updater):
    GMPHDTargetTracker(
        predictor=predictor,
        updater=updater)


def test_gmphd_multi_target_tracker_init_w_components(
        predictor, updater):
    association_threshold = 10
    merge_threshold = 4
    prune_threshold = 1e-6
    dim = 5
    num_states = 10
    states = [
        TaggedWeightedGaussianState(
            state_vector=np.random.rand(dim, 1),
            covar=np.eye(dim),
            weight=np.random.rand(),
            tag=i+1
        ) for i in range(num_states)
    ]
    GMPHDTargetTracker(
        predictor=predictor,
        updater=updater,
        association_threshold=association_threshold,
        merge_threshold=merge_threshold,
        prune_threshold=prune_threshold,
        components=states)


def test_gmphd_multi_target_tracker_card(
        predictor, updater):
    association_threshold = 10
    merge_threshold = 4
    prune_threshold = 1e-6
    tracker = GMPHDTargetTracker(
        predictor=predictor,
        updater=updater,
        association_threshold=association_threshold,
        merge_threshold=merge_threshold,
        prune_threshold=prune_threshold)
    tracker.estimated_number_of_targets
    state = TaggedWeightedGaussianState(
            state_vector=np.random.rand(4, 1),
            covar=np.eye(4),
            weight=1,
            tag=2
        )
    tracker.gaussian_mixture.append(state)
    tracker.estimated_number_of_targets


def test_gmphd_multi_target_tracker_cycle(
        predictor, updater):
    association_threshold = 10
    merge_threshold = 0.001
    prune_threshold = 1e-6
    timestamp = datetime.datetime.now()
    components = [
        TaggedWeightedGaussianState(
            np.array([[0]]),
            np.array([[0.7]]),
            timestamp,
            0.4,
            tag=2),
        TaggedWeightedGaussianState(
            np.array([[5]]),
            np.array([[0.7]]),
            timestamp,
            0.3,
            tag=3)]
    birth_mean = np.array([[30]])
    birth_covar = np.array([[50]])
    birth_component = TaggedWeightedGaussianState(
        birth_mean,
        birth_covar,
        weight=1,
        tag=1,
        timestamp=timestamp)
    tracker = GMPHDTargetTracker(
        predictor=predictor,
        updater=updater,
        association_threshold=association_threshold,
        merge_threshold=merge_threshold,
        prune_threshold=prune_threshold,
        components=components,
        birth_component=birth_component
        )
    tracker.predict(timestamp+datetime.timedelta(seconds=1))
    # Check predicted values
    for component in tracker.gaussian_mixture:
        if component.tag == 2:
            assert component.state_vector == components[0].state_vector+1
        if component.tag == 3:
            assert component.state_vector == components[1].state_vector+1
    detection1 = Detection(np.array([[1.1]]),
                           timestamp=timestamp+datetime.timedelta(seconds=1))
    detection2 = Detection(np.array([[6]]),
                           timestamp=timestamp+datetime.timedelta(seconds=1))
    detections = {detection1, detection2}
    tracker.update(detections, timestamp+datetime.timedelta(seconds=1))
    # Check tracks
    for key in tracker.target_tracks:
        if key == 2:
            assert tracker.target_tracks[key] == tracker.gaussian_mixture[0]
        elif key == 3:
            assert tracker.target_tracks[key] == tracker.gaussian_mixture[1]
