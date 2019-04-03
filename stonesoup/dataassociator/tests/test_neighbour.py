# -*- coding: utf-8 -*-
import datetime

import pytest
import numpy as np

from ..neighbour import (NearestNeighbour, GlobalNearestNeighbour,
                         GaussianMixtureAssociator)
from ...types import (Track, Detection, GaussianState, SingleHypothesis,
                      MultipleHypothesis, TaggedWeightedGaussianState,
                      GaussianMixtureState)
from ...hypothesiser import GMMahalanobisDistanceHypothesiser


@pytest.fixture(params=[NearestNeighbour, GlobalNearestNeighbour])
def associator(request, hypothesiser):
    return request.param(hypothesiser)


def test_nearest_neighbour(associator):

    timestamp = datetime.datetime.now()
    t1 = Track([GaussianState(np.array([[0]]), np.array([[1]]), timestamp)])
    t2 = Track([GaussianState(np.array([[3]]), np.array([[1]]), timestamp)])
    d1 = Detection(np.array([[2]]))
    d2 = Detection(np.array([[5]]))

    tracks = {t1, t2}
    detections = {d1, d2}

    associations = associator.associate(tracks, detections, timestamp)

    # There should be 2 associations
    assert len(associations) == 2

    # Each track should associate with a unique detection
    associated_measurements = [hypothesis.measurement
                               for hypothesis in associations.values()
                               if hypothesis.measurement is not None]
    assert len(associated_measurements) == len(set(associated_measurements))


def test_missed_detection_nearest_neighbour(associator):

    timestamp = datetime.datetime.now()
    t1 = Track([GaussianState(np.array([[0]]), np.array([[1]]), timestamp)])
    t2 = Track([GaussianState(np.array([[3]]), np.array([[1]]), timestamp)])
    d1 = Detection(np.array([[20]]))

    tracks = {t1, t2}
    detections = {d1}

    associations = associator.associate(tracks, detections, timestamp)

    # Best hypothesis should be missed detection hypothesis
    assert all(hypothesis.measurement is None
               for hypothesis in associations.values())


def test_gm_associator(probability_predictor, probability_updater):

    hypothesiser = GMMahalanobisDistanceHypothesiser(
        probability_predictor, probability_updater, 10)
    associator = GaussianMixtureAssociator(hypothesiser)

    timestamp = datetime.datetime.now()
    gaussian_mixture = GaussianMixtureState(
        [TaggedWeightedGaussianState(
            np.array([[0]]), np.array([[0.7]]), timestamp, 0.4),
            TaggedWeightedGaussianState(
                np.array([[5]]), np.array([[0.7]]), timestamp, 0.3)])
    detection1 = Detection(np.array([[1.1]]),
                           timestamp=timestamp+datetime.timedelta(seconds=1))
    detection2 = Detection(np.array([[6.2]]),
                           timestamp=timestamp+datetime.timedelta(seconds=1))
    detections = {detection1, detection2}

    hypotheses = associator.associate(gaussian_mixture, detections, timestamp)

    # There are 4 hypotheses - 2 each associated with detection1/detection2
    assert all(isinstance(multi_hyp, MultipleHypothesis)
               for multi_hyp in hypotheses)
    assert all(isinstance(hyp, SingleHypothesis)
               for multi_hyp in hypotheses for hyp in multi_hyp)
    assert len(hypotheses) == 2
    assert len(hypotheses[0]) == 2
    assert len(hypotheses[1]) == 2

    # each SingleHypothesis has a distance attribute
    assert all(hyp.distance >= 0
               for multi_hyp in hypotheses for hyp in multi_hyp)
