# -*- coding: utf-8 -*-
from operator import attrgetter
import datetime

import numpy as np

from ..distance import DistanceHypothesiser
from ...types.detection import Detection
from ...types.state import GaussianState
from ...types.track import Track
from ... import measures


def test_mahalanobis(predictor, updater):

    timestamp = datetime.datetime.now()
    track = Track([GaussianState(np.array([[0]]), np.array([[1]]), timestamp)])
    detection1 = Detection(np.array([[2]]))
    detection2 = Detection(np.array([[3]]))
    detections = {detection1, detection2}

    measure = measures.Mahalanobis()
    hypothesiser = DistanceHypothesiser(predictor, updater, measure=measure)

    hypotheses = hypothesiser.hypothesise(track, detections, timestamp)

    # There are 3 hypotheses - Detection 1, Detection 2, Missed Dectection
    assert len(hypotheses) == 3

    # There is a missed detection hypothesis
    assert any(not hypothesis.measurement for hypothesis in hypotheses)

    # Each hypothesis has a distance attribute
    assert all(hypothesis.distance >= 0 for hypothesis in hypotheses)

    # The hypotheses are sorted correctly
    assert min(hypotheses, key=attrgetter('distance')) is hypotheses[0]


def test_gm_mahalanobis(predictor, updater):

    timestamp = datetime.datetime.now()
    gaussian_mixture = GaussianMixtureState(
        [WeightedGaussianState(
            np.array([[0.3]]), np.array([[1]]), timestamp, 0.4),
            WeightedGaussianState(
                np.array([[5]]), np.array([[0.5]]), timestamp, 0.3)])
    detection1 = Detection(np.array([[1]]),
                           timestamp=timestamp+datetime.timedelta(seconds=1))
    detection2 = Detection(np.array([[6.2]]),
                           timestamp=timestamp+datetime.timedelta(seconds=1))
    detections = {detection1, detection2}

    hypothesiser = GMMahalanobisDistanceHypothesiser(predictor, updater, 10)

    hypotheses = hypothesiser.hypothesise(gaussian_mixture,
                                          detections, timestamp)

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

    # sanity-check the values returned by the hypothesiser
    assert hypotheses[0][0].distance < 10
    assert hypotheses[0][1].distance > 0
    assert hypotheses[1][0].distance > 0
    assert hypotheses[1][1].distance < 10
