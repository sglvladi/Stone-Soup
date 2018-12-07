# -*- coding: utf-8 -*-
from operator import attrgetter
import datetime

import numpy as np

from ..probability import PDAHypothesiser
from ...types import Track, Detection, GaussianState


def test_pda(predictor, updater):

    timestamp = datetime.datetime.now()
    track = Track([GaussianState(np.array([[0]]), np.array([[1]]), timestamp)])
    detection1 = Detection(np.array([[2]]))
    detection2 = Detection(np.array([[8]]))
    detections = {detection1, detection2}

    hypothesiser = PDAHypothesiser(predictor, updater,
                                   clutter_spatial_density=1.2e-2,
                                   prob_detect=0.9, prob_gate=0.99)

    hypotheses = hypothesiser.hypothesise(track, detections, timestamp)

    # There are 3 hypotheses - Detection 1, Detection 2, Missed Dectection
    assert len(hypotheses) == 3

    # Each hypothesis has a probability attribute
    assert all(hypothesis.probability >= 0 for hypothesis in hypotheses)

    # highest-probability hypothesis is Detection 1
    assert hypotheses[0].measurement is detection1

    # second-highest-probability hypothesis is Missed Detection
    assert hypotheses[1].measurement is None

    # lowest-probability hypothesis is Detection 2
    assert hypotheses[-1].measurement is detection2

    # The hypotheses are sorted correctly
    assert max(hypotheses, key=attrgetter('probability')) is hypotheses[0]
