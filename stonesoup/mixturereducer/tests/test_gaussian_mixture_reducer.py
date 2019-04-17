# -*- coding: utf-8 -*-
import numpy as np

from stonesoup.mixturereducer import GaussianMixtureReducer
from stonesoup.types.mixture import GaussianMixtureState
from stonesoup.types.state import (TaggedWeightedGaussianState,
                                   WeightedGaussianState)


def test_gaussianmixture_reducer_w_tags():
    dim = 4
    num_states = 10
    low_weight_states = [
        TaggedWeightedGaussianState(
            state_vector=np.array([[1.1], [2.1], [1.1], [2.1]]),
            covar=np.eye(dim),
            weight=1e-10,
            tag=i+1
        ) for i in range(num_states)
    ]
    states_to_be_merged = [
        TaggedWeightedGaussianState(
            state_vector=np.array([[1], [2], [1], [2]]) +
            np.random.rand(dim, 1)*1e-4,
            covar=np.eye(dim)*5,
            weight=0.05,
            tag=i+1
        ) for i in range(num_states, num_states*2)
    ]

    mixturestate = GaussianMixtureState(
        components=low_weight_states+states_to_be_merged
    )
    merge_threshold = 0.9
    prune_threshold = 1e-6
    mixturereducer = GaussianMixtureReducer(prune_threshold=prune_threshold,
                                            merge_threshold=merge_threshold)
    reduced_mixture_state = mixturereducer.reduce(mixturestate)
    assert len(reduced_mixture_state) == 1


def test_gaussianmixture_reducer():
    dim = 4
    num_states = 10
    low_weight_states = [
        WeightedGaussianState(
            state_vector=np.array([[1.1], [2.1], [1.1], [2.1]]),
            covar=np.eye(dim),
            weight=1e-10,
        ) for i in range(num_states)
    ]
    states_to_be_merged = [
        WeightedGaussianState(
            state_vector=np.array([[1], [2], [1], [2]]) +
            np.random.rand(dim, 1)*1e-4,
            covar=np.eye(dim)*5,
            weight=0.05,
        ) for i in range(num_states, num_states*2)
    ]

    mixturestate = GaussianMixtureState(
        components=low_weight_states+states_to_be_merged
    )
    merge_threshold = 0.9
    prune_threshold = 1e-6
    mixturereducer = GaussianMixtureReducer(prune_threshold=prune_threshold,
                                            merge_threshold=merge_threshold)
    reduced_mixture_state = mixturereducer.reduce(mixturestate)
    assert len(reduced_mixture_state) == 1
