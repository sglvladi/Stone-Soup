# -*- coding: utf-8 -*-
import numpy as np
import pytest


from ..mixture import GaussianMixtureState
from ..state import (GaussianState, WeightedGaussianState,
                     TaggedWeightedGaussianState)


def test_gaussianmixturestate_empty_components():
    mixturestate = GaussianMixtureState(components=[])
    assert len(mixturestate) == 0


def test_gaussianmixturestate():
    dim = 5
    num_states = 10
    states = [
        WeightedGaussianState(
            state_vector=np.random.rand(dim, 1),
            covar=np.eye(dim),
            weight=np.random.rand()
        ) for _ in range(num_states)
    ]
    mixturestate = GaussianMixtureState(components=states)
    assert len(mixturestate) == num_states
    for component1, component2 in zip(mixturestate, states):
        assert component1 == component2

    # Test iterator functionality implemented with __iter__ and __next__
    index = 0
    for component in mixturestate:
        assert component == states[index]
        index += 1

    # Test number_of_components
    assert len(mixturestate) == len(states)


def test_gaussianmixturestate_append():
    dim = 5
    state = WeightedGaussianState(
            state_vector=np.random.rand(dim, 1),
            covar=np.eye(dim),
            weight=np.random.rand()
        )

    mixturestate = GaussianMixtureState(components=[])
    mixturestate.append(state)
    assert len(mixturestate) == 1
    for component in mixturestate:
        assert component == state

    # Test iterator functionality implemented with __iter__ and __next__
    index = 0
    for component in mixturestate:
        assert component == state
        index += 1

    # Test number_of_components
    assert len(mixturestate) == 1


def test_gaussianmixturestate_with_tags():
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
    mixturestate = GaussianMixtureState(components=states)
    assert len(mixturestate) == num_states
    # Check equality of components of GaussianMixtureState and states
    for component1, component2 in zip(mixturestate, states):
        assert component1 == component2
    # Check each component has its proper tag assigned
    for i in range(num_states):
        assert mixturestate[i].tag == i+1


def test_gaussianmixturestate_with_single_component():
    dim = 5
    state = TaggedWeightedGaussianState(
            state_vector=np.random.rand(dim, 1),
            covar=np.eye(dim),
            weight=np.random.rand(),
            tag=2
        )
    mixturestate = GaussianMixtureState(components=state)
    assert len(mixturestate) == 1
    # Check equality of components of GaussianMixtureState and states
    for component in mixturestate:
        assert component == state


def test_gaussianmixturestate_wrong_type():
    dim = 5
    state = GaussianState(
            state_vector=np.random.rand(dim, 1),
            covar=np.eye(dim),
        )
    with pytest.raises(ValueError):
        GaussianMixtureState(components=state)


def test_gaussianmixturestate_extract_states_all():
    dim = 5
    num_states = 10
    states = [
        WeightedGaussianState(
            state_vector=np.random.rand(dim, 1),
            covar=np.eye(dim),
            weight=1
        ) for _ in range(num_states)
    ]
    mixturestate = GaussianMixtureState(components=states)
    extracted_states = mixturestate.extract_states()
    for component1, component2 in zip(mixturestate, extracted_states):
        assert component1 == component2


def test_gaussianmixturestate_extract_states_some():
    dim = 5
    num_states = 10
    weights = np.linspace(0, 1.0, num=num_states)
    states = [
        WeightedGaussianState(
            state_vector=np.random.rand(dim, 1),
            covar=np.eye(dim),
            weight=weights[i]
        ) for i in range(num_states)
    ]
    extraction_threshold = 0.5
    expected_states = np.array(states)
    expected_states = []
    for state in states:
        if state.weight > extraction_threshold:
            expected_states.append(state)
    number_of_expected_states = len(expected_states)

    mixturestate = \
        GaussianMixtureState(components=states,
                             extraction_threshold=extraction_threshold)
    extracted_states = mixturestate.extract_states()
    assert len(extracted_states) == number_of_expected_states
    for component1, component2 in zip(expected_states, extracted_states):
        assert component1 == component2


def test_gaussianmixturestate_get_and_set_item():
    dim = 5
    num_states = 10
    weights = np.linspace(0, 1.0, num=num_states)
    states = [
        WeightedGaussianState(
            state_vector=np.random.rand(dim, 1),
            covar=np.eye(dim),
            weight=weights[i]
        ) for i in range(num_states)
    ]
    mixturestate = GaussianMixtureState(components=states)
    new_component = WeightedGaussianState(
        state_vector=np.random.rand(dim, 1),
        covar=np.eye(dim),
        weight=1
        )
    mixturestate[0] = new_component
    assert mixturestate[0] == new_component
    with pytest.raises(ValueError):
        assert mixturestate["Test"]


def test_gaussianmixturestate_contains_item():
    dim = 5
    num_states = 10
    weights = np.linspace(0, 1.0, num=num_states)
    states = [
        WeightedGaussianState(
            state_vector=np.random.rand(dim, 1),
            covar=np.eye(dim),
            weight=weights[i]
        ) for i in range(num_states)
    ]
    mixturestate = GaussianMixtureState(components=states)
    check_component_in = mixturestate[0]
    check_component_not_in = WeightedGaussianState(
        state_vector=np.random.rand(dim, 1),
        covar=np.eye(dim),
        weight=np.random.rand(1, 1)
    )
    assert check_component_in in mixturestate
    assert check_component_not_in not in mixturestate
    with pytest.raises(ValueError):
        assert 1 in mixturestate
