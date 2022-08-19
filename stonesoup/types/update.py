# -*- coding: utf-8 -*-

from stonesoup.types.state import CreatableFromState, TwoStateParticleState
from ..base import Property
from .base import Type
from .hypothesis import Hypothesis
from .state import State, GaussianState, ParticleState, SqrtGaussianState, \
    WeightedGaussianState, TaggedWeightedGaussianState, InformationState, TwoStateGaussianState
from .mixture import GaussianMixture


class Update(Type, CreatableFromState):
    """ Update type

    The base update class. Updates are returned by :class:'~.Updater' objects
    and contain the information that was used to perform the updating"""

    hypothesis: Hypothesis = Property(doc="Hypothesis used for updating")


class StateUpdate(Update, State):
    """ StateUpdate type

    Most simple state update type, where everything only has time
    and a state vector. Requires a prior state that was updated,
    and the hypothesis used to update the prior.
    """


class GaussianStateUpdate(Update, GaussianState):
    """ GaussianStateUpdate type

    This is a simple Gaussian state update object, which, as the name
    suggests, is described by a Gaussian distribution.
    """


class SqrtGaussianStateUpdate(Update, SqrtGaussianState):
    """ SqrtGaussianStateUpdate type

    This is equivalent to a Gaussian state update object, but with the
    covariance of the Gaussian distribution stored in matrix square root
    form.
    """


class WeightedGaussianStateUpdate(Update, WeightedGaussianState):
    """ WeightedGaussianStateUpdate type

    This is a simple Gaussian state update object, which, as the name suggests, is described
    by a Gaussian distribution with an associated weight.
    """


class TaggedWeightedGaussianStateUpdate(Update, TaggedWeightedGaussianState):
    """ TaggedWeightedGaussianStateUpdate type

    This is a simple Gaussian state update object, which, as the name suggests, is described
    by a Gaussian distribution, with an associated weight and unique tag.
    """


class TwoStateGaussianStateUpdate(Update, TwoStateGaussianState):
    """ A Gaussian state object representing the predicted distribution
    :math:`p(x_{k+T}, x_{k} | Y)` """


class GaussianMixtureUpdate(Update, GaussianMixture):
    """ GaussianMixtureUpdate type

    This is a Gaussian mixture update object, which, as the name
    suggests, is described by a Gaussian mixture.
    """


class ParticleStateUpdate(Update, ParticleState):
    """ParticleStateUpdate type

    This is a simple Particle state update object.
    """


class InformationStateUpdate(Update, InformationState):
    """ InformationUpdate type

    This is a simple Information state update object, which, as the name
    suggests, is described by a precision matrix and its corresponding state vector.
    """

class TwoStateParticleStateUpdate(Update, TwoStateParticleState):
    """ A particle state object representing the updated distribution
    :math:`p(x_{k+T}, x_{k} | Y)` """
