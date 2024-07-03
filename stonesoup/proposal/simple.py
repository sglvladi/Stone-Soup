from typing import Union

import numpy as np

from stonesoup.base import Property
from stonesoup.models.measurement import MeasurementModel
from stonesoup.models.transition import TransitionModel
from stonesoup.proposal.base import Proposal
from stonesoup.types.array import StateVector, StateVectors
from stonesoup.types.detection import Detection
from stonesoup.types.numeric import Probability
from stonesoup.types.state import State


class PriorAsProposal(Proposal):
    """Proposal that uses the dynamics model as the importance density.

    This proposal uses the dynamics model to predict the next state, and then
    uses the predicted state as the prior for the measurement model.
    """
    transition_model: TransitionModel = Property(
        doc="The transition model used to make the prediction")
    measurement_model: MeasurementModel = Property(
        doc="The measurement model used to evaluate the likelihood")

    def rvs(self, state: State, **kwargs) -> Union[StateVector, StateVectors]:
        """Generate samples from the proposal.

        Parameters
        ----------
        state: :class:`~.State`
            The state to generate samples from.

        Returns
        -------
        : 2-D array of shape (:attr:`ndim`, ``num_samples``)
            A set of Np samples, generated from the model's noise
            distribution.
        """
        return self.transition_model.function(state, **kwargs)

    def pdf(self, new_state: State, old_state: State, **kwargs) -> Union[Probability, np.ndarray]:
        """Evaluate the probability density function of a state given the proposal.
        Parameters
        ----------
        state1: :class:`~.State`
            The state to evaluate the probability density function of a state given the proposal.
        state2: :class:`~.State`
            The state to evaluate the probability density function of a state given the proposal.
        Returns
        -------
        : float
            The probability density function of the state given the proposal.
        """
        return self.prior_pdf(new_state, old_state, **kwargs)

    def prior_pdf(self, new_state: State, old_state: State, **kwargs) -> Union[Probability, np.ndarray]:
        """Evaluate the probability density function of a state given the proposal.
        Parameters
        ----------
        state1: :class:`~.State`
            The state to evaluate the probability density function of a state given the proposal.
        state2: :class:`~.State`
            The state to evaluate the probability density function of a state given the proposal.
        Returns
        -------
        : float
            The probability density function of the state given the proposal.
        """
        return self.transition_model.pdf(new_state, old_state, **kwargs)

    def prior_logpdf(self, new_state: State, old_state: State, **kwargs) -> Union[float, np.ndarray]:
        """Evaluate the log likelihood of a state given the proposal.
        Parameters
        ----------
        state: :class:`~.StateVector`
            The state to evaluate the log likelihood of a state given the proposal.
        Returns
        -------
        : float
            The log likelihood of the state given the proposal.
        """
        return self.transition_model.logpdf(new_state, old_state, **kwargs)

    def logpdf(self, state1: State, state2: State, **kwargs) -> Union[float, np.ndarray]:
        """Evaluate the log probability density function of a state given the proposal.
        Parameters
        ----------
        state1: :class:`~.State`
            The state to evaluate the log probability density function of a state given the proposal.
        state2: :class:`~.State`
            The state to evaluate the log probability density function of a state given the proposal.
        Returns
        -------
        : float
            The log probability density function of the state given the proposal.
        """
        return self.prior_logpdf(state1, state2, **kwargs)
