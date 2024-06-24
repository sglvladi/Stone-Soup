from abc import abstractmethod
from typing import Union

import numpy as np

from stonesoup.base import Base, Property
from stonesoup.models.measurement import MeasurementModel
from stonesoup.models.transition import TransitionModel
from stonesoup.types.array import StateVector, StateVectors
from stonesoup.types.detection import Detection
from stonesoup.types.numeric import Probability
from stonesoup.types.state import State


class Proposal(Base):

    transition_model: TransitionModel = Property(doc="The transition model")
    measurement_model: MeasurementModel = Property(doc="The measurement model")

    def _get_measurement_model(self, detection: Detection) -> MeasurementModel:
        """Get the measurement model for the given detection.
        Parameters
        ----------
        detection: :class:`~.Detection`
            The detection to get the measurement model for.
        Returns
        -------
        : :class:`~.MeasurementModel`
            The measurement model for the given detection.
        """
        if detection.measurement_model is not None:
            return detection.measurement_model
        return self.measurement_model

    @abstractmethod
    def rvs(self, state: State, measurement: Detection = None, **kwargs) \
            -> Union[StateVector, StateVectors]:
        r"""Proposal noise/sample generation function
        Generates samples from the proposal.
        Parameters
        ----------
        state: :class:`~.State`
            The state to generate samples from.
        measurement: :class:`~.State`
            The measurement to generate samples from.
        num_samples: scalar, optional
            The number of samples to be generated (the default is 1)
        Returns
        -------
         : 2-D array of shape (:attr:`ndim`, ``num_samples``)
            A set of Np samples, generated from the model's noise
            distribution.
        """
        raise NotImplementedError

    @abstractmethod
    def pdf(self, new_state: State, old_state: State, measurement: Detection = None, **kwargs) \
            -> Union[Probability, np.ndarray]:
        r"""Proposal probability density function
        Evaluates the probability density function of a state given the proposal.
        Parameters
        ----------
        new_state: :class:`~.State`
            The state to evaluate the probability density function of a state given the proposal.
        old_state: :class:`~.State`
            The state to evaluate the probability density function of a state given the proposal.
        measurement: :class:`~.State`
            The measurement to evaluate the probability density function of a state given the
            proposal.
        Returns
        -------
        : float
            The probability density function of the state given the proposal.
        """
        raise NotImplementedError

    def likelihood(self, measurement: Detection, state: State, **kwargs) -> Union[Probability, np.ndarray]:
        r"""Proposal likelihood function
        Evaluates the likelihood of a state given the proposal.
        Parameters
        ----------
        measurement: :class:`~.State`
            The state to evaluate the likelihood of a state given the proposal.
        state: :class:`~.State`
            The state to evaluate the likelihood of a state given the proposal.
        Returns
        -------
        : float
            The likelihood of the state given the proposal.
        """
        return self._get_measurement_model(measurement).pdf(measurement, state, **kwargs)

    def prior_pdf(self, new_state: State, old_state: State, **kwargs) -> Union[Probability, np.ndarray]:
        r"""Proposal prior probability density function
        Evaluates the prior probability density function of a state given the proposal.
        Parameters
        ----------
        new_state: :class:`~.State`
            The state to evaluate the prior probability density function of a state given the proposal.
        old_state: :class:`~.State`
            The state to evaluate the prior probability density function of a state given the proposal.
        Returns
        -------
        : float
            The prior probability density function of the state given the proposal.
        """
        return self.transition_model.pdf(new_state, old_state, **kwargs)

    def loglikelihood(self, measurement: Detection, state: State, **kwargs) -> Union[float, np.ndarray]:
        r"""Proposal log likelihood function
        Evaluates the log likelihood of a state given the proposal.
        Parameters
        ----------
        measurement: :class:`~.State`
            The state to evaluate the log likelihood of a state given the proposal.
        state: :class:`~.State`
            The state to evaluate the log likelihood of a state given the proposal.
        Returns
        -------
        : float
            The log likelihood of the state given the proposal.
        """
        return np.log(self.likelihood(measurement, state, **kwargs))

    def logpdf(self, new_state: State, old_state: State, measurement: Detection = None, **kwargs) \
            -> Union[float, np.ndarray]:
        r"""Proposal log probability density function
        Evaluates the log probability density function of a state given the proposal.
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
        return np.log(self.pdf(new_state, old_state, measurement, **kwargs))

    def prior_logpdf(self, new_state: State, old_state: State, **kwargs) -> Union[float, np.ndarray]:
        r"""Proposal prior log probability density function
        Evaluates the prior log probability density function of a state given the proposal.
        Parameters
        ----------
        new_state: :class:`~.State`
            The state to evaluate the prior log probability density function of a state given the proposal.
        old_state: :class:`~.State`
            The state to evaluate the prior log probability density function of a state given the proposal.
        Returns
        -------
        : float
            The prior log probability density function of the state given the proposal.
        """
        return np.log(self.prior_pdf(new_state, old_state, **kwargs))

    def weight_update(self, measurement: Detection, new_state: State, old_state: State, **kwargs) \
            -> Union[Probability, np.ndarray]:
        r"""Proposal weight update function
        Evaluates the weight update of a state given the proposal.
        Parameters
        ----------
        measurement: :class:`~.State`
            The state to evaluate the weight update of a state given the proposal.
        new_state: :class:`~.State`
            The state to evaluate the weight update of a state given the proposal.
        old_state: :class:`~.State`
            The state to evaluate the weight update of a state given the proposal.
        Returns
        -------
        : float
            The weight update of the state given the proposal.
        """
        return (self.prior_pdf(new_state, old_state, **kwargs)
                * self.likelihood(measurement, new_state, **kwargs)
                / self.pdf(new_state, old_state, measurement, **kwargs))

    def log_weight_update(self, measurement: Detection, new_state: State, old_state: State, **kwargs) \
            -> Union[float, np.ndarray]:
        r"""Proposal log weight update function
        Evaluates the log weight update of a state given the proposal.
        Parameters
        ----------
        measurement: :class:`~.State`
            The state to evaluate the log weight update of a state given the proposal.
        new_state: :class:`~.State`
            The state to evaluate the log weight update of a state given the proposal.
        old_state: :class:`~.State`
            The state to evaluate the log weight update of a state given the proposal.
        Returns
        -------
        : float
            The log weight update of the state given the proposal.
        """
        return (self.prior_logpdf(new_state, old_state, **kwargs)
                + self.loglikelihood(measurement, new_state, **kwargs)
                - self.logpdf(new_state, old_state, measurement, **kwargs))
    