from typing import Union

import numpy as np
from scipy.stats import multivariate_normal

from stonesoup.base import Property
from stonesoup.models.measurement import MeasurementModel
from stonesoup.models.transition import TransitionModel
from stonesoup.proposal.base import Proposal
from stonesoup.types.array import StateVector, StateVectors
from stonesoup.types.detection import Detection
from stonesoup.types.numeric import Probability
from stonesoup.types.state import State, GaussianState, SqrtGaussianState, ParticleState
from stonesoup.types.prediction import Prediction
from stonesoup.updater.base import Updater
from stonesoup.predictor.base import Predictor
from stonesoup.predictor.kalman import SqrtKalmanPredictor
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.numeric import Probability
from stonesoup.types.particle import Particle


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

        new_state = self.transition_model.function(state, **kwargs)
        # temp_state = ParticleState(state_vector=new_state)
        # print(kwargs['detection'])
        # prior_logpdf = self.prior_logpdf(temp_state, state, **kwargs)
        # loglikelihood = self.measurement_model.logpdf(kwargs['detection'], temp_state, **kwargs)
        # prop_logpdf = self.logpdf(temp_state, state, **kwargs)
        # new_weights = state.log_weights + prior_logpdf + loglikelihood - prop_logpdf
        return new_state # , new_weights


    def pdf(self, new_state: State, old_state: State, measurement: Detection = None,
            **kwargs) -> Union[Probability, np.ndarray]:
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

    def prior_pdf(self, new_state: State, old_state: State, measurement: Detection = None, **kwargs) \
            -> Union[Probability, np.ndarray]:
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

    def prior_logpdf(self, new_state: State, old_state: State, measurement: Detection = None, **kwargs) \
            -> Union[float, np.ndarray]:
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

    def logpdf(self, state1: State, state2: State, measurement: Detection = None, **kwargs) \
            -> Union[float, np.ndarray]:
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
        return self.prior_logpdf(state1, state2, measurement, **kwargs)


class KFasProposal(Proposal):
    """This proposal should inherit the Kalman properties
        to perform the various steps required
    """
    predictor: Predictor = Property(
        doc="predictor to use the various values")
    updater: Updater = Property(
        doc="Updater used for update the values")

    def rvs(self, state: State, **kwargs) -> Union[StateVector, StateVectors]:
        """Generate samples from the proposal.
            Use the kalman filter predictor
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

        # get the number of particles
        number_particles = state.state_vector.shape[1]
        old_weight = state.log_weight

        if not isinstance(kwargs['detection'], type(None)):
            temp_timestamp = kwargs['detection'].timestamp
        else:
            temp_timestamp = state.timestamp

        if isinstance(self.predictor, SqrtKalmanPredictor):
            kalman_prediction = self.predictor.predict(
                SqrtGaussianState(state.mean, state.covar, state.timestamp),
                timestamp=temp_timestamp, noise=kwargs['noise'])
        else:
            kalman_prediction = self.predictor.predict(
                GaussianState(state.mean, state.covar, state.timestamp),
                timestamp=temp_timestamp, noise=kwargs['noise'])

        # ok now I have the new state in KF shape
        posterior_state = self.updater.update(SingleHypothesis(kalman_prediction, kwargs['detection']))
        # need to sample from the posterior now

        samples = multivariate_normal.rvs(
            np.array(posterior_state.state_vector).reshape(-1),
            posterior_state.covar,
            size=number_particles)

#        print(samples.shape)
        particles = [Particle(sample.reshape(-1, 1),
                              weight=1) for sample in samples]

        pred_state = ParticleState(state_vector=None,
                                   particle_list=particles,
                                   timestamp=temp_timestamp)

        # p(y_k|x_k)
        # loglikelihood = measurement_model.logpdf(kwargs['detection'], pred_state,
        #                                          **kwargs)
        #
        # # p(x_k|x_k-1)
        # prior_logpdf = self.prior_logpdf(pred_state,
        #                                  state,  # acting as prior
        #                                  kwargs['detection'],
        #                                  **kwargs)
        #
        # # q(x_k|x_k-1, y_k)
        # prop_logpdf = self.logpdf(predicted_state,
        #                           state,
        #                           kwargs['detection'],
        #                           **kwargs)
        #
        # new_weights = old_weight +  prior_logpdf + loglikelihood - prop_logpdf

        return pred_state.state_vector

    def prior_pdf(self, new_state: State, old_state: State, measurement: Detection = None, **kwargs) \
            -> Union[Probability, np.ndarray]:
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
        return self.predictor.transition_model.pdf(new_state, old_state, **kwargs)

    def pdf(self, new_state: State, old_state: State, measurement: Detection = None,
            **kwargs) -> Union[Probability, np.ndarray]:
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

        return self.prior_pdf(new_state, old_state, measurement,  **kwargs)

    def prior_logpdf(self, new_state: State, old_state: State, measurement: Detection = None, **kwargs) \
            -> Union[float, np.ndarray]:
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
        return self.predictor.transition_model.logpdf(new_state, old_state, **kwargs)

    def logpdf(self, state1: State, state2: State, measurement: Detection = None, **kwargs) \
            -> Union[float, np.ndarray]:
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
        return self.prior_logpdf(state1, state2, measurement, **kwargs)
