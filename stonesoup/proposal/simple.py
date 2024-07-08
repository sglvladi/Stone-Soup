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


class KFasProposal(Proposal):
    pass


class RWProposal(PriorAsProposal, Proposal):
    """Random Walk proposal, proposal that uses the random walk to propagate
    the particles states.

    """

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
#        return self.transition_model.function(state, **kwargs)  # this is the transition model
        return multivariate_normal.rvs(
            np.zeros(state.ndim), state.covar, 1, random_state=None)


# class HMCProposal(Proposal):
#     """Hamiltonian monte carlo proposal
#         Fixed step Hamiltonian Monte Carlo proposal distribution for an SMC-sampler.
#         Moves samples around the target using the leapfrog integration method over a fixed
#         number of steps and a fixed step-size. In HMC omentum is usually used in Hamiltonian
#         MCMC, but here we assume (for the time being) that the mass is the identity matrix,
#         we therefore refer to it as velocity since momentum = mass * velocity
#
#     """
#
#     D: int = Property(
#         default=None,
#         doc="Distribution dimension")
#     target: np.array = Property(
#         default=None,
#         doc="Target distribution")
#     h: float = Property(
#         default=0.5,
#         doc="Step size used by Leapfrog integration")
#     steps: int = Property(
#         default=100,
#         doc="Number of steps used by the leapfrog integration")
#     v_dist = Property(
#         default=None,
#         doc="Velocity distribution")
#
#     # missing at the moment both element wise gradient and v_distribution needs to
#     # be more specified, a well the target needs to be a function
#
#         # Set a gradient object which we call each time we require it inside Leapfrog
#     #    self.grad = egrad(self.target.logpdf)
#
#         # Define an initial velocity disitrbution
#      #   self.v_dist = multivariate_normal(mean=np.zeros(D), cov=Cov * np.eye(D))
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # maybe here I need to put the velocity distribution
#
#     def pdf(self, v, v_cond=None):
#         """
#             Calculate pdf of velocity distribution
#         """
#
#         return self.v_dist.pdf(v)
#
#     def logpdf(self, v, v_cond=None):
#         """
#             Calculate logpdf of velocity distribution
#         """
#
#         return self.v_dist.logpdf(v)
#
#     def rvs(self, x_cond):
#         """
#             Returns a new sample state at the end of the integer number
#             of Leapfrog steps.
#         """
#
#         # Unpack position, initial velocity, and initial gradient
#         x = x_cond[0, :]
#         v = x_cond[1, :]
#         grad_x = x_cond[2, :]
#
#         x_new, v_new = self.generate_HMC_samples(x, v, grad_x)
#         return x_new, v_new, self.h * self.steps
#
#     def generate_HMC_samples(self, x, v, grad_x):
#         """
#             Handles the fixed step HMC proposal by generating a new sample after a
#             number of Leapfrog steps.
#         """
#
#         # Main leapfrog loop
#         for k in range(0, self.steps):
#             x, v, grad_x = self.leapfrog(x, v, grad_x)
#
#         return x, v
#
#     def leapfrog(self, x, v, grad_x):
#         """
#             Performs a single Leapfrog step returning the final position,
#             velocity and gradient.
#         """
#
#         v = np.add(v, (self.h / 2) * grad_x)
#         x = np.add(x, self.h * v)
#         grad_x = self.grad(x)
#         v = np.add(v, (self.h / 2) * grad_x)
#
#         return x, v, grad_x
#
#     def v_rvs(self, size):
#         """
#             Draw a number of samples equal to size from the velocity-
#             momentum distribution
#         """
#
#         return self.v_dist.rvs(size)
