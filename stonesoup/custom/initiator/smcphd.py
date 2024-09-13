import logging
import warnings
from copy import copy
from typing import List, Any, Union, Callable, Optional

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from stonesoup.base import Base, Property
from stonesoup.initiator import Initiator
from stonesoup.models.measurement import MeasurementModel
from stonesoup.models.transition import TransitionModel
from stonesoup.resampler import Resampler
from stonesoup.types.angle import Bearing
from stonesoup.types.array import StateVectors
from stonesoup.types.detection import Detection, MissedDetection
from stonesoup.types.hypothesis import SingleProbabilityHypothesis
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.numeric import Probability
from stonesoup.types.prediction import Prediction
from stonesoup.types.state import State
from stonesoup.types.track import Track
from stonesoup.types.update import Update, GaussianStateUpdate, ParticleStateUpdate


class SMCPHDFilter(Base):
    """
    Sequential Monte-Carlo (SMC) PHD filter implementation, based on [1]_

     .. [1] Ba-Ngu Vo, S. Singh and A. Doucet, "Sequential Monte Carlo Implementation of the
            PHD Filter for Multi-target Tracking," Sixth International Conference of Information
            Fusion, 2003. Proceedings of the, 2003, pp. 792-799, doi: 10.1109/ICIF.2003.177320.
    .. [2]  P. Horridge and S. Maskell,  “Using a probabilistic hypothesis density filter to
            confirm tracks in a multi-target environment,” in 2011 Jahrestagung der Gesellschaft
            fr Informatik, October 2011.
    """

    transition_model: TransitionModel = Property(doc='The transition model')
    measurement_model: MeasurementModel = Property(doc='The measurement model')
    prob_detect: Union[Probability, Callable[[State], Probability]] = Property(
        default=Probability(0.85),
        doc="Target Detection Probability")
    prob_death: Probability = Property(doc='The probability of death')
    prob_birth: Probability = Property(doc='The probability of birth')
    birth_rate: float = Property(
        doc='The birth rate (i.e. number of new/born targets at each iteration(')
    birth_density: State = Property(
        doc='The birth density (i.e. density from which we sample birth particles)')
    clutter_intensity: float = Property(doc='The clutter intensity per unit volume')
    resampler: Resampler = Property(default=None, doc='Resampler to prevent particle degeneracy')
    num_samples: int = Property(doc='The number of samples. Default is 1024', default=1024)
    birth_scheme: str = Property(
        doc='The scheme for birth particles. Options are "expansion" | "mixture". '
            'Default is "expansion"',
        default='expansion'
    )
    scale_birth_weights: bool = Property(
        doc="Whether to scale the birth weights by their likelihood, given the birth density. "
            "Setting this to True can cause issues if the defined birth density is not a good "
            "approximation to the true birth density. On the other hand, setting this to False "
            "can lead to premature initialization of targets.",
        default=False
    )
    seed: Optional[Union[int, np.random.RandomState]] = Property(
        default=None,
        doc="Seed or state for random number generation. If defined as an integer, "
            "it will be used to create a numpy RandomState. Or it can be defined directly "
            "as a RandomState (useful if you want to pass one of the random state's "
            "functions as the :attr:`distribution`).")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not callable(self.prob_detect):
            prob_detect = copy(self.prob_detect)
            self.prob_detect = lambda state: prob_detect
        if isinstance(self.seed, int):
            self.random_state = np.random.RandomState(self.seed)
        elif isinstance(self.seed, np.random.RandomState):
            self.random_state = self.seed
        else:
            self.random_state = None

    def predict(self, state, timestamp):
        """
        Predict the next state of the target density

        Parameters
        ----------
        state: :class:`~.State`
            The current state of the target
        timestamp: :class:`datetime.datetime`
            The time at which the state is valid

        Returns
        -------
        : :class:`~.State`
            The predicted next state of the target
        """

        num_samples = len(state)
        log_prior_weights = state.log_weight
        time_interval = timestamp - state.timestamp

        # Predict particles forward
        pred_particles_sv = self.transition_model.function(state,
                                                           time_interval=time_interval,
                                                           noise=True)

        if self.birth_scheme == 'expansion':
            # Expansion birth scheme, as described in [1]
            # Compute number of birth particles (J_k) as a fraction of the number of particles
            num_birth = round(float(self.prob_birth) * self.num_samples)

            # Sample birth particles
            params = {'num_samples': num_birth}
            birth_particles = self.birth_density.sample(timestamp=timestamp, params=params)
            birth_particles_sv = birth_particles.state_vector
            log_birth_weights = birth_particles.log_weight + np.log(self.birth_rate)

            # Surviving particle weights
            log_prob_survive = -float(self.prob_death) * time_interval.total_seconds()
            log_pred_weights = log_prob_survive + log_prior_weights

            # Append birth particles to predicted ones
            pred_particles_sv = StateVectors(
                np.concatenate((pred_particles_sv, birth_particles_sv), axis=1))
            log_pred_weights = np.concatenate((log_pred_weights, log_birth_weights))
        else:
            # Flip a coin for each particle to decide if it gets replaced by a birth particle
            birth_inds = np.flatnonzero(self.random_state.binomial(1, float(self.prob_birth), num_samples,))

            # Sample birth particles and replace in original state vector matrix
            num_birth = len(birth_inds)
            params = {'num_samples': num_birth}
            birth_particles = self.birth_density.sample(timestamp=timestamp, params=params)
            birth_particles_sv = birth_particles.state_vector
            pred_particles_sv[:, birth_inds] = birth_particles_sv

            # Process weights
            prob_survive = np.exp(-float(self.prob_death) * time_interval.total_seconds())
            birth_weight = self.birth_rate / num_samples
            log_pred_weights = np.log(prob_survive + birth_weight) + log_prior_weights
            pred_particles_sv[:, birth_inds] = birth_particles

        prediction = Prediction.from_state(state, state_vector=pred_particles_sv,
                                           log_weight=log_pred_weights,
                                           timestamp=timestamp, particle_list=None,
                                           transition_model=self.transition_model)

        return prediction

    def update(self, prediction, detections, timestamp, meas_weights=None):
        """
        Update the predicted state of the target density with the given detections

        Parameters
        ----------
        prediction: :class:`~.State`
            The predicted state of the target
        detections: :class:`~.Detection`
            The detections at the current time step
        timestamp: :class:`datetime.datetime`
            The time at which the update is valid
        meas_weights: :class:`np.ndarray`
            The weights of the measurements

        Returns
        -------
        : :class:`~.State`
            The updated state of the target
        """

        log_weights_per_hyp = self.get_log_weights_per_hypothesis(prediction, detections,
                                                                  meas_weights)

        # Construct hypothesis objects (StoneSoup specific)
        single_hypotheses = [
            SingleProbabilityHypothesis(prediction,
                                        measurement=MissedDetection(timestamp=timestamp),
                                        probability=log_weights_per_hyp[:, 0])]
        for i, detection in enumerate(detections):
            single_hypotheses.append(
                SingleProbabilityHypothesis(prediction,
                                            measurement=detection,
                                            probability=log_weights_per_hyp[:, i + 1])
            )
        hypothesis = MultipleHypothesis(single_hypotheses, normalise=False)

        # Update weights Eq. (8) of [1]
        # w_k^i = \sum_{z \in Z_k}{w^{n,i}}, where i is the index of z in Z_k
        log_post_weights = logsumexp(log_weights_per_hyp, axis=1)

        # Resample
        log_num_targets = logsumexp(log_post_weights)  # N_{k|k}
        update = copy(prediction)
        # Normalize weights
        update.log_weight = log_post_weights - log_num_targets
        if self.resampler is not None:
            update = self.resampler.resample(update, self.num_samples)  # Resample
        # De-normalize
        update.log_weight = update.log_weight + log_num_targets

        return Update.from_state(
            state=prediction,
            state_vector=update.state_vector,
            log_weight=update.log_weight,
            particle_list=None,
            hypothesis=hypothesis,
            timestamp=timestamp)

    def iterate(self, state, detections: List[Detection], timestamp):
        """
        Iterate the filter over the given state and detections

        Parameters
        ----------
        state: :class:`~.State`
            The predicted state of the target
        detections: :class:`~.Detection`
            The detections at the current time step
        timestamp: :class:`datetime.datetime`
            The time at which the update is valid

        Returns
        -------
        : :class:`~.State`
            The updated state of the target
        """
        prediction = self.predict(state, timestamp)
        update = self.update(prediction, detections, timestamp)
        return update

    def get_measurement_loglikelihoods(self, prediction, detections, meas_weights):
        num_samples = prediction.state_vector.shape[1]
        # Compute g(z|x) matrix as in [1]
        g = np.zeros((num_samples, len(detections)))
        for i, detection in enumerate(detections):
            if not meas_weights[i]:
                g[:, i] = -np.inf
                continue
            g[:, i] = detection.measurement_model.logpdf(detection, prediction,
                                                         noise=True)
        return g

    def get_log_weights_per_hypothesis(self, prediction, detections, meas_weights, *args,
                                       **kwargs):
        num_samples = prediction.state_vector.shape[1]
        if meas_weights is None:
            meas_weights = np.array([Probability(1) for _ in range(len(detections))])

        # Compute g(z|x) matrix as in [1]
        g = self.get_measurement_loglikelihoods(prediction, detections, meas_weights)

        # Get probability of detection
        prob_detect = self.prob_detect(prediction)

        # Calculate w^{n,i} Eq. (20) of [2]
        Ck = prob_detect.log() + g + prediction.log_weight[:, np.newaxis]
        C = logsumexp(Ck, axis=0)
        k = np.log(self.clutter_intensity)
        C_plus = np.logaddexp(C, k)
        log_weights_per_hyp = np.full((num_samples, len(detections) + 1), -np.inf)
        log_weights_per_hyp[:, 0] = np.log(1 - prob_detect) + prediction.log_weight
        if len(detections):
            log_weights_per_hyp[:, 1:] = np.log(np.asfarray(meas_weights)) + Ck - C_plus

        return log_weights_per_hyp


class ISMCPHDFilter(SMCPHDFilter):

    def predict(self, state, timestamp):
        """
        Predict the next state of the target density

        Parameters
        ----------
        state: :class:`~.State`
            The current state of the target
        timestamp: :class:`datetime.datetime`
            The time at which the state is valid

        Returns
        -------
        : :class:`~.State`
            The predicted next state of the target
        """

        log_prior_weights = state.log_weight
        time_interval = timestamp - state.timestamp

        # Predict particles forward
        try:
            pred_particles_sv = self.transition_model.function(state,
                                                               time_interval=time_interval,
                                                               noise=True)
        except Exception as e:
            covar = self.transition_model.covar(time_interval=time_interval)
            logging.debug(f'Time interval: {time_interval} - Tmodel covar: {covar}')
            raise e

        # Surviving particle weights
        log_prob_survive = -float(self.prob_death) * time_interval.total_seconds()
        log_pred_weights = log_prob_survive + log_prior_weights

        prediction = Prediction.from_state(state, state_vector=pred_particles_sv,
                                           log_weight=log_pred_weights,
                                           timestamp=timestamp, particle_list=None,
                                           transition_model=self.transition_model)
        prediction.birth_idx = state.birth_idx if hasattr(state, 'birth_idx') else []
        return prediction

    def update(self, prediction, detections, timestamp, meas_weights=None):
        """
        Update the predicted state of the target density with the given detections

        Parameters
        ----------
        prediction: :class:`~.State`
            The predicted state of the target
        detections: :class:`~.Detection`
            The detections at the current time step
        timestamp: :class:`datetime.datetime`
            The time at which the update is valid
        meas_weights: :class:`np.ndarray`
            The weights of the measurements

        Returns
        -------
        : :class:`~.State`
            The updated state of the target
        """
        num_persistent = prediction.state_vector.shape[1]
        birth_state = self.get_birth_state(prediction, detections, timestamp)

        log_weights_per_hyp = self.get_log_weights_per_hypothesis(prediction, detections,
                                                                  meas_weights,
                                                                  birth_state)

        # Construct hypothesis objects (StoneSoup specific)
        single_hypotheses = [
            SingleProbabilityHypothesis(prediction,
                                        measurement=MissedDetection(timestamp=timestamp),
                                        probability=log_weights_per_hyp[:num_persistent, 0])]
        for i, detection in enumerate(detections):
            single_hypotheses.append(
                SingleProbabilityHypothesis(
                    prediction,
                    measurement=detection,
                    probability=log_weights_per_hyp[:num_persistent, i + 1]
                )
            )
        hypothesis = MultipleHypothesis(single_hypotheses, normalise=False)

        # Update weights Eq. (8) of [1]
        # w_k^i = \sum_{z \in Z_k}{w^{n,i}}, where i is the index of z in Z_k
        log_post_weights = logsumexp(log_weights_per_hyp, axis=1)
        log_post_weights_pers = log_post_weights[:num_persistent]
        log_post_weights_birth = log_post_weights[num_persistent:]

        # Resample persistent
        log_num_targets_pers = logsumexp(log_post_weights_pers)  # N_{k|k}
        update = copy(prediction)
        # Normalize weights
        update.log_weight = log_post_weights_pers - log_num_targets_pers
        if self.resampler is not None:
            update = self.resampler.resample(update, self.num_samples)  # Resample
        # De-normalize
        update.log_weight = update.log_weight + log_num_targets_pers

        if len(detections):
            # Resample birth
            log_num_targets_birth = logsumexp(log_post_weights_birth)  # N_{k|k}
            update2 = copy(birth_state)
            # Normalize weights
            update2.log_weight = log_post_weights_birth - log_num_targets_birth
            if self.resampler is not None:
                update2 = self.resampler.resample(update2,
                                                  update2.state_vector.shape[1])  # Resample
            # De-normalize
            update2.log_weight = update2.log_weight + log_num_targets_birth

            full_update = Update.from_state(
                state=prediction,
                state_vector=StateVectors(np.hstack((update.state_vector, update2.state_vector))),
                log_weight=np.hstack((update.log_weight, update2.log_weight)),
                particle_list=None,
                hypothesis=hypothesis,
                timestamp=timestamp)
        else:
            full_update = Update.from_state(
                state=prediction,
                state_vector=update.state_vector,
                log_weight=update.log_weight,
                particle_list=None,
                hypothesis=hypothesis,
                timestamp=timestamp)
        full_update.birth_idx = [i for i in range(len(update.weight), len(full_update.weight))]
        return full_update

    def get_birth_state(self, prediction, detections, timestamp):
        # Sample birth particles
        num_birth = round(float(self.prob_birth) * self.num_samples)
        birth_particles = np.zeros((prediction.state_vector.shape[0], 0))
        birth_weights = np.zeros((0,))
        if len(detections):
            num_birth_per_detection = num_birth // len(detections)
            for i, detection in enumerate(detections):
                if i == len(detections) - 1:
                    num_birth_per_detection += num_birth % len(detections)
                mu = np.zeros((prediction.state_vector.shape[0], 1))
                cov = self.birth_density.params['birth_density'].covar
                mu[0::2, :] = detection.state_vector
                cov[0::2, 0::2] = detection.measurement_model.covar()
                birth_particles_i = np.atleast_2d(
                    multivariate_normal.rvs(mu.ravel(),
                                            cov,
                                            num_birth_per_detection,
                                            random_state=self.random_state)).T
                birth_particles_i[0, :] = [Bearing(bearing) for bearing in birth_particles_i[0, :]]
                birth_particles_i = np.vstack((
                    [Bearing(bearing) for bearing in birth_particles_i[0, :]],
                    birth_particles_i[1:, :])
                )
                birth_weights_i = np.full((num_birth_per_detection,),
                                          np.log(self.birth_rate / num_birth))
                if self.scale_birth_weights:
                    birth_weights_i += multivariate_normal.logpdf(birth_particles_i.T,
                                                                  mu.ravel(),
                                                                  cov,
                                                                  allow_singular=True)
                birth_particles = np.hstack((birth_particles, birth_particles_i))
                birth_weights = np.hstack((birth_weights, birth_weights_i))
        else:
            birth_state = self.birth_density.sample(timestamp=timestamp,
                                                    params={'num_samples': num_birth})
            birth_particles = birth_state.state_vector
            birth_weights = np.full((num_birth,), np.log(self.birth_rate / num_birth))

        # birth_weights = np.full((num_birth,), Probability(self.birth_rate / num_birth))
        birth_particles = StateVectors(birth_particles)
        birth_state = Prediction.from_state(prediction,
                                            state_vector=birth_particles,
                                            log_weight=birth_weights,
                                            timestamp=timestamp, particle_list=None,
                                            transition_model=self.transition_model)
        return birth_state

    def get_log_weights_per_hypothesis(self, prediction, detections, meas_weights, birth_state,
                                       *args, **kwargs):
        num_samples = prediction.state_vector.shape[1]
        if meas_weights is None:
            meas_weights = np.array([Probability(1) for _ in range(len(detections))])

        # Compute g(z|x) matrix as in [1]
        g = self.get_measurement_loglikelihoods(prediction, detections, meas_weights)

        # Get probability of detection
        prob_detect = np.asfarray(self.prob_detect(prediction))

        # Catch divide by zero warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'divide by zero encountered in log')
            # Calculate w^{n,i} Eq. (20) of [2]
            try:
                Ck = np.log(prob_detect[:, np.newaxis]) + g \
                     + np.log(prediction.weight[:, np.newaxis].astype(float))
            except IndexError:
                Ck = np.log(prob_detect) + g \
                     + np.log(prediction.weight[:, np.newaxis].astype(float))
        C = logsumexp(Ck, axis=0)
        Ck_birth = np.tile(birth_state.log_weight[:, np.newaxis], len(detections))
        C_birth = logsumexp(Ck_birth, axis=0)
        k = np.log([detection.metadata['clutter_density']
                    if 'clutter_density' in detection.metadata else self.clutter_intensity
                    for detection in detections])
        C_plus = np.logaddexp(C, k)
        L = np.logaddexp(C_plus, C_birth)

        log_weights_per_hyp = np.full((num_samples + birth_state.state_vector.shape[1],
                                       len(detections) + 1), -np.inf)
        log_weights_per_hyp[:num_samples, 0] = np.log(1 - prob_detect) + prediction.log_weight
        if len(detections):
            log_weights_per_hyp[:num_samples, 1:] = np.log(np.asfarray(meas_weights)) + Ck - L
            log_weights_per_hyp[num_samples:, 1:] = np.log(
                np.asfarray(meas_weights)) + Ck_birth - L
        return log_weights_per_hyp


class SMCPHDInitiator(Initiator):
    filter: SMCPHDFilter = Property(doc='The phd filter')
    prior: Any = Property(doc='The prior state')
    threshold: Probability = Property(doc='The thrshold probability for initiation',
                                      default=Probability(0.9))
    num_samples: int = Property(doc='The number of samples. Default is 1024', default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._state = self.prior
        self._id = 0

    def _get_track_state(self, prediction, detections, log_weights_per_hyp, log_intensity_per_hyp,
                         idx, timestamp):
        particles_sv = copy(prediction.state_vector)
        weight = np.exp(log_weights_per_hyp[:, idx] - log_intensity_per_hyp[idx])

        hypothesis = SingleProbabilityHypothesis(
            prediction,
            measurement=detections[idx - 1],
            probability=Probability(log_weights_per_hyp[:, idx], log_value=True))

        if self.num_samples > 0 or self.num_samples is not None:
            track_state = ParticleStateUpdate(
                state_vector=particles_sv,
                log_weight=log_weights_per_hyp[:, idx] - log_intensity_per_hyp[idx],
                hypothesis=hypothesis,
                timestamp=timestamp,
            )

            track_state = self.filter.resampler.resample(track_state, self.num_samples)
        else:
            mu = np.average(particles_sv,
                            axis=1,
                            weights=weight)
            cov = np.cov(particles_sv, ddof=0, aweights=weight)

            track_state = GaussianStateUpdate(mu, cov, hypothesis=hypothesis,
                                              timestamp=timestamp)

        return track_state

    def initiate(self, detections, timestamp, weights=None, **kwargs):
        tracks = set()

        if self._state.timestamp is None:
            self._state.timestamp = timestamp
        # Predict forward
        prediction = self.filter.predict(self._state, timestamp)

        # Calculate weights per hypothesis
        log_weights_per_hyp = self.filter.get_log_weights_per_hypothesis(prediction, detections,
                                                                         weights)

        # Calculate intensity per hypothesis
        log_intensity_per_hyp = logsumexp(log_weights_per_hyp, axis=0)

        # Find detections with intensity above threshold and initiate
        valid_inds = np.flatnonzero(np.exp(log_intensity_per_hyp) > self.threshold)
        for idx in valid_inds:
            if not idx:
                continue

            track_state = self._get_track_state(prediction, detections, log_weights_per_hyp,
                                                log_intensity_per_hyp, idx, timestamp)

            # if np.trace(track_state.covar) < 10:
            log_weights_per_hyp[:, idx] = -np.inf
            track = Track([track_state], id=self._id)
            track.exist_prob = Probability(log_intensity_per_hyp[idx], log_value=True)
            tracks.add(track)
            self._id += 1

            weights[idx - 1] = 0

            # Calculate intensity per hypothesis
            log_intensity_per_hyp = logsumexp(log_weights_per_hyp, axis=0)

        # Update filter
        self._state = self.filter.update(prediction, detections, timestamp, weights)

        return tracks


class ISMCPHDInitiator(SMCPHDInitiator):
    filter: ISMCPHDFilter = Property(doc='The phd filter')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._state = self.prior
        self._id = 0

    def _get_track_state(self, prediction, detections, log_weights_per_hyp, log_intensity_per_hyp,
                         idx, timestamp):
        particles_sv = copy(
            prediction.state_vector[:, :len(prediction) - len(prediction.birth_idx)])
        weight = np.exp(
            log_weights_per_hyp[:self.filter.num_samples, idx] - log_intensity_per_hyp[idx])

        hypothesis = SingleProbabilityHypothesis(
            prediction,
            measurement=detections[idx - 1],
            probability=Probability(log_weights_per_hyp[:, idx], log_value=True))

        if self.num_samples is not None and self.num_samples > 0:
            track_state = ParticleStateUpdate(
                state_vector=particles_sv,
                log_weight=log_weights_per_hyp[:, idx] - log_intensity_per_hyp[idx],
                hypothesis=hypothesis,
                timestamp=timestamp,
            )

            track_state = self.filter.resampler.resample(track_state, self.num_samples)
        else:
            mu = np.average(particles_sv,
                            axis=1,
                            weights=weight)
            cov = np.cov(particles_sv, ddof=0, aweights=weight)

            track_state = GaussianStateUpdate(mu, cov, hypothesis=None,
                                              timestamp=timestamp)

        return track_state

    def initiate(self, detections, timestamp, weights=None, **kwargs):
        tracks = set()

        if self._state.timestamp is None:
            self._state.timestamp = timestamp
        # Predict forward
        prediction = self.filter.predict(self._state, timestamp)

        # Calculate weights per hypothesis
        birth_state = self.filter.get_birth_state(prediction, detections, timestamp)
        log_weights_per_hyp = self.filter.get_log_weights_per_hypothesis(prediction, detections,
                                                                         weights,
                                                                         birth_state)
        log_weights_per_hyp = log_weights_per_hyp[:self.filter.num_samples, :]

        # Calculate intensity per hypothesis
        log_intensity_per_hyp = logsumexp(log_weights_per_hyp, axis=0)
        logging.debug(np.exp(log_intensity_per_hyp))
        # Find detections with intensity above threshold and initiate
        valid_inds = np.flatnonzero(np.exp(log_intensity_per_hyp[1:]) > self.threshold)
        while len(valid_inds):
            idx = valid_inds[0] + 1

            track_state = self._get_track_state(prediction, detections, log_weights_per_hyp,
                                                log_intensity_per_hyp, idx, timestamp)

            # if np.trace(track_state.covar) < 10:
            log_weights_per_hyp[:, idx] = -np.inf
            track = Track([track_state], id=self._id)
            track.exist_prob = Probability(log_intensity_per_hyp[idx], log_value=True)
            tracks.add(track)
            self._id += 1

            weights[idx - 1] = 0
            log_intensity_per_hyp = logsumexp(log_weights_per_hyp, axis=0)
            valid_inds = np.flatnonzero(np.exp(log_intensity_per_hyp[1:]) > self.threshold)

        # Update filter
        self._state = self.filter.update(prediction, detections, timestamp, weights)

        return tracks