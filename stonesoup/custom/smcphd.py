import copy
from typing import List

import numpy as np
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal

from stonesoup.base import Base, Property
from stonesoup.functions import predict_state_to_two_state
from stonesoup.models.measurement import MeasurementModel
from stonesoup.models.transition import TransitionModel
from stonesoup.resampler import Resampler
from stonesoup.types.array import StateVectors
from stonesoup.types.detection import Detection, MissedDetection
from stonesoup.types.hypothesis import SingleProbabilityHypothesis
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.numeric import Probability
from stonesoup.types.particle import Particles
from stonesoup.types.prediction import Prediction
from stonesoup.types.state import GaussianState, TwoStateGaussianState
from stonesoup.types.update import Update


class SMCPHDFilter(Base):
    prior: GaussianState = Property(doc='The state prior')
    transition_model: TransitionModel = Property(doc='The transition model')
    measurement_model: MeasurementModel = Property(doc='The measurement model')
    prob_detect: Probability = Property(doc='The probability of detection')
    prob_death: Probability = Property(doc='The probability of death')
    prob_birth: Probability = Property(doc='The probability of birth')
    birth_rate: float = Property(doc='The birth rate')
    clutter_density: float = Property(doc='The clutter density')
    resampler: Resampler = Property(default=None, doc='Resampler to prevent particle degeneracy')
    num_samples: int = Property(doc='The number of samples. Default is 1024', default=1024)
    birth_scheme: str = Property(
        doc='The scheme for birth particles. Options are "expansion" | "mixture". '
            'Default is "expansion"',
        default='expansion'
    )

    def predict(self, state, timestamp):
        prior_weights = state.particles.weight
        time_interval = timestamp - state.timestamp

        # Predict particles forward
        pred_particles_sv = self.transition_model.function(state.particles,
                                                           time_interval=time_interval,
                                                           num_samples=self.num_samples,
                                                           noise=True)

        if self.birth_scheme == 'expansion':
            num_birth = round(float(self.prob_birth * self.num_samples))

            birth_particles = multivariate_normal.rvs(self.prior.mean.ravel(),
                                                      self.prior.covar,
                                                      num_birth)
            pred_particles_sv = StateVectors(
                np.concatenate((pred_particles_sv, birth_particles.T), axis=1))

            # Process weights
            pred_weights = (1 - self.prob_death) * prior_weights
            pred_weights = np.concatenate(
                (pred_weights, np.ones((num_birth,)) * Probability(self.birth_rate / num_birth)))
        else:

            # Perform birth
            birth_inds = np.flatnonzero(np.random.binomial(1, self.prob_birth, self.num_samples))
            birth_particles = multivariate_normal.rvs(self.prior.mean.ravel(), self.prior.covar,
                                                      len(birth_inds))
            pred_particles_sv[:, birth_inds] = birth_particles.T

            # Process weights
            pred_weights = (1 - self.prob_death) * prior_weights
            pred_weights[birth_inds] = self.birth_rate / len(birth_inds)

        pred_particles = Particles(state_vector=pred_particles_sv, weight=pred_weights)
        prediction = Prediction.from_state(state, particles=pred_particles, timestamp=timestamp,
                                           transition_model=self.transition_model)

        return prediction

    def update(self, prediction, detections, timestamp, meas_weights=None):
        pred_particles = prediction.particles
        pred_weights = pred_particles.weight
        weights_per_hyp = self.get_weights_per_hypothesis(prediction, detections, meas_weights)

        single_hypotheses = [
            SingleProbabilityHypothesis(prediction,
                                        measurement=MissedDetection(timestamp=timestamp),
                                        probability=weights_per_hyp[:, 0])]
        for i, detection in enumerate(detections):
            single_hypotheses.append(
                SingleProbabilityHypothesis(prediction,
                                            measurement=detection,
                                            probability=weights_per_hyp[:, i + 1])
            )
        hypothesis = MultipleHypothesis(single_hypotheses, normalise=False)

        # Update weights Eq. (28) of [1]
        post_weights = np.sum(weights_per_hyp, axis=1)
        prob_func = np.vectorize(lambda x: Probability(x, log_value=False))
        post_weights = prob_func(post_weights)

        # Resample
        num_targets = np.sum(post_weights)
        post_particles = copy.copy(pred_particles)
        post_particles.weight = post_weights / num_targets
        if self.resampler is not None:
            post_particles = self.resampler.resample(post_particles, self.num_samples)
        post_particles.weight = np.array(post_particles.weight) * num_targets

        return Update.from_state(
            prediction,
            particles=post_particles, hypothesis=hypothesis,
            timestamp=timestamp)

    def iterate(self, state, detections: List[Detection], timestamp):
        prior_weights = state.particles.weight
        time_interval = timestamp - state.timestamp
        detections_list = list(detections)

        # 1) Predict
        # =======================================================================================>

        # Predict particles forward
        pred_particles_sv = self.transition_model.function(state.particles,
                                                           time_interval=time_interval,
                                                           num_samples=self.num_samples,
                                                           noise=True)

        if self.birth_scheme == 'expansion':
            num_birth = round(float(self.prob_birth * self.num_samples))
            total_samples = self.num_samples + num_birth

            birth_particles = multivariate_normal.rvs(self.prior.mean.ravel(), self.prior.covar,
                                                      num_birth)
            pred_particles_sv = StateVectors(
                np.concatenate((pred_particles_sv, birth_particles.T), axis=1))

            # Process weights
            pred_weights = (1 - self.prob_death) * prior_weights
            pred_weights = np.concatenate(
                (pred_weights, np.ones((num_birth,)) * self.birth_rate / num_birth))
        else:
            total_samples = self.num_samples

            # Perform birth
            birth_inds = np.flatnonzero(np.random.binomial(1, self.prob_birth, self.num_samples))
            birth_particles = multivariate_normal.rvs(self.prior.mean.ravel(), self.prior.covar,
                                                      len(birth_inds))
            pred_particles_sv[:, birth_inds] = birth_particles.T

            # Process weights
            pred_weights = (1 - self.prob_death) * prior_weights
            pred_weights[birth_inds] = self.birth_rate / len(birth_inds)

        pred_particles = Particles(state_vector=pred_particles_sv, weight=pred_weights)
        prediction = Prediction.from_state(state, particles=pred_particles, timestamp=timestamp,
                                           transition_model=self.transition_model)

        # 2) Update
        # =======================================================================================>
        pred_particles = prediction.particles
        pred_weights = pred_particles.weight

        # Compute g(z|x) matrix as in [1]
        g = np.zeros((total_samples, len(detections)), dtype=np.object_)
        for i, detection in enumerate(detections_list):
            g[:, i] = detection.measurement_model.pdf(detection, pred_particles,
                                                      num_samples=total_samples,
                                                      noise=True)

        # Calculate w^{n,i} Eq. (20) of [2]
        Ck = self.prob_detect * g * pred_weights[:, np.newaxis]
        C = np.sum(Ck, axis=0)
        C_plus = C + self.clutter_density

        weights_per_hyp = np.zeros((total_samples, len(detections) + 1))
        weights_per_hyp[:, 0] = (1 - self.prob_detect) * pred_weights
        if len(detections):
            weights_per_hyp[:, 1:] = Ck / C_plus

        intensity_per_hyp = np.sum(weights_per_hyp, axis=0)
        single_hypotheses = [
            SingleProbabilityHypothesis(prediction,
                                        measurement=MissedDetection(timestamp=timestamp),
                                        probability=Probability(intensity_per_hyp[0]))]
        for i, detection in enumerate(detections_list):
            single_hypotheses.append(
                SingleProbabilityHypothesis(prediction,
                                            measurement=detection,
                                            probability=Probability(intensity_per_hyp[i + 1]))
            )
        hypothesis = MultipleHypothesis(single_hypotheses, normalise=False)

        # Update weights Eq. (28) of [1]
        post_weights = np.sum(weights_per_hyp, axis=1)
        prob_func = np.vectorize(lambda x: Probability(x, log_value=False))
        post_weights = prob_func(post_weights)

        # Resample
        num_targets = np.sum(post_weights)
        post_particles = copy.copy(pred_particles)
        post_particles.weight = post_weights / num_targets
        if self.resampler is not None:
            post_particles = self.resampler.resample(post_particles, self.num_samples)
        post_particles.weight = np.array(post_particles.weight) * num_targets

        return Update.from_state(
            prediction,
            particles=post_particles, hypothesis=hypothesis,
            timestamp=timestamp)

    def get_measurement_likelihoods(self, particles, detections, meas_weights):
        num_samples = len(particles)
        # Compute g(z|x) matrix as in [1]
        g = np.zeros((num_samples, len(detections)), dtype=Probability)
        for i, detection in enumerate(detections):
            if not meas_weights[i]:
                g[:, i] = Probability(0)
                continue
            g[:, i] = detection.measurement_model.pdf(detection, particles,
                                                      num_samples=num_samples,
                                                      noise=True)
        return g

    def get_weights_per_hypothesis(self, prediction, detections, meas_weights):
        num_samples = len(prediction.particles)
        pred_particles = prediction.particles
        pred_weights = pred_particles.weight
        if meas_weights is None:
            meas_weights = np.array([Probability(1) for _ in range(len(detections))])

        # Compute g(z|x) matrix as in [1]
        g = self.get_measurement_likelihoods(pred_particles, detections, meas_weights)

        # Calculate w^{n,i} Eq. (20) of [2]
        Ck = meas_weights * self.prob_detect * g * pred_weights[:, np.newaxis]
        C = np.sum(Ck, axis=0)
        k = np.array([detection.metadata['clutter_density']
                      if 'clutter_density' in detection.metadata else self.clutter_density
                      for detection in detections])
        C_plus = C + k

        weights_per_hyp = np.zeros((num_samples, len(detections) + 1), dtype=Probability)
        weights_per_hyp[:, 0] = (1 - self.prob_detect) * pred_weights
        if len(detections):
            weights_per_hyp[:, 1:] = Ck / C_plus

        return weights_per_hyp


class TwoStateSMCPHDFilter2(Base):
    prior: GaussianState = Property(doc='The state prior')
    transition_model: TransitionModel = Property(doc='The transition model')
    measurement_model: MeasurementModel = Property(doc='The measurement model')
    prob_detect: Probability = Property(doc='The probability of detection')
    prob_death: Probability = Property(doc='The probability of death')
    prob_birth: Probability = Property(doc='The probability of birth')
    birth_rate: float = Property(doc='The birth rate')
    clutter_density: float = Property(doc='The clutter density')
    resampler: Resampler = Property(default=None, doc='Resampler to prevent particle degeneracy')
    num_samples: int = Property(doc='The number of samples. Default is 1024', default=1024)
    birth_scheme: str = Property(
        doc='The scheme for birth particles. Options are "expansion" | "mixture". '
            'Default is "expansion"',
        default='expansion'
    )

    def predict(self, state, start_time, end_time):
        prior_weights = state.particles.weight
        time_interval = end_time - start_time

        init_mean = self.prior.mean
        init_cov = self.prior.covar
        init_mean, init_cov = predict_state_to_two_state(init_mean, init_cov,
                                                         self.transition_model,
                                                         end_time - start_time)

        prior = TwoStateGaussianState(init_mean, init_cov, start_time=start_time,
                                      end_time=end_time)

        # 1) Predict
        # =======================================================================================>

        # Predict particles forward

        statedim = self.transition_model.ndim_state
        A = self.transition_model.matrix(time_interval=time_interval)
        Q = self.transition_model.covar(time_interval=time_interval)
        # construct twostate transition and covar matrices
        AA = np.concatenate((np.eye(statedim), A))
        AAA = np.concatenate((np.zeros((2 * statedim, statedim)), AA), axis=1)
        QQ = block_diag(np.eye(statedim) * np.finfo(np.float32).eps, Q)

        pred_particles_sv = AAA @ state.particles.state_vector
        pred_particles_sv += multivariate_normal.rvs(np.zeros((2 * statedim,)), QQ,
                                                     self.num_samples).T

        if self.birth_scheme == 'expansion':
            num_birth = round(float(self.prob_birth * self.num_samples))
            total_samples = self.num_samples + num_birth

            birth_particles = multivariate_normal.rvs(prior.mean.ravel(), prior.covar,
                                                      num_birth)
            pred_particles_sv = StateVectors(
                np.concatenate((pred_particles_sv, birth_particles.T), axis=1))

            # Process weights
            pred_weights = (1 - self.prob_death) * prior_weights
            pred_weights = np.concatenate(
                (pred_weights, np.ones((num_birth,)) * self.birth_rate / num_birth))
        else:
            total_samples = self.num_samples

            # Perform birth
            birth_inds = np.flatnonzero(np.random.binomial(1, self.prob_birth, self.num_samples))
            birth_particles = multivariate_normal.rvs(prior.mean.ravel(), prior.covar,
                                                      len(birth_inds))
            pred_particles_sv[:, birth_inds] = birth_particles.T

            # Process weights
            pred_weights = (1 - self.prob_death) * prior_weights
            pred_weights[birth_inds] = self.birth_rate / len(birth_inds)

        pred_particles = Particles(state_vector=pred_particles_sv, weight=pred_weights)
        prediction = Prediction.from_state(state, particles=pred_particles,
                                           start_time=start_time,
                                           end_time=end_time)
        return prediction

    def update(self, prediction, detections, start_time, end_time, meas_weights=None):
        weights_per_hyp = self.get_weights_per_hypothesis(prediction, detections, meas_weights)

        single_hypotheses = [
            SingleProbabilityHypothesis(prediction,
                                        measurement=MissedDetection(timestamp=end_time),
                                        probability=weights_per_hyp[:, 0])]
        for i, detection in enumerate(detections):
            single_hypotheses.append(
                SingleProbabilityHypothesis(prediction,
                                            measurement=detection,
                                            probability=weights_per_hyp[:, i + 1])
            )
        hypothesis = MultipleHypothesis(single_hypotheses, normalise=False)

        # Update weights Eq. (28) of [1]
        post_weights = np.sum(weights_per_hyp, axis=1)
        prob_func = np.vectorize(lambda x: Probability(x, log_value=False))
        post_weights = prob_func(post_weights)

        # Resample
        num_targets = np.sum(post_weights)
        post_particles = copy.copy(prediction.particles)
        post_particles.weight = post_weights / num_targets
        if self.resampler is not None:
            post_particles = self.resampler.resample(post_particles, self.num_samples)
        post_particles.weight = np.array(post_particles.weight) * num_targets

        return Update.from_state(
            prediction,
            particles=post_particles,
            start_time=start_time,
            end_time=end_time,
            hypothesis=hypothesis)

    def iterate(self, state, detections: List[Detection], start_time, end_time, meas_weights=None):
        prediction = self.predict(state, start_time, end_time)
        return self.update(prediction, detections, start_time, end_time, meas_weights)

    def get_measurement_likelihoods(self, particles, detections, meas_weights):
        num_samples = len(particles)
        # Compute g(z|x) matrix as in [1]
        g = np.zeros((num_samples, len(detections)), dtype=Probability)
        for i, detection in enumerate(detections):
            if not meas_weights[i]:
                g[:, i] = Probability(0)
                continue
            g[:, i] = detection.measurement_model.pdf(detection, particles,
                                                      num_samples=num_samples,
                                                      noise=True)
        return g

    def get_weights_per_hypothesis(self, prediction, detections, meas_weights):
        num_samples = len(prediction.particles)
        pred_particles = prediction.particles
        pred_weights = pred_particles.weight
        if meas_weights is None:
            meas_weights = np.array([Probability(1) for _ in range(len(detections))])

        # Compute g(z|x) matrix as in [1]
        g = self.get_measurement_likelihoods(pred_particles, detections, meas_weights)

        # Calculate w^{n,i} Eq. (20) of [2]
        Ck = meas_weights * self.prob_detect * g * pred_weights[:, np.newaxis]
        C = np.sum(Ck, axis=0)
        k = np.array([detection.metadata['clutter_density']
                      if 'clutter_density' in detection.metadata else self.clutter_density
                      for detection in detections])
        C_plus = C + k

        weights_per_hyp = np.zeros((num_samples, len(detections) + 1), dtype=Probability)
        weights_per_hyp[:, 0] = (1 - self.prob_detect) * pred_weights
        if len(detections):
            weights_per_hyp[:, 1:] = Ck / C_plus

        return weights_per_hyp