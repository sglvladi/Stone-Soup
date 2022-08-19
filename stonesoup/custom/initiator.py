from copy import copy
from typing import Any

import numpy as np

from stonesoup.base import Property
from stonesoup.initiator import Initiator
from stonesoup.types.numeric import Probability
from stonesoup.types.particle import Particles
from stonesoup.types.track import Track
from stonesoup.types.update import TwoStateGaussianStateUpdate, GaussianStateUpdate


class SMCPHDInitiator(Initiator):
    filter: Any = Property(doc='The phd filter')
    prior: Any = Property(doc='The prior state')
    threshold: Probability = Property(doc='The thrshold probability for initiation',
                                      default=Probability(0.9))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._state = self.prior
        self._prediction = None

    def initiate(self, detections, timestamp, weights=None, **kwargs):
        tracks = set()
        detections_list = list(detections)
        self._prediction = self.filter.predict(self._state, timestamp)
        self._state = self.filter.update(self._prediction, detections_list, timestamp, weights)
        weights_per_hyp = np.zeros((len(self._state.hypothesis[0].weight), len(detections) + 1), dtype=Probability)
        for i, hyp in enumerate(self._state.hypothesis):
            weights_per_hyp[:, i] = hyp.weight
        intensity_per_hyp = np.sum(weights_per_hyp, axis=0)
        # print(intensity_per_hyp)
        valid_inds = np.flatnonzero(intensity_per_hyp > self.threshold)
        for idx in valid_inds:
            if not idx:
                continue

            particles_sv = copy(self._prediction.particles.state_vector)
            weight = weights_per_hyp[:, idx] / intensity_per_hyp[idx]

            mu = np.average(particles_sv,
                            axis=1,
                            weights=weight)
            cov = np.cov(particles_sv, ddof=0, aweights=np.array(weight))

            track_state = GaussianStateUpdate(mu, cov, hypothesis=self._state.hypothesis[idx],
                                              timestamp=timestamp)

            # if np.trace(track_state.covar) < 10:
            weights_per_hyp[:, idx] = Probability(0)
            track = Track([track_state])
            track.exist_prob = intensity_per_hyp[idx]
            tracks.add(track)
        self._state.particles.weight = np.sum(weights_per_hyp, axis=1)
        num_targets = np.sum(self._state.particles.weight)
        post_particles = copy(self._prediction.particles)
        post_particles.weight = self._state.particles.weight / num_targets
        if self.filter.resampler is not None:
            post_particles = self.filter.resampler.resample(post_particles, self.filter.num_samples)
        post_particles.weight = np.array(post_particles.weight) * num_targets
        self._state.particles = post_particles
        self._prediction.particles = post_particles
        return tracks


class TwoStateSMCPHDInitiator(Initiator):
    filter: Any = Property(doc='The phd filter')
    prior: Any = Property(doc='The prior state')
    threshold: Probability = Property(doc='The thrshold probability for initiation',
                                      default=Probability(0.9))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._state = self.prior
        self._prediction = None

    def predict(self, start_time, end_time):
        self._prediction = self.filter.predict(self._state, start_time, end_time)

    def initiate(self, detections, start_time, end_time,
                 weights=None, **kwargs):
        tracks = set()
        detections_list = list(detections)
        self._state = self.filter.update(self._prediction, detections_list, start_time, end_time, weights)
        weights_per_hyp = np.zeros((len(self._state.hypothesis[0].weight), len(detections) + 1), dtype=Probability)
        for i, hyp in enumerate(self._state.hypothesis):
            weights_per_hyp[:, i] = hyp.weight
        intensity_per_hyp = np.sum(weights_per_hyp, axis=0)
        # print(intensity_per_hyp)
        valid_inds = np.flatnonzero(intensity_per_hyp > self.threshold)
        for idx in valid_inds:
            if not idx:
                continue

            particles_sv = copy(self._prediction.particles.state_vector)
            weight = weights_per_hyp[:, idx] / intensity_per_hyp[idx]

            mu = np.average(particles_sv,
                            axis=1,
                            weights=weight)
            cov = np.cov(particles_sv, ddof=0, aweights=np.array(weight))

            track_state = TwoStateGaussianStateUpdate(mu, cov,
                                                      hypothesis=self._state.hypothesis[idx],
                                                      start_time=start_time,
                                                      end_time=end_time)

            # if np.trace(track_state.covar) < 10:
            weights_per_hyp[:, idx] = Probability(0)
            track = Track([track_state])
            track.exist_prob = intensity_per_hyp[idx]
            tracks.add(track)
        self._state.particles.weight = np.sum(weights_per_hyp, axis=1)
        num_targets = np.sum(self._state.particles.weight)
        post_particles = copy(self._prediction.particles)
        post_particles.weight = self._state.particles.weight / num_targets
        if self.filter.resampler is not None:
            post_particles = self.filter.resampler.resample(post_particles, self.filter.num_samples)
        post_particles.weight = np.array(post_particles.weight) * num_targets
        self._state.particles = post_particles
        self._prediction.particles = post_particles
        return tracks



# class TwoStateSMCPHDInitiator2(Initiator):
#     filter: Any = Property(doc='The phd filter')
#     prior: Any = Property(doc='The prior state')
#     threshold: Probability = Property(doc='The thrshold probability for initiation',
#                                       default=Probability(0.9))
#     num_samples: int = Property(doc='', default=1024)
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._state = self.prior
#
#     def predict(self, start_time, end_time):
#         self._state = self.filter.predict(self._state, start_time, end_time)
#
#     def initiate(self, detections, start_time, end_time, weights=None, **kwargs):
#         tracks = set()
#
#         weights_per_hyp = self.filter.get_weights_per_hypothesis(self._state, detections,
#                                                                  weights)
#         intensity_per_hyp = np.sum(weights_per_hyp, axis=0)
#         print(intensity_per_hyp)
#         valid_inds = np.flatnonzero(intensity_per_hyp > self.threshold)
#         for idx in valid_inds:
#             if not idx:
#                 continue
#
#             particles_sv = copy(self._state.particles.state_vector)
#             weight = weights_per_hyp[:, idx] / intensity_per_hyp[idx]
#             particles = Particles(state_vector=particles_sv, weight=weight)
#             particles = self.filter.resampler.resample(particles, self.num_samples)
#
#             hypothesis = SingleProbabilityHypothesis(self._state,
#                                             measurement=detections[idx-1],
#                                             probability=weights_per_hyp[:, idx])
#             track_state = TwoStateParticleStateUpdate(
#                 particles,
#                 hypothesis=hypothesis,
#                 start_time=start_time,
#                 end_time=end_time)
#
#             track = Track([track_state])
#             track.exist_prob = intensity_per_hyp[idx]
#             tracks.add(track)
#
#             weights[idx-1] = 0
#
#         self._state = self.filter.update(self._state, detections, start_time, end_time, weights)
#         return tracks