# -*- coding: utf-8 -*-
from functools import lru_cache

import numpy as np
from .base import Updater
from ..base import Property
from ..resampler import Resampler
from ..types.numeric import Probability
from ..types.particle import Particle
from ..types.prediction import ParticleMeasurementPrediction, ParticleMeasurementPrediction2
from ..types.update import ParticleStateUpdate, ParticleStateUpdate2


class ParticleUpdater(Updater):
    """Simple Particle Updater

        Perform measurement update step in the standard Kalman Filter.
        """

    resampler = Property(Resampler,
                         doc='Resampler to prevent particle degeneracy')

    def update(self, hypothesis, **kwargs):
        """Particle Filter update step

        Parameters
        ----------
        hypothesis : :class:`~.Hypothesis`
            Hypothesis with predicted state and associated detection used for
            updating.

        Returns
        -------
        : :class:`~.ParticleState`
            The state posterior
        """
        if hypothesis.measurement.measurement_model is None:
            measurement_model = self.measurement_model
        else:
            measurement_model = hypothesis.measurement.measurement_model

        for particle in hypothesis.prediction.particles:
            particle.weight *= measurement_model.pdf(
                hypothesis.measurement, particle,
                **kwargs)[0]

        # Normalise the weights
        sum_w = Probability.sum(
            i.weight for i in hypothesis.prediction.particles)
        for particle in hypothesis.prediction.particles:
            particle.weight /= sum_w

        # Resample
        new_particles = self.resampler.resample(
            hypothesis.prediction.particles)

        return ParticleStateUpdate(new_particles,
                                   hypothesis,
                                   timestamp=hypothesis.measurement.timestamp)

    @lru_cache()
    def predict_measurement(self, state_prediction, measurement_model=None,
                            **kwargs):

        if measurement_model is None:
            measurement_model = self.measurement_model

        new_particles = []
        for particle in state_prediction.particles:
            new_state_vector = measurement_model.function(
                particle, noise=0, **kwargs)
            new_particles.append(
                Particle(new_state_vector,
                         weight=particle.weight,
                         parent=particle.parent))

        return ParticleMeasurementPrediction(
            new_particles, timestamp=state_prediction.timestamp)


class ParticleUpdater2(Updater):
    """Simple Particle Updater

        Perform measurement update step in the standard Kalman Filter.
        """

    resampler = Property(Resampler,
                         doc='Resampler to prevent particle degeneracy')

    def update(self, hypothesis, **kwargs):
        """Particle Filter update step

        Parameters
        ----------
        hypothesis : :class:`~.Hypothesis`
            Hypothesis with predicted state and associated detection used for
            updating.

        Returns
        -------
        : :class:`~.ParticleState`
            The state posterior
        """
        if hypothesis.measurement.measurement_model is None:
            measurement_model = self.measurement_model
        else:
            measurement_model = hypothesis.measurement.measurement_model

        weights = measurement_model.pdf(hypothesis.measurement, hypothesis.prediction, **kwargs)
        weights *= np.array(hypothesis.prediction.weights)

        # Normalise the weights
        sum_w = Probability.sum(weight for weight in weights)
        weights = [weight/sum_w for weight in weights]

        # Resample
        new_particles, new_weights = self.resampler.resample(
            hypothesis.prediction.particles, weights)

        new_weights = [weight*sum_w for weight in new_weights]
        return ParticleStateUpdate2(new_particles,
                                    hypothesis,
                                    weights=new_weights,
                                    timestamp=hypothesis.measurement.timestamp)

    @lru_cache()
    def predict_measurement(self, state_prediction, measurement_model=None,
                            **kwargs):

        if measurement_model is None:
            measurement_model = self.measurement_model

        new_particle_sv = measurement_model.function(state_prediction, **kwargs)

        return ParticleMeasurementPrediction2(
            new_particle_sv, state_prediction.weights, timestamp=state_prediction.timestamp)