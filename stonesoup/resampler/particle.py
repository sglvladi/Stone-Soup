# -*- coding: utf-8 -*-
import numpy as np

from .base import Resampler
from ..types.numeric import Probability
from ..types.particle import Particle


class SystematicResampler(Resampler):

    def resample(self, particles):
        """Resample the particles

        Parameters
        ----------
        particles : list of :class:`~.Particle`
            The particles to be resampled according to their weight

        Returns
        -------
        particles : list of :class:`~.Particle`
            The resampled particles
        """

        n_particles = len(particles)
        weight = Probability(1/n_particles)
        cdf = np.cumsum([float(p.weight) for p in particles])
        particles_listed = list(particles)
        # Pick random starting point
        u_i = np.random.uniform(0, 1 / n_particles)
        new_particles = []

        # Cycle through the cumulative distribution and copy the particle
        # that pushed the score over the current value
        for j in range(n_particles):

            u_j = u_i + (1 / n_particles) * j

            particle = particles_listed[np.argmax(u_j < cdf)]
            new_particles.append(
                Particle(particle.state_vector,
                         weight=weight,
                         parent=particle))

        return new_particles

class SystematicResampler2(Resampler):

    def resample(self, particles, weights):
        """Resample the particles

        Parameters
        ----------
        particles : list of :class:`~.Particle`
            The particles to be resampled according to their weight

        Returns
        -------
        particles : list of :class:`~.Particle`
            The resampled particles
        """

        n_particles = len(weights)

        # Sort the particles by weight (is this necessary?)
        idx = np.argsort(weights)
        weights = list(np.array(weights)[idx])
        particles = particles[:, idx]

        # Compute cumsum
        cdf = np.cumsum([float(weight) for weight in weights])

        # Pick random starting point
        u_i = np.random.uniform(0, 1 / n_particles)

        # Cycle through the cumulative distribution and copy the particle
        # that pushed the score over the current value
        new_particles = np.zeros(particles.shape)
        new_weights = [Probability(1/n_particles) for i in range(n_particles)]
        for j in range(n_particles):

            u_j = u_i + (1 / n_particles) * j

            new_particles[:, j] = particles[:, np.argmax(u_j < cdf)]

        return new_particles, new_weights
