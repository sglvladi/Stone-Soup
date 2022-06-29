# -*- coding: utf-8 -*-
import numpy as np

from .base import Resampler
from ..base import Property
from ..types.numeric import Probability
from ..types.particle import Particles


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

        if not isinstance(particles, Particles):
            particles = Particles(particle_list=particles)
        n_particles = len(particles)
        weight = Probability(1 / n_particles)

        log_weights = np.array([weight.log_value for weight in particles.weight])
        weight_order = np.argsort(log_weights, kind='stable')
        max_log_value = log_weights[weight_order[-1]]
        with np.errstate(divide='ignore'):
            cdf = np.log(np.cumsum(np.exp(log_weights[weight_order] - max_log_value)))
        cdf += max_log_value

        # Pick random starting point
        u_i = np.random.uniform(0, 1 / n_particles)

        # Cycle through the cumulative distribution and copy the particle
        # that pushed the score over the current value
        u_j = u_i + (1 / n_particles) * np.arange(n_particles)
        index = weight_order[np.searchsorted(cdf, np.log(u_j))]
        new_particles = Particles(state_vector=particles.state_vector[:, index],
                                  weight=[weight] * n_particles,
                                  parent=Particles(state_vector=particles.state_vector[:, index],
                                                   weight=particles.weight[index]))
        return new_particles


class ESSResampler(Resampler):
    """ This wrapper uses a :class:`~.Resampler` to resample the particles inside
        an instant of :class:`~.Particles`, but only after checking if this is necessary
        by comparing Effective Sample Size (ESS) with a supplied threshold (numeric).
        Kish's ESS is used, as recommended in Section 3.5 of this tutorial [1]_.

        References
        ----------
        .. [1] Doucet A., Johansen A.M., 2009, Tutorial on Particle Filtering \
        and Smoothing: Fifteen years later, Handbook of Nonlinear Filtering, Vol. 12.

        """

    threshold: float = Property(default=None,
                                doc='Threshold compared with ESS to decide whether to resample. \
                                    Default is number of particles divided by 2, \
                                        set in resample method')
    resampler: Resampler = Property(default=SystematicResampler,
                                    doc='Resampler to wrap, which is called \
                                        when ESS below threshold')

    def resample(self, particles):
        """
        Parameters
        ----------
        particles : list of :class:`~.Particle`
            The particles to be resampled according to their weight

        Returns
        -------
        particles : list of :class:`~.Particle`
            The particles, either unchanged or resampled, depending on weight degeneracy
        """
        if not isinstance(particles, Particles):
            particles = Particles(particle_list=particles)
        if self.threshold is None:
            self.threshold = len(particles) / 2
        if 1 / np.sum(np.square(particles.weight)) < self.threshold:  # If ESS too small, resample
            return self.resampler.resample(self.resampler, particles)
        else:
            return particles


class ESSResampler2(Resampler):
    """ This wrapper uses a :class:`~.Resampler` to resample the particles inside
        an instant of :class:`~.Particles`, but only after checking if this is necessary
        by comparing Effective Sample Size (ESS) with a supplied threshold (numeric).
        Kish's ESS is used, as recommended in Section 3.5 of this tutorial [1]_.

        References
        ----------
        .. [1] Doucet A., Johansen A.M., 2009, Tutorial on Particle Filtering \
        and Smoothing: Fifteen years later, Handbook of Nonlinear Filtering, Vol. 12.

        """

    resampler: Resampler = Property(default=SystematicResampler,
                                    doc='Resampler to wrap, which is called \
                                        when ESS below threshold')
    threshold: float = Property(default=None,
                                doc='Threshold compared with ESS to decide whether to resample. \
                                    Default is number of particles divided by 2, \
                                        set in resample method')

    def resample(self, particles, weights):
        """
        Parameters
        ----------
        particles : list of :class:`~.Particle`
            The particles to be resampled according to their weight

        Returns
        -------
        particles : list of :class:`~.Particle`
            The particles, either unchanged or resampled, depending on weight degeneracy
        """
        if self.threshold is None:
            self.threshold = len(weights) / 2
        if 1 / np.sum(np.square(weights)) < self.threshold:  # If ESS too small, resample
            return self.resampler.resample(particles, weights)
        else:
            return particles, weights


class SystematicResampler2(Resampler):

    def resample(self, particles, weights_l):
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

        weights = np.array(weights_l).astype(float)
        n_particles = len(weights)

        # Sort the particles by weight (is this necessary?)
        idx = np.argsort(weights)
        weights = weights[idx]
        particles = particles[:, idx]

        # Compute cumsum
        cdf = np.cumsum(weights)

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


class StratifiedResampler(Resampler):

    def resample(self, particles, weights_l):
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

        weights = np.array(weights_l).astype(float)
        n_particles = len(weights)

        # Compute the strata
        e = particles[2, :]
        strata = np.unique(e)
        num_strata = len(strata)
        inds_per_stratum = dict()
        for stratum in strata:
            inds_per_stratum[stratum] = np.flatnonzero(e == stratum)

        # Remove strata with low probability
        strata_cp = strata.copy()
        for stratum in strata_cp:
            if np.sum(weights[inds_per_stratum[stratum]]) < 0.01:
                strata = strata[strata != stratum]
                inds_per_stratum.pop(stratum, None)

        # Resample
        n_particles_per_stratum = n_particles//num_strata
        new_particles_idx = []
        for stratum in strata:
            stratum_inds = inds_per_stratum[stratum]
            stratum_weights = weights[stratum_inds]
            stratum_weights /= np.sum(stratum_weights)
            new_idx = np.random.choice(stratum_inds, n_particles_per_stratum, p=stratum_weights)
            new_particles_idx += list(new_idx)

        if len(new_particles_idx) < n_particles:
            num_rem = n_particles - len(new_particles_idx)
            new_idx = np.random.choice([i for i in range(n_particles)], num_rem, p=weights)
            new_particles_idx += list(new_idx)

        new_particles = particles[:, new_particles_idx]
        # new_weights = weights[:, new_particles_idx]
        new_weights = [Probability(1 / n_particles) for i in range(n_particles)]

        return new_particles, new_weights
