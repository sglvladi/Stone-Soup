# -*- coding: utf-8 -*-
from scipy.stats import multivariate_normal

from .gaussianmixture import GaussianMixtureMultiTargetTracker
from ..types import TaggedWeightedGaussianState


class GMPHDTargetTracker(GaussianMixtureMultiTargetTracker):
    """A implementation of the Gaussian Mixture
    Probability Hypothesis Density (GM-PHD) multi-target filter

    References
    ----------

    .. [1] B.-N. Vo and W.-K. Ma, “The Gaussian Mixture Probability Hypothesis
    Density Filter,” Signal Processing,IEEE Transactions on, vol. 54, no. 11,
    pp. 4091–4104, 2006..
    """

    def predict(self, timestamp, control_input=None, **kwargs):
        """
        Predicts the current components in the
        :state:`GaussianMixtureState` according to the GM-PHD prediction
        step.

        Parameters
        ==========
        self : :state:`GMPHDTargetTracker`
            Current GMPHD Tracker at time :math:`k`

        Returns
        =======
        self : :state:`GMPHDTargetTracker`
            GMPHD Tracker with predicted components to time :math:`k | k+1`

        Note
        ======
        This is an iteration over a list (:class:`GaussianMixtureState`).
        It predicts each component :math:`i` according to the underlying
        :class:`Predictor` class and multiplies its weight :math:`w_i`
        by the probability of survival :math:`P_s`
        """
        if len(self.gaussian_mixture) > 0:
            for i in range(len(self.gaussian_mixture)):
                gaussian_state_prediction = self.predictor.predict(
                    prior=self.gaussian_mixture[i],
                    control_input=control_input,
                    timestamp=timestamp,
                )
                self.gaussian_mixture[i] = TaggedWeightedGaussianState(
                    tag=self.gaussian_mixture[i].tag,
                    weight=self.gaussian_mixture[i].weight *
                    self.prob_of_survival,
                    state_vector=gaussian_state_prediction.state_vector,
                    covar=gaussian_state_prediction.covar,
                    timestamp=timestamp
                )
        # Birth component simply gets added to the components
        self.birth_component.timestamp = timestamp
        self.gaussian_mixture.append(self.birth_component)

    def update(self, measurements, timestamp, **kwargs):
        """
        Updates the current components in the
        :state:`GaussianMixtureState` according to the GM-PHD update
        step with the supplied measurements.

        Parameters
        ==========
        self : :state:`GMPHDTargetTracker`
            GMPHD Tracker with predicted components to time :math:`k | k+1`
        measurements : list
            Measurements obtained at time :math:`k+1`

        Returns
        =======
        self : :state:`GMPHDTargetTracker`
            GMPHD Tracker with updated components at time :math:`k+1`
        """
        # Get all valid target-measurement hypotheses
        hypotheses = self.data_associator.associate(
                                self.gaussian_mixture.components,
                                measurements,
                                timestamp
                                )
        updated_components = []
        # Loop over all measurements
        for i in range(len(hypotheses)):
            updated_measurement_components = []
            # Initialise weight sum for measurement to clutter intensity
            weight_sum = self.clutter_spatial_density
            # For every valid single hypothesis, update that component with
            # measurements and calculate new weight
            for j in range(len(hypotheses[i])):
                measurement_prediction = \
                    hypotheses[i][j].measurement_prediction
                measurement = hypotheses[i][j].measurement
                prediction = hypotheses[i][j].prediction
                # Calculate new weight and add to weight sum
                q = multivariate_normal.pdf(
                    measurement.state_vector.flatten(),
                    mean=measurement_prediction.mean.flatten(),
                    cov=measurement_prediction.covar
                )
                new_weight = self.prob_of_detection*prediction.weight*q
                weight_sum += new_weight
                # Perform single target Kalman Update
                temp_updated_component = self.updater.update(hypotheses[i][j])
                updated_component = TaggedWeightedGaussianState(
                    tag=prediction.tag,
                    weight=new_weight,
                    state_vector=temp_updated_component.mean,
                    covar=temp_updated_component.covar,
                    timestamp=timestamp
                )
                # Assign new tag if spawned from birth component
                if updated_component.tag == 1:
                    updated_component.tag = self.gaussian_mixture.current_tag
                    self.gaussian_mixture.current_tag += 1
                # Add updated component to mixture
                updated_measurement_components.append(updated_component)
            for component in updated_measurement_components:
                component.weight /= weight_sum
                updated_components.append(component)
        for component in self.gaussian_mixture:
            # Add all active components except birth component back into
            # mixture
            if not component.tag == 1:
                component.weight *= 1 - self.prob_of_detection
                updated_components.append(component)
        # Perform pruning and merging
        self.gaussian_mixture.components = updated_components
        self.gaussian_mixture.components = \
            self.reducer.reduce(self.gaussian_mixture.components)
        # Update the tracks
        self.tracks_gen()
