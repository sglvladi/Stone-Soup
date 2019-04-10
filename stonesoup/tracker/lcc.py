# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import multivariate_normal

from ..base import Property
from .gaussianmixture import GaussianMixtureMultiTargetTracker
from ..types import TaggedWeightedGaussianState


class GMLCCTargetTracker(GaussianMixtureMultiTargetTracker):
    """A implementation of the Gaussian Mixture
    Linear-Complexity Cumulant (GM-LCC) multi-target filter

    References
    ----------

    .. [1]  D. E. Clark and F. De Melo. A Linear-Complexity Second-Order Multi-
            Object Filter via Factorial Cumulants". In: 2018 21st International
            Conference on Information Fusion (FUSION). 2018. doi: 10.23919/ICIF
            .2018.8455331.
    """
    first_order_cumulant = Property(
        float,
        default=1,
        doc="The first order cumulant of the filter. This is equal to the mean"
            "estimated number of targets"
        )
    second_order_cumulant = Property(
        float,
        default=1,
        doc="The second order cumulant of the filter. This is equal to the "
            "variance on the estimated number of targets minus the mean of "
            "the estimated number of targets over a region. "
        )
    mean_FA = Property(
        float,
        default=0,
        doc="The mean number of false alarms per timestep"
        )
    var_FA = Property(
        float,
        default=0,
        doc="The variance on the number of false alarms per timestep"
        )
    c2_FA = Property(
        float,
        default=0,
        doc="The false alarm component of the second order cumulant"
        )
    var_birth = Property(
        float,
        default=0,
        doc="The variance on the number of target births per timestep"
        )
    c2_birth = Property(
        float,
        default=0,
        doc="The birth component of the second order cumulant"
        )

    def predict(self, timestamp, control_input=None, **kwargs):
        """
        Predicts the current components in the
        :state:`GaussianMixtureState` according to the GM-LCC prediction
        step.

        Parameters
        ==========
        self : :state:`GMLCCTargetTracker`
            Current GMLCC Tracker at time :math:`k`

        Returns
        =======
        self : :state:`GMLCCTargetTracker`
            GMLCC Tracker with predicted components to time :math:`k | k+1`

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
        self.second_order_cumulant *= self.prob_of_survival**2
        self.second_order_cumulant += self.c2_birth

    def update(self, measurements, timestamp, **kwargs):
        """
        Updates the current components in the
        :state:`GaussianMixtureState` according to the GM-LCC update
        step with the supplied measurements.

        Parameters
        ==========
        self : :state:`GMLCCTargetTracker`
            GMLCC Tracker with predicted components to time :math:`k | k+1`
        measurements : list
            Measurements obtained at time :math:`k+1`

        Returns
        =======
        self : :state:`GMLCCTargetTracker`
            GMLCC Tracker with updated components at time :math:`k+1`
        """
        # Misdetected intensity
        mu_phi = (1-self.prob_of_detection)*self.first_order_cumulant
        # Detected intensity
        mu_d = self.prob_of_detection*self.first_order_cumulant
        # Calculate the beta value of the Panjer
        beta = 0
        try:
            beta = float(self.first_order_cumulant + self.mean_FA) \
                / float(self.second_order_cumulant+self.c2_FA)
        except ZeroDivisionError:
            beta = 1
        # Calculate the alpha value of the Panjer
        alpha_pred = ((self.first_order_cumulant + self.mean_FA)**2) \
            / (self.second_order_cumulant+self.c2_FA)

        number_of_measurements = len(measurements)
        # Get all valid target-measurement hypotheses
        hypotheses = self.data_associator.associate(
                                self.gaussian_mixture.components,
                                measurements,
                                timestamp
                                )
        updated_components = []
        weight_sum_array = []
        # Loop over all measurements
        for i in range(len(hypotheses)):
            updated_measurement_components = []
            # Initialise weight sum for measurement to clutter intensity
            weight_sum = 0
            # For every valid single hypothesis, update that component with
            # measurements and calculate new weight
            for j in range(len(hypotheses[i])):
                measurement_prediction = \
                    hypotheses[i][j].measurement_prediction
                measurement = hypotheses[i][j].measurement
                prediction = hypotheses[i][j].prediction
                # Calculate new weight and add to weight sum
                try:
                    q = multivariate_normal.pdf(
                        measurement.state_vector.flatten(),
                        mean=measurement_prediction.mean.flatten(),
                        cov=measurement_prediction.covar
                    )
                except ValueError:
                    q = 1e-9
                    print(measurement.state_vector.flatten())
                    print(measurement_prediction.mean.flatten())
                    print(measurement_prediction.covar)
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
            weight_sum_array.append(weight_sum)
            for component in updated_measurement_components:
                component.weight /= weight_sum+self.clutter_spatial_density
                updated_components.append(component)
        denominator = alpha_pred + mu_d + self.mean_FA
        numerator = alpha_pred + number_of_measurements
        l1 = numerator/denominator
        l2 = numerator/(denominator**2)

        for component in self.gaussian_mixture:
            # Add all active components except birth component back into
            # mixture
            if not component.tag == 1:
                component.weight *= (1 - self.prob_of_detection) * l1
                updated_components.append(component)

        # Perform pruning and merging
        self.gaussian_mixture.components = updated_components
        self.gaussian_mixture.components = \
            self.reducer.reduce(self.gaussian_mixture.components)

        # Calulate the updated second order cumulant
        weight_sum_array_w_clutter = \
            np.array(
                [x+self.clutter_spatial_density for x in weight_sum_array])
        weight_sum_array = np.array(weight_sum_array)

        detected_c2 = sum((weight_sum_array / weight_sum_array_w_clutter)**2)
        misdetected_c2 = (mu_phi**2) * l2
        self.second_order_cumulant = misdetected_c2 - detected_c2
        # Calulate the first order cumulant
        self.first_order_cumulant = self.estimated_number_of_targets
        # Update the tracks
        # self.tracks_gen()

    @property
    def estimated_target_number_variance(self):
        """
        The estimated variance on the number of hypothesised targets.
        """
        return self.second_order_cumulant + self.first_order_cumulant
