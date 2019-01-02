# -*- coding: utf-8 -*-

import numpy as np

from ..base import Property
from ..types import Hypothesis
from .numeric import Probability
from .prediction import MeasurementPrediction, Prediction
from .detection import Detection, MissedDetection


class MultipleMeasurementHypothesis(Hypothesis):
    """Multiple Measurement Hypothesis base type

    A Multiple Measurement Hypothesis relates to 1 Track with 1 Prediction,
    and then multiple measurements ('weighted_measurements') that *could* be
    the correct association for this Track.  Each 'weighted_measurement' is
    recorded as a 'dict' with a 'measurement' and a 'weight'.  If one of the
    measurements has been recorded as the *correct* measurement, then it is
    linked to 'selected_measurement' (associated weight is not linked).

    Properties
    ----------
    prediction : :class:`Prediction`
        prediction of Track position at prediction.timestamp
    measurement_prediction : :class:`MeasurementPrediction`
        prediction of the correct measurement with covariance matrix
    weighted_measurements : :class:`list`
        list of dict{"measurement": Detection, "weight": float or int}
    """

    prediction = Property(
        Prediction,
        doc="Predicted track state")
    measurement_prediction = Property(
        MeasurementPrediction,
        default=None,
        doc="Optional track prediction in measurement space")
    weighted_measurements = Property(
        list,
        default=list(),
        doc="Weighted measurements used for hypothesis and updating")
    selected_measurement = Property(
        Detection,
        default=None,
        doc="The measurement that was selected to associate with a track.")

    @property
    def measurement(self):
        return self.get_selected_measurement()

    def add_weighted_detections(self, measurements, weights, normalize=False):

        # verify that 'measurements' and 'weights' are the same size and the
        # correct data types
        if any(not (isinstance(measurement, Detection))
               for measurement in measurements):
            raise Exception('measurements must all be of type Detection!')
        if any(not (isinstance(weight, float) or isinstance(weight, int))
               for weight in weights):
            raise Exception('weights must all be of type float or int!')
        if len(measurements) != len(weights):
            raise Exception('There must be the same number of weights '
                            'and measurements!')

        # normalize the weights to sum up to 1 if indicated
        if normalize is True:
            sum_weights = sum(weights)
            for index in range(0, len(weights)):
                weights[index] /= sum_weights

        # store weights and measurements in 'weighted_measurements'
        for index in range(0, len(measurements)):
            self.weighted_measurements.append(
                {"measurement": measurements[index],
                 "weight": weights[index]})

    def __bool__(self):
        if self.selected_measurement is not None:
            return not isinstance(self.selected_measurement, MissedDetection)
        else:
            raise Exception('Cannot check whether a '
                            'MultipleMeasurementHypothesis.'
                            'selected_measurement is a MissedDetection before'
                            ' it has been set!')

    def set_selected_measurement(self, detection):
        if any(np.array_equal(detection.state_vector,
                              measurement["measurement"].state_vector)
               for measurement in self.weighted_measurements):
            self.selected_measurement = detection
        else:
            raise Exception('Cannot set MultipleMeasurementHypothesis.'
                            'selected_measurement to a value not contained in'
                            ' MultipleMeasurementHypothesis.'
                            'weighted_detections!')

    def get_selected_measurement(self):
        if self.selected_measurement is not None:
            return self.selected_measurement
        else:
            raise Exception('best measurement in MultipleMeasurementhypothesis'
                            ' not selected, so it cannot be returned!')


class ProbabilityMultipleMeasurementHypothesis(MultipleMeasurementHypothesis):
    """Probability-scored multiple measurement hypothesis.

    Sub-type of MultipleMeasurementHypothesis where 'weight' must be of
    type Probability.  One of the 'weighted_measurements' MUST be a
    MissedDetection.  Used primarily with Probabilistic Data Association (PDA).
    """

    def __init__(self, prediction, measurement_prediction, *args, **kwargs):
        super().__init__(prediction, measurement_prediction, *args, **kwargs)

    def add_weighted_detections(self, measurements, weights, normalize=False):
        self.weighted_measurements = list()

        # verify that 'measurements' and 'weights' are the same size and the
        # correct data types
        if any(not (isinstance(measurement, Detection))
               for measurement in measurements):
            raise Exception('measurements must all be of type Detection!')
        if any(not isinstance(weight, Probability) for weight in weights):
            raise Exception('weights must all be of type Probability!')
        if len(measurements) != len(weights):
            raise Exception('There must be the same number of weights '
                            'and measurements!')

        # normalize the weights to sum up to 1 if indicated
        if normalize is True:
            sum_weights = Probability.sum(weights)
            for index in range(0, len(weights)):
                weights[index] /= sum_weights

        # store probabilities and measurements in 'weighted_measurements'
        for index in range(0, len(measurements)):
            self.weighted_measurements.append(
                {"measurement": measurements[index],
                 "weight": weights[index]})

    def get_missed_detection_probability(self):
        for measurement in self.weighted_measurements:
            if isinstance(measurement["measurement"], MissedDetection):
                return measurement["weight"]
        return None
