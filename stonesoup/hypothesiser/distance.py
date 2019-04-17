# -*- coding: utf-8 -*-
from .base import Hypothesiser
from ..base import Property
from ..measures import Measure
from ..predictor import Predictor
from ..types.hypothesis import SingleDistanceHypothesis
from ..types.multihypothesis import MultipleHypothesis
from ..types.detection import MissedDetection
from ..updater import Updater


class DistanceHypothesiser(Hypothesiser):
    """Prediction Hypothesiser based on a Measure

    Generate track predictions at detection times and score each hypothesised
    prediction-detection pair using the distance of the supplied
    :class:`Measure` class.
    """

    predictor = Property(
        Predictor,
        doc="Predict tracks to detection times")
    updater = Property(
        Updater,
        doc="Updater used to get measurement prediction")
    measure = Property(
        Measure,
        doc="Measure class used to calculate the distance between two states.")
    missed_distance = Property(
        float,
        default=float('inf'),
        doc="Distance for a missed detection. Default is set to infinity")

    def hypothesise(self, track, detections, timestamp):

        hypotheses = list()

        for detection in detections:
            prediction = self.predictor.predict(
                track.state, timestamp=detection.timestamp)
            measurement_prediction = self.updater.get_measurement_prediction(
                prediction, detection.measurement_model)
            distance = self.measure(measurement_prediction, detection)

            hypotheses.append(
                SingleDistanceHypothesis(
                    prediction, detection, distance, measurement_prediction))

        # Missed detection hypothesis with distance as 'missed_distance'
        prediction = self.predictor.predict(track.state, timestamp=timestamp)
        hypotheses.append(SingleDistanceHypothesis(
            prediction,
            MissedDetection(timestamp=timestamp),
            self.missed_distance))

        return sorted(hypotheses, reverse=True)


class GMMahalanobisDistanceHypothesiser(Hypothesiser):
    """Gaussian Mixture Prediction Hypothesiser based on a Distance Measure

    Generate Gaussian Mixture component predictions at detection times and
    score each hypothesised prediction-detection pair using the Mahalanobis
    Distance.
    """

    predictor = Property(
        Predictor,
        doc="Predict tracks to detection times")
    updater = Property(
        Updater,
        doc="Updater used to get measurement prediction")
    association_distance = Property(
        int,
        default=4,
        doc="Distance in standard deviations at which association between "
            "Gaussian Mixture component and detection is low enough to be "
            "ignored.")
    measure = Property(
        Measure,
        doc="Measure class used to calculate the distance between two states.")

    def hypothesise(self, predict_state, detections, timestamp):
        """Form hypotheses for associations between Detections and Gaussian
        Mixture components, discard those with too great a distance.

        Parameters
        ----------
        predict_state : :class:`list`
            List of :class:`WeightedGaussianState` components
            representing the predicted state of the space
        detections : list of :class:`Detection`
            Retrieved measurements
        timestamp : datetime
            time of the detections/predicted state

        Returns
        -------
        list of :class:`MultipleHypothesis`
            each MultipleHypothesis in the list contains SingleHypotheses
            pertaining to the same Detection
        """

        hypotheses = list()

        for detection in detections:

            this_detect_hypotheses = list()

            for component in predict_state:
                measurement_prediction = \
                    self.updater.get_measurement_prediction(component)
                distance = self.measure(measurement_prediction,
                                        detection)

                if distance < self.association_distance:
                    this_detect_hypotheses.append(
                        SingleDistanceHypothesis(
                            component, detection, distance,
                            measurement_prediction=measurement_prediction))

            if len(this_detect_hypotheses) > 0:
                hypotheses.append(MultipleHypothesis(this_detect_hypotheses))

        return hypotheses
