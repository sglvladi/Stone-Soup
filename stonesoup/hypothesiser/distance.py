# -*- coding: utf-8 -*-
from .base import Hypothesiser
from ..base import Property
from ..measures import Measure
from ..predictor import Predictor
from ..types.detection import MissedDetection
from ..types.prediction import Prediction
from ..types.hypothesis import SingleDistanceHypothesis
from ..types.multihypothesis import MultipleHypothesis
from ..updater import Updater
from copy import copy


class DistanceHypothesiser(Hypothesiser):
    """Prediction Hypothesiser based on a Measure

    Generate track predictions at detection times and score each hypothesised
    prediction-detection pair using the distance of the supplied
    :class:`~.Measure` class.
    """

    predictor: Predictor = Property(doc="Predict tracks to detection times")
    updater: Updater = Property(doc="Updater used to get measurement prediction")
    measure: Measure = Property(
        doc="Measure class used to calculate the distance between two states.")
    missed_distance: float = Property(
        default=float('inf'),
        doc="Distance for a missed detection. Default is set to infinity")
    include_all: bool = Property(
        default=False,
        doc="If `True`, hypotheses beyond missed distance will be returned. Default `False`")

    def hypothesise(self, track, detections, timestamp, missed_detection=None, mult=None, **kwargs):
        """ Evaluate and return all track association hypotheses.

        For a given track and a set of N available detections, return a
        MultipleHypothesis object with N+1 detections (first detection is
        a 'MissedDetection'), each with an associated distance measure..

        Parameters
        ----------
        track : Track
            The track object to hypothesise on
        detections : set of :class:`~.Detection`
            The available detections
        timestamp : datetime.datetime
            A timestamp used when evaluating the state and measurement
            predictions. Note that if a given detection has a non empty
            timestamp, then prediction will be performed according to
            the timestamp of the detection.

        Returns
        -------
        : :class:`~.MultipleHypothesis`
            A container of :class:`~SingleDistanceHypothesis` objects

        """
        hypotheses = list()

        if missed_detection is None:
            missed_detection = MissedDetection(timestamp=timestamp)

        # Common state & measurement prediction
        mmsis = [detection.metadata["MMSI"] for detection in detections]
        if track.metadata["MMSI"] in mmsis:
            prediction = self.predictor.predict(track, timestamp=timestamp, **kwargs)
            measurement_prediction = self.updater.predict_measurement(prediction, **kwargs)

            # Missed detection hypothesis with distance as 'missed_distance'
            hypotheses.append(
                SingleDistanceHypothesis(
                    prediction,
                    MissedDetection(timestamp=timestamp),
                    self.missed_distance
                    ))

            # True detection hypotheses
            distances = {detection: self.measure(measurement_prediction, detection)
                         for detection in detections}
            hypotheses += [SingleDistanceHypothesis(
                                prediction,
                                detection,
                                distances[detection],
                                measurement_prediction) for detection in detections
                                if self.include_all
                                   or distances[detection] < self.missed_distance]
        else:
            # Missed detection hypothesis with distance as 'missed_distance'
            if isinstance(track.state, Prediction):
                prediction = track.state
                hypotheses.append(
                    SingleDistanceHypothesis(
                        prediction,
                        MissedDetection(timestamp=prediction.timestamp),
                        self.missed_distance))
            else:
                prediction = self.predictor.predict(track.state,
                                                    timestamp=timestamp)
                hypotheses.append(
                    SingleDistanceHypothesis(
                        prediction,
                        MissedDetection(timestamp=timestamp),
                        self.missed_distance))
        # for detection in detections:
        #
        #     # Re-evaluate prediction
        #     prediction = self.predictor.predict(
        #         track, timestamp=detection.timestamp, **kwargs)
        #
        #     # Compute measurement prediction and distance measure
        #     measurement_prediction = self.updater.predict_measurement(
        #         prediction, detection.measurement_model, **kwargs)
        #     distance = self.measure(measurement_prediction, detection)
        #
        #     if self.include_all or distance < self.missed_distance:
        #         # True detection hypothesis
        #         hypotheses.append(
        #             SingleDistanceHypothesis(
        #                 prediction,
        #                 detection,
        #                 distance,
        #                 measurement_prediction))

        mult2 = copy(mult)
        mult2.single_hypotheses = sorted(hypotheses, reverse=True)
        return mult2  # MultipleHypothesis(sorted(hypotheses, reverse=True))
