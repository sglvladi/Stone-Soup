# -*- coding: utf-8 -*-

from .base import DataAssociator
from ..base import Property
from ..hypothesiser import Hypothesiser


class NearestNeighbour(DataAssociator):
    """Nearest Neighbour Associator

    Scores and associates detections to a predicted state using the Nearest
    Neighbour method.
    """

    hypothesiser = Property(
        Hypothesiser,
        doc="Generate a set of hypotheses for each prediction-detection pair")

    def associate(self, tracks, detections, time):
        """Associate detections with predicted states.

        Parameters
        ----------
        tracks : list of :class:`Track`
            Current tracked objects
        detections : list of :class:`Detection`
            Retrieved measurements
        time : datetime
            Detection time to predict to

        Returns
        -------
        dict
            Key value pair of tracks with associated detection
        """

        # Generate a set of hypotheses for each track on each detection
        hypotheses = {
            track: self.hypothesiser.hypothesise(track, detections, time)
            for track in tracks}

        associations = {}
        associated_measurements = set()
        while tracks > associations.keys():
            # Define a 'greedy' association
            best_hypothesis = None
            for track in tracks - associations.keys():
                for hypothesis in hypotheses[track]:
                    # A measurement may only be associated with a single track
                    if hypothesis.measurement in associated_measurements:
                        continue
                    # best_hypothesis is 'greater than' other
                    if (best_hypothesis is None
                            or hypothesis > best_hypothesis):
                        best_hypothesis = hypothesis
                        best_hypothesis_track = track

            associations[best_hypothesis_track] = best_hypothesis
            if best_hypothesis.measurement is not None:
                associated_measurements.add(best_hypothesis.measurement)

        return associations


class GlobalNearestNeighbour(DataAssociator):
    """Global Nearest Neighbour Associator

    Scores and associates detections to a predicted state using the Global
    Nearest Neighbour method, assuming a distance-based hypothesis score.
    """

    hypothesiser = Property(
        Hypothesiser,
        doc="Generate a set of hypotheses for each prediction-detection pair")

    def associate(self, tracks, detections, time):
        """Associate a set of detections with predicted states.

        Parameters
        ----------
        tracks : list of :class:`Track`
            Current tracked objects
        detections : list of :class:`Detection`
            Retrieved measurements
        time : datetime
            Detection time to predict to

        Returns
        -------
        dict
            Key value pair of tracks with associated detection
        """

        # Generate a set of hypotheses for each track on each detection
        hypotheses = {
            track: self.hypothesiser.hypothesise(track, detections, time)
            for track in tracks}

        # Link hypotheses into a set of joint_hypotheses and evaluate
        joint_hypotheses = self.enumerate_joint_hypotheses(hypotheses)
        associations = max(joint_hypotheses)

        return associations


class GaussianMixtureAssociator(DataAssociator):
    """Gaussian Mixture (GM) Associator

    Hypothesiser returns a list of MultipleHypothesis objects where each
    object contains SingleHypotheses with the Mahalanobis Distance between a
    Detection and each component of the predicted Gaussian Mixture state, and
    each MultipleHypothesis contains SingleHypotheses related to a SINGLE
    Detection.  This associator just passes through the results.
    """

    hypothesiser = Property(
        Hypothesiser,
        doc="Generates hypotheses for each Gaussian Mixture component-"
            "detection pair")

    def associate(self, predict_state, detections, time):
        """Returns the result of calling the hypothesiser.

        Parameters
        ----------
        predict_state : :class:`list`
            List of :class:`WeightedGaussianState` components
            representing the predicted state of the space
        detections : list of :class:`Detection`
            Retrieved measurements
        time : datetime
            time of the detections/predicted state

        Returns
        -------
        list of :class:`MultipleHypothesis`
            each MultipleHypothesis in the list contains SingleHypotheses
            pertaining to the same Detection
        """

        return self.hypothesiser.hypothesise(predict_state, detections, time)
