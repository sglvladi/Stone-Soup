# -*- coding: utf-8 -*-

import numpy as np

from .base import DataAssociator
from ..base import Property
from ..hypothesiser import Hypothesiser
from ..types.detection import MissedDetection
from ..types.hypothesis import JointHypothesis
from ..functions import auction


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

        # Assign an index to each detection
        for ind, detection in enumerate(detections):
            detection.metadata["ind"] = ind + 1

        # Generate a set of hypotheses for each track on each detection
        hypotheses = {
            track: self.hypothesiser.hypothesise(track, detections, time)
            for track in tracks}

        # Construct cost matrix
        num_tracks, num_detections = (len(tracks), len(detections))
        cost_matrix = np.full((num_tracks, num_detections + 1), -np.inf)
        for t_ind, track in enumerate(tracks):
            for hypothesis in hypotheses[track]:
                if isinstance(hypothesis.measurement, MissedDetection):
                    cost_matrix[t_ind, 0] = -hypothesis.distance
                else:
                    cost_matrix[t_ind, hypothesis.measurement.metadata["ind"]] = -hypothesis.distance

        # Perform auction
        assignments, _ = auction(cost_matrix)

        # Get joint hypothesis
        joint_hypothesis = dict()
        for t_ind, track in enumerate(tracks):
            joint_hypothesis[track] = [x for x in hypotheses[track]
                                       if (assignments[t_ind] == 0 and isinstance(x.measurement, MissedDetection))
                                       or (not isinstance(x.measurement, MissedDetection)
                                           and x.measurement.metadata["ind"] == assignments[t_ind])][0]
        associations = JointHypothesis(joint_hypothesis)

        # Link hypotheses into a set of joint_hypotheses and evaluate
        # joint_hypotheses = self.enumerate_joint_hypotheses(hypotheses)
        # associations = max(joint_hypotheses)

        return associations
