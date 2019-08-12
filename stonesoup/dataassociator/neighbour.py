# -*- coding: utf-8 -*-

import numpy as np
import concurrent.futures
from scipy.optimize import linear_sum_assignment

from .base import DataAssociator
from ..base import Property
from ..types.hypothesis import JointHypothesis
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

        # Only associate tracks with one or more hypotheses
        associate_tracks = {track
                            for track, track_hypotheses in hypotheses.items()
                            if track_hypotheses}

        associations = {}
        associated_measurements = set()
        while associate_tracks > associations.keys():
            # Define a 'greedy' association
            best_hypothesis = None
            for track in associate_tracks - associations.keys():
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
            if best_hypothesis:
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

        # Assign a hypothesis index to each detection
        for ind, detection in enumerate(detections):
            detection.metadata["hyp_ind"] = ind

        # Generate a set of hypotheses for each track on each detection
        track_list = list(tracks)
        hypotheses = {
            track: self.hypothesiser.hypothesise(track, detections, time)
            for track in track_list}

        # The cost matrix has a size (num_tracks, num_detections + num_tracks)
        # where, each row contains the association costs between a given track
        # and all detections, including the missed detection. The first
        # num_detections columns contain costs of association to detections,
        # while the remaining num_tracks columns are used to store the missed
        # detection cost for each track.
        num_tracks, num_detections = (len(track_list), len(detections))
        pseudo_inf = np.finfo(float).max  # Pseudo Inf value needed for
                                            # scipy.linear_sum_assignment
        cost_matrix = np.full((num_tracks, num_detections + num_tracks),
                              pseudo_inf)
        # Store column index of missed hypothesis cost for each track
        missed_hyp_indices = [i for i in range(num_detections,
                                               num_detections + num_tracks)]
        # Construct cost matrix
        for track_ind, track in enumerate(track_list):
            missed_hyp_ind = missed_hyp_indices[track_ind]
            for hypothesis in hypotheses[track]:
                hyp_ind = missed_hyp_ind if not hypothesis else \
                    hypothesis.measurement.metadata["hyp_ind"]
                cost = min((pseudo_inf, -np.log(hypothesis.weight)))
                cost = max((-pseudo_inf, cost))
                cost_matrix[track_ind, hyp_ind] = cost

        # Solve the linear sum assignment problem.
        track_inds, hyp_inds = linear_sum_assignment(cost_matrix)

        # Get joint hypothesis
        joint_hypothesis = dict()
        for i, track_ind in enumerate(track_inds):
            track = track_list[track_ind]
            hyp_ind = hyp_inds[i]
            hypothesis = next(
                hyp for hyp in hypotheses[track]
                if (not hyp and hyp_ind == missed_hyp_indices[track_ind])
                or (hyp and hyp_ind == hyp.measurement.metadata["hyp_ind"]))
            joint_hypothesis[track] = hypothesis
        associations = JointHypothesis(joint_hypothesis)

        return associations
