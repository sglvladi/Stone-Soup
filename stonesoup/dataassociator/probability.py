# -*- coding: utf-8 -*-

from .base import DataAssociator
from ..base import Property
from ..hypothesiser import Hypothesiser
from ..types.multiplehypothesis import ProbabilityMultipleHypothesis


class SimplePDA(DataAssociator):
    """Simple Probabilistic Data Associatoion (PDA)

    Given a set of detections and a set of tracks, each detection has a
    probability that it is associated each specific track.  For each track,
    associate the highest probability (remaining) detection hypothesis with
    that track.

    This particular data associator assumes no gating; all detections have the
    possibility to be associated with any track.  This can lead to excessive
    computation time.
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

            # form the 'best multihypothesis' - MultipleHypothesis that
            # contains all hypotheses related to 'best_hypothesis_track' (but
            # different detections), with the information relevant to
            # 'best_hypothesis' residing in the top level of the
            # MultipleHypothesis
            # - the null hypothesis (hypothesis.measurement == None) must
            #   be the first Hypothesis in the MultipleHypothesis
            multihypothesis = [hypothesis for hypothesis in
                               hypotheses[best_hypothesis_track] if
                               hypothesis.measurement is None] + \
                              [hypothesis for hypothesis in
                               hypotheses[best_hypothesis_track] if
                               hypothesis.measurement is not None]

            best_multihypothesis = \
                ProbabilityMultipleHypothesis(
                    best_hypothesis.prediction, best_hypothesis.measurement,
                    measurement_prediction=best_hypothesis.
                    measurement_prediction,
                    probability=best_hypothesis.probability,
                    hypotheses=multihypothesis)

            associations[best_hypothesis_track] = best_multihypothesis
            if best_hypothesis.measurement is not None:
                associated_measurements.add(best_hypothesis.measurement)

        return associations
