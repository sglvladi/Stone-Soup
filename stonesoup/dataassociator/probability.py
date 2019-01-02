# -*- coding: utf-8 -*-
import itertools

from .base import DataAssociator
from ..base import Property
from ..hypothesiser import Hypothesiser
from ..hypothesiser.probability import PDAHypothesiser
from ..types import Probability, MissedDetection, \
    SingleMeasurementProbabilityHypothesis, ProbabilityJointHypothesis
from ..types.multimeasurementhypothesis import \
    ProbabilityMultipleMeasurementHypothesis


def associate_highest_probability_hypotheses(tracks, hypotheses):
    """Associate Detections with Tracks according to highest probability hypotheses

        Parameters
        ----------
        tracks : list of :class:`Track`
            Current tracked objects
        hypotheses : list of :class:`ProbabilityMultipleMeasurementHypothesis`
            Hypothesis containing probability each of the Detections is
            associated with the specified Track (or MissedDetection)

        Returns
        -------
        dict
            Key value pair of tracks with associated detection
    """
    associations = {}
    associated_measurements = set()
    while tracks > associations.keys():
        # Define a 'greedy' association
        highest_probability_detection = None
        highest_probability = Probability(0)
        for track in tracks - associations.keys():
            for weighted_measurement in \
                    hypotheses[track].weighted_measurements:
                # A measurement may only be associated with a single track
                current_probability = weighted_measurement["weight"]
                if weighted_measurement["measurement"] in \
                        associated_measurements:
                    continue
                # best_hypothesis is 'greater than' other
                if (highest_probability_detection is None
                        or current_probability > highest_probability):
                    highest_probability_detection = \
                        weighted_measurement["measurement"]
                    highest_probability = current_probability
                    highest_probability_track = track

        hypotheses[highest_probability_track]. \
            set_selected_measurement(highest_probability_detection)
        associations[highest_probability_track] = \
            hypotheses[highest_probability_track]
        if not isinstance(highest_probability_detection, MissedDetection):
            associated_measurements.add(highest_probability_detection)

    return associations


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

        return associate_highest_probability_hypotheses(tracks, hypotheses)


class JPDA(DataAssociator):
    """Joint Probabilistic Data Associatoion (JPDA)

    Given a set of detections and a set of tracks, each detection has a
    probability that it is associated with each specific track.  However,
    when a detection could be associated with one of several tracks, this
    must be calculated via a joint probability.  In the end, the highest-
    probability Joint Hypothesis is returned as the correct Track/Detection
    association set.

    This particular data associator has no configurable gating; therefore,
    all detections have the possibility to be associated with any track
    (although the probability of association could be very close to 0).  This
    can lead to excessive computation time due to combinatorial explosion.  To
    address this problem, some rudimentary gating is implemented.  If

    .. math::

          prob_association(Detection, Track) <
          \frac{prob_association(MissedDetection, Track);gate_ratio}

    then Detection is assumed to be outside Track's gate ('gate_ratio'
    arbitrarily set to 5).  This calculation takes place in the function
    'enumerate_JPDA_hypotheses()'.
    """

    hypothesiser = Property(
        PDAHypothesiser,
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

        # Calculate MultipleMeasurementHypothesis for each Track over all
        # available Detections
        hypotheses = {
            track: self.hypothesiser.hypothesise(track, detections, time)
            for track in tracks}

        # enumerate the Joint Hypotheses of track/detection associations
        joint_hypotheses = self.enumerate_JPDA_hypotheses(tracks,
                                                          detections,
                                                          hypotheses)

        # Calculate MultiMeasurementHypothesis for each Track over all
        # available Detections with probabilities drawn from JointHypotheses
        new_hypotheses = []

        for track_index, track in enumerate(tracks):
            weighted_detections = list()
            probabilities = list()

            # record the MissedDetection hypothesis for this track
            prob_misdetect = Probability(
                sum([joint_hypothesis.probability
                     for joint_hypothesis in joint_hypotheses
                     if isinstance(joint_hypothesis.hypotheses[track].
                                   measurement, MissedDetection)]))
            weighted_detections.append(MissedDetection(timestamp=time))
            probabilities.append(prob_misdetect)

            # record hypothesis for any given Detection being associated with
            # this track
            for detection in detections:
                pro_detect_assoc = Probability(
                    sum([joint_hypothesis.probability
                         for joint_hypothesis in joint_hypotheses
                         if joint_hypothesis.
                        hypotheses[track].measurement is detection]))
                weighted_detections.append(detection)
                probabilities.append(pro_detect_assoc)

            result = ProbabilityMultipleMeasurementHypothesis(
                hypotheses[track].prediction,
                hypotheses[track].measurement_prediction)
            result.add_weighted_detections(
                weighted_detections, probabilities, normalize=True)

            new_hypotheses.append(result)

        new_hypotheses_result = {
            track: new_hypothesis
            for track, new_hypothesis in zip(tracks, new_hypotheses)}

        return associate_highest_probability_hypotheses(tracks,
                                                        new_hypotheses_result)

    @classmethod
    def enumerate_JPDA_hypotheses(cls, tracks, input_detections, multihypths):

        detections = list(input_detections)
        joint_hypotheses = list()

        num_tracks = len(tracks)
        num_detections = len(detections)

        if num_detections <= 0 or num_tracks <= 0:
            return joint_hypotheses

        # perform a simple level of gating - all track/detection pairs for
        # which the probability of association is a certain multiple less
        # than the probability of missed detection - detection is outside the
        # gating region, association is impossible
        gate_ratio = 5
        possible_assoc = list()

        for track_index, track in enumerate(tracks):
            this_track_possible_assoc = list()
            this_track_missed_detection_probability = multihypths[track].\
                get_missed_detection_probability()
            for detect_index, detection in enumerate(
                    multihypths[track].weighted_measurements):
                if this_track_missed_detection_probability / \
                        detection["weight"] <= gate_ratio:
                    this_track_possible_assoc.append(detect_index)
            possible_assoc.append(tuple(this_track_possible_assoc))

        # enumerate all valid JPDA joint hypotheses: position in character
        # string is the track, digit is the assigned detection
        # (0 is missed detection)
        enum_JPDA_hypotheses = [joint_hypothesis
                                for joint_hypothesis in
                                list(itertools.product(*possible_assoc))
                                if cls.isvalid(joint_hypothesis)]

        # turn the valid JPDA joint hypotheses into 'JointHypothesis'
        for elem in enum_JPDA_hypotheses:
            local_hypotheses = {}

            for detection, track in zip(elem, tracks):
                source_multihypothesis = multihypths[track]
                assoc_detection = detections[detection-1] if detection > 0 \
                    else MissedDetection(
                    timestamp=detections[detection].timestamp)

                local_hypothesis = \
                    SingleMeasurementProbabilityHypothesis(
                        source_multihypothesis.prediction, assoc_detection,
                        measurement_prediction=source_multihypothesis.
                        measurement_prediction,
                        probability=source_multihypothesis.
                        weighted_measurements[detection]["weight"])

                local_hypotheses[track] = local_hypothesis

            joint_hypotheses.append(
                ProbabilityJointHypothesis(local_hypotheses))

        # normalize ProbabilityJointHypotheses relative to each other
        sum_probabilities = sum([hypothesis.probability
                                 for hypothesis in joint_hypotheses])
        for hypothesis in joint_hypotheses:
            hypothesis.probability /= sum_probabilities

        return joint_hypotheses

    @staticmethod
    def isvalid(joint_hypothesis):

        # 'joint_hypothesis' represents a valid joint hypothesis if:
        #   1) no digit is repeated (except 0)

        # check condition #1
        uniqueList = []
        for elem in joint_hypothesis:
            if elem in uniqueList and elem != 0:
                return False
            else:
                uniqueList.append(elem)

        return True
