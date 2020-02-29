import numpy as np
from copy import copy
from scipy.stats import multivariate_normal as mn
from scipy.linalg import expm

from stonesoup.types.prediction import Prediction
from .base import Hypothesiser
from ..base import Property
from ..types.detection import MissedDetection
from ..types.hypothesis import SingleProbabilityHypothesis
from ..types.multihypothesis import MultipleHypothesis
from ..types.numeric import Probability
from ..predictor import Predictor
from ..updater import Updater


class PDAHypothesiser(Hypothesiser):
    """Hypothesiser based on Probabilistic Data Association (PDA)

    Generate track predictions at detection times and calculate probabilities
    for all prediction-detection pairs for single prediction and multiple
    detections.
    """

    predictor = Property(
        Predictor,
        doc="Predict tracks to detection times")
    updater = Property(
        Updater,
        doc="Updater used to get measurement prediction")
    clutter_spatial_density = Property(
        float,
        doc="Spatial density of clutter - tied to probability of false "
            "detection")
    prob_detect = Property(
        Probability,
        default=Probability(0.85),
        doc="Target Detection Probability")
    prob_gate = Property(
        Probability,
        default=Probability(0.95),
        doc="Gate Probability - prob. gate contains true measurement "
            "if detected")

    def hypothesise(self, track, detections, timestamp):
        r"""Evaluate and return all track association hypotheses.

        For a given track and a set of N detections, return a
        MultipleHypothesis with N+1 detections (first detection is
        a 'MissedDetection'), each with an associated probability.
        Probabilities are assumed to be exhaustive (sum to 1) and mutually
        exclusive (two detections cannot be the correct association at the
        same time).

        Detection 0: missed detection, none of the detections are associated
        with the track.
        Detection :math:`i, i \in {1...N}`: detection i is associated
        with the track.

        The probabilities for these detections are calculated as follow:

        .. math::

          \beta_i(k) = \begin{cases}
                \frac{\mathcal{L}_{i}(k)}{1-P_{D}P_{G}+\sum_{j=1}^{m(k)}
                  \mathcal{L}_{j}(k)}, \quad i=1,...,m(k) \\
                \frac{1-P_{D}P_{G}}{1-P_{D}P_{G}+\sum_{j=1}^{m(k)}
                  \mathcal{L}_{j}(k)}, \quad i=0
                \end{cases}

        where

        .. math::

          \mathcal{L}_{i}(k) = \frac{\mathcal{N}[z_{i}(k);\hat{z}(k|k-1),
          S(k)]P_{D}}{\lambda}

        :math:`\lambda` is the clutter density

        :math:`P_{D}` is the detection probability

        :math:`P_{G}` is the gate probability

        :math:`\mathcal{N}[z_{i}(k);\hat{z}(k|k-1),S(k)]` is the likelihood
        ratio of the measurement :math:`z_{i}(k)` originating from the track
        target rather than the clutter.

        NOTE: Since all probabilities have the same denominator and are
        normalized later, the denominator can be discarded.

        References:

        [1] "The Probabilistic Data Association Filter: Estimation in the
        Presence of Measurement Origin Uncertainty" -
        https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5338565

        [2] "Robotics 2 Data Association" (Lecture notes) -
        http://ais.informatik.uni-freiburg.de/teaching/ws10/robotics2/pdfs/rob2-15-dataassociation.pdf

        Parameters
        ----------
        track: :class:`~.Track`
            The track object to hypothesise on
        detections: :class:`list`
            A list of :class:`~Detection` objects, representing the available
            detections.
        timestamp: :class:`datetime.datetime`
            A timestamp used when evaluating the state and measurement
            predictions. Note that if a given detection has a non empty
            timestamp, then prediction will be performed according to
            the timestamp of the detection.

        Returns
        -------
        : :class:`~.MultipleHypothesis`
            A container of :class:`~SingleProbabilityHypothesis` objects

        """

        hypotheses = list()

        # Common state & measurement prediction
        prediction = self.predictor.predict(track.state, timestamp=timestamp)

        # Missed detection hypothesis
        probability = Probability(1 - self.prob_detect*self.prob_gate)
        hypotheses.append(
            SingleProbabilityHypothesis(
                prediction,
                MissedDetection(timestamp=timestamp),
                probability))

        # True detection hypotheses
        for detection in detections:

            # Re-evaluate prediction
            prediction = self.predictor.predict(
                track.state, timestamp=detection.timestamp)

            # Compute measurement prediction and probability measure
            measurement_prediction = self.updater.predict_measurement(
                prediction, detection.measurement_model)
            log_pdf = mn.logpdf(detection.state_vector.ravel(),
                                measurement_prediction.state_vector.ravel(),
                                measurement_prediction.covar)
            pdf = Probability(log_pdf, log_value=True)
            probability = (pdf * self.prob_detect)/self.clutter_spatial_density

            # True detection hypothesis
            hypotheses.append(
                SingleProbabilityHypothesis(
                    prediction,
                    detection,
                    probability,
                    measurement_prediction))

        return MultipleHypothesis(hypotheses, normalise=True, total_weight=1)


class PDAHypothesiserFast(PDAHypothesiser):

    def hypothesise(self, track, detections, timestamp, missed_detection=None, mult=None):
        r"""Evaluate and return all track association hypotheses.

        For a given track and a set of N detections, return a
        MultipleHypothesis with N+1 detections (first detection is
        a 'MissedDetection'), each with an associated probability.
        Probabilities are assumed to be exhaustive (sum to 1) and mutually
        exclusive (two detections cannot be the correct association at the
        same time).

        Detection 0: missed detection, none of the detections are associated
        with the track.
        Detection :math:`i, i \in {1...N}`: detection i is associated
        with the track.

        The probabilities for these detections are calculated as follow:

        .. math::

          \beta_i(k) = \begin{cases}
                \frac{\mathcal{L}_{i}(k)}{1-P_{D}P_{G}+\sum_{j=1}^{m(k)}
                  \mathcal{L}_{j}(k)}, \quad i=1,...,m(k) \\
                \frac{1-P_{D}P_{G}}{1-P_{D}P_{G}+\sum_{j=1}^{m(k)}
                  \mathcal{L}_{j}(k)}, \quad i=0
                \end{cases}

        where

        .. math::

          \mathcal{L}_{i}(k) = \frac{\mathcal{N}[z_{i}(k);\hat{z}(k|k-1),
          S(k)]P_{D}}{\lambda}

        :math:`\lambda` is the clutter density

        :math:`P_{D}` is the detection probability

        :math:`P_{G}` is the gate probability

        :math:`\mathcal{N}[z_{i}(k);\hat{z}(k|k-1),S(k)]` is the likelihood
        ratio of the measurement :math:`z_{i}(k)` originating from the track
        target rather than the clutter.

        NOTE: Since all probabilities have the same denominator and are
        normalized later, the denominator can be discarded.

        References:

        [1] "The Probabilistic Data Association Filter: Estimation in the
        Presence of Measurement Origin Uncertainty" -
        https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5338565

        [2] "Robotics 2 Data Association" (Lecture notes) -
        http://ais.informatik.uni-freiburg.de/teaching/ws10/robotics2/pdfs/rob2-15-dataassociation.pdf

        Parameters
        ----------
        track: :class:`~.Track`
            The track object to hypothesise on
        detections: :class:`list`
            A list of :class:`~Detection` objects, representing the available
            detections.
        timestamp: :class:`datetime.datetime`
            A timestamp used when evaluating the state and measurement
            predictions. Note that if a given detection has a non empty
            timestamp, then prediction will be performed according to
            the timestamp of the detection.

        Returns
        -------
        : :class:`~.MultipleHypothesis`
            A container of :class:`~SingleProbabilityHypothesis` objects

        """

        hypotheses = list()

        if missed_detection is None:
            missed_detection = MissedDetection(timestamp=timestamp)

        # Common state & measurement prediction
        prediction = self.predictor.predict(track.state, timestamp=timestamp)
        measurement_prediction = self.updater.predict_measurement(
            prediction)

        # Missed detection hypothesis
        probability = Probability(1 - self.prob_detect*self.prob_gate)
        hypotheses.append(
            SingleProbabilityHypothesis(
                prediction,
                missed_detection,
                probability,
                measurement_prediction))

        # True detection hypotheses
        for detection in detections:

            # # Re-evaluate prediction
            # prediction = self.predictor.predict(
            #     track.state, timestamp=detection.timestamp)
            #
            # # Compute measurement prediction and probability measure
            # measurement_prediction = self.updater.predict_measurement(
            #     prediction, detection.measurement_model)
            log_pdf = mn.logpdf(detection.state_vector.ravel(),
                                measurement_prediction.state_vector.ravel(),
                                measurement_prediction.covar)
            pdf = Probability(log_pdf, log_value=True)
            probability = (pdf * self.prob_detect)/self.clutter_spatial_density

            # True detection hypothesis
            hypotheses.append(
                SingleProbabilityHypothesis(
                    prediction,
                    detection,
                    probability,
                    measurement_prediction))

        mult2 = copy(mult)
        mult2.single_hypotheses = sorted(hypotheses, reverse=True)
        return mult2  # MultipleHypothesis(hypotheses, normalise=True, total_weight=1)


class ELINTHypothesiser(Hypothesiser):
    predictor = Property(
        Predictor,
        doc="Predict tracks to detection times")
    updater = Property(
        Updater,
        doc="Updater used to get measurement prediction")
    prob_detect = Property(
        Probability,
        doc="Probability of detection"
    )
    deathRate = Property(float, doc="")
    logNullLikelihood = Property(float, doc="")

    def hypothesise(self, track, detections, timestamp, missed_detection=None, mult=None):
        hypotheses = list()
        if missed_detection is None:
            missed_detection = MissedDetection(timestamp=timestamp)

        # Existence propagation
        p_exist = track.metadata["existence"]["value"]
        dt = timestamp - track.metadata["existence"]["time"]
        p_exist = np.exp(-self.deathRate * dt.total_seconds()) * p_exist
        metadata = {
            "existence": {
                "value": p_exist,
                "time": timestamp
            }
        }
        track.metadata.update(metadata)

        # Common state & measurement prediction
        prediction = self.predictor.predict(track.state, timestamp=timestamp)

        # Missed detection hypothesis
        logNullPdf = Probability(self.logNullLikelihood, log_value=True)
        hypotheses.append(
            SingleProbabilityHypothesis(
                prediction,
                missed_detection,
                logNullPdf))

        # True detection hypotheses
        for detection in detections:
            # Compute measurement prediction and probability measure
            measurement_prediction = self.updater.predict_measurement(
                prediction, detection.measurement_model)
            log_meas = mn.logpdf(detection.state_vector.ravel(),
                                measurement_prediction.state_vector.ravel(),
                                measurement_prediction.covar)
            log_pdf = np.log(self.prob_detect) + np.log(p_exist) + log_meas
            pdf = Probability(log_pdf, log_value=True)

            # True detection hypothesis
            hypotheses.append(
                SingleProbabilityHypothesis(
                    prediction,
                    detection,
                    pdf,
                    measurement_prediction))

        mult2 = copy(mult)
        mult2.single_hypotheses = sorted(hypotheses, reverse=True)
        return mult2  # MultipleHypothesis(hypotheses, normalise=True, total_weight=1)


class ELINTHypothesiserFast(Hypothesiser):
    predictor = Property(
        Predictor,
        doc="Predict tracks to detection times")
    updater = Property(
        Updater,
        doc="Updater used to get measurement prediction")
    prob_detect = Property(
        Probability,
        doc="Probability of detection"
    )
    deathRate = Property(float, doc="")
    logNullLikelihood = Property(float, doc="")

    def hypothesise(self, track, detections, timestamp, missed_detection=None, mult=None):
        hypotheses = list()
        if missed_detection is None:
            missed_detection = MissedDetection(timestamp=timestamp)

        # Existence propagation
        p_exist = track.metadata["existence"]["value"]
        dt = timestamp-track.metadata["existence"]["time"]
        p_exist = np.exp(-self.deathRate*dt.total_seconds())*p_exist
        metadata = {
            "existence": {
                "value": p_exist,
                "time": timestamp
            }
        }
        track.metadata.update(metadata)

        # Only Hypothesise tracks whose mmsi appears in the detections
        if len(detections):
            # Common state & measurement prediction
            prediction = self.predictor.predict(track.state, timestamp=timestamp)

            # Missed detection hypothesis
            logNullPdf = Probability(self.logNullLikelihood, log_value=True)
            hypotheses.append(
                SingleProbabilityHypothesis(
                    prediction,
                    missed_detection,
                    logNullPdf))

            # True detection hypotheses
            for detection in detections:
                # Compute measurement prediction and probability measure
                measurement_prediction = self.updater.predict_measurement(
                    prediction, detection.measurement_model)
                log_meas = mn.logpdf(detection.state_vector.ravel(),
                                    measurement_prediction.state_vector.ravel(),
                                    measurement_prediction.covar)
                log_pdf = np.log(self.prob_detect) + np.log(p_exist) + log_meas
                pdf = Probability(log_pdf, log_value=True)

                # True detection hypothesis
                hypotheses.append(
                    SingleProbabilityHypothesis(
                        prediction,
                        detection,
                        pdf,
                        measurement_prediction))
        else:
            # If a prediction already exists simple set it as mis-detected
            if isinstance(track.state, Prediction):
                prediction = track.state
                # Missed detection hypothesis
                logNullPdf = Probability(self.logNullLikelihood, log_value=True)
                hypotheses.append(
                    SingleProbabilityHypothesis(
                        prediction,
                        missed_detection,
                        logNullPdf))
            else:
                # Else if the state is an update, then generate a prediction
                # and generate mis-detection
                prediction = self.predictor.predict(track.state,
                                                    timestamp=timestamp)
                # Missed detection hypothesis
                logNullPdf = Probability(self.logNullLikelihood, log_value=True)
                hypotheses.append(
                    SingleProbabilityHypothesis(
                        prediction,
                        missed_detection,
                        logNullPdf))

        mult2 = copy(mult)
        mult2.single_hypotheses = sorted(hypotheses, reverse=True)
        return mult2  # MultipleHypothesis(hypotheses, normalise=True, total_weight=1)


class AisElintHypothesiserFast(Hypothesiser):
    predictor = Property(
        Predictor,
        doc="Predict tracks to detection times")
    updater = Property(
        Updater,
        doc="Updater used to get measurement prediction")
    logNullLikelihoods = Property([float], doc="These can be precomputed")
    sensors = Property(list, doc="The sensors")
    visibility = Property(dict, doc="The visibility constants")

    def hypothesise(self, track, detections, timestamp, missed_detection=None, mult=None, sensor_idx=None, trans_matrix=None):
        hypotheses = list()
        if missed_detection is None:
            missed_detection = MissedDetection(timestamp=timestamp)

        # Visibility propagation
        vis_probs = track.metadata['visibility']['probs']
        # dt = timestamp-track.metadata["visibility"]["time"]
        # trans_matrix = self._get_vis_transitions(dt.total_seconds(), self.visibility['visStates'], self.sensors)
        metadata = {
            'visibility': {
                'probs': vis_probs @ trans_matrix, # Predicted visibility probs
                'time': timestamp
            }
        }
        track.metadata.update(metadata)

        # Only Hypothesise tracks whose mmsi appears in the detections
        if len(detections):
            # Common state & measurement prediction
            prediction = self.predictor.predict(track.state, timestamp=timestamp)

            # Missed detection hypothesis
            logNullPdf = Probability(self.logNullLikelihoods[sensor_idx], log_value=True)
            hypotheses.append(
                SingleProbabilityHypothesis(
                    prediction,
                    missed_detection,
                    logNullPdf))

            # True detection hypotheses
            for detection in detections:
                # Compute measurement prediction and probability measure
                measurement_prediction = self.updater.predict_measurement(
                    prediction, detection.measurement_model)
                log_meas = mn.logpdf(detection.state_vector.ravel(),
                                    measurement_prediction.state_vector.ravel(),
                                    measurement_prediction.covar)
                pv = self._get_sensor_vis_prob(track.metadata['visibility']['probs'], sensor_idx)
                log_pdf = np.log(self.sensors[sensor_idx]['rates']['meas']) + np.log(pv) + log_meas
                pdf = Probability(log_pdf, log_value=True)

                # True detection hypothesis
                hypotheses.append(
                    SingleProbabilityHypothesis(
                        prediction,
                        detection,
                        pdf,
                        measurement_prediction))
        else:
            # If a prediction already exists simple set it as mis-detected
            if isinstance(track.state, Prediction):
                prediction = track.state
                # Missed detection hypothesis
                logNullPdf = Probability(self.logNullLikelihoods[sensor_idx], log_value=True)
                hypotheses.append(
                    SingleProbabilityHypothesis(
                        prediction,
                        missed_detection,
                        logNullPdf))
            else:
                # Else if the state is an update, then generate a prediction
                # and generate mis-detection
                prediction = self.predictor.predict(track.state,
                                                    timestamp=timestamp)
                # Missed detection hypothesis
                logNullPdf = Probability(self.logNullLikelihoods[sensor_idx], log_value=True)
                hypotheses.append(
                    SingleProbabilityHypothesis(
                        prediction,
                        missed_detection,
                        logNullPdf))

        mult2 = copy(mult)
        mult2.single_hypotheses = sorted(hypotheses, reverse=True)
        return mult2  # MultipleHypothesis(hypotheses, normalise=True, total_weight=1)

    def _get_sensor_vis_prob(self, t_vis_probs, sensor_idx):
        return np.sum(t_vis_probs[self.visibility['visStates'][sensor_idx, :] > 0])