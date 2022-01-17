import numpy as np

from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..tracker import Tracker
from ..reader.tracklet import PseudoMeasExtractor
from ..predictor import Predictor
from ..updater import Updater
from ..dataassociator import DataAssociator
from ..types.numeric import Probability
from ..types.state import GaussianState, TwoStateGaussianState
from ..types.prediction import TwoStateGaussianStatePrediction, Prediction
from ..types.array import StateVector, CovarianceMatrix
from ..types.update import Update
from ..types.hypothesis import SingleHypothesis, MultiHypothesis
from ..types.track import Track

from ..functions import predict_state_to_two_state


class FuseTracker(Tracker):
    detector: PseudoMeasExtractor = Property(doc='The pseudo-measurement extractor')
    prior: GaussianState = Property(doc='The prior used to initiate fused tracks')
    predictor: Predictor = Property(doc='Predictor used to predict fused tracks')
    updater: Updater = Property(doc='Updater used to update fused tracks')
    associator: DataAssociator = Property(doc='Associator used to associate fused tracks with'
                                              'pseudomeasurements')
    death_rate: float = Property(doc='The exponential death rate of tracks. Default is 1e-4',
                                 default=1e-4)
    prob_detect: Probability = Property(doc='The probability of detection', default=0.9)
    delete_thresh: Probability = Property(doc='The existence probability deletion threshold',
                                          default=0.1)

    def __init__(self, *args, **kwargs):
        super(FuseTracker, self).__init__(*args, **kwargs)
        self._max_track_id = 0
        self.transition_model = self.predictor.transition_model

    @BufferedGenerator.generator_method
    def tracks_gen(self):
        """Returns a generator of tracks for each time step.

        Yields
        ------
        : :class:`datetime.datetime`
            Datetime of current time step
        : set of :class:`~.Track`
            Tracks existing in the time step
        """
        tracks = set()
        current_end_time = None
        for timestamp, scan in self.detector:
            new_start_time = scan.start_time
            new_end_time = scan.end_time
            if current_end_time and new_start_time < current_end_time:
                print('Skipping a scan')
                continue
            # Predict two-state tracks forward
            for track in tracks:
                self.predict_track(track, current_end_time, new_start_time, new_end_time,
                                   self.death_rate)
            current_start_time = new_start_time
            current_end_time = new_end_time

            for sensor_scan in scan.sensor_scans:
                detections = set(sensor_scan.detections)

                # Perform data association
                associations = self.associator.associate(tracks, detections,
                                                         timestamp=current_end_time)
                # Update tracks
                for track in tracks:
                    self.update_track(track, associations[track])

                # Initiate new tracks on unassociated detections
                assoc_detections = set(
                    [hyp.measurement for hyp in associations.values() if hyp])
                unassoc_detections = set(detections) - assoc_detections
                tracks |= self.init_tracks(unassoc_detections, current_start_time,
                                           current_end_time)

            tracks -= self.delete_tracks(tracks)
            yield timestamp, tracks

    def predict_track(self, track, current_end_time, new_start_time, new_end_time,
                      death_rate=0.):

        # Predict existence
        survive_prob = np.exp(-death_rate * (new_end_time - current_end_time).total_seconds())
        track.exist_prob = track.exist_prob * survive_prob

        # Predict forward
        # p(x_k, x_{k+\Delta} | y^{1:S}_{1:k})
        prediction = self.predictor.predict(track.state, current_end_time=current_end_time,
                                            new_start_time=new_start_time,
                                            new_end_time=new_end_time)
        # Append prediction to track history
        track.append(prediction)

    def update_track(self, track, hypothesis, time=None):
        last_state = track.states[-1]
        if hypothesis:
            # Perform update using the hypothesis
            update = self.updater.update(hypothesis)
            # Modify track states depending on type of last state
            if isinstance(last_state, Update) and last_state.timestamp == update.timestamp:
                # If the last scan was an update with the same timestamp, we need to modify this
                # state to reflect the computed mean and covariance, as well as the hypotheses that
                # resulted to this
                hyp = last_state.hypothesis
                try:
                    hyp.measurements.append(hypothesis.measurement)
                except AttributeError:
                    hyp = MultiHypothesis(prediction=hypothesis.prediction,
                                          measurements=[hyp.measurement, hypothesis.measurement])
                update.hypothesis = hyp  # Update the hypothesis
                track.states[-1] = update  # Replace the last state
            elif isinstance(last_state, Prediction) and last_state.timestamp == update.timestamp:
                # If the last state was a prediction with the same timestamp, it means that the
                # state was created by a sensor scan in the same overall scan, due to the track not
                # having been associated to any measurement. Therefore, we replace the prediction
                # with the update
                update.hypothesis = MultiHypothesis(prediction=hypothesis.prediction,
                                                    measurements=[hypothesis.measurement])
                track.states[-1] = update
            else:
                # Else simply append the update to the track history
                update.hypothesis = MultiHypothesis(prediction=hypothesis.prediction,
                                                    measurements=[hypothesis.measurement])
                track.append(update)
            # Set existence probability to 1
            track.exist_prob = 1
        else:
            # If the track was not associated to any measurement, simply update the existence
            # probability
            non_exist_weight = 1 - track.exist_prob
            non_det_weight = (1 - self.prob_detect) * track.exist_prob
            track.exist_prob = non_det_weight / (non_exist_weight + non_det_weight)

    def init_tracks(self, detections, start_time, end_time):
        init_mean = self.prior.mean
        init_cov = self.prior.covar
        init_mean, init_cov = predict_state_to_two_state(init_mean, init_cov,
                                                         self.transition_model,
                                                         end_time - start_time)

        prior = TwoStateGaussianState(init_mean, init_cov, start_time=start_time,
                                      end_time=end_time)
        new_tracks = set()
        for detection in detections:
            hyp = SingleHypothesis(prediction=prior, measurement=detection)
            state = self.updater.update(hyp)
            track = Track([state], id=self._max_track_id)
            track.exist_prob = Probability(1)
            self._max_track_id += 1
            new_tracks.add(track)

        return new_tracks

    def delete_tracks(self, tracks):
        del_tracks = set([track for track in tracks if track.exist_prob < self.delete_thresh])
        return del_tracks
