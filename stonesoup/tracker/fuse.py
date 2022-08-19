import numpy as np

from ..base import Base, Property
from ..buffered_generator import BufferedGenerator
from ..custom.initiator import TwoStateSMCPHDInitiator
from ..dataassociator.mfa import MFADataAssociator
from ..dataassociator.probability import JPDA
from ..tracker import Tracker
from ..reader.tracklet import PseudoMeasExtractor
from ..predictor import Predictor
from ..types.mixture import GaussianMixture
from ..types.multihypothesis import MultipleHypothesis
from ..updater import Updater
from ..dataassociator import DataAssociator
from ..types.numeric import Probability
from ..types.prediction import Prediction
from ..types.array import StateVectors
from ..types.update import Update, TwoStateGaussianStateUpdate, GaussianMixtureUpdate
from ..types.hypothesis import MultiHypothesis
from ..initiator import Initiator
from ..functions import gm_reduce_single


class _BaseFuseTracker(Base):
    initiator: Initiator = Property(doc='The initiator used to initiate fused tracks')
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
        super(_BaseFuseTracker, self).__init__(*args, **kwargs)
        self._max_track_id = 0

    def process_scan(self, scan, tracks, current_end_time):
        new_start_time = scan.start_time
        new_end_time = scan.end_time
        if current_end_time and new_start_time < current_end_time:
            print('Scans out of order! Skipping a scan...')
            return tracks, current_end_time

        if hasattr(self.initiator, 'predict'):
            self.initiator.predict(new_start_time, new_end_time)
            self.initiator.current_end_time = new_end_time

        # Predict two-state tracks forward
        for track in tracks:
            self.predict_track(track, current_end_time, new_start_time, new_end_time,
                               self.death_rate)

        current_start_time = new_start_time
        current_end_time = new_end_time

        for sensor_scan in scan.sensor_scans:
            tracks = list(tracks)
            detections = set(sensor_scan.detections)

            # Perform data association
            associations = self.associator.associate(tracks, detections,
                                                     timestamp=current_end_time)
            # Update tracks
            for track in tracks:
                self.update_track(track, associations[track], scan.id)

            # Initiate new tracks on unassociated detections
            if isinstance(self.associator, JPDA) or isinstance(self.associator, MFADataAssociator):
                assoc_detections = set(
                    [h.measurement for hyp in associations.values() for h in hyp if h])
            else:
                assoc_detections = set(
                    [hyp.measurement for hyp in associations.values() if hyp])

            num_tracks = len(tracks)
            num_detections = len(detections)
            assoc_prob_matrix = np.zeros((num_tracks, num_detections + 1))
            for i, track in enumerate(tracks):
                for hyp in associations[track]:
                    if not hyp:
                        assoc_prob_matrix[i, 0] = hyp.weight
                    else:
                        j = next(d_i for d_i, detection in enumerate(detections)
                                 if hyp.measurement == detection)
                        assoc_prob_matrix[i, j + 1] = hyp.weight

            rho = np.zeros((len(detections)))
            for j, detection in enumerate(detections):
                rho_tmp = 1
                if len(assoc_prob_matrix):
                    for i, track in enumerate(tracks):
                        rho_tmp *= 1 - assoc_prob_matrix[i, j + 1]
                rho[j] = rho_tmp


            if isinstance(self.initiator, TwoStateSMCPHDInitiator):
                # pass
                tracks = set(tracks)
                tracks |= self.initiator.initiate(detections, current_start_time,
                                                  current_end_time,
                                                  weights=rho,
                                                  sensor_id=sensor_scan.sensor_id)
            else:
                tracks = set(tracks)
                unassoc_detections = set(detections) - assoc_detections
                if isinstance(sensor_scan.sensor_id, str):
                    tracks |= self.initiator.initiate(unassoc_detections, sensor_scan.timestamp,
                                                      sensor_scan.timestamp,
                                                      sensor_id=sensor_scan.sensor_id)
                else:
                    tracks |= self.initiator.initiate(unassoc_detections, current_start_time,
                                                      current_end_time, sensor_id=sensor_scan.sensor_id)
        try:
            self.initiator.current_end_time = current_end_time
        except AttributeError:
            pass

        tracks -= self.delete_tracks(tracks)
        return tracks, current_end_time

    def predict_track(self, track, current_end_time, new_start_time, new_end_time,
                      death_rate=0.):

        # Predict existence
        survive_prob = np.exp(-death_rate * (new_end_time - current_end_time).total_seconds())
        track.exist_prob = track.exist_prob * survive_prob

        # Predict forward
        # p(x_k, x_{k+\Delta} | y^{1:S}_{1:k})
        if not isinstance(track.state, GaussianMixture):
            prediction = self.predictor.predict(track.state, current_end_time=current_end_time,
                                                new_start_time=new_start_time,
                                                new_end_time=new_end_time)
        else:
            pred_components = []
            for component in track.state:
                pred_components.append(self.predictor.predict(component,
                                                              current_end_time=current_end_time,
                                                              new_start_time=new_start_time,
                                                              new_end_time=new_end_time))
            prediction = GaussianMixture(pred_components)
        # Append prediction to track history
        track.append(prediction)

    def update_track(self, track, hypothesis, scan_id):
        last_state = track.states[-1]
        if isinstance(self.associator, MFADataAssociator):
            components = []
            dct = dict()    # Contains association weights for each distinct prediction component
            for hyp in hypothesis:
                if not hyp:
                    idx = 0
                    components.append(hyp.prediction)
                else:
                    idx = 1
                    update = self.updater.update(hyp)
                    components.append(update)
                    if 'track_id' in hyp.measurement.metadata:
                        try:
                            track.track_ids.add(hyp.measurement.metadata['track_id'])
                        except AttributeError:
                            track.track_ids = {hyp.measurement.metadata['track_id']}
                # Populate association weights for each prediction component
                # (i.e. components that share a common measurement history up to previous
                # (hence -1) timestep)
                try:
                    dct[tuple(hyp.prediction.tag[:-1])].insert(idx, hyp.probability)
                except KeyError:
                    dct[tuple(hyp.prediction.tag[:-1])] = [hyp.probability]

            # Ensure update contains all hypotheses
            # NOTE: EAFP (easier to ask for forgiveness than permission)
            # If the last state is not a GaussianMixtureUpdate, an AttributeError will be
            # raised since GaussianMixture does not have a "hypothesis" attribute.
            try:
                # New hypotheses = old hypotheses + (current-null)
                single_hyps = [hyp for hyp in last_state.hypothesis] \
                              + [hyp for hyp in hypothesis if hyp]
                hypothesis = MultipleHypothesis(single_hyps)
            except AttributeError:
                pass

            update = GaussianMixtureUpdate(components=components, hypothesis=hypothesis)
            track[-1] = update

            # Compute existence probability
            non_exist_weight = 1 - track.exist_prob
            non_det_weight = (1 - self.prob_detect) * track.exist_prob
            null_exist_weight = non_det_weight / (non_exist_weight + non_det_weight)
            exist_probs_list = []
            exist_probs_weights = []
            # Iterate over prediction components and compute existence prob for each
            for key, weights in dct.items():
                exist_probs = np.array([null_exist_weight, *[1. for i in range(len(weights) - 1)]])
                exist_probs_list.append(Probability.sum(exist_probs * weights))
                exist_probs_weights.append(Probability.sum(weights))

            # Existence prob is the max over them
            # NOTE: Might be better to use weighted average
            track.exist_prob = np.max(exist_probs_list)

        elif isinstance(self.associator, JPDA):
            # calculate each Track's state as a Gaussian Mixture of
            # its possible associations with each detection, then
            # reduce the Mixture to a single Gaussian State
            posterior_states = []
            posterior_state_weights = []
            for hyp in hypothesis:
                if not hyp:
                    posterior_states.append(hyp.prediction)
                    # Ensure null hyp weight is at index 0
                    posterior_state_weights.insert(0, hyp.probability)
                else:
                    posterior_states.append(
                        self.updater.update(hyp))
                    posterior_state_weights.append(
                        hyp.probability)
                    if 'track_id' in hyp.measurement.metadata:
                        try:
                            track.track_ids.add(hyp.measurement.metadata['track_id'])
                        except AttributeError:
                            track.track_ids = {hyp.measurement.metadata['track_id']}

            means = StateVectors([state.state_vector for state in posterior_states])
            covars = np.stack([state.covar for state in posterior_states], axis=2)
            weights = np.asarray(posterior_state_weights)

            post_mean, post_covar = gm_reduce_single(means, covars, weights)

            update = TwoStateGaussianStateUpdate(post_mean, post_covar,
                                                 start_time=posterior_states[0].start_time,
                                                 end_time=posterior_states[0].end_time,
                                                 hypothesis=hypothesis)
            track[-1] = update
            # Compute existence probability
            non_exist_weight = 1 - track.exist_prob
            non_det_weight = (1 - self.prob_detect) * track.exist_prob
            null_exist_weight = non_det_weight / (non_exist_weight + non_det_weight)
            exist_probs = np.array([null_exist_weight, *[1. for i in range(len(weights) - 1)]])
            track.exist_prob = Probability.sum(exist_probs * weights)
        else:
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
                                              measurements=[hyp.measurement,
                                                            hypothesis.measurement])
                    update.hypothesis = hyp  # Update the hypothesis
                    track[-1] = update  # Replace the last state
                elif isinstance(last_state,
                                Prediction) and last_state.timestamp == update.timestamp:
                    # If the last state was a prediction with the same timestamp, it means that the
                    # state was created by a sensor scan in the same overall scan, due to the track not
                    # having been associated to any measurement. Therefore, we replace the prediction
                    # with the update
                    update.hypothesis = MultiHypothesis(prediction=hypothesis.prediction,
                                                        measurements=[hypothesis.measurement])
                    track[-1] = update
                else:
                    # Else simply append the update to the track history
                    update.hypothesis = MultiHypothesis(prediction=hypothesis.prediction,
                                                        measurements=[hypothesis.measurement])
                    track.append(update)
                # Set existence probability to 1
                track.exist_prob = 1
                if 'track_id' in hypothesis.measurement.metadata:
                    try:
                        track.track_ids.add(hypothesis.measurement.metadata['track_id'])
                    except AttributeError:
                        track.track_ids = {hypothesis.measurement.metadata['track_id']}
            else:
                # If the track was not associated to any measurement, simply update the existence
                # probability
                non_exist_weight = 1 - track.exist_prob
                non_det_weight = (1 - self.prob_detect) * track.exist_prob
                track.exist_prob = non_det_weight / (non_exist_weight + non_det_weight)

    def delete_tracks(self, tracks):
        del_tracks = set([track for track in tracks if track.exist_prob < self.delete_thresh])
        return del_tracks


class FuseTracker(Tracker, _BaseFuseTracker):
    """

    """

    detector: PseudoMeasExtractor = Property(doc='The pseudo-measurement extractor')

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
        for timestamp, scans in self.detector:
            for scan in scans:
                tracks, current_end_time = self.process_scan(scan, tracks, current_end_time)
            yield timestamp, tracks


