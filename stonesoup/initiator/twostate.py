import numpy as np

from ..base import Base, Property
from ..dataassociator.mfa import MFADataAssociator
from ..dataassociator.probability import JPDA
from ..models.base import NonLinearModel, ReversibleModel
from ..models.transition import TransitionModel
from ..tracker.fuse import _BaseFuseTracker
from ..types.mixture import GaussianMixture
from ..types.state import GaussianState, TwoStateGaussianState, State
from ..types.update import Update
from ..updater import Updater
from ..functions import predict_state_to_two_state, nearestPD, isPD
from ..types.hypothesis import SingleHypothesis, MultiHypothesis
from ..types.track import Track
from ..types.numeric import Probability
from ..types.tracklet import SensorScan, Scan


class TwoStateInitiator(Base):

    def __init__(self, *args, **kwargs):
        super(TwoStateInitiator, self).__init__(*args, **kwargs)
        self._max_track_id = 0

    prior: GaussianState = Property(doc='The prior used to initiate fused tracks')
    transition_model: TransitionModel = Property(doc='The transition model')
    updater: Updater = Property(doc='Updater used to update fused tracks')

    def initiate(self, detections, start_time, end_time, **kwargs):
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


class TwoStateInitiatorMixture(TwoStateInitiator):

    def initiate(self, detections, start_time, end_time, **kwargs):
        sure_tracks = super().initiate(detections, start_time, end_time, **kwargs)
        for track in sure_tracks:
            prior = GaussianMixture([TwoStateGaussianState(track.state.state_vector,
                                                           track.state.covar,
                                                           start_time=track.state.start_time,
                                                           end_time=track.state.end_time,
                                                           weight=Probability(1),
                                                           tag=[])])
            track.states[-1] = prior
        return sure_tracks



class TwoStateMeasurementInitiator(TwoStateInitiator):

    skip_non_reversible: bool = Property(default=False)
    diag_load: float = Property(default=0.0, doc="Positive float value for diagonal loading")

    def initiate(self, detections, start_time, end_time, **kwargs):

        new_tracks = set()
        for detection in detections:
            measurement_model = detection.measurement_model

            if isinstance(measurement_model, NonLinearModel):
                if isinstance(measurement_model, ReversibleModel):
                    try:
                        state_vector = measurement_model.inverse_function(detection)
                    except NotImplementedError:
                        if not self.skip_non_reversible:
                            raise
                        else:
                            continue
                    model_matrix = measurement_model.jacobian(State(state_vector))
                    inv_model_matrix = np.linalg.pinv(model_matrix)
                elif self.skip_non_reversible:
                    continue
                else:
                    raise Exception("Invalid measurement model used.\
                                    Must be instance of linear or reversible.")
            else:
                model_matrix = measurement_model.matrix()
                inv_model_matrix = np.linalg.pinv(model_matrix)
                state_vector = inv_model_matrix @ detection.state_vector

            model_covar = measurement_model.covar()

            init_mean = self.prior.state_vector.copy()
            init_cov = self.prior.covar.copy()


            init_mean, init_cov = predict_state_to_two_state(init_mean, init_cov,
                                                             self.transition_model,
                                                             end_time - start_time)
            mapped_dimensions = measurement_model.mapping

            init_mean[mapped_dimensions, :] = 0
            init_cov[mapped_dimensions, :] = 0
            C0 = inv_model_matrix @ model_covar @ inv_model_matrix.T
            C0 = C0 + init_cov + np.diag(np.array([self.diag_load] * C0.shape[0]))
            if not isPD(C0):
                C0 = nearestPD(C0)
            init_mean = init_mean + state_vector
            prior = TwoStateGaussianState(init_mean, C0, start_time=start_time,
                                          end_time=end_time)
            hyp = SingleHypothesis(prediction=prior, measurement=detection)
            state = self.updater.update(hyp)
            track = Track([state], id=self._max_track_id)
            track.exist_prob = Probability(1)
            self._max_track_id += 1
            new_tracks.add(track)

        return new_tracks


class TwoStateMeasurementInitiatorMixture(TwoStateMeasurementInitiator):

    def initiate(self, detections, start_time, end_time, **kwargs):
        sure_tracks = super().initiate(detections, start_time, end_time, **kwargs)
        for track in sure_tracks:
            prior = GaussianMixture([TwoStateGaussianState(track.state.state_vector,
                                                           track.state.covar,
                                                           start_time=track.state.start_time,
                                                           end_time=track.state.end_time,
                                                           weight=Probability(1),
                                                           tag=[])])
            track.states[-1] = prior
        return sure_tracks


class FuseTrackerInitiator(_BaseFuseTracker):
    min_points: int = Property(
        default=2, doc="Minimum number of track points required to confirm a track.")
    output_mixture: bool = Property(default=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.holding_tracks = set()
        self.current_end_time = None
        self.current_start_time = None

    def initiate(self, detections, start_time, end_time, sensor_id=None, **kwargs):
        sure_tracks = set()

        if len(detections):
            sscan = SensorScan(sensor_id, list(detections), timestamp=end_time)
        else:
            sscan = SensorScan(None, [])

        scan = Scan(start_time, end_time, [sscan])
        self.holding_tracks, current_end_time = self.process_scan(scan, self.holding_tracks,
                                                                  self.current_end_time)
        if not self.current_end_time:
            self.current_end_time = current_end_time

        for track in self.holding_tracks:
            if sum(1 for state in track if isinstance(state, Update)) >= self.min_points \
                    and track.exist_prob > 0.9:
                sure_tracks.add(track)

        self.holding_tracks -= sure_tracks

        if self.output_mixture:
            for track in sure_tracks:
                prior = GaussianMixture([TwoStateGaussianState(track.state.state_vector,
                                                               track.state.covar,
                                                               start_time=track.state.start_time,
                                                               end_time=track.state.end_time,
                                                               weight=Probability(1),
                                                               tag=[])])
                track.states[-1] = prior
        return sure_tracks

    def predict(self, new_start_time, new_end_time):
        # Predict two-state tracks forward
        for track in self.holding_tracks:
            self.predict_track(track, self.current_end_time, new_start_time, new_end_time,
                               self.death_rate)

    def process_scan(self, scan, tracks, current_end_time):

        current_start_time = scan.start_time
        current_end_time = scan.end_time

        for sensor_scan in scan.sensor_scans:
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



