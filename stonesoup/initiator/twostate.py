
from ..base import Base, Property
from ..models.transition import TransitionModel
from ..types.state import GaussianState, TwoStateGaussianState
from ..updater import Updater
from ..functions import predict_state_to_two_state
from ..types.hypothesis import SingleHypothesis, MultiHypothesis
from ..types.track import Track
from ..types.numeric import Probability


class TwoStateInitiator(Base):

    def __init__(self, *args, **kwargs):
        super(TwoStateInitiator, self).__init__(*args, **kwargs)
        self._max_track_id = 0

    prior: GaussianState = Property(doc='The prior used to initiate fused tracks')
    transition_model: TransitionModel = Property(doc='The transition model')
    updater: Updater = Property(doc='Updater used to update fused tracks')

    def initiate(self, detections, start_time, end_time):
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
