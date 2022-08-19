import numpy as np

from stonesoup.base import Property
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.dataassociator import DataAssociator
from stonesoup.deleter import Deleter
from stonesoup.functions import gm_reduce_single
from stonesoup.initiator import Initiator
from stonesoup.reader import DetectionReader
from stonesoup.tracker import Tracker
from stonesoup.types.array import StateVectors
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.types.update import GaussianStateUpdate
from stonesoup.updater import Updater


class MultiTargetMixtureTrackerSMC(Tracker):
    """A simple multi target tracker that receives associations from a
    (Gaussian) Mixture associator.

    Track multiple objects using Stone Soup components. The tracker works by
    first calling the :attr:`data_associator` with the active tracks, and then
    either updating the track state with the result of the
    :attr:`data_associator` that reduces the (Gaussian) Mixture of all
    possible track-detection associations, or with the prediction if no
    detection is associated to the track. Tracks are then checked for deletion
    by the :attr:`deleter`, and remaining unassociated detections are passed
    to the :attr:`initiator` to generate new tracks.

    Parameters
    ----------
    """
    initiator: Initiator = Property(doc="Initiator used to initialise the track.")
    deleter: Deleter = Property(doc="Initiator used to initialise the track.")
    detector: DetectionReader = Property(doc="Detector used to generate detection objects.")
    data_associator: DataAssociator = Property(
        doc="Association algorithm to pair predictions to detections")
    updater: Updater = Property(doc="Updater used to update the track object to the new state.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @BufferedGenerator.generator_method
    def tracks_gen(self):
        tracks = set()

        for time, detections in self.detector:
            tracks = list(tracks)
            detections = list(detections)

            associations = self.data_associator.associate(
                tracks, detections, time)

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

            for track, multihypothesis in associations.items():

                # calculate each Track's state as a Gaussian Mixture of
                # its possible associations with each detection, then
                # reduce the Mixture to a single Gaussian State
                posterior_states = []
                posterior_state_weights = []
                for hypothesis in multihypothesis:
                    if not hypothesis:
                        posterior_states.append(hypothesis.prediction)
                    else:
                        posterior_states.append(
                            self.updater.update(hypothesis))
                    posterior_state_weights.append(
                        hypothesis.probability)

                means = StateVectors([state.state_vector for state in posterior_states])
                covars = np.stack([state.covar for state in posterior_states], axis=2)
                weights = np.asarray(posterior_state_weights)

                post_mean, post_covar = gm_reduce_single(means, covars, weights)

                missed_detection_weight = next(hyp.weight for hyp in multihypothesis if not hyp)

                # Check if at least one reasonable measurement...
                if any(hypothesis.weight > missed_detection_weight
                       for hypothesis in multihypothesis):
                    # ...and if so use update type
                    track.append(GaussianStateUpdate(
                        post_mean, post_covar,
                        multihypothesis,
                        multihypothesis[0].measurement.timestamp))
                else:
                    # ...and if not, treat as a prediction
                    track.append(GaussianStatePrediction(
                        post_mean, post_covar,
                        multihypothesis[0].prediction.timestamp))

            tracks = set(tracks)
            tracks -= self.deleter.delete_tracks(tracks)
            tracks |= self.initiator.initiate(detections, time, weights=rho)

            yield time, tracks