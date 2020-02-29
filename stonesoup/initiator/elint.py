import numpy as np
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal

from stonesoup.types.angle import Longitude, Latitude
from stonesoup.types.array import CovarianceMatrix, StateVector
from stonesoup.types.state import GaussianState
from .base import GaussianInitiator
from ..base import Property
from ..models.measurement import MeasurementModel
from ..types.hypothesis import SingleHypothesis
from ..types.track import Track
from ..updater.kalman import KalmanUpdater
from stonesoup.functions import degree2meters


class ELINTInitiator(GaussianInitiator):
    """Initiator that maps measurement space to state space

    This initiator utilises the :class:`~.MeasurementModel` matrix to convert
    :class:`~.Detection` state vector and model covariance into state space.
    This then replaces mapped values in the :attr:`prior_state` to form the
    initial :class:`~.GaussianState` of the :class:`~.Track`.
    """

    prior = Property(dict)
    measurement_model = Property(MeasurementModel, doc="Measurement model")

    def initiate(self, detections, **kwargs):
        updater = KalmanUpdater(self.measurement_model)

        tracks = set()

        for detection in detections:
            prior_state = self._get_init_state_prior(detection.state_vector)
            # Predict measurement
            measurement_prediction = updater.predict_measurement(
                prior_state, detection.measurement_model)
            # Perform Kalman update
            track_state = updater.update(SingleHypothesis(
                prior_state, detection, measurement_prediction))
            metadata = {
                "existence":{
                    "value":1,
                    "time":detection.timestamp
                }
            }
            # Create track
            track = Track([track_state])
            track.metadata = metadata
            tracks.add(track)

        return tracks

    def _get_init_state_prior(self, pos):
        """
        Get the prior on the state using the position to convert the velocity
        from m/s to deg/s

        Parameters
        ----------
        pos
        prior

        Returns
        -------

        """

        initvel_metres = self.prior["initspeed_sd_metres"] ** 2 * np.eye(2)
        H = np.diag(1/degree2meters(pos).ravel())
        priorcov_lonlatvel = H @ initvel_metres @ H
        priormeanLonLat = np.array([[Longitude(self.prior["lonlat_mean"][0, 0])],
                                    [0],
                                    [Latitude(self.prior["lonlat_mean"][1, 0])],
                                    [0]])
        priorcovLonLat = np.zeros((4,4))
        priorcovLonLat[0, 0] = np.array(self.prior["lonlat_cov"])[0, 0]
        priorcovLonLat[2, 2] = np.array(self.prior["lonlat_cov"])[1, 1]
        priorcovLonLat[1, 1] = priorcov_lonlatvel[0, 0]
        priorcovLonLat[3, 3] = priorcov_lonlatvel[1, 1]
        # priorcovLonLat[[0, 2], :][:, [0, 2]]= self.prior["lonlat_cov"]
        # priorcovLonLat[[1, 3], :][:, [1, 3]] = priorcov_lonlatvel

        priormean = StateVector(priormeanLonLat)
        priorcov = CovarianceMatrix(priorcovLonLat)

        return GaussianState(priormean, priorcov)


class AisElintVisibilityInitiator(GaussianInitiator):
    """Initiator that maps measurement space to state space

    This initiator utilises the :class:`~.MeasurementModel` matrix to convert
    :class:`~.Detection` state vector and model covariance into state space.
    This then replaces mapped values in the :attr:`prior_state` to form the
    initial :class:`~.GaussianState` of the :class:`~.Track`.
    """

    prior = Property(dict)
    measurement_model = Property(MeasurementModel, doc="Measurement model")
    sensors = Property(list)
    visibility = Property(dict)

    def initiate(self, detections, **kwargs):
        updater = KalmanUpdater(self.measurement_model)

        tracks = set()

        for detection in detections:
            prior_state = self._get_init_state_prior(detection.state_vector)
            # Predict measurement
            measurement_prediction = updater.predict_measurement(
                prior_state, detection.measurement_model)
            # Perform Kalman update
            track_state = updater.update(SingleHypothesis(
                prior_state, detection, measurement_prediction))
            vis_probs = self._get_prior_vis(detection)
            metadata = {
                'visibility': {
                    'probs': vis_probs,
                    'time': detection.timestamp
                },
                'existence': {
                    'value': self._get_existence_prob(vis_probs)
                }
            }
            # Create track
            track = Track([track_state])
            track.metadata = metadata
            tracks.add(track)

        return tracks

    def _get_init_state_prior(self, pos):

        initvel_metres = self.prior['initspeed_sd_metres'] ** 2 * np.eye(2)
        H = np.diag(1/degree2meters(pos).ravel())
        priorcov_lonlatvel = H @ initvel_metres @ H
        priormeanLonLat = np.array([Longitude(self.prior['lonlat_mean'][0, 0]), 0,
                                    Latitude(self.prior['lonlat_mean'][1, 0]), 0])
        priorcovLonLat = np.zeros((4,4))
        priorcovLonLat[0, 0] = np.array(self.prior['lonlat_cov'])[0, 0]
        priorcovLonLat[2, 2] = np.array(self.prior['lonlat_cov'])[1, 1]
        priorcovLonLat[1, 1] = priorcov_lonlatvel[0, 0]
        priorcovLonLat[3, 3] = priorcov_lonlatvel[1, 1]
        # priorcovLonLat[[0, 2], :][:, [0, 2]]= self.prior['lonlat_cov']
        # priorcovLonLat[[1, 3], :][:, [1, 3]] = priorcov_lonlatvel

        # Colour prior
        priormean = StateVector(np.concatenate((priormeanLonLat, self.prior['colour_mean'].ravel())))
        priorcov = CovarianceMatrix(block_diag(priorcovLonLat, np.diag(self.prior['colour_sd']**2)))

        return GaussianState(priormean, priorcov)

    def _get_prior_vis(self, detection):
        vis_states = self.visibility['visStates']
        if detection.metadata['sensor']['type'] == 'AIS':
            sensor_idx = 0
        else:
            sensor_idx = 1
        priorVisProbsPerSensor = np.array([sensor['priorVisProb'] for sensor in self.sensors])
        priorVisProbsPerSensor[sensor_idx] = 1
        nstates = vis_states.shape[1]
        priorJointVisProbs = np.zeros((nstates,))
        for i in range(nstates):
            isvis = vis_states[:, i].ravel()
            priorJointVisProbs[i] = np.prod(priorVisProbsPerSensor**isvis*(1-priorVisProbsPerSensor)**(np.logical_not(isvis)))
        priorJointVisProbs = priorJointVisProbs / np.sum(priorJointVisProbs)
        return priorJointVisProbs

    def _get_existence_prob(self, vis_probs):
        return np.sum(vis_probs[self.visibility['existStates'] > 0])