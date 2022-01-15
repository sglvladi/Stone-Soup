import datetime

import numpy as np
from scipy.linalg import block_diag, inv, cholesky
from scipy.stats import multivariate_normal as mn
import pickle

from stonesoup.base import Property
from stonesoup.types.base import Type
from stonesoup.hypothesiser import Hypothesiser
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.predictor import Predictor
from stonesoup.predictor._utils import predict_lru_cache
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.detection import Detection, MissedDetection
from stonesoup.models.measurement.linear import LinearGaussianPredefinedH
from stonesoup.types.hypothesis import SingleHypothesis, SingleProbabilityHypothesis
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.numeric import Probability
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.updater import Updater
from stonesoup.updater.kalman import ExtendedKalmanUpdater


class TwoStateGaussianState(GaussianState):
    """ A Gaussian state object representing the distribution :math:`p(x_{k+T}, x_{k} | Y)` """
    start_time: datetime.datetime = Property(doc='Timestamp at t_k')
    end_time: datetime.datetime = Property(doc='Timestamp at t_{k+T}')

    @property
    def timestamp(self):
        return self.end_time


class SensorTracks:
    def __init__(self, tracks, sensor_id):
        self.tracks = tracks
        self.sensor_id = sensor_id

    def __iter__(self):
        return (t for t in self.tracks)


class SensorTracklets(SensorTracks):
    pass


class Tracklet:
    def __init__(self, id=None, priors=None, posteriors=None):
        self.id = id
        self.priors = priors if priors else []
        self.posteriors = posteriors if posteriors else []


class Scan:
    pass


class SensorScan:
    pass


def predict_state_to_two_state(old_mean, old_cov, tx_model, dt):
    A = tx_model.matrix(time_interval=dt)
    Q = tx_model.covar(time_interval=dt)
    statedim = A.shape[0]
    AA = np.concatenate((np.eye(statedim), A))
    QQ = block_diag(np.zeros((statedim, statedim)), Q)
    return AA @ old_mean, AA @ old_cov @ AA.T + QQ


class TrackletExtractor:

    def __init__(self, transition_model):
        self._tracklets = []
        self._fuse_times = []
        self.transition_model = transition_model

    def get_tracklets_seq(self, alltracks, timestamp):
        # Append current fuse time to fuse times
        self._fuse_times.append(timestamp)
        # Iterate over the local tracks of each sensor
        for sensor_tracks in alltracks:
            sensor_id = sensor_tracks.sensor_id
            # Get tracklets for sensor
            idx = next((i for i, t in enumerate(self._tracklets)
                        if t.sensor_id == sensor_id), None)
            sensor_tracklets = self._tracklets[idx] if idx is not None else []
            # Temporary tracklet list
            tracklets_tmp = []
            # For each local track
            for track in sensor_tracks:
                tracklet = next((t for t in sensor_tracklets if track.id == t.id), None)
                # If the tracklet doesn't already exist
                if tracklet is None and len(self._fuse_times) > 1:
                    # Create it
                    tracklet = self.get_tracklet(track, self.transition_model,
                                                 np.array(self._fuse_times))
                elif tracklet is not None:
                    # Else simply augment
                    self.augment_tracklet(tracklet, track, timestamp)
                # Append tracklet to temporary tracklets
                if tracklet:
                    tracklets_tmp.append(tracklet)
            # If a tracklet set for the sensor doesn't already exist
            if idx is None:
                # Add it
                self._tracklets.append(SensorTracklets(tracklets_tmp, sensor_id))
            else:
                # Else replace the existing one
                self._tracklets[idx] = SensorTracklets(tracklets_tmp, sensor_id)
        # Return the stored tracklets
        return self._tracklets

    def augment_tracklet(self, tracklet, track, timestamp):
        track_times = np.array([s.timestamp for s in track])

        filtered_means = np.concatenate([s.mean for s in track], 1)
        filtered_covs = np.stack([s.covar for s in track], 2)
        filtered_times = np.array([s.timestamp for s in track])

        start_time = tracklet.posteriors[-1].timestamp
        end_time = timestamp
        nupd = np.sum(np.logical_and(track_times > start_time, track_times <= end_time))
        if nupd > 0:
            # Indices of end-states that are just before the start and end times
            ind0 = np.flatnonzero(filtered_times <= start_time)[-1]
            ind1 = np.flatnonzero(filtered_times <= end_time)[-1]
            # The end states
            end_states = [track.states[ind0], track.states[ind1]]
            # All means, covs and times that fall inbetween
            means = filtered_means[:, ind0 + 1:ind1 + 1]
            covs = filtered_covs[:, :, ind0 + 1: ind1 + 1]
            times = filtered_times[ind0 + 1:ind1 + 1]
            # Compute interval distribution
            post_mean, post_cov, prior_mean, prior_cov = \
                self.get_interval_dist(means, covs, times, end_states,
                                       self.transition_model, start_time, end_time)
            prior = TwoStateGaussianState(prior_mean, prior_cov, start_time=start_time,
                                          end_time=end_time)
            posterior = TwoStateGaussianState(post_mean, post_cov, start_time=start_time,
                                              end_time=end_time)
            tracklet.priors.append(prior)
            tracklet.posteriors.append(posterior)

    @classmethod
    def get_tracklets(cls, alltracks, tx_model, fuse_times):
        tracklets = []
        for tracks in alltracks:
            tracklets_tmp = []
            for track in tracks:
                tracklet = cls.get_tracklet(track, tx_model, fuse_times)
                if tracklet:
                    tracklets_tmp.append(tracklet)
            tracklets.append(tracklets_tmp)
        return tracklets

    @classmethod
    def get_tracklet(cls, track, tx_model, fuse_times):
        track_times = np.array([s.timestamp for s in track])
        idx0 = np.flatnonzero(fuse_times >= track_times[0])
        idx1 = np.flatnonzero(fuse_times <= track_times[-1])

        if not len(idx0) or not len(idx1):
            return None
        else:
            idx0 = idx0[0]
            idx1 = idx1[-1]

        priors = []
        posteriors = []

        filtered_means = np.concatenate([s.mean for s in track], 1)
        filtered_covs = np.stack([s.covar for s in track], 2)
        filtered_times = np.array([s.timestamp for s in track])

        cnt = 0
        for i in range(idx0, idx1):
            start_time = fuse_times[i]
            end_time = fuse_times[i + 1]
            nupd = np.sum(np.logical_and(track_times > start_time, track_times <= end_time))
            if nupd > 0:
                cnt += 1
                # Indices of end-states that are just before the start and end times
                ind0 = np.flatnonzero(filtered_times <= start_time)[-1]
                ind1 = np.flatnonzero(filtered_times <= end_time)[-1]
                # The end states
                end_states = [track.states[ind0], track.states[ind1]]
                # All means, covs and times that fall inbetween
                means = filtered_means[:, ind0 + 1:ind1 + 1]
                covs = filtered_covs[:, :, ind0 + 1: ind1 + 1]
                times = filtered_times[ind0 + 1:ind1 + 1]
                # Compute interval distribution
                post_mean, post_cov, prior_mean, prior_cov = \
                    cls.get_interval_dist(means, covs, times, end_states,
                                          tx_model, start_time, end_time)

                prior = TwoStateGaussianState(prior_mean, prior_cov, start_time=start_time,
                                              end_time=end_time)
                posterior = TwoStateGaussianState(post_mean, post_cov, start_time=start_time,
                                                  end_time=end_time)
                priors.append(prior)
                posteriors.append(posterior)

        if not cnt:
            return None

        tracklet = Tracklet(id=track.id, priors=priors, posteriors=posteriors)

        return tracklet

    @classmethod
    def get_interval_dist(cls, filtered_means, filtered_covs, filtered_times, states, tx_model,
                          start_time, end_time):

        # Get filtered distributions at start and end of interval
        predictor = ExtendedKalmanPredictor(tx_model)
        pred0 = predictor.predict(states[0], start_time)
        pred1 = predictor.predict(states[1], end_time)

        # Predict prior mean
        prior_mean, prior_cov = predict_state_to_two_state(pred0.mean, pred0.covar, tx_model,
                                                           end_time - start_time)

        # Get posterior mean by running smoother
        mn = np.concatenate([pred0.mean, filtered_means, pred1.mean], 1)
        cv = np.stack([pred0.covar, *list(np.swapaxes(filtered_covs, 0, 2)), pred1.covar], 2)
        t = np.array([start_time, *filtered_times, end_time])
        post_mean, post_cov = cls.rts_smoother_endpoints(mn, cv, t, tx_model)

        return post_mean, post_cov, prior_mean, prior_cov

    @classmethod
    def rts_smoother_endpoints(cls, filtered_means, filtered_covs, times, tx_model):
        statedim, ntimesteps = filtered_means.shape

        joint_smoothed_mean = np.tile(filtered_means[:, -1], (1, 2)).T
        joint_smoothed_cov = np.tile(filtered_covs[:, :, -1], (2, 2))

        for k in reversed(range(ntimesteps - 1)):
            dt = times[k + 1] - times[k]
            A = tx_model.matrix(time_interval=dt)
            Q = tx_model.covar(time_interval=dt)
            # Filtered distribution
            m = filtered_means[:, k][:, np.newaxis]
            P = filtered_covs[:, :, k]
            # Get transition model x_{k+1} -> x_k
            # p(x_k | x_{k+1}, y_{1:T}) = Norm(x_k; Fx_{k+1} + b, Omega)
            F = P @ A.T @ inv(A @ P @ A.T + Q)
            b = m - F @ A @ m
            Omega = P - F @ (A @ P @ A.T + Q) @ F.T
            # Two-state transition model (x_{k+1}, x_T) -> (x_k, x_T)
            F2 = block_diag(F, np.eye(statedim))
            b2 = np.concatenate((b, np.zeros((statedim, 1))))
            Omega2 = block_diag(Omega, np.zeros((statedim, statedim)))
            # Predict back
            joint_smoothed_mean = F2 @ joint_smoothed_mean + b2
            joint_smoothed_cov = F2 @ joint_smoothed_cov @ F2.T + Omega2
        return joint_smoothed_mean, joint_smoothed_cov


class PseudoMeasExtractor:

    def __init__(self):
        self._last_scan = None

    def get_pseudomeas_seq(self, all_tracklets):
        measurements = []
        for i, tracklets in enumerate(all_tracklets):
            for j, tracklet in enumerate(tracklets):
                measdata = self.get_pseudomeas_single(tracklet, i, self._last_scan)
                measurements += measdata

        measurements.sort(key=lambda x: x.end_time)
        return measurements


    @classmethod
    def get_pseudomeas(cls, all_tracklets):

        measurements = []
        for i, tracklets in enumerate(all_tracklets):
            for j, tracklet in enumerate(tracklets):
                measdata = cls.get_pseudomeas_single(tracklet, i)
                measurements += measdata

        measurements.sort(key=lambda x: x.end_time)
        return measurements

    @classmethod
    def get_pseudomeas_single(cls, tracklet, sensor_id, last_scan=None):
        if last_scan is None:
            inds = (i for i in range(len(tracklet.posteriors)))
        else:
            inds = (i for i, p in enumerate(tracklet.posteriors) if p.timestamp > last_scan)

        measdata = []

        for k in inds:
            post_mean = tracklet.posteriors[k].mean
            post_cov = tracklet.posteriors[k].covar
            prior_mean = tracklet.priors[k].mean
            prior_cov = tracklet.priors[k].covar

            H, z, R, _ = cls.get_pseudomeasurement(post_mean, post_cov, prior_mean, prior_cov)

            if len(H):
                meas_model = LinearGaussianPredefinedH(h_matrix=H, noise_covar=R,
                                                       mapping=[i for i in range(H.shape[0])])
                detection = Detection(state_vector=StateVector(z), measurement_model=meas_model,
                                      timestamp=tracklet.posteriors[k].timestamp)
                detection.start_time = tracklet.posteriors[k].start_time
                detection.end_time = tracklet.posteriors[k].end_time
                detection.sensor_id = sensor_id
                measdata.append(detection)

        return measdata

    @classmethod
    def get_pseudomeasurement(cls, mu1, C1, mu2, C2):
        return cls.get_pm(mu1, C1, mu2, C2)

    @classmethod
    def get_pm(cls, mu1, C1, mu2, C2):
        eigthresh = 1e-6
        matthresh = 1e-6

        invC1 = inv(C1)
        invC2 = inv(C2)
        # Ensure inverses are symmetric
        invC1 = (invC1 + invC1.T) / 2
        invC2 = (invC2 + invC2.T) / 2
        invC = invC1 - invC2

        D, v = np.linalg.eig(invC)
        D = np.diag(D)
        Htilde = v.T
        evals = np.diag(D)

        idx = np.flatnonzero(np.abs(evals) > eigthresh)

        H = Htilde[idx, :]

        statedim = mu1.shape[0]
        if np.max(np.abs(C1.flatten() - C2.flatten())) < matthresh:
            print('Discarded - matrices too similar')
            H = np.zeros((0, statedim))
            z = np.zeros((0, 1))
            R = np.zeros((0, 0))
            return H, z, R, evals

        if np.all(np.abs(evals) <= eigthresh):
            print('Discarded - all eigenvalues zero')
            H = np.zeros((0, statedim))
            z = np.zeros((0, 1))
            R = np.zeros((0, 0))
            return H, z, R, evals

        R = inv(D[idx, :][:, idx])
        z = R @ (H @ invC1 @ mu1 - H @ invC2 @ mu2)

        # Discard measurement if R is not positive definite
        try:
            np.linalg.cholesky(R)
        except np.linalg.LinAlgError:
            # if not np.all(np.linalg.eigvals(R) > 0):
            print('Discarded - singular R')
            H = np.zeros((0, statedim))
            z = np.zeros((0, 1))
            R = np.zeros((0, 0))
            return H, z, R, evals

        return H, z, R, evals

    @classmethod
    def get_scans_from_measdata(cls, measdata):
        if not len(measdata):
            return []

        start = np.min([m.start_time for m in measdata])
        times = np.array([[(m.end_time - start).total_seconds(),
                           (m.start_time - start).total_seconds()] for m in measdata])
        true_times = np.array([[m.end_time, m.start_time] for m in measdata])
        end_start_times, idx = np.unique(times, return_index=True, axis=0)
        idx2 = []
        for previous, current in zip(idx, idx[1:]):
            idx2.append([i for i in range(previous, current)])
        else:
            idx2.append([i for i in range(idx[-1], len(measdata))])
        nscans = len(idx)

        scans = []
        for i in range(nscans):
            thesescans = [measdata[j] for j in idx2[i]]
            if not len(thesescans):
                continue
            scan = Scan()
            scan.start_time = true_times[idx[i], 1]  # end_start_times[i, 1]
            scan.end_time = true_times[idx[i], 0]
            sens_ids = [m.sensor_id for m in thesescans]
            sens_ids, sidx = np.unique(sens_ids, return_index=True)
            sidx2 = []
            for previous, current in zip(sidx, sidx[1:]):
                sidx2.append([i for i in range(previous, current)])
            else:
                sidx2.append([i for i in range(sidx[-1], len(thesescans))])
            nsensscans = len(sidx)

            scan.sensor_scans = []
            for s in range(nsensscans):
                sscan = SensorScan()
                sscan.sensor_id = sens_ids[s]
                sscan.detections = [thesescans[j] for j in sidx2[s]]
                scan.sensor_scans.append(sscan)
            scans.append(scan)
        return scans

    @classmethod
    def get_scans_from_tracklets(cls, tracklets):
        measdata = cls.get_pseudomeas(tracklets)
        return cls.get_scans_from_measdata(measdata)

    def get_scans_from_tracklets_seq(self, tracklets, timestamp):
        measdata = self.get_pseudomeas_seq(tracklets)
        self._last_scan = timestamp
        return self.get_scans_from_measdata(measdata)


class FuseTracker:

    def __init__(self, prior, predictor, updater, associator, death_rate, prob_detect,
                 delete_thresh):
        self.prior = prior
        self.predictor = predictor
        self.updater = updater
        self.associator = associator
        self.transition_model = predictor.transition_model
        self.death_rate = death_rate
        self.prob_detect = prob_detect
        self.delete_thresh = delete_thresh
        self._max_track_id = 0
        self._tracks = set()
        self._tracklet_extractor = TrackletExtractor(self.transition_model)
        self._pseudomeas_extractor = PseudoMeasExtractor()
        self._current_end_time = None

    def run_seq(self, sensor_tracks, timestamp):
        tracklets = self._tracklet_extractor.get_tracklets_seq(sensor_tracks, timestamp)
        # tracklets = pickle.load(open('tracklets.pickle', 'rb'))
        scans = self._pseudomeas_extractor.get_scans_from_tracklets_seq(tracklets, timestamp)

        scans.sort(key=lambda x: x.start_time)
        for scan in scans:
            new_start_time = scan.start_time
            new_end_time = scan.end_time
            if self._current_end_time and new_start_time < self._current_end_time:
                print('Skipping a scan')
                continue
            # Predict two-state tracks forward
            for track in self._tracks:
                self.predict_track(track, self.transition_model, self._current_end_time, new_start_time,
                                   new_end_time, self.death_rate)
            # for track in tracks:
            #     # Predict existence
            #     survive_prob = np.exp(-self.death_rate * (new_end_time - old_end_time))
            #     track.exist_prob = track.exist_prob * survive_prob
            self._current_start_time = new_start_time
            self._current_end_time = new_end_time

            for sensor_scan in scan.sensor_scans:
                detections = set(sensor_scan.detections)

                # TODO: Data association here
                associations = self.associator.associate(self._tracks, detections,
                                                         timestamp=self._current_end_time)

                # Update tracks
                for track in self._tracks:
                    self.update_track(track, associations[track])

                # Initiate new tracks on unassociated detections
                assoc_detections = set([hyp.measurement for hyp in associations.values() if hyp])
                unassoc_detections = set(detections) - assoc_detections
                self._tracks |= self.init_tracks(unassoc_detections, self._current_start_time,
                                           self._current_end_time)

            self._tracks -= self.delete_tracks(self._tracks)
        return self._tracks

    def run_batch(self, tracks, fuse_times):
        tracklets = TrackletExtractor.get_tracklets(tracks, self.transition_model, fuse_times)
        # tracklets = pickle.load(open('tracklets.pickle', 'rb'))
        scans = PseudoMeasExtractor.get_scans_from_tracklets(tracklets)

        all_tracks = set()
        tracks = set()
        current_end_time = None
        for scan in scans:
            new_start_time = scan.start_time
            new_end_time = scan.end_time
            # Predict two-state tracks forward
            for track in tracks:
                self.predict_track(track, self.transition_model, current_end_time, new_start_time,
                                   new_end_time, self.death_rate)
            # for track in tracks:
            #     # Predict existence
            #     survive_prob = np.exp(-self.death_rate * (new_end_time - old_end_time))
            #     track.exist_prob = track.exist_prob * survive_prob
            current_start_time = new_start_time
            current_end_time = new_end_time

            for sensor_scan in scan.sensor_scans:
                detections = set(sensor_scan.detections)

                # TODO: Data association here
                associations = self.associator.associate(tracks, detections,
                                                         timestamp=current_end_time)

                # Update tracks
                for track in tracks:
                    self.update_track(track, associations[track])

                # Initiate new tracks on unassociated detections
                assoc_detections = set([hyp.measurement for hyp in associations.values() if hyp])
                unassoc_detections = set(detections) - assoc_detections
                tracks |= self.init_tracks(unassoc_detections, current_start_time,
                                           current_end_time)

            tracks -= self.delete_tracks(tracks)
            all_tracks.update(tracks)
        return all_tracks

    @classmethod
    def predict_track(cls, track, tx_model, current_end_time, new_start_time, new_end_time,
                      death_rate=0):
        statedim = tx_model.ndim_state

        # Predict existence
        survive_prob = np.exp(-death_rate * (new_end_time - current_end_time).total_seconds())
        track.exist_prob = track.exist_prob * survive_prob

        # Predict forward
        # p(x_k, x_{k+\Delta} | y^{1:S}_{1:k})
        mu = track.mean[-statedim:]
        C = track.covar[-statedim:, -statedim:]
        if new_start_time > current_end_time:
            dt = new_start_time - current_end_time
            A = tx_model.matrix(time_interval=dt)
            Q = tx_model.covar(time_interval=dt)
            mu = A @ mu
            C = A @ C @ A.T + Q
        elif new_start_time < current_end_time:
            raise ValueError('newStartTime < currentEndTime - scan times messed up!')

        two_state_mu, two_state_cov = predict_state_to_two_state(mu, C, tx_model,
                                                                 new_end_time - new_start_time)

        track.append(GaussianStatePrediction(StateVector(two_state_mu),
                                             CovarianceMatrix(two_state_cov),
                                             timestamp=new_end_time))

    def update_track(self, track, hypothesis, time=None):
        if hypothesis:
            update = self.updater.update(hypothesis)
            track.states[-1] = update
            track.exist_prob = 1
        else:
            non_exist_weight = 1 - track.exist_prob
            non_det_weight = (1 - self.prob_detect) * track.exist_prob
            track.exist_prob = non_det_weight / (non_exist_weight + non_det_weight)

    def init_tracks(self, detections, start_time, end_time):
        init_mean = self.prior.mean
        init_cov = self.prior.covar
        init_mean, init_cov = predict_state_to_two_state(init_mean, init_cov,
                                                         self.transition_model,
                                                         end_time - start_time)

        prior = GaussianState(init_mean, init_cov, timestamp=end_time)
        new_tracks = set()
        for detection in detections:
            hyp = SingleHypothesis(prediction=prior, measurement=detection)
            state = self.updater.update(hyp)
            track = Track([state], id=self._max_track_id)
            track.exist_prob = 1
            self._max_track_id += 1
            new_tracks.add(track)

        return new_tracks

    def delete_tracks(self, tracks):
        return set([track for track in tracks if track.exist_prob < self.delete_thresh])


class TwoStatePredictor(Predictor):

    @predict_lru_cache()
    def predict(self, prior, timestamp=None, current_end_time=None,
                new_start_time=None, new_end_time=None, **kwargs):
        statedim = self.transition_model.ndim_state
        mu = prior.mean[-statedim:]
        C = prior.covar[-statedim:, -statedim:]
        if new_start_time > current_end_time:
            dt = new_start_time - current_end_time
            A = self.transition_model.matrix(time_interval=dt)
            Q = self.transition_model.covar(time_interval=dt)
            mu = A @ mu
            C = A @ C @ A.T + Q
        else:
            raise ValueError('newStartTime < currentEndTime - scan times messed up!')

        two_state_mu, two_state_cov = predict_state_to_two_state(mu, C, self.transition_model,
                                                                 new_end_time - new_start_time)
        return GaussianStatePrediction(StateVector(two_state_mu),
                                       CovarianceMatrix(two_state_cov),
                                       timestamp=new_end_time)


class CustomPDAHypothesiser(PDAHypothesiser):
    """Hypothesiser based on Probabilistic Data Association (PDA)

    Generate track predictions at detection times and calculate probabilities
    for all prediction-detection pairs for single prediction and multiple
    detections.
    """
    prob_new_targets: Probability = Property(doc='New target probability',
                                             default=Probability(0.01))

    # mean_new_target_lik: Probability = Property(doc='Mean new target likelihood',
    #                                             default=Probability(-70, log_value=True))

    def hypothesise(self, track, detections, timestamp, **kwargs):

        hypotheses = list()

        # Common state & measurement prediction
        prediction = track.states[-1]
        # Missed detection hypothesis
        probability = self.prob_new_targets * Probability(1 - self.prob_detect * self.prob_gate)
        hypotheses.append(
            SingleProbabilityHypothesis(
                prediction,
                MissedDetection(timestamp=timestamp),
                probability
            ))

        # True detection hypotheses
        pdfs = [probability]
        for detection in detections:
            # Compute measurement prediction and probability measure
            measurement_prediction = self.updater.predict_measurement(
                prediction, detection.measurement_model, **kwargs)
            # Calculate difference before to handle custom types (mean defaults to zero)
            # This is required as log pdf coverts arrays to floats
            try:
                log_pdf = mn.logpdf(
                    (detection.state_vector - measurement_prediction.state_vector).ravel(),
                    cov=measurement_prediction.covar)
            except:
                print('Had to allow singular when evaluating likelihood!!!')
                log_pdf = mn.logpdf(
                    (detection.state_vector - measurement_prediction.state_vector).ravel(),
                    cov=measurement_prediction.covar, allow_singular=True)
            pdf = Probability(log_pdf, log_value=True)
            probability = (pdf * self.prob_detect) / self.clutter_spatial_density
            pdfs.append(pdf)
            # True detection hypothesis
            hypotheses.append(
                SingleProbabilityHypothesis(
                    prediction,
                    detection,
                    probability,
                    measurement_prediction))

        return MultipleHypothesis(hypotheses, normalise=True, total_weight=1)
