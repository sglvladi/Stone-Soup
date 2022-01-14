import numpy as np
from scipy.linalg import block_diag, inv, cholesky
from scipy.stats import multivariate_normal as mn
import pickle

from stonesoup.base import Property
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


class Tracklet:
    pass


class PseudoMeas:
    pass


class Scan:
    pass


class SensorScan:
    pass

def nan_array(shape):
    a = np.empty(shape)
    a[:] = np.nan
    return a

def predict_state_to_two_state(old_mean, old_cov, tx_model, dt):
    A = tx_model.matrix(time_interval=dt)
    Q = tx_model.covar(time_interval=dt)
    statedim = A.shape[0]
    AA = np.concatenate((np.eye(statedim), A))
    QQ = block_diag(np.zeros((statedim, statedim)), Q)
    return AA@old_mean, AA@old_cov@AA.T + QQ


class TrackletExtractor:

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

        tracklet = Tracklet()
        tracklet.id = track.id

        if not len(idx0) or not len(idx1):
            return None
        else:
            idx0 = idx0[0]
            idx1 = idx1[-1]

        tracklet.twoStatePostMeans = []
        tracklet.twoStatePostCovs = []
        tracklet.twoStatePriorMeans = []
        tracklet.twoStatePriorCovs = []
        tracklet.startTimes = []
        tracklet.endTimes = []
        tracklet.targetIds = []
        tracklet.numUpdates = []

        filtered_times = np.array([s.timestamp for s in track])
        filtered_means = np.concatenate([s.mean for s in track], 1)
        filtered_covs = np.stack([s.covar for s in track], 2)

        cnt = 0
        for i in range(idx0, idx1):
            start_time = fuse_times[i]
            end_time = fuse_times[i+1]
            nupd = np.sum(np.logical_and(track_times > start_time, track_times <= end_time))
            if nupd > 0:
                cnt += 1
                post_mean, post_cov, prior_mean, prior_cov = \
                    cls.get_interval_dist(filtered_means, filtered_covs, filtered_times, track,
                                           tx_model, start_time, end_time)
                tracklet.twoStatePostMeans.append(post_mean)
                tracklet.twoStatePostCovs.append(post_cov)
                tracklet.twoStatePriorMeans.append(prior_mean)
                tracklet.twoStatePriorCovs.append(prior_cov)
                tracklet.startTimes.append(start_time)
                tracklet.endTimes.append(end_time)
                tracklet.numUpdates.append(nupd)
                tracklet.targetIds.append(track.id)

        if not cnt:
            return None

        tracklet.twoStatePostMeans = np.concatenate(tracklet.twoStatePostMeans, 1)
        tracklet.twoStatePostCovs = np.stack(tracklet.twoStatePostCovs, 2)
        tracklet.twoStatePriorMeans = np.concatenate(tracklet.twoStatePriorMeans, 1)
        tracklet.twoStatePriorCovs = np.stack(tracklet.twoStatePriorCovs, 2)
        tracklet.startTimes = np.array(tracklet.startTimes)
        tracklet.endTimes = np.array(tracklet.endTimes)
        tracklet.targetIds = np.array(tracklet.targetIds)
        tracklet.numUpdates = np.array(tracklet.numUpdates)

        return tracklet

    @classmethod
    def get_interval_dist(cls, filtered_means, filtered_covs, filtered_times, track, tx_model,
                          start_time, end_time):

        idx0 = np.flatnonzero(filtered_times <= start_time)[-1]
        idx1 = np.flatnonzero(filtered_times <= end_time)[-1]

        kf = ExtendedKalmanPredictor(tx_model)

        # Get filtered distributions at start and end of interval
        pred0 = kf.predict(track.states[idx0], start_time)
        pred1 = kf.predict(track.states[idx1], end_time)

        # Predict prior mean
        prior_mean, prior_cov = predict_state_to_two_state(pred0.mean, pred0.covar, tx_model,
                                                           end_time-start_time)

        # Get posterior mean by running smoother
        mn = np.concatenate([pred0.mean, filtered_means[:, idx0+1:idx1+1], pred1.mean], 1)
        cv_tmp = filtered_covs[:, :, idx0+1: idx1+1]
        cv = np.stack((pred0.covar, *[cv_tmp[:, :, i] for i in range(cv_tmp.shape[2])], pred1.covar), 2)
        t = np.array([start_time, *filtered_times[idx0+1:idx1+1], end_time])
        post_mean, post_cov = cls.rts_smoother_endpoints(mn, cv, t, tx_model)

        return post_mean, post_cov, prior_mean, prior_cov

    @classmethod
    def rts_smoother_endpoints(cls, filtered_means, filtered_covs, times, tx_model):
        statedim, ntimesteps = filtered_means.shape

        joint_smoothed_mean = np.tile(filtered_means[:, -1], (1, 2)).T
        joint_smoothed_cov = np.tile(filtered_covs[:, :, -1], (2, 2))

        rng = [i for i in range(ntimesteps-1)]
        for k in reversed(rng):
            dt = times[k + 1] - times[k]
            A = tx_model.matrix(time_interval=dt)
            Q = tx_model.covar(time_interval=dt)
            # Filtered distribution
            m = filtered_means[:, k][:, np.newaxis]
            P = filtered_covs[:, :, k]
            # Get transition model x_{k+1} -> x_k
            # p(x_k | x_{k+1}, y_{1:T}) = Norm(x_k; Fx_{k+1} + b, Omega)
            F = P@A.T@inv(A@P@A.T+Q)
            b = m - F@A@m
            Omega = P-F@(A@P@A.T+Q)@F.T
            # Two-state transition model (x_{k+1}, x_T) -> (x_k, x_T)
            F2 = block_diag(F, np.eye(statedim))
            b2 = np.concatenate((b, np.zeros((statedim, 1))))
            Omega2 = block_diag(Omega, np.zeros((statedim, statedim)))
            # Predict back
            joint_smoothed_mean = F2@joint_smoothed_mean + b2
            joint_smoothed_cov = F2@joint_smoothed_cov@F2.T + Omega2
        return joint_smoothed_mean, joint_smoothed_cov


class PseudoMeasExtractor:

    @classmethod
    def get_pseudomeas(cls, all_tracklets):

        measurements = []
        for i, tracklets in enumerate(all_tracklets):
            for j, tracklet in enumerate(tracklets):
                measdata = cls.get_pseudomeas_single(tracklet, i, j)
                measurements += measdata

        measurements.sort(key=lambda x: x.endTime)
        return measurements

    @classmethod
    def get_pseudomeas_single(cls, tracklet, sensor_id, tracklet_id):
        twostatedim, nmeas = tracklet.twoStatePostMeans.shape

        measdata = []

        for k in range(nmeas):
            post_mean = tracklet.twoStatePostMeans[:, k][:, np.newaxis]
            post_cov = tracklet.twoStatePostCovs[:, :, k]
            prior_mean = tracklet.twoStatePriorMeans[:, k][:, np.newaxis]
            prior_cov = tracklet.twoStatePriorCovs[:, :, k]

            H, z, R, evals = cls.get_pseudomeasurement(post_mean, post_cov, prior_mean, prior_cov)

            if len(H):
                meas_model = LinearGaussianPredefinedH(h_matrix=H, noise_covar=R,
                                                       mapping=[i for i in range(H.shape[0])])
                detection = Detection(state_vector=StateVector(z), measurement_model=meas_model,
                                      timestamp=tracklet.endTimes[k])
                detection.startTime = tracklet.startTimes[k]
                detection.endTime = tracklet.endTimes[k]
                detection.sensorId = sensor_id
                detection.trackletId = tracklet_id
                detection.trackletTimestamp = k
                detection.evals = evals
                detection.targetId = tracklet.targetIds[k]
                detection.numUpdates = tracklet.numUpdates[k]
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
        if np.max(np.abs(C1.flatten()-C2.flatten())) < matthresh:
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
        z = R@(H@invC1@mu1 - H@invC2@mu2)

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
            nscans = 0
            return []
        else:
            start = np.min([m.startTime for m in measdata])
            times = np.array([[(m.endTime-start).total_seconds(), (m.startTime-start).total_seconds()] for m in measdata])
            true_times = np.array([[m.endTime, m.startTime] for m in measdata])
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
            scan = Scan()
            scan.startTime = true_times[idx[i], 1]  # end_start_times[i, 1]
            scan.endTime = true_times[idx[i], 0]
            sens_ids = [m.sensorId for m in thesescans]
            sens_ids, sidx = np.unique(sens_ids, return_index=True)
            sidx2 = []
            for previous, current in zip(sidx, sidx[1:]):
                sidx2.append([i for i in range(previous, current)])
            else:
                sidx2.append([i for i in range(sidx[-1], len(thesescans))])
            nsensscans = len(sidx)

            scan.sensorScans = []
            for s in range(nsensscans):
                sscan = SensorScan()
                sscan.sensorId = sens_ids[s]
                sscan.meas = [thesescans[j] for j in sidx2[s]]
                scan.sensorScans.append(sscan)
            scans.append(scan)
        return scans

    @classmethod
    def get_scans_from_tracklets(cls, tracklets):
        measdata = cls.get_pseudomeas(tracklets)
        return cls.get_scans_from_measdata(measdata)


class FuseTracker:

    def __init__(self, prior, predictor, updater, associator, death_rate, prob_detect, delete_thresh):
        self.prior = prior
        self.predictor = predictor
        self.updater = updater
        self.associator = associator
        self.transition_model = predictor.transition_model
        self.death_rate = death_rate
        self.prob_detect = prob_detect
        self.delete_thresh = delete_thresh
        self._max_track_id = 0

    def run_batch(self, tracks, fuse_times):
        tracklets = TrackletExtractor.get_tracklets(tracks, self.transition_model, fuse_times)
        # tracklets = pickle.load(open('tracklets.pickle', 'rb'))
        scans = PseudoMeasExtractor.get_scans_from_tracklets(tracklets)

        all_tracks = set()
        tracks = set()
        current_start_time = None
        current_end_time = None
        for scan in scans:
            new_start_time = scan.startTime
            new_end_time = scan.endTime
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

            for sensor_scan in scan.sensorScans:
                detections = set(sensor_scan.meas)

                # TODO: Data association here
                associations = self.associator.associate(tracks, detections,
                                                         timestamp=current_end_time)

                # Update tracks
                for track in tracks:
                    self.update_track(track, associations[track])

                # Initiate new tracks on unassociated detections
                assoc_detections = set([hyp.measurement for hyp in associations.values() if hyp])
                unassoc_detections = set(detections)-assoc_detections
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
            mu = A@mu
            C = A@C@A.T + Q
        elif new_start_time < current_end_time:
            raise ValueError('newStartTime < currentEndTime - scan times messed up!')

        two_state_mu, two_state_cov = predict_state_to_two_state(mu, C, tx_model,
                                                                 new_end_time-new_start_time)

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
                                                         end_time-start_time)

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
        probability = self.prob_new_targets * Probability(1 - self.prob_detect*self.prob_gate)
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
