################################################################################
# IMPORTS                                                                      #
################################################################################
# General imports
import sys
import numpy as np
from datetime import timedelta
from scipy.io import loadmat
from scipy.linalg import expm
from matplotlib import pyplot as plt
from matplotlib import colormaps

# Stone-Soup imports
from stonesoup.deleter.elint import ELINTDeleter
from stonesoup.hypothesiser.probability import ELINTHypothesiser, ELINTHypothesiserFast, AisElintHypothesiserFast, PDAHypothesiserFast
from stonesoup.initiator.elint import ELINTInitiator, AisElintVisibilityInitiator, ElintVisibilityInitiator
from stonesoup.models.transition.linear import RandomWalk, CombinedLinearGaussianTransitionModel
# from stonesoup.initiator.simple import LinearMeasurementInitiator
from stonesoup.deleter.time import UpdateTimeDeleter
from stonesoup.dataassociator.neighbour import (
    NearestNeighbour, GlobalNearestNeighbour, GNNWith2DAssignment)
from stonesoup.hypothesiser.distance import DistanceHypothesiser, DistanceHypothesiserFast
from stonesoup.measures import Mahalanobis
from stonesoup.hypothesiser.filtered import FilteredDetectionsHypothesiser
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.update import GaussianStateUpdate
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.reader.elint import ElintDetectionReader

if __name__ == '__main__':
    SEED = 42  # Random seed for reproducibility
    plt.rcParams['figure.figsize'] = (12, 8)

    def print_progress(i: int, total: int, *, prefix: str = "", suffix: str = "", width: int = 30):
        """
        Simple terminal progress bar.
        i: current iteration index (0-based)
        total: total number of iterations
        """
        total = max(int(total), 1)
        i = max(0, min(int(i), total - 1))
        done = i + 1

        frac = done / total
        filled = int(width * frac)
        bar = "#" * filled + "-" * (width - filled)

        msg = f"{prefix}[{bar}] {done:>5}/{total:<5} ({frac:>6.1%}) {suffix}"
        # \r returns to start of line; pad with spaces to overwrite leftovers
        sys.stdout.write("\r" + msg + " " * 10)
        sys.stdout.flush()

        if done == total:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def get_prior(file_path, pdw_features, sd_val, scaling_factors=None):
        wp = loadmat(file_path)
        elintdata = np.column_stack([wp[feature].ravel() for feature in pdw_features])
        if scaling_factors is not None:
            elintdata = elintdata / scaling_factors
        prior = {'colour_min': np.min(elintdata, axis=0),
            'colour_max': np.max(elintdata, axis=0),
            'colour_mean': np.mean(elintdata, axis=0),
            'colour_sd': np.array(sd_val)}
        return prior

    def update_vis_probs(tracks, visibility, sensor_idx, dt):
        for track in tracks:
            loglik = visibility['loglik'] * dt

            # if track was detected
            if isinstance(track.state, GaussianStateUpdate) and track.state.timestamp == scan_time:
                # Invisible states to the sensor which detected the target are impossible
                loglik[np.logical_not(visibility['visStates'][sensor_idx, :])] = -np.inf

            vis_probs = track.metadata['visibility']['probs']
            vis_probs = vis_probs * np.exp(loglik)
            vis_probs = vis_probs / np.sum(vis_probs)
            exist_prob = np.sum(vis_probs[visibility['existStates'] > 0])
            track.metadata['visibility']['probs'] = vis_probs
            track.metadata['existence'] = {
                'value': exist_prob
            }
        return tracks

    def _get_vis_transitions(dt, vis_states, sensors):
        # Get probability matrix of transitioning between hidden and visible for
        # each sensor
        reveal_rates = [sensor['rates']['reveal'] for sensor in sensors]
        hide_rates = [sensor['rates']['hide'] for sensor in sensors]
        trans_matrices = _get_vis_trans_matrices(hide_rates, reveal_rates, dt)
        num_trans = vis_states.shape[1]
        num_sensors = len(sensors)

        # Get p_trans(i,j) = probability of transitioning from visibility state
        # vis_states(:,i) to vis_states(:,j)
        p_trans = np.ones((num_trans, num_trans))
        for i in range(num_trans):
            for j in range(num_trans):
                for s in range(num_sensors):
                    p = trans_matrices[vis_states[s, i], vis_states[s, j], s]
                    p_trans[i, j] = p_trans[i, j] * p
        for i in range(1, num_trans):
            p_trans[i,:] = p_trans[i,:]/np.sum(p_trans[i,:])
        p_trans[0, :] = np.concatenate(([1], np.zeros((num_trans-1,)))) # Target can't resurrect!
        return p_trans

    def _get_vis_trans_matrices(hide_rates, reveal_rates, dt):
        # Get transition probability matrix from hide and reveal rates for each
        # sensor
        num_sensors = len(hide_rates)
        trans_matrices = np.zeros((2, 2, num_sensors))
        for i in range(num_sensors):
            # Get transition matrix from exponential rates
            # https://cs.nyu.edu/mishra/COURSES/09.HPGP/scribe3
            a = np.array([[-reveal_rates[i], reveal_rates[i]],
                          [hide_rates[i], -hide_rates[i]]])
            trans_matrices[:, :, i] = expm(dt * a)
        return trans_matrices

    ##########################################################################
    # Tracking components                                                    #
    ##########################################################################
    file_name = 'IFM4_PDW_t_pulseFreq_pulseLen_bandWidth.mat'
    file_path = r'C:\Users\marfon\OneDrive - The University of Liverpool\Code\ESA_BIC\data\{}'.format(
        file_name)
    pdw_features = ['pulseFreq','pulseLen','bandWidth'] # time is always considered to generate timestamps
    scaling_factors = [1e9,1e-3,1e6]
    units_labels = ['GHz', 'ms', 'Mhz']
    pdw_features_process_noise = [1, 0.001, 200] #  this should be manually aligned with 'pdw_features'
    pdw_features_meas_noise = [1, 0.001, 200]
    sd_val_prior = [5, 5, 100]

    # select features
    sel_feat = [1,2]
    pdw_features = [pdw_features[i] for i in sel_feat]
    scaling_factors = [scaling_factors[i] for i in sel_feat]
    units_labels = [units_labels[i] for i in sel_feat]
    pdw_features_process_noise = [pdw_features_process_noise[i] for i in sel_feat]
    pdw_features_meas_noise = [pdw_features_meas_noise[i] for i in sel_feat]
    sd_val_prior = [sd_val_prior[i] for i in sel_feat]
    print('PDW features: ', pdw_features)

    # Event driven parameters
    # =======================
    mins2sec = 60
    hours2sec = 60 * mins2sec
    days2sec = 24 * hours2sec
    rates = {
        'birth': 1 / (10 * hours2sec),
        'killProbThresh': 0.1
    }
    sensors = [
        {
            'type': 'ELINT',
            'rates': {
                'meas': 1 / (90*mins2sec),
                'reveal': 1 / days2sec,
                'hide': 1 / (2 * days2sec)
            },
            'priorVisProb': 0.5,
            'colour_error_sd': np.array([0.1])
        }
    ]
    sensors[0]['rates']['firstmeas'] = (rates['birth'] * sensors[0]['rates']['meas'])

    # Compute prior and visibility constants
    measrates = np.array([[sensor["rates"]["meas"]] for sensor in sensors])
    visibility = {
        'visStates': np.array([[1]]),
        'existStates': np.array([1])
    }
    visibility['loglik'] = -np.sum(measrates * visibility['visStates'], 0)
    logNullLogLikelihoods = [-8.669524141543720 + np.log(sensor['rates']['firstmeas']) for sensor in sensors]

    # Transition, Measurement models & Detector
    # =========================================
    trans_model = [RandomWalk(x) for x in pdw_features_process_noise]
    transition_model = CombinedLinearGaussianTransitionModel(trans_model)
    measurement_model = LinearGaussian(ndim_state=len(pdw_features_process_noise), mapping=list(range(0, len(pdw_features))),
                                       noise_covar=np.diag(pdw_features_meas_noise)) #0.001

    detector = ElintDetectionReader(path=file_path, sensors=sensors, meas_model=measurement_model,
                                    timestamp=True, start_offset=timedelta(seconds=36.5),#36.5
                                    length=timedelta(seconds=3), pdw_features=pdw_features,
                                    scaling_factors=scaling_factors)

    # Predictor & Updater
    # ===================
    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)

    # Hypothesiser & Data Associator
    # ==============================
    EVENT_DRIVEN_ON = True
    print('EVENT_DRIVEN_ON = ' + str(EVENT_DRIVEN_ON))
    if EVENT_DRIVEN_ON:
        hypothesiser = AisElintHypothesiserFast(predictor, updater,
                                                logNullLogLikelihoods,
                                                sensors, visibility,
                                                Mahalanobis(),
                                                missed_distance=1
                                                )
        hypothesiser = FilteredDetectionsHypothesiser(hypothesiser, 'MMSI')
    else:
        hypothesiser = PDAHypothesiserFast(predictor=predictor,
                                       updater=updater,
                                       clutter_spatial_density=0.125,
                                       prob_detect=0.9)
    # hypothesiser = DistanceHypothesiser(predictor, updater, Mahalanobis(), 100)
    # class TPRGNN(GNNWith2DAssignment, TPRTreeMixIn):
    #     pass
    #associator = TPRGNN(hypothesiser, measurement_model, timedelta(hours=24), [1, 3], std_thresh=600)
    associator = GNNWith2DAssignment(hypothesiser)

    # Track Initiator
    # ===============
    prior = get_prior(file_path, pdw_features, sd_val_prior, scaling_factors)
    initiator = ElintVisibilityInitiator(prior, measurement_model, sensors, visibility, EVENT_DRIVEN_ON)

    # Track Deleter
    # =============
    if EVENT_DRIVEN_ON:
        deleter = ELINTDeleter(rates['killProbThresh'])
    else:
        deleter = UpdateTimeDeleter(time_since_update=timedelta(hours=2))

    ################################################################################
    # Main Tracking process                                                        #
    ################################################################################

    # Set random seed for reproducibility
    np.random.seed(SEED)
    num_colors = 9  # Adjust based on expected number of tracks
    colormap = colormaps['Set1']  # or 'tab10', 'Set3', 'hsv', etc.

    tracks = set()  # Main set of tracks
    id_dict = {}  # Dictionary to map track IDs to unique integer values

    alldetections = np.array(list(detector.detections_gen()))
    # Pre-extract the detection sets once to avoid redundant indexing
    detection_sets = alldetections[:, 1]
    # Use list comprehension without pop() if data can be reused, or vectorize if possible
    alltimesteps = np.array([next(iter(x)).timestamp for x in detection_sets])
    allmeas = np.array([next(iter(x)).state_vector for x in detection_sets])

    prev_time = None
    k = 0
    sensor_idx = 0
    num_dims = 1
    mapping = [0]
    fig, axes = plt.subplots(len(pdw_features), 1, figsize=(12, 4 * len(pdw_features)))
    if EVENT_DRIVEN_ON:
        fig.suptitle('Event-driven ON')
    else:
        fig.suptitle('Event-driven OFF')
    if len(pdw_features) == 1:
        axes = [axes]
    for scan_time, detections in detector.detections_gen():
        if prev_time is None:
            prev_time = scan_time
        print_progress(
            k, len(alltimesteps),
            prefix="",
            suffix=f"ts={scan_time} | tracks={len(tracks)} | elapsed={scan_time - alltimesteps[0]}s",
            width=40
        )

        # Perform data association
        dt = scan_time - prev_time
        if EVENT_DRIVEN_ON:
            trans_matrix = _get_vis_transitions(dt.total_seconds(), visibility['visStates'], sensors)
            associations = associator.associate(tracks, detections, scan_time, trans_matrix=trans_matrix)
        else:
            associations = associator.associate(tracks, detections, scan_time)

        # Update tracks based on association hypotheses
        associated_detections = set()
        for track, hypothesis in associations.items():
            if hypothesis:
                state_post = updater.update(hypothesis)
                track.append(state_post)
                associated_detections.add(hypothesis.measurement)
            else:
                track.append(hypothesis.prediction)
        # Update visibility probs
        if EVENT_DRIVEN_ON:
            tracks = update_vis_probs(tracks, visibility, sensor_idx, dt.total_seconds())
        prev_time = scan_time

        # Initiate new tracks
        unassociated_detections = detections - associated_detections
        new_tracks = initiator.initiate(unassociated_detections)
        if len(new_tracks):
            print('Initiating tracks!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        tracks |= new_tracks
        # # Delete invalid tracks
        del_tracks = deleter.delete_tracks(tracks, timestamp=scan_time)
        if len(del_tracks):
            print('Deleting tracks!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        tracks -= del_tracks

        # Update id_dict with track IDs
        for track in tracks:
            if track.id not in id_dict:
                id_dict[track.id] = len(id_dict) + 1

        # Generate plots (assuming one track)
        # Plot true trajectory
        for track in tracks:
            if isinstance(track.state_vector, GaussianStatePrediction):
                continue
            if k == 0:
                # Scatter all measurements for each feature
                for feat_idx in range(len(pdw_features)):
                    axes[feat_idx].scatter(alltimesteps, allmeas[:, feat_idx],
                                           c='b', marker='o', s=20, label='Measurements')

            # Plot estimated trajectory for each feature
            track_state = np.array(track.state_vector)
            for feat_idx in range(len(pdw_features)):
                color = colormap((id_dict[track.id] - 1) % num_colors / num_colors)
                axes[feat_idx].scatter(scan_time, track_state[feat_idx],
                                       c=[color], marker='x', s=10,
                                       label='Track ' + str(id_dict[track.id]-1) if len(track) == 1 else '')
                #axes[feat_idx].set_yscale('log')
                axes[feat_idx].grid(True)
                axes[feat_idx].set_xlabel('Time')
                axes[feat_idx].set_ylabel(pdw_features[feat_idx] + ' [' + units_labels[feat_idx] + ']')

                if len(track) == 1:
                    axes[feat_idx].legend()

        plt.pause(0.0001)
        k = k + 1

    plt.show()
