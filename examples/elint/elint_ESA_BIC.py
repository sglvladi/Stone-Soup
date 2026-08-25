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
from stonesoup.hypothesiser.probability import ELINTHypothesiserFast
from stonesoup.initiator.elint import ELINTInitiator
from stonesoup.models.transition.linear import RandomWalk, CombinedLinearGaussianTransitionModel
from stonesoup.deleter.time import UpdateTimeDeleter
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
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

    ##########################################################################
    # Scenario selection and data path                                       #
    ##########################################################################
    scenario_number = 1  # 1, 2 or 3
    vm_val = [1, 0.25, 0.1]
    file_name = f'IFM4_PDW_t_freq_PW_BW_tm10_vm{vm_val[scenario_number-1]}_stos2.5_noise0_drop0.5.mat'
    file_path = r'Stone-Soup/examples/elint/{}'.format(file_name)

    ##########################################################################
    # Tracking components                                                    #
    ##########################################################################

    # Standard filter parameters
    # =======================
    PLOT_ONLY_UPD = True
    EVENT_DRIVEN_ON = True
    print('EVENT_DRIVEN_ON = ' + str(EVENT_DRIVEN_ON))

    pdw_features = ['pulseFreq','pulseLen','bandWidth'] # time is always considered to generate timestamps
    scaling_factors = [1e9,1e-3,1e6]
    units_labels = ['GHz', 'ms', 'Mhz']
    pdw_features_process_noise = [0.001, 0.00001, 0.01]
    pdw_features_meas_noise = [0.001, 0.00001, 0.01]
    sd_val_prior = [5, 5, 100]
    gating_th = 5
    # select features
    sel_feat = [0,1,2]
    pdw_features = [pdw_features[i] for i in sel_feat]
    scaling_factors = [scaling_factors[i] for i in sel_feat]
    units_labels = [units_labels[i] for i in sel_feat]
    pdw_features_process_noise = [pdw_features_process_noise[i] for i in sel_feat]
    pdw_features_meas_noise = [pdw_features_meas_noise[i] for i in sel_feat]
    sd_val_prior = [sd_val_prior[i] for i in sel_feat]

    # Event driven parameters
    # =======================
    mins2sec = 60
    hours2sec = 60 * mins2sec
    days2sec = 24 * hours2sec
    rates = {
        "birth": 1 / (10 * hours2sec),
        "death": 1 / (10 * days2sec),
        "killProbThresh": 0.1
    }
    sensors = {
        "ELINT": {
            "rates": {
            }
        }
    }

    sensors["ELINT"]["rates"]["meas"] = 1 / (60 * mins2sec) #(60 * mins2sec)
    sensors["ELINT"]["rates"]["firstmeas"] = (rates["birth"] * sensors["ELINT"]["rates"]["meas"] /
                                              (rates["death"] + sensors["ELINT"]["rates"]["meas"]))
    logNullLogLikelihood = 7 + np.log(sensors["ELINT"]['rates']['firstmeas']) # 7 good with vm1
    print('rates:', rates)
    print('sensors["ELINT"]:', sensors["ELINT"])
    print('logNullLogLikelihood:', np.exp(logNullLogLikelihood))

    # Transition, Measurement models & Detector
    # =========================================
    trans_model = [RandomWalk(x) for x in pdw_features_process_noise]
    transition_model = CombinedLinearGaussianTransitionModel(trans_model)
    measurement_model = LinearGaussian(ndim_state=len(pdw_features_process_noise), mapping=list(range(0, len(pdw_features))),
                                       noise_covar=np.diag(pdw_features_meas_noise)) #0.001

    detector = ElintDetectionReader(path=file_path, sensors=sensors, meas_model=measurement_model,
                                    timestamp=True, start_offset=timedelta(seconds=0),#36.5
                                    length=timedelta(seconds=5), pdw_features=pdw_features,
                                    scaling_factors=scaling_factors)

    # Predictor & Updater
    # ===================
    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)

    # Hypothesiser & Data Associator
    # ==============================
    if EVENT_DRIVEN_ON:
        hypothesiser = ELINTHypothesiserFast(predictor, updater,
                                     sensors["ELINT"]["rates"]["meas"],
                                     rates["death"], logNullLogLikelihood,
                                     Mahalanobis(), missed_distance=gating_th)
        # hypothesiser = DistanceHypothesiser(predictor, updater, Mahalanobis(), 100)
    else:
        # hypothesiser = PDAHypothesiserFast(predictor=predictor,
        #                                updater=updater,
        #                                clutter_spatial_density=0.125,
        #                                prob_detect=sensors["ELINT"]["rates"]["meas"])
        hypothesiser = DistanceHypothesiser(predictor, updater, Mahalanobis(), gating_th)
    # class TPRGNN(GNNWith2DAssignment, TPRTreeMixIn):
    #     pass
    # associator = TPRGNN(hypothesiser, measurement_model, timedelta(hours=24), [1, 3], std_thresh=600)
    associator = GNNWith2DAssignment(hypothesiser)

    # Track Initiator
    # ===============
    prior = get_prior(file_path, pdw_features, sd_val_prior, scaling_factors)
    initiator = ELINTInitiator(prior, measurement_model)

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
    if 'id' in next(iter(detection_sets[0])).metadata:
        allids = np.array([next(iter(x)).metadata['id'] for x in detection_sets])
    else:
        allids = None

    prev_time = None
    k = 0
    sensor_idx = 0
    fig, axes = plt.subplots(len(pdw_features), 1, figsize=(12, 4 * len(pdw_features)))
    if EVENT_DRIVEN_ON:
        fig.suptitle(file_name + ', event-driven ON')
    else:
        fig.suptitle(file_name + ', event-driven OFF')
    if len(pdw_features) == 1:
        axes = [axes]
    # RECURSION
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
            for track in tracks:
                if isinstance(track.state, GaussianStateUpdate) and track.state.timestamp == scan_time:
                    # since you've assumed that the target definitely generated the measurement
                    track.metadata["existence"]["value"] = 1
                else:
                    pe = track.metadata["existence"]["value"]
                    dt = scan_time - prev_time
                    logpnotdetect = -sensors["ELINT"]["rates"]["meas"] * dt.total_seconds()
                    pnotdetgivenexist = np.exp(logpnotdetect)
                    pnotdet = pnotdetgivenexist * pe + (1 - pe)
                    track.metadata["existence"]["value"] = pe * pnotdetgivenexist / pnotdet
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
                    if allids is None:
                        axes[feat_idx].scatter(alltimesteps, allmeas[:, feat_idx],
                                               c='b', marker='o', s=20, label='Measurements')
                    else:
                        indices_zero = np.where(allids == 0)[0]
                        axes[feat_idx].scatter(alltimesteps[indices_zero], allmeas[indices_zero, feat_idx],
                                               c='b', marker='o', s=20, label='Measurements emitter 0')
                        indices_zero = np.where(allids == 1)[0]
                        axes[feat_idx].scatter(alltimesteps[indices_zero], allmeas[indices_zero, feat_idx],
                                               c='orange', marker='o', s=20, label='Measurements emitter 1')

            # Plot estimated trajectory for each feature
            if isinstance(track.state, GaussianStateUpdate) or not PLOT_ONLY_UPD:
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
