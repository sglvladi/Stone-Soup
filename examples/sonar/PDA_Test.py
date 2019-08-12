import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from copy import copy
from datetime import datetime, timedelta

from stonesoup.functions import gm_reduce_single

from stonesoup.initiator.simple import LinearMeasurementInitiator
from stonesoup.deleter.time import UpdateTimeDeleter

from stonesoup.dataassociator.probability import PDAHypothesiser
from stonesoup.dataassociator.probability import PDA

from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater

from stonesoup.models.transition.linear import ConstantVelocity, CombinedLinearGaussianTransitionModel
from stonesoup.models.measurement.linear import LinearGaussian

from stonesoup.types.state import GaussianState
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.detection import Detection
from stonesoup.types.state import StateVector, CovarianceMatrix
from stonesoup.types.update import Update, GaussianStateUpdate
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.types.track import Track

from stonesoup.simulator.simple import SingleTargetGroundTruthSimulator,\
    MultiTargetGroundTruthSimulator, DetectionSimulator

from matplotlib.patches import Ellipse
def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta,
                    **kwargs)

    ax.add_artist(ellip)
    return ellip

def plot_paths(paths, ax=None):
    # Mercator projection map
    for path in paths:
        data = np.array([state.state_vector for state in path.states])
        if ax is not None:
            ax.plot(data[:, 0], data[:, 2], 'b-')
        else:
            plt.plot(data[:, 0], data[:, 2], 'b-')
        # plot_cov_ellipse(track.state.covar[[0, 2], :][:, [0, 2]],
        #                  track.state.mean[[0, 2], :], edgecolor='r',
        #                  facecolor='none', ax=ax)
        # v_x, v_y = (track.last_update.state_vector[1, 0],
        #             track.last_update.state_vector[3,0])
        # txt = "{} - V: {}".format(
        #     np.around(track.last_update.weights[0, 0], 2),
        #     np.around(np.sqrt(v_x**2 + v_y**2)*1.944, 2))
        # plt.text(x[-1], y[-1], txt, fontsize=6)

def plot_tracks(tracks, show_error=True, ax=None):
    # Mercator projection map
    for track in tracks:
        data = np.array([state.state_vector for state in track.states if
                         isinstance(state, Update)])
        if ax is not None:
            ax.plot(data[:, 0], data[:, 2], 'r-')
        else:
            plt.plot(data[:, 0], data[:, 2], 'r-')
        plot_cov_ellipse(track.state.covar[[0, 2], :][:, [0, 2]],
                         track.state.mean[[0, 2], :], edgecolor='r',
                         facecolor='none', ax=ax)
        # v_x, v_y = (track.last_update.state_vector[1, 0],
        #             track.last_update.state_vector[3,0])
        # txt = "{} - V: {}".format(
        #     np.around(track.last_update.weights[0, 0], 2),
        #     np.around(np.sqrt(v_x**2 + v_y**2)*1.944, 2))
        # plt.text(x[-1], y[-1], txt, fontsize=6)

def plot_weights(tracks, ax=None):
    for i, track in enumerate(tracks):
        x = i*2
        y = i*2 + 1
        ax.bar([x,y], track.state.weights.ravel())

# Transition & Measurement models
# ===============================
transition_model = CombinedLinearGaussianTransitionModel(
    (ConstantVelocity(0.1 ** 2),
     ConstantVelocity(0.1 ** 2)))
measurement_model = LinearGaussian(ndim_state=4, mapping=[0, 2],
                                   noise_covar=np.diag([2 ** 2,
                                                        2 ** 2]))

# Predictor & Updater
# ===================
predictor = KalmanPredictor(transition_model)
updater = KalmanUpdater(measurement_model)

# Hypothesiser & Data Associator
# ==============================
hypothesiser = PDAHypothesiser(predictor, updater, 0.0001)
associator = PDA(hypothesiser)

# Track Initiator
# ===============
state_vector = StateVector([[0], [0], [0], [0]])
covar = CovarianceMatrix(np.diag([10 ** 2, 2 ** 2,
                                  10 ** 2, 2 ** 2]))
prior_state = GaussianStatePrediction(state_vector, covar)
initiator = LinearMeasurementInitiator(prior_state, measurement_model)

# Track Deleter
# =============
deleter = UpdateTimeDeleter(time_since_update=timedelta(minutes=5))

# Simulator
# =========
timestamp_init = datetime.now()
state_init = GaussianState(StateVector([[1],[0],[0],[0]]),
                           CovarianceMatrix(
                               np.diag([10 ** 2, 0.02 ** 2,
                                        10 ** 2, 0.02 ** 2])),
                           timestamp=timestamp_init)
# gndt = SingleTargetGroundTruthSimulator(transition_model, state_init, number_steps=500)
gndt = MultiTargetGroundTruthSimulator(transition_model, state_init,
                                       birth_rate=0.1,
                                       death_probability=0,
                                       number_steps=500)

# TODO: Add detection simulator
# meas_range = [[-1000, 1000], [-1000, 1000]]
# mdt = DetectionSimulator(gndt, measurement_model,
#                          meas_range,
#                          detection_probability=hypothesiser.prob_detect,
#                          clutter_rate=hypothesiser.clutter_spatial_density
#                                       *np.prod(np.diff(meas_range)))
fig, (ax1) = plt.subplots(1,1)
tracks = set()
for time, gnd_path in gndt.groundtruth_paths_gen():
    detections = set()
    for path in gnd_path:
        measurement = Detection(measurement_model.function(
            path.state.state_vector, measurement_model.rvs(1)),
            time)
        detections.add(measurement)

    # Perform data association
    print("Tracking....")
    associations = associator.associate(tracks, detections, time)

    # Update tracks based on association hypotheses
    associated_detections = set()
    for track, multihypothesis in associations.items():

        # calculate each Track's state as a Gaussian Mixture of
        # its possible associations with each detection, then
        # reduce the Mixture to a single Gaussian State
        posterior_states = []
        posterior_weights = []
        for hypothesis in multihypothesis:
            posterior_weights.append(
                hypothesis.probability)
            if hypothesis:
                posterior_states.append(updater.update(hypothesis))
                associated_detections.add(hypothesis.measurement)
            else:
                posterior_states.append(hypothesis.prediction)

        # Merge/Collapse to single Gaussian
        means = np.concatenate([state.state_vector
                                for state in posterior_states], 1).T
        covars = np.array([state.covar for state in posterior_states])
        weights = np.array([weight for weight in posterior_weights])
        post_mean, post_covar = gm_reduce_single(means, covars, weights)

        track.append(GaussianStateUpdate(
            np.array(post_mean), np.array(post_covar),
            multihypothesis,
            multihypothesis[0].measurement_prediction.timestamp))

    # Delete invalid tracks
    del_tracks = deleter.delete_tracks(tracks, timestamp=time)
    tracks -= del_tracks

    # Initiate new tracks
    unassociated_detections = detections - associated_detections
    tracks |= initiator.initiate(unassociated_detections)

    ax1.cla()
    plot_paths(gnd_path, ax=ax1)
    plot_tracks(tracks, ax=ax1)
    plt.pause(0.0001)
    # i+=1

