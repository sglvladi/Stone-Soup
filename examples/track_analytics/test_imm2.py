import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from copy import copy
from datetime import datetime, timedelta

from stonesoup.functions import gm_reduce_single
from stonesoup.functions import imm_merge as imm_merge2

from stonesoup.predictor.kalman import KalmanPredictor, IMMPredictor
from stonesoup.updater.kalman import KalmanUpdater, IMMUpdater

from stonesoup.models.transition.linear import ConstantVelocity, \
    CombinedLinearGaussianTransitionModel, RandomWalk, OrnsteinUhlenbeck

from stonesoup.models.measurement.linear import LinearGaussian

from stonesoup.types.state import GaussianState
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.detection import Detection
from stonesoup.types.state import StateVector, CovarianceMatrix, \
    GaussianMixtureState, WeightedGaussianState
from stonesoup.types.track import Track

from stonesoup.simulator.simple import SingleTargetGroundTruthSimulator

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

# t1 = CombinedLinearGaussianTransitionModel((OrnsteinUhlenbeck(1**2, 0.1),
#                                             OrnsteinUhlenbeck(1**2, 0.1)))
t1 = CombinedLinearGaussianTransitionModel(
        (OrnsteinUhlenbeck(0.00001 ** 2, 2e-2),
         OrnsteinUhlenbeck(0.00001 ** 2, 2e-2)))
# t2 = CombinedLinearGaussianTransitionModel(
#         (OrnsteinUhlenbeck(0.00001 ** 2, 2e-2),
#          OrnsteinUhlenbeck(0.00001 ** 2, 2e-2)))
# t2 = CombinedLinearGaussianTransitionModel((ConstantVelocity(11**2),
#                                             ConstantVelocity(11**2)))
t2 = CombinedLinearGaussianTransitionModel((RandomWalk(0.000000001**2),
                                            RandomWalk(np.finfo(float).eps),
                                            RandomWalk(0.000000001**2),
                                            RandomWalk(np.finfo(float).eps)))
timestamp_init = datetime.now()
state_init = WeightedGaussianState(StateVector([[1],
                                                [0],
                                                [0],
                                                [0]]),
                                   CovarianceMatrix(
                                       np.diag([0.0001 ** 2, 0.02 ** 2,
                                                0.0001 ** 2, 0.02 ** 2])),
                                   timestamp=timestamp_init,
                                   weight=0.5)

lg = LinearGaussian(ndim_state=4, mapping=[0, 2],
                                       noise_covar=np.diag([0.0001 ** 2,
                                                            0.0001 ** 2]))


p = np.array([[0.95, 0.05],
              [0.05, 0.95]])
kp1 = KalmanPredictor(t1)
kp2 = KalmanPredictor(t2)
imm_predictor = IMMPredictor([kp1, kp2], p)
ku1 = KalmanUpdater(lg)
ku2 = KalmanUpdater(lg)
imm_updater = IMMUpdater([ku1, ku2], p)

gndt = SingleTargetGroundTruthSimulator(t1, state_init, number_steps=500)

def imm_predict(p, predictors, state, timestamp):
    nm = p.shape[0]

    # Extract means, covars and weights
    means = state.means
    covars = state.covars
    weights = state.weights

    # Step 1) Calculation of mixing probabilities
    c_j = p @ weights
    MU_ij = p * (weights @ (1 / c_j).T)

    # Step 2) Mixing (Mixture Reduction)
    means_k, covars_k = imm_merge(means, covars, MU_ij)

    predictions = []
    # Step 3) Mode-matched filtering
    for i in range(nm):
        prior = GaussianState(means_k[:,[i]],
                              np.squeeze(covars_k[[i],:,:]),
                              timestamp = state.timestamp)
        prediction = predictors[i].predict(prior, timestamp = timestamp)
        predictions.append(WeightedGaussianState(prediction.mean,
                                                 prediction.covar,
                                                 weight = weights[i, 0],
                                                 timestamp=prediction.timestamp))
    state_k = GaussianMixtureState(predictions)
    return state_k

def imm_update(p, updaters, prediction, measurement):
    nm = p.shape[0]
    nx = prediction.ndim

    # Extract means, covars and weights
    means, covars, weights = (prediction.means,
                              prediction.covars,
                              prediction.weights)

    # Step 3) Mode-matched filtering (ctn'd)
    Lj = np.zeros((nm,1))
    posteriors = []
    for i in range(nm):
        pred = GaussianState(means[:,[i]],
                             np.squeeze(covars[[i],:,:]),
                             timestamp = prediction.timestamp)
        meas_prediction = updaters[i].get_measurement_prediction(pred)
        hyp = SingleHypothesis(pred, measurement, meas_prediction)
        posterior = updaters[i].update(hyp)
        posteriors.append(posterior)
        S = meas_prediction.covar
        Lj[[i], 0] = multivariate_normal.pdf(measurement.state_vector.T,
                                             meas_prediction.mean.ravel(),
                                             (S+S.T)/2)

    #Step 4) Mode Probability update
    c_j = p @ weights # (11.6.6-8)
    weights = Lj * c_j # (11.6.6-15)
    weights = weights/np.sum(weights) # Normalise
    posteriors_w = [WeightedGaussianState(posteriors[i].mean,
                                          posteriors[i].covar,
                                          weight=weights[i,0],
                                          timestamp=posteriors[i].timestamp)
                    for i in range(nm)]

    posterior = GaussianMixtureState(posteriors_w)
    means = posterior.means
    covars = posterior.covars
    m, P = gm_reduce_single(means.T, covars, weights.ravel())
    estimate = GaussianState(m, P, timestamp=posterior.timestamp)
    return posterior, estimate

def imm_merge(means, covars, weights):
    nx, nm = means.shape
    x_0j = means @ weights
    v = means - x_0j
    P_t = np.zeros((nm, nx, nx))
    for i in range(nm):
        P_t[[i], :, :,] = covars[[i], :, :] + v[:, [i]]@v[:, [i]].T

    x_0j = np.zeros((nx, nm))
    P_0j = np.zeros((nm, nx, nx))
    for j in range(nm):
        x_0j[:, [j]] = means @ weights[:, [j]]
        v_0 = means[:, [j]] - x_0j[:, [j]]
        for i in range(nm):
            P_0j[[j], :,:] = P_0j[[j], :, :] \
                             + weights[i, j]*(covars[[j], :, :] + v_0@v_0.T)
    return [x_0j, P_0j]

def imm(meas, s1, s2, mu, p, kp1, kp2, ku):
    # getting no.of rows and coloumns for transprob matrix
    r, c = p.shape

    # Mixing probabilities calculation
    c1 = p[0, 0]*mu[0, 0] + p[0, 1]*mu[1, 0]
    c2 = p[1, 0]*mu[0, 0] + p[1, 1]*mu[1, 0]
    c_j = p@mu
    MU_ij = p*(mu@(1/c_j).T)

    # Mixed estimates of means and Covariances
    xm = np.concatenate((s1.mean, s2.mean),1)
    temp_x = xm @ MU_ij # mixing mean equation 11.6.6 - 9

    xk_1k_1 = xm # initialization for mixed means difference
    xp = np.array([s1.covar, s2.covar])
    temp_p = xp  # initialization for mixed covariances
    m, n, d = temp_p.shape      # getting dimension for mixed corivances
    P = np.zeros((m * n, d))    # defination for the mixed covariance

    for i in range(c):
        xk_1k_1[:, [i]] = xm[:, [i]] - temp_x[:, [i]]  # mixed mean difference
        temp_p[[i], :, :] = xp[[i], :, :] + xk_1k_1[:, [i]]@xk_1k_1[:, [i]].T #
        # mixed
        # covarince without MU_ij multiplication
    a=2


prior1 = copy(state_init)
track1 = Track([prior1])
track2 = Track([prior1])
prior = GaussianMixtureState([track1.state, track2.state])
prior_1 = GaussianMixtureState([track1.state, track2.state])
i=1
track = Track([copy(prior)])

fig, (ax1, ax2) = plt.subplots(2,1)
for time, gnd_path in gndt.groundtruth_paths_gen():
    gn_p = gnd_path.pop()
    measurement = Detection(lg.function(gn_p.state.state_vector, lg.rvs(1)),
                          time)

    prediction = imm_predictor.predict(track.state, timestamp=time)
    meas_prediction = imm_updater.get_measurement_prediction(prediction)
    hyp = SingleHypothesis(prediction, measurement)

    # prediction_1 = imm_predict(p, [kp1, kp2], prior_1, time)
    # if (not np.all(np.equal(prediction_1.means, prediction.means))):
    #     a = 2

    prior = imm_updater.update(hyp)
    # prior_1, estimate = imm_update(p, [ku1, ku2], prediction_1, measurement)
    x, P = gm_reduce_single(prior.means.T, prior.covars,
                            prior.weights.ravel())
    estimate = GaussianState(x, P)
    track.append(prior)
    # if(not np.all(np.equal(prior_1.means, prior.means))):
    #     a=2
    ax1.cla()
    ax2.cla()
    data = np.array([state.state_vector for state in track.states])
    ax1.plot(data[:, 0], data[:, 2], 'r-')
    data = np.array([state.state_vector for state in gn_p.states])
    ax1.plot(data[:, 0], data[:, 2], 'b-')
    print(prior.weights)
    v_x, v_y = (estimate.state_vector[1,0], estimate.state_vector[3,0])
    plt.title("Speed: {} knots".format(np.sqrt(v_x**2 + v_y**2)*1.944))
    # x, P = gm_reduce_single(meas_prediction.means.T, meas_prediction.covars, meas_prediction.weights.ravel())
    # estimate = GaussianState(x, P)
    plot_cov_ellipse(meas_prediction.covar,
                     meas_prediction.mean, edgecolor='b',
                     facecolor='none', ax=ax1)
    plot_cov_ellipse(track.state.covar[[0, 2], :][:, [0, 2]],
                     track.state.mean[[0, 2], :], edgecolor='r',
                     facecolor='none', ax=ax1)
    ax2.bar([1,2], prior.weights.ravel())
    plt.pause(0.0001)
    i+=1

