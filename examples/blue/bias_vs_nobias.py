import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D

from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.array import CovarianceMatrix
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity, RandomWalk
from stonesoup.models.measurement.blue import SimpleBlueMeasurementModel
from stonesoup.reader.blue import BlueDetectionReaderMatlab, BlueDetectionReaderFile
from stonesoup.initiator.blue import SimpleBlueInitiator
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.updater.kalman import UnscentedKalmanUpdater, ExtendedKalmanUpdater

# Helper functions ====================================================================================================

def plot_3d_covariance(mean, cov, std=1.,
                       ax=None, color=None, alpha=1.,
                       N=60, shade=True,
                       **kwargs):
    """
    Plots a covariance matrix `cov` as a 3D ellipsoid centered around
    the `mean`.
    Parameters
    ----------
    mean : 3-vector
        mean in x, y, z. Can be any type convertable to a row vector.
    cov : ndarray 3x3
        covariance matrix
    std : double, default=1
        standard deviation of ellipsoid
    ax : matplotlib.axes._subplots.Axes3DSubplot, optional
        Axis to draw on. If not provided, a new 3d axis will be generated
        for the current figure
    title : str, optional
        If provided, specifies the title for the plot
    color : any value convertible to a color
        if specified, color of the ellipsoid.
    alpha : float, default 1.
        Alpha value of the ellipsoid. <1 makes is semi-transparent.
    label_xyz: bool, default True
        Gives labels 'X', 'Y', and 'Z' to the axis.
    N : int, default=60
        Number of segments to compute ellipsoid in u,v space. Large numbers
        can take a very long time to plot. Default looks nice.
    shade : bool, default=True
        Use shading to draw the ellipse
    limit_xyz : bool, default=True
        Limit the axis range to fit the ellipse
    **kwargs : optional
        keyword arguments to supply to the call to plot_surface()
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]
    # The idea is simple - find the 3 axis of the covariance matrix
    # by finding the eigenvalues and vectors. The eigenvalues are the
    # radii (squared, since covariance has squared terms), and the
    # eigenvectors give the rotation. So we make an ellipse with the
    # given radii and then rotate it to the proper orientation.

    eigval, eigvec = eigsorted(cov)
    radii = std * np.sqrt(np.real(eigval))

    if eigval[0] < 0:
        raise ValueError("covariance matrix must be positive definite")


    # calculate cartesian coordinates for the ellipsoid surface
    u = np.linspace(0.0, 2.0 * np.pi, N)
    v = np.linspace(0.0, np.pi, N)
    x = np.outer(np.cos(u), np.sin(v)) * radii[0]
    y = np.outer(np.sin(u), np.sin(v)) * radii[1]
    z = np.outer(np.ones_like(u), np.cos(v)) * radii[2]

    # rotate data with eigenvector and center on mu
    a = np.kron(eigvec[:, 0], x)
    b = np.kron(eigvec[:, 1], y)
    c = np.kron(eigvec[:, 2], z)

    data = a + b + c
    N = data.shape[0]
    x = data[:,   0:N]   + mean[0]
    y = data[:,   N:N*2] + mean[1]
    z = data[:, N*2:]    + mean[2]

    fig = plt.gcf()
    if ax is None:
        ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x, y, z,
                    rstride=3, cstride=3, linewidth=0.1, alpha=alpha,
                    shade=shade, color=color, **kwargs)

    return ax

def plot_cov_ellipse(cov, pos, nstd=1, ax=None, **kwargs):
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
                    alpha=0.4, **kwargs)

    ax.add_artist(ellip)
    return ellip


def plot_bias(track_ekf, track_ukf, truthdata):
    fig = plt.figure()
    t = truthdata.hit_times
    idxs = np.array([1, 4, 2, 5, 3, 6]) + 5
    labels = ['thetaBias1 (rad)', 'psiBias1 (rad)', 'dtBias1 (sec)', 'thetaBias2 (rad)', 'psiBias2 (rad)', 'dtBias2 (sec)']
    titles = ['Elevation (Sensor 1)', 'Bearing (Sensor 1)', 'Time delay (Sensor 1)',
              'Elevation (Sensor 2)', 'Bearing (Sensor 2)', 'Time delay (Sensor 2)']
    for i, idx in enumerate(idxs):
        ax = fig.add_subplot(2, 3, i + 1)
        mn = np.array([state.state_vector[idx, 0] for state in track_ekf.states])
        sd = np.sqrt(np.squeeze([state.covar[idx, idx] for state in track_ekf.states]))
        l1, = ax.plot(t, mn, 'gx-')
        l2 = ax.fill_between(t, mn - sd, mn + sd, facecolor='g', alpha=0.2)
        mn = np.array([state.state_vector[idx, 0] for state in track_ukf.states])
        sd = np.sqrt(np.squeeze([state.covar[idx, idx] for state in track_ukf.states]))
        l3, = ax.plot(t, mn, 'rx-')
        l4 = ax.fill_between(t, mn - sd, mn + sd, facecolor='r', alpha=0.2)
        ax.set_title(titles[i])
        ax.legend((l1, l2, l3, l4), ('EKF - Mean', r'EKF - $\pm 1\sigma$', 'UKF - Mean', r'UKF - $\pm 1\sigma$'))
        ax.set_xlim((0, np.max(t)))
        ax.set_xlabel('Hit time (sec)')
        ax.set_ylabel(labels[i])
    st = plt.suptitle('Bias estimation', fontsize=18)
    st.set_y(0.92)
    plt.show()


def plot_coords(track_bias, track_nobias, truthdata):
    fig = plt.figure()
    t = truthdata.hit_times.to_numpy()
    target_pos_hit = np.stack(truthdata.target_pos_hit).T
    idxs = [0, 2, 4]
    labels = ['x', 'y', 'z']
    titles = ['X coordinate', 'Y coordinate', 'Z coordinate']
    for i, idx in enumerate(idxs):
        ax = fig.add_subplot(2, 2, i + 1)
        l1, = ax.plot(t, target_pos_hit[i, :], 'k+-')
        mn = np.array([state.state_vector[idx, 0] for state in track_bias.states])
        sd = np.sqrt(np.squeeze([state.covar[idx, idx] for state in track_bias.states]))
        l2, = ax.plot(t, mn, 'gx-')
        l3 = ax.fill_between(t, mn - sd, mn + sd, facecolor='g', alpha=0.2)
        mn = np.array([state.state_vector[idx, 0] for state in track_nobias.states])
        sd = np.sqrt(np.squeeze([state.covar[idx, idx] for state in track_nobias.states]))
        l4, = ax.plot(t, mn, 'rx-')
        l5 = ax.fill_between(t, mn - sd, mn + sd, facecolor='r', alpha=0.2)
        # ax.plot(t, mn + sd, 'r--')
        # ax.plot(t, mn - sd, 'r--')
        ax.set_title(titles[i])
        ax.legend((l1, l2, l3, l4, l5), ('True', 'Bias - Mean', r'Bias - $\pm 1\sigma$', 'No bias - Mean', r'No bias - $\pm 1\sigma$'))
        ax.set_xlim((0, np.max(t)))
        ax.set_xlabel('Hit time (sec)')
        ax.set_ylabel('{} (m)'.format(labels[i]))
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    l1, = ax.plot(target_pos_hit[0, :], target_pos_hit[1, :], target_pos_hit[2, :], 'k+-')
    mn = np.array([state.state_vector[[0, 2, 4], 0] for state in track_bias.states]).T
    l2, = ax.plot(mn[0, :], mn[1, :], mn[2,:], 'gx-')
    for state in track_bias.states:
        plot_3d_covariance(state.mean[[0, 2, 4], :],
                           state.covar[[0, 2, 4], :][:, [0, 2, 4]],
                           color='g',
                           alpha=0.1, ax=ax)
        # plot_cov_ellipse(state.covar[[0, 2], :][:, [0, 2]],
        #                  state.mean[[0, 2], :], edgecolor='c',
        #                  facecolor='none', ax=ax)
    # ax = fig.add_subplot(111, projection='3d')
    mn = np.array([state.state_vector[[0, 2, 4], 0] for state in track_nobias.states]).T
    l3, = ax.plot(mn[0, :], mn[1, :], mn[2, :], 'rx-')
    for state in track_nobias.states:
        plot_3d_covariance(state.mean[[0, 2, 4], :],
                           state.covar[[0, 2, 4], :][:, [0, 2, 4]],
                           color='r',
                           alpha=0.1, ax=ax)
        # plot_cov_ellipse(state.covar[[0, 2], :][:, [0, 2]],
        #                  state.mean[[0, 2], :], edgecolor='r',
        #                  facecolor='none', ax=ax)
    ax.set_title('3D visualisation')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    st = plt.suptitle('Position Estimation', fontsize=18)
    st.set_y(0.92)
    plt.show()


def plot_track(track, show_error=True, ax=None):
    data = np.array([state.state_vector for state in track.states])
    if ax is not None:
        ax.plot(data[:, 0].ravel(), data[:, 2].ravel(), data[:, 4].ravel(), 'r-')
    else:
        plt.plot(data[:, 0].ravel(), data[:, 2].ravel(), data[:, 4].ravel(), 'r-')

# =====================================================================================================================

matlab_path = r'C:\Users\sglvladi\OneDrive\Workspace\PostDoc\CADMURI\MATLAB\BLUE\UKF_tracker'

# Detection Reader
file_path = r'C:\Users\sglvladi\OneDrive\Workspace\PostDoc\CADMURI\MATLAB\BLUE\UKF_tracker\simdata.mat'
# detector = BlueDetectionReaderFile(path=file_path)
detector = BlueDetectionReaderMatlab(dir_path=matlab_path)

# Models
tx_model_bias = CombinedLinearGaussianTransitionModel([ConstantVelocity(1),  # x
                                                       ConstantVelocity(1),  # y
                                                       ConstantVelocity(0.1),  # z
                                                       RandomWalk(0.02 ** 2 / 400),  # el-bias 1
                                                       RandomWalk(0.02 ** 2 / 400),  # el-bias 2
                                                       RandomWalk(0),  # az-bias 1
                                                       RandomWalk(0),  # az-bias 2
                                                       RandomWalk(0.002 ** 2 / 400),  # dt-bias 1
                                                       RandomWalk(0.002 ** 2 / 400)  # dt-bias 2
                                                      ])
tx_model_nobias = CombinedLinearGaussianTransitionModel([ConstantVelocity(1),  # x
                                                         ConstantVelocity(1),  # y
                                                         ConstantVelocity(0.1),  # z
                                                        ])
R = CovarianceMatrix(np.diag([10, 10, 1, 0.1, 0.01, 0.2]))
meas_model_bias = SimpleBlueMeasurementModel(ndim_state=12, mapping=[0, 2, 4], noise_covar=R)
meas_model_nobias = SimpleBlueMeasurementModel(ndim_state=6, mapping=[0, 2, 4], noise_covar=R, with_bias=False)

# Predictor
predictor_bias = KalmanPredictor(tx_model_bias)
# updater_bias = UnscentedKalmanUpdater(meas_model_bias)
updater_bias = ExtendedKalmanUpdater(meas_model_bias)

predictor_nobias = KalmanPredictor(tx_model_nobias)
# updater_nobias = UnscentedKalmanUpdater(meas_model_nobias)
updater_nobias = ExtendedKalmanUpdater(meas_model_nobias)

# Initiator
initiator_bias = SimpleBlueInitiator(dir_path=matlab_path)
initiator_nobias = SimpleBlueInitiator(dir_path=matlab_path, with_bias=False)


track_bias = None
track_nobias = None
for i, (timestamp, detections) in enumerate(detector):
    print(timestamp)

    if i == 0:
        tracks = initiator_bias.initiate(detections)
        track_bias = tracks.pop()
        tracks_nobias = initiator_nobias.initiate(detections)
        track_nobias = tracks_nobias.pop()
    else:
        detection = detections.pop()

        # Bias
        prediction = predictor_bias.predict(track_bias.state, timestamp=timestamp)
        meas_prediction = updater_bias.predict_measurement(prediction, detection.measurement_model)
        hypothesis = SingleHypothesis(prediction, detection, meas_prediction)
        posterior = updater_bias.update(hypothesis)
        track_bias.append(posterior)

        # No bias
        prediction = predictor_nobias.predict(track_nobias.state, timestamp=timestamp)
        detection.measurement_model.with_bias = False
        meas_prediction = updater_nobias.predict_measurement(prediction, detection.measurement_model)
        hypothesis = SingleHypothesis(prediction, detection, meas_prediction)
        posterior = updater_nobias.update(hypothesis)
        track_nobias.append(posterior)

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# plot_track(track, ax=ax)
# plt.show()

plot_coords(track_bias, track_nobias, detector._truthdata)