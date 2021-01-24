import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.array import CovarianceMatrix
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity, RandomWalk
from stonesoup.models.measurement.blue import SimpleBlueMeasurementModel2
from stonesoup.reader.blue import BlueDetectionReaderMatlab, BlueDetectionReaderFile, BlueMultiDetectionReaderFile2
from stonesoup.initiator.blue import SingleMeasurementBlueInitiator
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.updater.kalman import UnscentedKalmanUpdater, ExtendedKalmanUpdater


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
    x = data[:, 0:N] + mean[0]
    y = data[:, N:N*2] + mean[1]
    z = data[:, N*2:] + mean[2]

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

def plot_bias(track, truthdata):
    fig = plt.figure()
    t = truthdata.hit_times
    idxs = np.array([1, 4, 2, 5, 3, 6]) + 5
    labels = ['thetaBias1', 'psiBias1', 'dtBias1', 'thetaBias2', 'psiBias2', 'dtBias2']
    for i, idx in enumerate(idxs):
        ax = fig.add_subplot(2,3,i+1)
        mn = np.array([state.state_vector[idx, 0] for state in track.states])
        sd = np.sqrt(np.squeeze([state.covar[idx, idx] for state in track.states]))
        l1, = ax.plot(t, mn, 'rx-')
        # ax.plot(t, mn+sd, 'r--')
        # ax.plot(t, mn-sd, 'r--')
        l2 = ax.fill_between(t, mn-sd, mn+sd, facecolor='r', alpha=0.2)
        ax.legend((l1, l2), ('Mean', r'$\pm 1\sigma$'))
        ax.set_xlim((0, np.max(t)))
        ax.set_xlabel('Hit time (sec)')
        ax.set_ylabel(labels[i])
    st = plt.suptitle('UKF - Bias estimation', fontsize=18)
    st.set_y(0.92)
    plt.show()


def plot_coords(track, truthdata):
    fig = plt.figure()
    t = truthdata.hit_times.to_numpy()
    target_pos_hit = np.stack(truthdata.target_pos_hit).T
    idxs = [0, 2, 4]
    labels = ['x', 'y', 'z']
    for i, idx in enumerate(idxs):
        ax = fig.add_subplot(2,2,i+1)
        mn = np.array([state.state_vector[idx, 0] for state in track.states])
        sd = np.sqrt(np.squeeze([state.covar[idx, idx] for state in track.states]))
        l1, = ax.plot(t, target_pos_hit[i,:], 'k+-')
        l2, = ax.plot(t, mn, 'rx-')
        l3 = ax.fill_between(t, mn-sd, mn+sd, facecolor='r', alpha=0.2)
        # ax.plot(t, mn + sd, 'r--')
        # ax.plot(t, mn - sd, 'r--')
        ax.legend((l1, l2, l3),('True', 'Estimated Mean', r'$\pm 1\sigma$'))
        ax.set_xlim((0, np.max(t)))
        ax.set_xlabel('Hit time (sec)')
        ax.set_ylabel('{} (m)'.format(labels[i]))
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    l1, = ax.plot(target_pos_hit[0, :], target_pos_hit[1, :], target_pos_hit[2, :], 'k+-')
    mn = np.array([state.state_vector[[0, 2, 4], 0] for state in track.states]).T
    l2, = ax.plot(mn[0, :], mn[1, :], mn[2, :], 'gx-')
    ax.legend((l1, l2), ('True', 'Estimated'))
    for state in track.states:
        plot_3d_covariance(state.mean[[0, 2, 4], :],
                           state.covar[[0, 2, 4], :][:, [0, 2, 4]],
                           color='g',
                           alpha=0.1, ax=ax)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    st = plt.suptitle('UKF - Position Estimation', fontsize=18)
    st.set_y(0.92)
    plt.show()

def plot_truth(target_pos_hit, hit_times, ax, timestamp):
    idx = [i for i in range(len(hit_times)) if hit_times[i]<=timestamp]
    if len(idx):
        ax.plot(target_pos_hit[0, idx], target_pos_hit[1, idx], target_pos_hit[2, idx], 'k+-')
        ax.plot(target_pos_hit[0, 0], target_pos_hit[1, 0], target_pos_hit[2, 0], 'go')
        ax.plot(target_pos_hit[0, idx[-1]], target_pos_hit[1, idx[-1]], target_pos_hit[2, idx[-1]], 'rs')

def plot_track(track, show_error=True, ax=None):
    data = np.array([state.state_vector for state in track.states])
    if ax is not None:
        ax.plot(data[:, 0].ravel(), data[:, 2].ravel(), data[:, 4].ravel(), 'r-')
    else:
        plt.plot(data[:, 0].ravel(), data[:, 2].ravel(), data[:, 4].ravel(), 'r-')

    plot_3d_covariance(track.state.mean[[0, 2, 4], :],
                       track.state.covar[[0, 2, 4], :][:, [0, 2, 4]],
                       color='g',
                       alpha=0.1, ax=ax)
    # for state in track.states:
    #     plot_3d_covariance(state.mean[[0, 2, 4], :],
    #                        state.covar[[0, 2, 4], :][:, [0, 2, 4]],
    #                        color='g',
    #                        alpha=0.1, ax=ax)


matlab_path = r'C:\Users\sglvladi\OneDrive\Workspace\PostDoc\CADMURI\MATLAB\BLUE\InitiatorStacked'

# Detection Reader
file_path = r'C:\Users\sglvladi\OneDrive\Workspace\PostDoc\CADMURI\MATLAB\BLUE\Simulator\TestData\data3targets_final2.mat'
detector = BlueMultiDetectionReaderFile2(path=file_path)
# detector = BlueDetectionReaderMatlab(dir_path=matlab_path)
# file_path = r'C:\Users\sglvladi\OneDrive\Workspace\PostDoc\CADMURI\MATLAB\BLUE\UKF_tracker\simdata.mat'
# detector = BlueDetectionReaderFile(path=file_path)

# Models
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(1),  # x
                                                          ConstantVelocity(1),  # y
                                                          ConstantVelocity(0.1),  # z
                                                          RandomWalk(0.02 ** 2 / 400),  # el-bias 1
                                                          RandomWalk(0.02 ** 2 / 400),  # el-bias 2
                                                          RandomWalk(0),  # az-bias 1
                                                          RandomWalk(0),  # az-bias 2
                                                          RandomWalk(0.002 ** 2 / 400),  # dt-bias 1
                                                          RandomWalk(0.002 ** 2 / 400)  # dt-bias 2
                                                          ])
R = CovarianceMatrix(np.diag([10, 10, 1, 0.1, 0.01, 0.2]))
measurement_model = SimpleBlueMeasurementModel2(ndim_state=12, mapping=[0, 2, 4], noise_covar=R)

# Predictor
predictor = KalmanPredictor(transition_model)
# updater = UnscentedKalmanUpdater(measurement_model)
updater = ExtendedKalmanUpdater(measurement_model)

# Initiator
initiator = SingleMeasurementBlueInitiator(dir_path=matlab_path)

# Data association
hypothesiser = DistanceHypothesiser(predictor, updater, Mahalanobis(), 10)
associator = GNNWith2DAssignment(hypothesiser)

tracks = set()
fig = plt.figure()
ax = fig.gca(projection='3d')
for i, (timestamp, detections) in enumerate(detector):
    print(timestamp)
    associations = associator.associate(tracks, detections, timestamp)

    # Update tracks based on association hypotheses
    associated_detections = set()
    for track, hypothesis in associations.items():
        if hypothesis:
            state_post = updater.update(hypothesis)
            track.append(state_post)
            associated_detections.add(hypothesis.measurement)
        else:
            track.append(hypothesis.prediction)

    unassociated_detections = detections - associated_detections

    tracks |= initiator.initiate(unassociated_detections)

    ax.cla()
    for track in tracks:
        plot_track(track, ax=ax)

    for j in range(3):
        plot_truth(detector._truthdata.target_pos_hit[j], detector._truthdata.hit_times[j].ravel(), ax, timestamp)

    plt.pause(0.0001)
    a = 2

    # if i == 0:
    #     tracks = initiator.initiate(detections)
    #     track = tracks.pop()
    # else:
    #     detection = detections.pop()
    #     prediction = predictor.predict(track.state, timestamp=timestamp)
    #     meas_prediction = updater.predict_measurement(prediction, detection.measurement_model)
    #     hypothesis = SingleHypothesis(prediction, detection, meas_prediction)
    #     posterior = updater.update(hypothesis)
    #     track.append(posterior)

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# plot_track(track, ax=ax)
plt.show()