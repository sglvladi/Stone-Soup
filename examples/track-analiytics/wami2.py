import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from datetime import timedelta

from stonesoup.reader.generic import CSVDetectionReader, CSVGroundTruthReader
from stonesoup.models.transition.linear import ConstantVelocity, CombinedLinearGaussianTransitionModel
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.hypothesiser.distance import DistanceHypothesiser, DistanceHypothesiserFast
from stonesoup.measures import Mahalanobis
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.initiator.simple import SimpleMeasurementInitiator, SinglePointInitiator, MultiMeasurementInitiator
from stonesoup.deleter.time import UpdateTimeStepsDeleter, UpdateTimeDeleter
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.deleter.multi import CompositeDeleter
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.dataassociator.tree import TPRTreeMixIn
from stonesoup.feeder.filter import BoundingBoxReducer

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


def plot_gnd_tracks(tracks, ax, color=None, linestyle=None):
    for track in tracks:
        if color is None:
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
        if linestyle is None:
            linestyle = '.-'
        states = [state.state_vector for state in track.states]
        if len(states) == 0:
            continue
        data = np.array(states)
        lat = data[:, 1]
        lon = data[:, 0]
        ax.plot(lon, lat, linestyle, c=color)


def plot_tracks(tracks, ax=None, show_error=True):
    for track in tracks:
        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b)
        linestyle = '.-'
        states = [state.state_vector for state in track.states]
        if len(states) == 0:
            continue
        data = np.array(states)
        lat = data[:, 2]
        lon = data[:, 0]
        ax.plot(lon, lat, linestyle, c=color)
        if show_error:
            plot_cov_ellipse(track.covar[[0, 2], :][:, [0, 2]],
                             track.mean[[0, 2], :], edgecolor=color,
                             facecolor='none', ax=ax, linestyle='-')

def plot_detections(detections, ax=None):
    for detection in detections:
        ax.plot(detection.state_vector[0], detection.state_vector[1], 'r+')


# Models
# ======
transition_model = CombinedLinearGaussianTransitionModel((ConstantVelocity(0.00001 ** 2),
                                                          ConstantVelocity(0.00001 ** 2)))
measurement_model = LinearGaussian(ndim_state=4, mapping=[0, 2],
                                   noise_covar=np.diag([0.00002 ** 2, 0.00002 ** 2]))

# Predictor & Updater
# ===================
predictor = KalmanPredictor(transition_model)
updater = KalmanUpdater(measurement_model)

# Hypothesiser & Data Associator
# ==============================
hypothesiser = DistanceHypothesiser(predictor, updater, Mahalanobis(), 10)
# associator = GNNWith2DAssignment(hypothesiser)


# hypothesiser = DistanceHypothesiserFast(predictor, updater, Mahalanobis(), 10)
class TPRGNN(GNNWith2DAssignment, TPRTreeMixIn):
    pass
associator = TPRGNN(hypothesiser, measurement_model, timedelta(hours=1), [1, 3])

# Track Initiator
# ===============
state_vector = StateVector([[0.], [0.], [0.], [0.]])
covar = CovarianceMatrix(np.diag([0.00002 ** 2, 0.00002 ** 2,
                                  0.00002 ** 2, 0.00002 ** 2]))
prior_state = GaussianStatePrediction(state_vector, covar)

deleter1 = UpdateTimeStepsDeleter(5)
# deleter1 = UpdateTimeDeleter(timedelta(seconds=20))
deleter2 = CovarianceBasedDeleter(np.trace(covar)*2)
deleter_init = CompositeDeleter([deleter1, deleter2], intersect=False)
associator_init = TPRGNN(hypothesiser, measurement_model, timedelta(hours=1), [1, 3])
initiator = MultiMeasurementInitiator(prior_state, measurement_model, deleter_init, associator_init, updater, min_points=5)

# Track Deleter
# =============
deleter1 = UpdateTimeStepsDeleter(10)
# deleter1 = UpdateTimeDeleter(timedelta(seconds=20))
deleter2 = CovarianceBasedDeleter(np.trace(covar)*3)
deleter = CompositeDeleter([deleter1, deleter2], intersect=False)

detector = CSVDetectionReader(
    "results_latlon_2.csv",
    state_vector_fields=("Longitude", "Latitude"),
    time_field="Time",
    timestamp=True
)


gnd_reader = CSVGroundTruthReader(
    "groundtruth_latlon.csv",
    state_vector_fields=("Longitude", "Latitude"),
    time_field="Time",
    timestamp=True,
    path_id_field="ID"
)

# lim = [[-73.56377588140325, -73.56154040888711], [45.498546252115666, 45.500988506790314]]
# # lim = [(-73.55356670242087, -73.54936843295665), (45.49023070839193, 45.494422002333586)]
# # lim = [(-73.56758784232056, -73.56485474548387), (45.49798743681181, 45.50075713314169)]
# # lim = [(-73.558246656375, -73.55610901768146), (45.4962286626693, 45.49757840490672)]
# detector = BoundingBoxReducer(detector, lim)
# gnd_reader = BoundingBoxReducer(gnd_reader, lim)

tracks = set()
fig = plt.figure()
ax = fig.add_subplot(111)
i = 0
for ((time, gnd_tracks), (t, detections)) in zip(gnd_reader, detector):
    print('{} - Num. Detections: {} - Num. True Tracks: {} - Num. Est. Tracks: {}'.format(time, len(detections), len(gnd_tracks), len(tracks)))

    associated_detections = set()
    associations = associator.associate(tracks, detections, time)
    for track, hypothesis in associations.items():
        if hypothesis:
            state_post = updater.update(hypothesis)
            track.append(state_post)
            associated_detections.add(hypothesis.measurement)
        else:
            track.append(hypothesis.prediction)

    # Initiate new tracks
    print("Track Initiation...")
    unassociated_detections = detections - associated_detections
    new_tracks = initiator.initiate(unassociated_detections)
    tracks |= new_tracks

    # Delete invalid tracks
    print("Track Deletion...")
    del_tracks = deleter.delete_tracks(tracks, timestamp=time)
    tracks -= del_tracks

    ax.cla()
    plot_gnd_tracks(gnd_tracks, ax, 'k', '--')
    plot_tracks(tracks, ax)
    plot_detections(detections, ax)
    plt.pause(0.1)

    if i == 10:
        a=2
