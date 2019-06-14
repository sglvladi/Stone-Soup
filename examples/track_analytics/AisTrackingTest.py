# coding: utf-8

################################################################################
# IMPORTS                                                                      #
################################################################################

# General imports
import os
import pickle
import numpy as np
from datetime import datetime, timedelta
from copy import copy

# Stone-Soup imports
from stonesoup.models.transition.linear import (
    ConstantVelocity, OrnsteinUhlenbeck, CombinedLinearGaussianTransitionModel)
from stonesoup.initiator.simple import SinglePointInitiator
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.dataassociator.neighbour import (
    NearestNeighbour, GlobalNearestNeighbour)
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.hypothesiser.filtered import FilteredDetectionsHypothesiser
from stonesoup.updater.kalman import (
    UnscentedKalmanUpdater, ExtendedKalmanUpdater)
from stonesoup.predictor.kalman import (
    UnscentedKalmanPredictor, ExtendedKalmanPredictor)
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.update import Update
from stonesoup.types.state import GaussianState
from stonesoup.writer.mongo import MongoWriter
from stonesoup.reader.generic import CSVDetectionReader
from stonesoup.feeder.time import TimeSyncFeeder
from stonesoup.feeder.filter import BoundingBoxReducer


################################################################################
# Plotting functions                                                           #
################################################################################

from matplotlib.patches import Ellipse
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (12, 8)
plt.style.use('seaborn-colorblind')
fig = plt.figure()


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
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


def plot_tracks(tracks):
    for track in tracks:
        data = np.array([state.state_vector for state in track.states if
                         isinstance(state, Update)])
        plt.plot(data[:, 0], data[:, 2], 'r-', label="AIS Tracks")
        plot_cov_ellipse(track.state.covar[[0, 2], :][:, [0, 2]],
                         track.state.mean[[0, 2], :], edgecolor='r',
                         facecolor='none')


def plot_data(detections=None):
    if len(detections) > 0:
        x = [s.state_vector[0] for s in detections]
        y = [s.state_vector[1] for s in detections]
        plt.plot(x, y, linestyle='', marker='x')


################################################################################
# Tracking components                                                          #
################################################################################

# Transition & Measurement models
# ===============================
transition_model = CombinedLinearGaussianTransitionModel(
                            (OrnsteinUhlenbeck(0.00001 ** 2, 1e-3),
                             OrnsteinUhlenbeck(0.00001 ** 2, 1e-3)))
measurement_model = LinearGaussian(ndim_state=4, mapping=[0, 2],
                                   noise_covar=np.diag([0.001 ** 2,
                                                        0.001 ** 2]))

# Predictor & Updater
# ===================
predictor = UnscentedKalmanPredictor(transition_model)
updater = UnscentedKalmanUpdater(measurement_model)

# Hypothesiser & Data Associator
# ==============================
hypothesiser = DistanceHypothesiser(predictor, updater, Mahalanobis(), 4)
hypothesiser = FilteredDetectionsHypothesiser(hypothesiser, 'mmsi',
                                              match_missing=False)
associator = NearestNeighbour(hypothesiser)

# Track Initiator
# ===============
state_vector = StateVector([[0], [0], [0], [0]])
covar = CovarianceMatrix(np.diag([10 ** 2, 2 ** 2, 10 ** 2, 2 ** 2]))
prior_state = GaussianState(state_vector, covar)
initiator = SinglePointInitiator(prior_state, measurement_model)

# Track Deleter
# =============
deleter = UpdateTimeStepsDeleter(time_steps_since_update=2)

# Track Recycler
# ==============
def recycle_tracks(tracks, detections):
    recycled_tracks = set()
    for detection in detections:
        for track in tracks:
            if detection.metadata["mmsi"] == track.metadata["mmsi"]:
                recycled_tracks.add(track)
                break
    return recycled_tracks

# Writer
# ======
writer = MongoWriter()

# Initialise DB collections
collections = ["Live_SS_Tracks", "Live_SS_Points"]
writer.reset_collections(
    host_name="138.253.118.175",
    host_port=27017,
    db_name="TA_IHS",
    collection_names=["Live_SS_Tracks", "Live_SS_Points"],
)

################################################################################
# Main Tracking process                                                        #
################################################################################

filenames = ['AIS_2017-01-23_2017-01-28_id',
             'AIS_2017-01-28_2017-02-02_id',
             'AIS_2017-02-01_2017-02-06_id']
tracks = set()
deleted_tracks = set()

# We process the files sequentially
for i, filename in enumerate(filenames):

    # Detection reader
    # ================
    # (Needs to be re-instantiated for each file)
    detector = CSVDetectionReader(
        path=os.path.join(os.getcwd(),
                          r'\\bh-fs01\rdm01\TrackAnalytics\Phase 2a\Technical\Work in Progress\Stone Soup\SS_input_data\{}.csv'.format(filename)),
        state_vector_fields=["Longitude", "Latitude"],
        time_field="Time",
        metadata_fields=['ID', 'LRIMOShipNo', 'ShipType', 'ShipName', 'MMSI',
                         'CallSign', 'Beam', 'Draught', 'Length', 'Latitude',
                         'Longitude', 'Speed', 'Heading', 'ETA', 'Destination',
                         'DestinationTidied', 'AdditionalInfo', 'MovementDateTime',
                         'MovementID', 'MoveStatus'])
    detector = TimeSyncFeeder(detector,
                              time_window=timedelta(minutes=1))
    detector = BoundingBoxReducer(detector,
                                  # bounding_box=np.array([-6, -5, 48, 50]),
                                  bounding_box=np.array([-6, 0, 48, 56]),
                                  mapping=[0, 1])

    # We process each scan sequentially
    for scan_time, detections in detector.detections_gen():

        # Skip iteration if there are no available detections
        if len(detections) == 0:
            continue

        print("Measurements: " + str(len(detections)))
        # Recycle AIS tracks
        # tracks |= recycle_tracks(deleted_tracks, detections)

        # Perform data association
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

        # Delete invalid tracks
        del_tracks = deleter.delete_tracks(tracks)
        deleted_tracks |= del_tracks
        tracks -= del_tracks

        # Initiate new tracks
        unassociated_detections = detections - associated_detections
        tracks |= initiator.initiate(unassociated_detections)

        # Write data
        writer.write(tracks,
                     detections,
                     host_name="138.253.118.175",
                     host_port=27017,
                     db_name="TA_IHS",
                     collection_name=collections,
                     drop=False)

        # Plot the data
        # plt.clf()
        # plot_data(detections)
        # plot_tracks(tracks)
        # plt.pause(0.01)

        print("Time: " + scan_time.strftime('%H:%M:%S %d/%m/%Y')
              + " - Measurements: " + str(len(detections))
              + " - Tracks: " + str(len(tracks)))

# Back-up data
data = {"tracks": tracks,
        "deleted_tracks": deleted_tracks}
with open('data/{}_tracks.pickle'.format(filename), 'wb') as f:
    pickle.dump(data, f)
