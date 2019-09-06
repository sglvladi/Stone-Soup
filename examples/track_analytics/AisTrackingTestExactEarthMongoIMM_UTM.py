# coding: utf-8

################################################################################
# IMPORTS                                                                      #
################################################################################
# General imports
import os
import pickle
import numpy as np
from datetime import datetime, timedelta
import msvcrt
import time
from copy import copy

# Stone-Soup imports
from stonesoup.models.transition.linear import (
    RandomWalk, OrnsteinUhlenbeck, CombinedLinearGaussianTransitionModel)
from stonesoup.initiator.simple import SinglePointInitiator, \
    LinearMeasurementInitiator, LinearMeasurementInitiatorMixture
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.deleter.time import UpdateTimeStepsDeleter, UpdateTimeDeleter
from stonesoup.dataassociator.neighbour import (
    NearestNeighbour, GlobalNearestNeighbour)
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.hypothesiser.filtered import FilteredDetectionsHypothesiser
from stonesoup.updater.kalman import (KalmanUpdater,
                                      IMMUpdater,
                                      UnscentedKalmanUpdater,
                                      ExtendedKalmanUpdater)
from stonesoup.predictor.kalman import (KalmanPredictor,
                                        IMMPredictor,
                                        UnscentedKalmanPredictor,
                                        ExtendedKalmanPredictor)
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.types.update import Update
from stonesoup.types.state import GaussianState
from stonesoup.writer.mongo import MongoWriter, MongoWriter_EE
from stonesoup.reader.generic import CSVDetectionReader
from stonesoup.reader.mongo import MongoDetectionReader
from stonesoup.feeder.time import TimeSyncFeeder, TimeWindowReducer
from stonesoup.feeder.filter import BoundingBoxReducer

if __name__ == '__main__':
    ##############################################################################
    # TRACKING LIMIT SELECTION                                                   #
    ##############################################################################
    TARGET = "GREECE"
    LIMITS = {
        "TEST": {
            "LON_MIN": -62.,
            "LON_MAX": -61.5,
            "LAT_MIN": 11.8,
            "LAT_MAX": 12.2
        },
        "GLOBAL": {
            "LON_MIN": -180.,
            "LON_MAX": 180.,
            "LAT_MIN": -80.,
            "LAT_MAX": 80.
        },
        "FULL": {
            "LON_MIN": -84.,
            "LON_MAX": 34.5,
            "LAT_MIN": 9.5,
            "LAT_MAX": 62.
        },
        "CARIBBEAN": {
            "LON_MIN": -90.,
            "LON_MAX": -60.,
            "LAT_MIN": 10.,
            "LAT_MAX": 22.
        },
        "MEDITERRANEAN": {
            "LON_MIN": -6.,
            "LON_MAX": 36.5,
            "LAT_MIN": 30.,
            "LAT_MAX": 46.
        },
        "GREECE": {
            "LON_MIN": 20.,
            "LON_MAX": 28.2,
            "LAT_MIN": 34.6,
            "LAT_MAX": 41.
        },
        "UK": {
            "LON_MIN": -12.,
            "LON_MAX": 3.5,
            "LAT_MIN": 48.,
            "LAT_MAX": 60.
        }
    }
    LON_MIN = LIMITS[TARGET]["LON_MIN"]
    LON_MAX = LIMITS[TARGET]["LON_MAX"]
    LAT_MIN = LIMITS[TARGET]["LAT_MIN"]
    LAT_MAX = LIMITS[TARGET]["LAT_MAX"]
    END_TIME = None  # datetime(2017, 8, 10, 0, 48,13)
    LOAD = False
    BACKUP = False
    LOAD_OFFSET = 3

    ################################################################################
    # Plotting functions                                                           #
    ################################################################################

    from matplotlib.patches import Ellipse
    from matplotlib import pyplot as plt
    from mpl_toolkits.basemap import Basemap

    plt.rcParams['figure.figsize'] = (12, 8)
    plt.style.use('seaborn-colorblind')
    fig = plt.figure()


    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())

    def plot_map(timestamp):
        # Mercator projection map
        m = Basemap(llcrnrlon=LON_MIN, llcrnrlat=LAT_MIN, urcrnrlon=LON_MAX,
                    urcrnrlat=LAT_MAX, projection='merc', resolution='h')
        m.drawcoastlines()
        m.drawcountries()
        m.drawmapboundary(fill_color='#99ffff')
        m.fillcontinents(color='#cc9966', lake_color='#99ffff')
        m.drawparallels(np.arange(-90., 91., 20.), labels=[1, 1, 0, 0])
        m.drawmeridians(np.arange(-180., 181., 20.), labels=[0, 0, 0, 1])
        plt.title(
            'Exact Earth AIS dataset Tracking\n'
            + "({})\n".format(TARGET)
            + timestamp.strftime('%H:%M:%S %d/%m/%Y'))


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


    def plot_tracks(tracks, show_error=True):
        # Mercator projection map
        m = Basemap(llcrnrlon=LON_MIN, llcrnrlat=LAT_MIN, urcrnrlon=LON_MAX,
                    urcrnrlat=LAT_MAX, projection='merc', resolution='c')
        for track in tracks:
            data = np.array([state.state_vector for state in track.states if
                             isinstance(state, Update)])
            lat = data[:, 2]
            lon = data[:, 0]

            x, y = m(lon, lat)
            m.plot(x, y, 'b-o', linewidth=1, markersize=1)
            m.plot(x[-1], y[-1], 'ro', markersize=1)
            #plt.text(x[-1], y[-1], track.metadata["Vessel_Name"], fontsize=12)
            plt.text(x[-1], y[-1], np.around(track.last_update.weights[0,0],2), fontsize=12)
            # plt.plot(data[:, 0], data[:, 2], '-', label="AIS Tracks")
            # if show_error:
            #     plot_cov_ellipse(track.state.covar[[0, 2], :][:, [0, 2]],
            #                      track.state.mean[[0, 2], :], edgecolor='r',
            #                      facecolor='none')


    def plot_data(detections=None):
        if len(detections) > 0:
            x = [s.state_vector[0] for s in detections]
            y = [s.state_vector[1] for s in detections]
            plt.plot(x, y, linestyle='', marker='x')


    ##########################################################################
    # Tracking components                                                    #
    ##########################################################################

    # Transition & Measurement models
    # ===============================
    transition_model = CombinedLinearGaussianTransitionModel(
        (OrnsteinUhlenbeck(0.00001 ** 2, 2e-2),
         OrnsteinUhlenbeck(0.00001 ** 2, 2e-2)))
    transition_model2 = CombinedLinearGaussianTransitionModel(
        (RandomWalk(0.00002 ** 2),
         RandomWalk(np.finfo(float).eps),
         RandomWalk(0.00002 ** 2),
         RandomWalk(np.finfo(float).eps)))
    measurement_model = LinearGaussian(ndim_state=4, mapping=[0, 2],
                                       noise_covar=np.diag([0.001 ** 2,
                                                            0.001 ** 2]))

    # Predictor & Updater
    # ===================
    p = np.array([[0.95, 0.05],
                  [0.05, 0.95]])
    predictor1 = KalmanPredictor(transition_model)
    predictor2 = KalmanPredictor(transition_model2)
    predictor = IMMPredictor([predictor1, predictor2], p)
    updater1 = KalmanUpdater(measurement_model)
    updater2 = KalmanUpdater(measurement_model)
    updater = IMMUpdater([updater1, updater2], p)

    # Hypothesiser & Data Associator
    # ==============================
    hypothesiser = DistanceHypothesiser(predictor, updater, Mahalanobis(), 5)
    hypothesiser = FilteredDetectionsHypothesiser(hypothesiser, 'MMSI',
                                                  match_missing=False)
    associator = GlobalNearestNeighbour(hypothesiser)

    # Track Initiator
    # ===============
    state_vector = StateVector([[0], [0], [0], [0]])
    covar = CovarianceMatrix(np.diag([0.0001 ** 2, 0.02 ** 2,
                                      0.0001 ** 2, 0.02 ** 2]))
    prior_state = GaussianStatePrediction(state_vector, covar)
    # initiator = SinglePointInitiator(prior_state, measurement_model)
    initiator = LinearMeasurementInitiatorMixture(prior_state,
                                                  measurement_model)

    # Track Deleter
    # =============
    # deleter = UpdateTimeStepsDeleter(time_steps_since_update=2)
    deleter = UpdateTimeDeleter(time_since_update=timedelta(minutes=5))
    rec_deleter = UpdateTimeDeleter(time_since_update=timedelta(hours=2))


    # Track Recycler
    # ==============
    def recycle_tracks(tracks, temp_deleted_tracks, deleted_tracks, detections,
                       timestamp):
        # Recycle tracks
        temp_del_tracks = set()
        unmatched_detections = copy(detections)

        for track in temp_deleted_tracks:
            # if len(unmatched_detections)==0:
            #     break
            mmsi = track.metadata["MMSI"]
            for detection in unmatched_detections:
                # If a temporary deleted track exists with the given MMSI
                if detection.metadata["MMSI"] == mmsi:
                    temp_del_tracks.add(track)  # Remove from temp deleted
                    # unmatched_detections.remove(detection)
                    break

        # Delete permanently
        tracks |= temp_del_tracks
        temp_deleted_tracks = list(temp_deleted_tracks - temp_del_tracks)

        timestamps = [time.mktime(track.last_update.timestamp.timetuple())
                      for track in temp_deleted_tracks]
        ts = time.mktime((timestamp - timedelta(hours=10)).timetuple())
        t_inds = np.argwhere(np.array(timestamps) > ts).flatten()
        # del_tracks = rec_deleter.delete_tracks(temp_deleted_tracks, timestamp=timestamp)
        temp_deleted_tracks = set([temp_deleted_tracks[t_ind]
                                   for t_ind in t_inds])
        # deleted_tracks |= del_tracks

        # trackw = [track for track in tracks
        #           if track.metadata["MMSI"] == "775000000"
        #           and len([state for state in track.states
        #                    if isinstance(state,Update)])>1]
        # if len(trackw)>0:
        #     s = 2
        return tracks, temp_deleted_tracks, deleted_tracks

    # Detection readers
    # ================
    static_detector = MongoDetectionReader(host_name="138.253.118.175",
                                           host_port=27017,
                                           db_name="TA_ExactEarth",
                                           collection_name="Raw_Static_Reports",
                                           state_vector_fields=["MMSI"],
                                           time_field="ReceivedTime",
                                           # time_field_format="%Y%m%d_%H%M%S",
                                           timestamp=True)
    static_detector = TimeSyncFeeder(static_detector,
                                     time_window=timedelta(seconds=1))
    static_gen = static_detector.detections_gen()
    dynamic_detector = MongoDetectionReader(host_name="138.253.118.175",
                                            host_port=27017,
                                            db_name="TA_ExactEarth",
                                            collection_name="Raw_Position_Reports",
                                            state_vector_fields=[
                                                "Longitude",
                                                "Latitude"],
                                            time_field="ReceivedTime",
                                            # time_field_format="%Y%m%d_%H%M%S",
                                            timestamp=True)
    dynamic_detector = TimeSyncFeeder(dynamic_detector,
                                      time_window=timedelta(seconds=1))
    dynamic_detector = BoundingBoxReducer(dynamic_detector,
                                          bounding_box=np.array([LON_MIN, LON_MAX,
                                                         LAT_MIN, LAT_MAX]),
                                          mapping=[0, 1])
    dynamic_gen = dynamic_detector.detections_gen()
    # detector = TimeWindowReducer(detector)
    # detector = BoundingBoxReducer(detector,
    #                               bounding_box=np.array([LON_MIN, LON_MAX,
    #                                                      LAT_MIN, LAT_MAX]),
    #                               mapping=[0, 1])

    # Writer
    # ======
    writer = MongoWriter_EE()

    # Initialise DB collections
    collections = ["Live_SS_Tracks", "Live_SS_Points"]
    writer.reset_collections(
        host_name="138.253.118.175",
        host_port=27017,
        db_name="TA_ExactEarth",
        collection_names=["Live_SS_Tracks", "Live_SS_Points"],
    )

    ################################################################################
    # Main Tracking process                                                        #
    ################################################################################
    from stonesoup.types.sets import TrackSet

    tracks = TrackSet()  # Main set of tracks
    temp_deleted_tracks = TrackSet()  # Tentatively deleted tracks
    deleted_tracks = TrackSet()  # Fully deleted tracks

    # We process each scan sequentially
    j = 0
    last_static_time, last_static_detections = next(static_gen)
    for scan_time, detections in dynamic_gen:
        j = j+1
        # Skip iteration if there are no available detections
        if len(detections) == 0:
            continue

        print("Measurements: " + str(len(detections)))

        # Recycle tracks
        tracks, temp_deleted_tracks, deleted_tracks = \
            recycle_tracks(tracks, temp_deleted_tracks, deleted_tracks,
                           detections, scan_time)

        # Process static AIS
        while last_static_time < scan_time:
            for track in tracks:
                for detection in last_static_detections:
                    if detection.metadata["MMSI"] == track.metadata["MMSI"]:
                        track.metadata.update(detection.metadata)
            for track in temp_deleted_tracks:
                for detection in last_static_detections:
                    if detection.metadata["MMSI"] == track.metadata["MMSI"]:
                        track.metadata.update(detection.metadata)
            last_static_time, last_static_detections = next(static_gen)

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
        del_tracks = deleter.delete_tracks(tracks, timestamp=scan_time)
        # mmsis = [track.metadata["MMSI"] for track in del_tracks]
        # temp_deleted_tracks -= set([track for track in temp_deleted_tracks
        #                             if track.metadata["MMSI"] in mmsis])
        temp_deleted_tracks |= del_tracks
        tracks -= del_tracks

        # Initiate new tracks
        unassociated_detections = detections - associated_detections
        tracks |= initiator.initiate(unassociated_detections)

        # Write data
        writer.write(tracks,
                     detections,
                     host_name="138.253.118.175",
                     host_port=27017,
                     db_name="TA_ExactEarth",
                     collection_name=collections,
                     drop=False,
                     timestamp=scan_time)

        print("Time: " + scan_time.strftime('%H:%M:%S %d/%m/%Y')
              + " - Measurements: " + str(len(detections))
              + " - Main Tracks: " + str(len(tracks))
              + " - Tentative Tracks: " + str(len(temp_deleted_tracks))
              + " - Total Tracks: " + str(
            len(tracks) + len(temp_deleted_tracks)))

        # Only plot if 'f' or 'p' button is pressed
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch == b'f' or ch == b'p':
                print("[INFO]: Plotting Tracks")
                # Plot the data
                plt.clf()
                plot_map(scan_time)
                plot_data(detections)
                plot_tracks(tracks, False)
                plot_tracks(temp_deleted_tracks, False)
                if ch == b'f':
                    plt.pause(0.01)
                elif ch == b'p':
                    plt.show()
        if j >= 500:
            a = 2
        if END_TIME is not None and scan_time >= END_TIME:
            break
