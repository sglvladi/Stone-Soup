# coding: utf-8

# AisTrackingExactEarthIMM_Single.py
# ==================================
# Run a Global Nearest Neighbour with IMM Prediction & Update, for each of
# the MMSI files in the "per_mmsi" folder.
# -

################################################################################
# IMPORTS                                                                      #
################################################################################
# General imports
import glob
import os
import pickle
import numpy as np
from datetime import datetime, timedelta
import msvcrt
import time
from copy import copy
import cProfile as profile
# In outer section of code
pr = profile.Profile()
pr.disable()

# Stone-Soup imports
from stonesoup.dataassociator.tree import TPRTreeMixIn
from stonesoup.feeder.filter import BoundingBoxDetectionReducer
from stonesoup.models.transition.linear import (
    RandomWalk, OrnsteinUhlenbeck, CombinedLinearGaussianTransitionModel)
from stonesoup.initiator.simple import LinearMeasurementInitiator
from stonesoup.deleter.time import UpdateTimeDeleter
from stonesoup.dataassociator.neighbour import (
    NearestNeighbour, GlobalNearestNeighbour, GNNWith2DAssignment)
from stonesoup.hypothesiser.distance import DistanceHypothesiser, DistanceHypothesiserFast
from stonesoup.measures import Mahalanobis
from stonesoup.hypothesiser.filtered import FilteredDetectionsHypothesiser
from stonesoup.types.update import Update
from stonesoup.updater.kalman import (KalmanUpdater)
from stonesoup.predictor.kalman import (KalmanPredictor)
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.prediction import GaussianStatePrediction, Prediction
from stonesoup.types.angle import Longitude, Latitude
from stonesoup.reader.generic import CSVDetectionReader_EE
from stonesoup.feeder.time import TimeSyncFeeder, TimeBufferedFeeder

if __name__ == '__main__':
    ##############################################################################
    # TRACKING LIMIT SELECTION                                                   #
    ##############################################################################
    TARGET = "UK"
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
            "LAT_MAX": 80.,
            "RES": 'c'
        },
        "FULL": {
            "LON_MIN": -84.,
            "LON_MAX": 34.5,
            "LAT_MIN": 9.5,
            "LAT_MAX": 62.,
            "RES": 'c'
        },
        "CARIBBEAN": {
            "LON_MIN": -90.,
            "LON_MAX": -60.,
            "LAT_MIN": 10.,
            "LAT_MAX": 22.,
            "RES": 'h'
        },
        "MEDITERRANEAN": {
            "LON_MIN": -6.,
            "LON_MAX": 36.5,
            "LAT_MIN": 30.,
            "LAT_MAX": 46.,
            "RES": 'l'
        },
        "GREECE": {
            "LON_MIN": 20.,
            "LON_MAX": 28.2,
            "LAT_MIN": 34.6,
            "LAT_MAX": 41.,
            "RES": 'h'
        },
        "UK": {
            "LON_MIN": -12.,
            "LON_MAX": 3.5,
            "LAT_MIN": 48.,
            "LAT_MAX": 60.,
            "RES": 'l'
        }
    }
    LON_MIN = LIMITS[TARGET]["LON_MIN"]
    LON_MAX = LIMITS[TARGET]["LON_MAX"]
    LAT_MIN = LIMITS[TARGET]["LAT_MIN"]
    LAT_MAX = LIMITS[TARGET]["LAT_MAX"]
    RES = LIMITS[TARGET]["RES"]
    END_TIME = None  # datetime(2017, 8, 10, 0, 48,13)
    LOAD = False
    BACKUP = False
    LOAD_OFFSET = 3
    LIMIT_STATES = False
    SHOW_MAP = True

    ################################################################################
    # Plotting functions                                                           #
    ################################################################################

    from matplotlib.patches import Ellipse
    from matplotlib import pyplot as plt
    from mpl_toolkits.basemap import Basemap

    plt.rcParams['figure.figsize'] = (12, 8)
    plt.style.use('seaborn-colorblind')
    fig = plt.figure()


    def plot_map(timestamp):
        # Mercator projection map
        m = Basemap(llcrnrlon=LON_MIN, llcrnrlat=LAT_MIN, urcrnrlon=LON_MAX,
                    urcrnrlat=LAT_MAX, projection='merc', resolution=RES)
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


    def plot_cov_ellipse(cov, pos, nstd=40, ax=None, **kwargs):
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


    def plot_tracks(tracks, show_mmsis=False, show_probs=False,
                    show_error=True, show_map=False):
        # Mercator projection map
        if show_map:
            m = Basemap(llcrnrlon=LON_MIN, llcrnrlat=LAT_MIN, urcrnrlon=LON_MAX,
                        urcrnrlat=LAT_MAX, projection='merc', resolution='c')
        if show_mmsis:
            mmsis = list({track.metadata["MMSI"] for track in tracks})
        for track in tracks:
            states = [state.state_vector for state in track.states]
            if len(states) == 0:
                continue
            data = np.array(states)
            lat = data[:, 2]
            lon = data[:, 0]
            if show_map:
                x, y = m(lon, lat)
                m.plot(x, y, 'b-o', linewidth=1, markersize=1)
                m.plot(x[-1], y[-1], 'ro', markersize=1)
                # plt.text(x[-1], y[-1], track.metadata["Vessel_Name"], fontsize=12)
                if show_probs:
                    plt.text(x[-1], y[-1],
                             np.around(track.last_update.weights[0, 0], 2),
                             fontsize=6)
                elif show_mmsis:
                    ind = mmsis.index(track.metadata["MMSI"])
                    plt.text(x[-1], y[-1],
                             str(ind),
                             fontsize=6)
            else:
                plt.plot(data[:, 0], data[:, 2], '-', label="AIS Tracks")
                # if show_error:
                plot_cov_ellipse(track.state.covar[[0, 2], :][:, [0, 2]],
                                 track.state.mean[[0, 2], :], edgecolor='r',
                                 facecolor='none')

        shared_mmsi = dict()
        for track in tracks:
            for track2 in tracks:
                if track.id != track2.id and track.metadata["MMSI"] == track2.metadata["MMSI"]:
                    mmsi = track.metadata["MMSI"]
                    if mmsi in shared_mmsi:
                        if track not in shared_mmsi[mmsi]:
                            shared_mmsi[mmsi].push(track)
                        if track2 not in shared_mmsi[mmsi]:
                            shared_mmsi[mmsi].push(track2)
                    else:
                        shared_mmsi[mmsi] = [track, track2]

        for mmsi in shared_mmsi:
            states = [track.state.state_vector for track in shared_mmsi[mmsi]]
            data = np.array(states)
            lat = data[:, 2]
            lon = data[:, 0]
            if show_map:
                x, y = m(lon, lat)
                m.plot(x, y, 'y-o', linewidth=0.5, markersize=1)
            else:
                plt.plot(data[:, 0], data[:, 2], '-', label="AIS Tracks")

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
        (OrnsteinUhlenbeck(0.00001 ** 2, 2e-3),
         OrnsteinUhlenbeck(0.00001 ** 2, 2e-3)))
    measurement_model = LinearGaussian(ndim_state=4, mapping=[0, 2],
                                       noise_covar=np.diag([0.001 ** 2,
                                                            0.001 ** 2]))

    # Predictor & Updater
    # ===================
    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)

    # Hypothesiser & Data Associator
    # ==============================
    hypothesiser = DistanceHypothesiserFast(predictor, updater, Mahalanobis(), 20)
    hypothesiser = FilteredDetectionsHypothesiser(hypothesiser, 'MMSI',
                                                  match_missing=False)


    class TPRGNN(GNNWith2DAssignment, TPRTreeMixIn):
        pass


    associator = TPRGNN(hypothesiser, measurement_model, timedelta(hours=24), [1, 3])

    # Track Initiator
    # ===============
    state_vector = StateVector([[Longitude(0)], [0], [Latitude(0)], [0]])
    covar = CovarianceMatrix(np.diag([0.0001 ** 2, 0.0003 ** 2,
                                      0.0001 ** 2, 0.0003 ** 2]))
    prior_state = GaussianStatePrediction(state_vector, covar)
    initiator = LinearMeasurementInitiator(prior_state, measurement_model)

    # Track Deleter
    # =============
    deleter = UpdateTimeDeleter(time_since_update=timedelta(hours=5))

    ################################################################################
    # Main Tracking process                                                        #
    ################################################################################

    tracks = set()  # Main set of tracks
    for file_path in glob.iglob(
            os.path.join('C:/Users/sglvladi/Documents/TrackAnalytics/data'
                         '/exact_earth/id_sorted/', r'*.csv')):
        print(file_path)
        # Detection readers
        # ================
        detector = CSVDetectionReader_EE(
            path=file_path,
            state_vector_fields=["Longitude", "Latitude"],
            time_field="Time",
            time_field_format="%Y%m%d_%H%M%S")
        detector = TimeBufferedFeeder(detector)
        detector = TimeSyncFeeder(detector,
                                  time_window=timedelta(seconds=1))

        # Writer
        # ======
        filename = os.path.basename(file_path)
        # writer = CSV_Writer_EE_MMSI(
        #     'C:/Users/sglvladi/Documents/TrackAnalytics/output/exact_earth'
        #     '/per_mmsi', mmsi)

        # We process each scan sequentially
        date = datetime.now().date()
        # last_static_time, last_static_detections = next(static_gen)

        for scan_time, detections in detector.detections_gen():

            # Skip iteration if there are no available detections
            if len(detections) == 0:
                continue
            static_fields = ['Vessel_Name', 'Call_sign', 'IMO', 'Ship_Type', 'Dimension_to_Bow', 'Dimension_to_stern',
                             'Dimension_to_port', 'Dimension_to_starboard', 'Draught', 'Destination', 'AIS_version',
                             'Fixing_device', 'ETA_month', 'ETA_day', 'ETA_hour', 'ETA_minute', 'Data_terminal']
            dynamic_detections = [detection for detection in detections
                                  if detection.metadata["type"] == "dynamic"]
            for detection in dynamic_detections:
                for field in static_fields:
                    del detection.metadata[field]
            static_detections = [detection for detection in detections
                                 if detection.metadata["type"] == "static"]
            detections = set(dynamic_detections)

            # BBox reducer low patch
            limits = np.array([[LON_MIN, LON_MAX],
                               [LAT_MIN, LAT_MAX]])
            num_dims = len(limits)
            mapping = [0, 1]
            outlier_detections = set()
            for detection in detections:
                state_vector = detection.state_vector
                for i in range(num_dims):
                    min = limits[i][0]
                    max = limits[i][1]
                    value = state_vector[mapping[i]]
                    if value < min or value > max:
                        outlier_detections.add(detection)
                        break
            detections -= outlier_detections

            # Remove potential duplicate detections
            # This has been added to solve the problem where the same ship may broadcast
            # its position more than once within the set time-window. If not dealt with
            # this will result in new false tracks being spawned
            unique_detections = set([])
            for detection in detections:
                unique = True
                for detection_2 in unique_detections:
                    if (detection.metadata["MMSI"] == detection_2.metadata["MMSI"]
                            and np.allclose(np.array(detection.state_vector, dtype=np.float32),
                                            np.array(detection_2.state_vector, dtype=np.float32),
                                            atol=0.005)):
                        unique = False
                if unique:
                    unique_detections.add(detection)
            detections = unique_detections

            print("Measurements: " + str(len(detections)) + " - Time: " + scan_time.strftime('%H:%M:%S %d/%m/%Y'))

            # Process static AIS
            for track in tracks:
                for detection in static_detections:
                    if detection.metadata["MMSI"] == track.metadata["MMSI"]:
                        static_fields = ['Vessel_Name', 'Ship_Type',
                                         'Destination', 'IMO']
                        track._metadata.update({x: detection.metadata[x] for
                                                x in static_fields})

            # Perform data association
            print("Tracking.... NumTracks: {}".format(str(len(tracks))))
            associated_detections = set()
            if len(detections) > 0:
                pr.enable()
                associations = associator.associate(tracks, detections, scan_time)
                pr.disable()

                # Update tracks based on association hypotheses
                print("Updating...")
                associated_detections = set()
                for track, hypothesis in associations.items():
                    if hypothesis:
                        state_post = updater.update(hypothesis)
                        track.append(state_post)
                        associated_detections.add(hypothesis.measurement)
                    else:
                        # Only append new prediction if previous is an Update
                        # or is a Prediction with different timestamp
                        if (isinstance(track.state, Update)
                                or (isinstance(track.state, Prediction)
                                    and track.state.timestamp != hypothesis.prediction.timestamp)):
                            track.append(hypothesis.prediction)
                    if LIMIT_STATES and len(track.states) > 10:
                        track.states = track.states[-10:]

            # Write data
            # print("Writing....")
            # writer.write(tracks,
            #              detections,
            #              timestamp=scan_time)

            # Initiate new tracks
            print("Track Initiation...")
            unassociated_detections = detections - associated_detections
            new_tracks = initiator.initiate(unassociated_detections)
            tracks |= new_tracks

            # Delete invalid tracks
            print("Track Deletion...")
            del_tracks = deleter.delete_tracks(tracks, timestamp=scan_time)
            tracks -= del_tracks

            print("{}".format(filename)
                  + " - Time: " + scan_time.strftime('%H:%M:%S %d/%m/%Y')
                  + " - Measurements: " + str(len(detections))
                  + " - Tracks: " + str(len(tracks)))

            # Only plot if 'f' or 'p' button is pressed
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch == b'f' or ch == b'p' or ch == b'm':
                    print("[INFO]: Plotting Tracks")
                    # Plot the data
                    plt.clf()
                    if SHOW_MAP:
                        plot_map(scan_time)
                    plot_data(detections)
                    if ch == b'f':
                        plot_tracks(tracks, show_probs=False, show_map=SHOW_MAP)
                        plt.pause(0.01)
                    elif ch == b'p':
                        plot_tracks(tracks, show_probs=False, show_map=SHOW_MAP)
                        plt.show()
                    elif ch == b'm':
                        t_tracks = tracks
                        plot_tracks(t_tracks, show_mmsis=False, show_map=SHOW_MAP)
                        plt.show()
                elif ch == b'r':
                    print("[INFO]: Dumping Profiler stats")
                    pr.dump_stats('profile_{}.pstat'.format(scan_time.strftime('%H-%M-%S_%d-%m-%Y')))

            if END_TIME is not None and scan_time >= END_TIME:
                break
