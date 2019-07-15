# coding: utf-8

################################################################################
# IMPORTS                                                                      #
################################################################################
# 22:24
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
    ConstantVelocity, OrnsteinUhlenbeck, CombinedLinearGaussianTransitionModel)
from stonesoup.initiator.simple import SinglePointInitiator, LinearMeasurementInitiator
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.deleter.time import UpdateTimeStepsDeleter, UpdateTimeDeleter
from stonesoup.dataassociator.neighbour import (
    NearestNeighbour, GlobalNearestNeighbour)
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.hypothesiser.filtered import FilteredDetectionsHypothesiser
from stonesoup.updater.kalman import ( KalmanUpdater,
    UnscentedKalmanUpdater, ExtendedKalmanUpdater)
from stonesoup.predictor.kalman import ( KalmanPredictor,
    UnscentedKalmanPredictor, ExtendedKalmanPredictor)
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.types.update import Update
from stonesoup.types.state import GaussianState
from stonesoup.writer.mongo import MongoWriter
from stonesoup.reader.generic import CSVDetectionReader
from stonesoup.feeder.time import TimeSyncFeeder, TimeWindowReducer
from stonesoup.feeder.filter import BoundingBoxReducer

if __name__ == '__main__':
    ##############################################################################
    # TRACKING LIMIT SELECTION                                                   #
    ##############################################################################
    TARGET = "FULL"
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
    END_TIME = None #datetime(2017, 8, 10, 0, 48,13)
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
                    urcrnrlat=LAT_MAX, projection='merc', resolution='c')
        m.drawcoastlines()
        m.drawcountries()
        m.drawmapboundary(fill_color='#99ffff')
        m.fillcontinents(color='#cc9966', lake_color='#99ffff')
        m.drawparallels(np.arange(-90., 91., 20.), labels=[1,1,0,0])
        m.drawmeridians(np.arange(-180., 181., 20.),labels=[0,0,0,1])
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
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

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
            # plt.text(x[-1], y[-1], track.metadata["MMSI"], fontsize=12)
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


    ################################################################################
    # Tracking components                                                          #
    ################################################################################

    # Transition & Measurement models
    # ===============================
    transition_model = CombinedLinearGaussianTransitionModel(
                        (OrnsteinUhlenbeck(0.00001 ** 2, 2e-2),
                         OrnsteinUhlenbeck(0.00001 ** 2, 2e-2)))
    measurement_model = LinearGaussian(ndim_state=4, mapping=[0, 2],
                                       noise_covar=np.diag([0.001 ** 2,
                                                            0.001 ** 2]))

    # Predictor & Updater
    # ===================
    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)

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
    initiator = LinearMeasurementInitiator(prior_state, measurement_model)

    # Track Deleter
    # =============
    #deleter = UpdateTimeStepsDeleter(time_steps_since_update=2)
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
                    temp_del_tracks.add(track) # Remove from temp deleted
                    # unmatched_detections.remove(detection)
                    break

        # Delete permanently
        tracks |= temp_del_tracks
        temp_deleted_tracks = list(temp_deleted_tracks-temp_del_tracks)

        timestamps = [time.mktime(track.last_update.timestamp.timetuple())
                      for track in temp_deleted_tracks]
        ts = time.mktime((timestamp - timedelta(hours=10)).timetuple())
        t_inds = np.argwhere(np.array(timestamps) > ts).flatten()
        # del_tracks = rec_deleter.delete_tracks(temp_deleted_tracks, timestamp=timestamp)
        temp_deleted_tracks = set([temp_deleted_tracks[t_ind]
                                   for t_ind in t_inds])
        #deleted_tracks |= del_tracks

        # trackw = [track for track in tracks
        #           if track.metadata["MMSI"] == "775000000"
        #           and len([state for state in track.states
        #                    if isinstance(state,Update)])>1]
        # if len(trackw)>0:
        #     s = 2
        return tracks, temp_deleted_tracks, deleted_tracks

    # Writer
    # ======
    writer = MongoWriter()

    # Initialise DB collections
    # collections = ["Live_SS_Tracks", "Live_SS_Points"]
    # writer.reset_collections(
    #     host_name="138.253.118.175",
    #     host_port=27017,
    #     db_name="TA_IHS",
    #     collection_names=["Live_SS_Tracks", "Live_SS_Points"],
    # )

    ################################################################################
    # Main Tracking process                                                        #
    ################################################################################
    from stonesoup.types.sets import TrackSet

    filenames = [
                 'exactEarth_historical_data_2017-08-10',
                 'exactEarth_historical_data_2017-08-11',
                 'exactEarth_historical_data_2017-08-12',
                 'exactEarth_historical_data_2017-08-13',
                 'exactEarth_historical_data_2017-08-14',
                 'exactEarth_historical_data_2017-08-15',
                 'exactEarth_historical_data_2017-08-16',
                 'exactEarth_historical_data_2017-08-17',
                 'exactEarth_historical_data_2017-08-18',
                 'exactEarth_historical_data_2017-08-19',
                 'exactEarth_historical_data_2017-08-20',
                 'exactEarth_historical_data_2017-08-21',
                 'exactEarth_historical_data_2017-08-22',
                 'exactEarth_historical_data_2017-08-23',
                 'exactEarth_historical_data_2017-08-24',
                 'exactEarth_historical_data_2017-08-25',
                 'exactEarth_historical_data_2017-08-26',
                 'exactEarth_historical_data_2017-08-27',
                 'exactEarth_historical_data_2017-08-28',
                 'exactEarth_historical_data_2017-08-29',
                 'exactEarth_historical_data_2017-08-30',
                 'exactEarth_historical_data_2017-08-31',
                 'exactEarth_historical_data_2017-09-01',
                 'exactEarth_historical_data_2017-09-02',
                 'exactEarth_historical_data_2017-09-03',
                 'exactEarth_historical_data_2017-09-04',
                 'exactEarth_historical_data_2017-09-05',
                 'exactEarth_historical_data_2017-09-06',
                 'exactEarth_historical_data_2017-09-07',
                 'exactEarth_historical_data_2017-09-08',
                 'exactEarth_historical_data_2017-09-09',
                 'exactEarth_historical_data_2017-09-10']
    tracks = TrackSet()  # Main set of tracks
    temp_deleted_tracks = TrackSet()   # Tentatively deleted tracks
    deleted_tracks = TrackSet()   # Fully deleted tracks

    # We process the files sequentially
    for i, filename in enumerate(filenames):
        if LOAD and i < LOAD_OFFSET:
            continue
        if i > 0 and LOAD:
            write_dir = 'output/exact_earth/'
            with open(os.path.join(write_dir, '{}_tracks.pickle'.format(
                    filenames[i - 1])), 'rb') as f:
                data = pickle.load(f)
                tracks = data["tracks"]
                temp_deleted_tracks = data["temp_deleted_tracks"]
                deleted_tracks = TrackSet()
                # deleted_tracks = data["deleted_tracks"]
        # Detection reader
        # ================
        # (Needs to be re-instantiated for each file)
        detector = CSVDetectionReader(
            path=os.path.join(os.getcwd(),
                              r'data\exact_earth\id\{}_id.csv'.format(filename)),
            state_vector_fields=["Longitude", "Latitude"],
            time_field="Time",
            time_field_format="%Y%m%d_%H%M%S",
            metadata_fields = ['ID', 'MMSI', 'Message_ID', 'Repeat_indicator',
                               'Time', 'Millisecond', 'Group_code', 'Channel',
                               'Data_length', 'Navigational_status', 'ROT',
                               'SOG', 'Accuracy', 'Longitude', 'Latitude',
                               'COG', 'Heading', 'Maneuver', 'RAIM_flag'])
        detector = TimeSyncFeeder(detector,
                                  time_window=timedelta(seconds=1))
        detector = TimeWindowReducer(detector)
        detector = BoundingBoxReducer(detector,
                                      bounding_box=np.array([LON_MIN, LON_MAX,
                                                             LAT_MIN, LAT_MAX]),
                                      mapping=[0, 1])

        # We process each scan sequentially
        j = 0
        for scan_time, detections in detector.detections_gen():

            # Skip iteration if there are no available detections
            if len(detections) == 0:
                continue

            print("Measurements: " + str(len(detections)))

            # Recycle tracks
            tracks, temp_deleted_tracks, deleted_tracks = \
                recycle_tracks(tracks, temp_deleted_tracks, deleted_tracks,
                               detections, scan_time)

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
            del_tracks = deleter.delete_tracks(tracks, timestamp = scan_time)
            mmsis =[track.metadata["MMSI"]for track in del_tracks]
            temp_deleted_tracks -= set([track for track in temp_deleted_tracks
                                        if track.metadata["MMSI"] in mmsis])
            temp_deleted_tracks |= del_tracks
            tracks -= del_tracks

            # Initiate new tracks
            unassociated_detections = detections - associated_detections
            tracks |= initiator.initiate(unassociated_detections)

            # Write data
            # writer.write(tracks,
            #              detections,
            #              host_name="138.253.118.175",
            #              host_port=27017,
            #              db_name="TA_IHS",
            #              collection_name=collections,
            #              drop=False)

            print("Time: " + scan_time.strftime('%H:%M:%S %d/%m/%Y')
                  + " - Measurements: " + str(len(detections))
                  + " - Main Tracks: " + str(len(tracks))
                  + " - Tentative Tracks: " + str(len(temp_deleted_tracks))
                  + " - Total Tracks: " + str(len(tracks)+len(temp_deleted_tracks)))

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

            if END_TIME is not None and scan_time>=END_TIME:
                break

        # Back-up data
        if BACKUP:
            write_dir = 'output/exact_earth/'
            os.makedirs(write_dir, exist_ok=True)
            data = {"tracks": tracks,
                    "temp_deleted_tracks": temp_deleted_tracks,
                    "deleted_tracks": deleted_tracks}
            with open(os.path.join(write_dir,'{}_tracks.pickle'.format(
                    filename)), 'wb') as f:
                pickle.dump(data, f)
