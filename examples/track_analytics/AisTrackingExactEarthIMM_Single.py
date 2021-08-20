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

# Stone-Soup imports
from stonesoup.models.transition.linear import (
    RandomWalk, OrnsteinUhlenbeck, CombinedLinearGaussianTransitionModel)
from stonesoup.initiator.simple import LinearMeasurementInitiatorMixture
from stonesoup.deleter.time import UpdateTimeDeleter
from stonesoup.dataassociator.neighbour import (GlobalNearestNeighbour, GNNWith2DAssignment)
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.gater.filtered import FilteredDetectionsGater
from stonesoup.updater.kalman import (KalmanUpdater,
                                      IMMUpdater)
from stonesoup.predictor.kalman import (KalmanPredictor,
                                        IMMPredictor)
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.writer.csv import CSV_Writer_EE_MMSI
from stonesoup.reader.generic import CSVDetectionReader_EE
from stonesoup.feeder.time import TimeSyncFeeder

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
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

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
                #plt.text(x[-1], y[-1], track.metadata["Vessel_Name"], fontsize=12)
                if show_probs:
                    plt.text(x[-1], y[-1],
                             np.around(track.last_update.weights[0,0], 2),
                             fontsize=6)
                elif show_mmsis:
                    ind = mmsis.index(track.metadata["MMSI"])
                    plt.text(x[-1], y[-1],
                             str(ind),
                             fontsize=6)
            else:
                plt.plot(data[:, 0], data[:, 2], '-', label="AIS Tracks")
                #if show_error:
                plot_cov_ellipse(track.state.covar[[0, 2], :][:, [0, 2]],
                                 track.state.mean[[0, 2], :], edgecolor='r',
                                 facecolor='none')


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
    transition_model2 = CombinedLinearGaussianTransitionModel(
        (RandomWalk(0.00001 ** 2),
         RandomWalk(np.finfo(float).eps),
         RandomWalk(0.00001 ** 2),
         RandomWalk(np.finfo(float).eps)))
    measurement_model = LinearGaussian(ndim_state=4, mapping=[0, 2],
                                       noise_covar=np.diag([0.001 ** 2,
                                                            0.001 ** 2]))

    # Predictor & Updater
    # ===================
    p = np.array([[0.8, 0.2],
                  [0.2, 0.8]])
    predictor1 = KalmanPredictor(transition_model)
    predictor2 = KalmanPredictor(transition_model2)
    predictor = IMMPredictor([predictor1, predictor2], p)
    updater1 = KalmanUpdater(measurement_model)
    updater2 = KalmanUpdater(measurement_model)
    updater = IMMUpdater([updater1, updater2], p)

    # Hypothesiser & Data Associator
    # ==============================
    hypothesiser = DistanceHypothesiser(predictor, updater, Mahalanobis(), 20)
    hypothesiser = FilteredDetectionsGater(hypothesiser, 'MMSI',
                                                  match_missing=False)
    # associator = GlobalNearestNeighbour(hypothesiser)
    associator = GNNWith2DAssignment(hypothesiser)

    # Track Initiator
    # ===============
    state_vector = StateVector([[0], [0], [0], [0]])
    covar = CovarianceMatrix(np.diag([0.0001 ** 2, 0.0003 ** 2,
                                      0.0001 ** 2, 0.0003 ** 2]))
    prior_state = GaussianStatePrediction(state_vector, covar)
    initiator = LinearMeasurementInitiatorMixture(prior_state,
                                                  measurement_model)

    # Track Deleter
    # =============
    deleter = UpdateTimeDeleter(time_since_update=timedelta(hours=24))

    ################################################################################
    # Main Tracking process                                                        #
    ################################################################################

    # Possible parallelisation point
    #   - e.g. Run each file in a separate thread
    for file_path in glob.iglob(
            os.path.join(r'C:\Users\sglvladi\OneDrive\Workspace\PostDoc\CADMURI\Python\Stone-Soup\examples\track_analytics\data\exact_earth\mmsi', r'*.csv')):

        # Detection readers
        # ================
        detector = CSVDetectionReader_EE(
            path=file_path,
            state_vector_fields=["Longitude", "Latitude"],
            time_field="Time",
            time_field_format="%Y%m%d_%H%M%S")
        detector = TimeSyncFeeder(detector,
                                  time_window=timedelta(seconds=1))

        # Writer
        # ======
        filename = os.path.basename(file_path)
        mmsi = filename.replace(".csv","")
        writer = CSV_Writer_EE_MMSI(
            r'C:\Users\sglvladi\OneDrive\Workspace\PostDoc\CADMURI\Python\Stone-Soup\examples\track_analytics\output\exact_earth\per_mmsi', mmsi)
        collections = ["Live_SS_Tracks", "Live_SS_Points"]

        tracks = set()  # Main set of tracks

        # We process each scan sequentially
        date = datetime.now().date()
        # last_static_time, last_static_detections = next(static_gen)

        for scan_time, detections in detector.detections_gen():

            # Skip iteration if there are no available detections
            if len(detections) == 0:
                continue

            dynamic_detections = [detection for detection in detections
                                  if detection.metadata["type"] == "dynamic"]
            static_detections = [detection for detection in detections
                                 if detection.metadata["type"] == "static"]
            detections = set(dynamic_detections)

            unique_detections = set([])
            for detection in detections:
                unique = True
                for detection_2 in unique_detections:
                    if (detection.metadata["MMSI"]==detection_2.metadata["MMSI"]
                        and np.allclose(detection.state_vector,
                                        detection_2.state_vector,
                                        atol=0.005)):
                        unique = False
                if unique:
                    unique_detections.add(detection)
            detections = unique_detections

            print("Measurements: " + str(len(detections)))

            # Process static AIS
            for track in tracks:
                for detection in static_detections:
                    if detection.metadata["MMSI"] == track.metadata["MMSI"]:
                        static_fields = ['Vessel_Name', 'Ship_Type',
                                         'Destination', 'IMO']
                        track.metadata.update({x: detection.metadata[x] for x in static_fields})

            # Perform data association
            print("Tracking.... NumTracks: {}".format(str(len(tracks))))
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
                if LIMIT_STATES and len(track.states) > 10:
                    track.states = track.states[-10:]

            # Write data
            # print("Writing....")
            # writer.write(tracks,
            #              detections,
            #              timestamp=scan_time)

            # Initiate new tracks
            unassociated_detections = detections - associated_detections
            new_tracks = initiator.initiate(unassociated_detections)
            tracks |= new_tracks

            # Delete invalid tracks
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
                        plot_tracks(tracks, show_probs=True, show_map=SHOW_MAP)
                        plt.pause(0.01)
                    elif ch == b'p':
                        plot_tracks(tracks, show_probs=True, show_map=SHOW_MAP)
                        plt.show()
                    elif ch == b'm':
                        t_tracks = tracks
                        plot_tracks(t_tracks, show_mmsis=True, show_map=SHOW_MAP)
                        plt.show()

            if END_TIME is not None and scan_time >= END_TIME:
                break
