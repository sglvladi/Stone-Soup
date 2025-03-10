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

pr = profile.Profile()
pr.disable()

# Stone-Soup imports
from stonesoup.dataassociator.tree import TPRTreeMixIn
from stonesoup.deleter.elint import ELINTDeleter
from stonesoup.feeder.filter import BoundingBoxDetectionReducer
from stonesoup.hypothesiser.probability import ELINTHypothesiser, ELINTHypothesiserFast
from stonesoup.initiator.elint import ELINTInitiator
from stonesoup.models.transition.linear import (
    RandomWalk, OrnsteinUhlenbeck, CombinedLinearGaussianTransitionModel)
# from stonesoup.initiator.simple import LinearMeasurementInitiator
from stonesoup.deleter.time import UpdateTimeDeleter
from stonesoup.dataassociator.neighbour import (
    NearestNeighbour, GlobalNearestNeighbour, GNNWith2DAssignment)
from stonesoup.hypothesiser.distance import DistanceHypothesiser, DistanceHypothesiserFast
from stonesoup.measures import Mahalanobis
from stonesoup.hypothesiser.filtered import FilteredDetectionsHypothesiser
from stonesoup.updater.kalman import (KalmanUpdater)
from stonesoup.predictor.kalman import (KalmanPredictor)
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.update import StateUpdate, GaussianStateUpdate
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.types.angle import Bearing
from stonesoup.reader.generic import CSVDetectionReader_EE
from stonesoup.reader.elint import BasicELINTDetectionReader
from stonesoup.feeder.time import TimeSyncFeeder, TimeBufferedFeeder

if __name__ == '__main__':
    ##############################################################################
    # TRACKING LIMIT SELECTION                                                   #
    ##############################################################################
    TARGET = "GLOBAL"
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
    # plt.style.use('seaborn-colorblind')
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
            lat = np.array(data[:, 2], dtype=np.float32)
            lon = np.array(data[:, 0], dtype=np.float32)
            if show_map:
                x, y = m(lon, lat)
                m.plot(x, y, '-o', linewidth=1, markersize=1)
                m.plot(x[-1], y[-1], 'ro', markersize=1)
                # print([lon[-1], lat[-1]])
                # print([x[-1], y[-1]])
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
        m = Basemap(llcrnrlon=LON_MIN, llcrnrlat=LAT_MIN, urcrnrlon=LON_MAX,
                    urcrnrlat=LAT_MAX, projection='merc', resolution='c')
        if len(detections) > 0:
            lon = [s.state_vector[0] for s in detections]
            lat = [s.state_vector[1] for s in detections]
            x, y = m(lon, lat)
            m.plot(x, y, linestyle='', marker='x', markersize=4)


    ##########################################################################
    # Tracking components                                                    #
    ##########################################################################

    mins2sec = 60
    hours2sec = 60 * mins2sec
    days2sec = 24 * hours2sec
    rates = {
        "birth": 1 / (10 * hours2sec),
        "death": 1 / (10 * days2sec),
        "killProbThresh": 0.1
    }
    sensors = {
        "ELINT": {
            "rates": {
            }
        }
    }
    sensors["ELINT"]["rates"]["meas"] = 1/(60*mins2sec)
    sensors["ELINT"]["rates"]["firstmeas"] = (rates["birth"]*sensors["ELINT"]["rates"]["meas"] /
                                              (rates["death"] + sensors["ELINT"]["rates"]["meas"]))

    prior = {
        "lonlat_min": [108, 5],
        "lonlat_max": [119, 16],
        "lonlat_mean": StateVector([[113.5], [10.5]]),
        "lonlat_cov": CovarianceMatrix([[10.0833, 0], [0, 10.083]]),
        "initspeed_sd_metres": 10
    }
    # Transition & Measurement models
    # ===============================
    transition_model = CombinedLinearGaussianTransitionModel(
        (OrnsteinUhlenbeck(5e-11, 5e-4),
         OrnsteinUhlenbeck(1e-11, 5e-4)))
    measurement_model = LinearGaussian(ndim_state=4, mapping=[0, 2],
                                       noise_covar=np.diag([0.001 ** 2,
                                                            0.001 ** 2]))

    # Predictor & Updater
    # ===================
    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)

    # Hypothesiser & Data Associator
    # ==============================
    logNullLogLikelihood = -4.795790545596741 + np.log(sensors["ELINT"]["rates"]["firstmeas"])
    hypothesiser = ELINTHypothesiser(predictor, updater,
                                     sensors["ELINT"]["rates"]["meas"],
                                     rates["death"], logNullLogLikelihood)
    # hypothesiser = DistanceHypothesiser(predictor, updater, Mahalanobis(), 100)

    class TPRGNN(GNNWith2DAssignment, TPRTreeMixIn):
        pass
    associator = TPRGNN(hypothesiser, measurement_model, timedelta(hours=24), [1, 3], std_thresh=300)
    # associator = GNNWith2DAssignment(hypothesiser)

    # Track Initiator
    # ===============
    state_vector = StateVector([[113.5], [0], [10.5], [0]])
    covar = CovarianceMatrix(np.diag([10.0833, 3e-8,
                                      3e-8, 10.0833]))
    prior_state = GaussianStatePrediction(state_vector, covar)
    initiator = ELINTInitiator(prior, measurement_model)
    # initiator = LinearMeasurementInitiator(prior_state, measurement_model)

    # Track Deleter
    # =============
    # deleter = UpdateTimeDeleter(time_since_update=timedelta(hours=2))
    deleter = ELINTDeleter(rates["killProbThresh"])

    ################################################################################
    # Main Tracking process                                                        #
    ################################################################################

    tracks = set()  # Main set of tracks
    file_path = r"C:\Users\sglvladi\OneDrive\Workspace\PostDoc\CADMURI\MATLAB\ELINT\smalldata20tracks_elint_LV.mat"
    detector = BasicELINTDetectionReader(path=file_path, )


    # Writer
    # ======
    filename = os.path.basename(file_path)
    # writer = CSV_Writer_EE_MMSI(
    #     'C:/Users/sglvladi/Documents/TrackAnalytics/output/exact_earth'
    #     '/per_mmsi', mmsi)

    # We process each scan sequentially
    date = datetime.now().date()
    # last_static_time, last_static_detections = next(static_gen)
    alldetections = set()
    prev_time = None
    for scan_time, detections in detector.detections_gen():
        if prev_time is None:
            prev_time = scan_time

        alldetections |= detections
        # Skip iteration if there are no available detections

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

        print("Measurements: " + str(len(detections))+" - Time: " + scan_time.strftime('%H:%M:%S %d/%m/%Y'))

        # Perform data association
        print("Tracking.... NumTracks: {}".format(str(len(tracks))))

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
                track.append(hypothesis.prediction)
            if LIMIT_STATES and len(track.states) > 10:
                track.states = track.states[-10:]

        for track in tracks:
            if isinstance(track.state, GaussianStateUpdate) and track.state.timestamp==scan_time:
                # since you've assumed that the target definitely generated the measurement
                track.metadata["existence"]["value"] = 1
            else:
                pe = track.metadata["existence"]["value"]
                dt = scan_time-prev_time
                logpnotdetect = -sensors["ELINT"]["rates"]["meas"]*dt.total_seconds()
                pnotdetgivenexist = np.exp(logpnotdetect)
                pnotdet = pnotdetgivenexist * pe + (1 - pe)
                track.metadata["existence"]["value"] = pe * pnotdetgivenexist / pnotdet
        prev_time = scan_time
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

        if scan_time.timestamp() == 1502400890.0:
            plot_map(scan_time)
            plot_data(alldetections)
            plot_tracks(tracks, show_probs=False, show_map=SHOW_MAP)
            plt.show()
            a=2
        # Only plot if 'f' or 'p' button is pressed
        # plt.clf()
        # plot_map(scan_time)
        # plot_tracks(tracks, show_probs=False, show_map=SHOW_MAP)
        # plt.pause(0.01)
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch == b'f' or ch == b'p' or ch == b'm':
                print("[INFO]: Plotting Tracks")
                # Plot the data
                plt.clf()
                if SHOW_MAP:
                    plot_map(scan_time)
                plot_data(alldetections)
                if ch == b'f':
                    plot_tracks(tracks, show_probs=False, show_map=SHOW_MAP)
                    plt.pause(0.01)
                elif ch == b'p':
                    plot_tracks(tracks, show_probs=False, show_map=SHOW_MAP)
                    # plot_data(detections)
                    plt.show()
                elif ch == b'm':
                    t_tracks = tracks
                    plot_tracks(t_tracks, show_mmsis=False, show_map=SHOW_MAP)
                    plt.show()
            elif ch == b'r':
                print("[INFO]: Dumping Profiler stats")
                pr.dump_stats('profile_ELINT_{}.pstat'.format(scan_time.strftime('%H-%M-%S_%d-%m-%Y')))

        if END_TIME is not None and scan_time >= END_TIME:
            break

print(scan_time.timestamp())