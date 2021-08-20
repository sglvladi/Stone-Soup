# coding: utf-8

################################################################################
# IMPORTS                                                                      #
################################################################################
# General imports
import os
import numpy as np
from datetime import timedelta
import glob
# In outer section of code


# Stone-Soup imports
from stonesoup.models.transition.linear import RandomWalk, OrnsteinUhlenbeck, CombinedLinearGaussianTransitionModel
from stonesoup.initiator.simple import LinearMeasurementInitiatorMixture
from stonesoup.initiator.wrapper import StatesLengthLimiter
from stonesoup.deleter.time import UpdateTimeDeleter
from stonesoup.dataassociator.tree import LongLatTPRTreeMixIn
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.gater.filtered import FilteredDetectionsGater
from stonesoup.updater.kalman import KalmanUpdater, IMMUpdater
from stonesoup.predictor.kalman import KalmanPredictor, IMMPredictor
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.prediction import GaussianStatePrediction, Prediction
from stonesoup.types.update import Update
from stonesoup.types.angle import Longitude, Latitude
from stonesoup.reader.generic import CSVDetectionReader_EE
from stonesoup.writer.mongo import MongoWriter_EE
from stonesoup.reader.mongo import MongoDetectionReader
from stonesoup.feeder.time import TimeSyncFeeder, TimeBufferedFeeder

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
            "RES": 'h'
        }
    }
    LON_MIN = LIMITS[TARGET]["LON_MIN"]
    LON_MAX = LIMITS[TARGET]["LON_MAX"]
    LAT_MIN = LIMITS[TARGET]["LAT_MIN"]
    LAT_MAX = LIMITS[TARGET]["LAT_MAX"]
    lims = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]
    RES = LIMITS[TARGET]["RES"]

    ################################################################################
    # Tracking components                                                          #
    ################################################################################

    # Transition & Measurement models
    # ===============================
    transition_model = CombinedLinearGaussianTransitionModel(
        (OrnsteinUhlenbeck(np.deg2rad(0.00001 ** 2), 2e-3),
         OrnsteinUhlenbeck(np.deg2rad(0.00001 ** 2), 2e-3)))
    transition_model2 = CombinedLinearGaussianTransitionModel(
        (RandomWalk(np.deg2rad(0.00001 ** 2)),
         RandomWalk(np.finfo(float).eps),
         RandomWalk(np.deg2rad(0.00001 ** 2)),
         RandomWalk(np.finfo(float).eps)))
    measurement_model = LinearGaussian(ndim_state=4, mapping=[0, 2],
                                       noise_covar=np.diag([np.deg2rad(0.001 ** 2),
                                                            np.deg2rad(0.001 ** 2)]))

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
    hypothesiser = FilteredDetectionsGater(hypothesiser, 'MMSI', match_missing=False)
    class TPRGNN(GNNWith2DAssignment, LongLatTPRTreeMixIn):
        pass
    associator = TPRGNN(hypothesiser, measurement_model, timedelta(hours=24), [0, 2])

    # Track Initiator
    # ===============
    state_vector = StateVector([[Longitude(0.)], [0.], [Latitude(0.)], [0.]])
    covar = CovarianceMatrix(np.diag([np.deg2rad(0.0001 ** 2), np.deg2rad(0.0003 ** 2),
                                      np.deg2rad(0.0001 ** 2), np.deg2rad(0.0003 ** 2)]))
    prior_state = GaussianStatePrediction(state_vector, covar)
    initiator = LinearMeasurementInitiatorMixture(prior_state, measurement_model)
    initiator = StatesLengthLimiter(initiator, max_length=10)

    # Track Deleter
    # =============
    deleter = UpdateTimeDeleter(time_since_update=timedelta(hours=5))

    ################################################################################
    # Main Tracking process                                                        #
    ################################################################################

    tracks = set()  # Main set of tracks
    temp_deleted_tracks = set()  # Tentatively deleted tracks
    deleted_tracks = set()  # Fully deleted tracks

    # Possible parallelisation point
    #   - e.g. Run each file in a separate thread
    file_paths = glob.iglob(os.path.join(r'data\exact_earth\id_sorted',  r'*.csv'))
    for file_path in file_paths:
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

        for scan_time, detections in detector:

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
                    value = np.rad2deg(state_vector[mapping[i]])
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

            if len(detections) == 0:
                continue

            print("Measurements: " + str(len(detections)) + " - Time: " + scan_time.strftime('%H:%M:%S %d/%m/%Y'))

            # Process static AIS
            for track in tracks:
                for detection in static_detections:
                    if detection.metadata["MMSI"] == track.metadata["MMSI"]:
                        static_fields = ['Vessel_Name', 'Ship_Type',
                                         'Destination', 'IMO']
                        track.metadata.update({x: detection.metadata[x] for x in static_fields})

            # Perform data association
            print("Tracking.... NumTracks: {}".format(str(len(tracks))))
            associated_detections = set()
            if len(detections) > 0:

                associations = associator.associate(tracks, detections, scan_time)

                # Update tracks based on association hypotheses
                print("Updating...")
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

            # Initiate new tracks
            print("Track Initiation...")
            unassociated_detections = detections - associated_detections
            new_tracks = initiator.initiate(unassociated_detections, timestamp=scan_time)
            tracks |= new_tracks

            # Delete invalid tracks
            print("Track Deletion...")
            del_tracks = deleter.delete_tracks(tracks, timestamp=scan_time)
            tracks -= del_tracks

            # Write data
            # writer.write(tracks,
            #              detections,
            #              host_name="138.253.118.175",
            #              host_port=27017,
            #              db_name="TA_ExactEarth",
            #              collection_name=collections,
            #              drop=False,
            #              timestamp=scan_time)

            print("Time: " + scan_time.strftime('%H:%M:%S %d/%m/%Y')
                  + " - Measurements: " + str(len(detections))
                  + " - Main Tracks: " + str(len(tracks))
                  + " - Tentative Tracks: " + str(len(temp_deleted_tracks))
                  + " - Total Tracks: " + str(
                len(tracks) + len(temp_deleted_tracks)))
