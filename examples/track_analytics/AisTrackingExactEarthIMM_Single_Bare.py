# coding: utf-8

# AisTrackingExactEarthIMM_Single.py
# ==================================
# Run a Global Nearest Neighbour with IMM Prediction & Update, for each of
# the MMSI files in the "per_mmsi" folder.

################################################################################
# IMPORTS                                                                      #
################################################################################
# General imports
import glob
import os
import numpy as np
from datetime import datetime, timedelta
import argparse


# Stone-Soup imports
from stonesoup.models.transition.linear import (
    RandomWalk, OrnsteinUhlenbeck, CombinedLinearGaussianTransitionModel)
from stonesoup.initiator.simple import LinearMeasurementInitiatorMixture
from stonesoup.deleter.time import UpdateTimeDeleter
from stonesoup.dataassociator.neighbour import GlobalNearestNeighbour
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.hypothesiser.filtered import FilteredDetectionsHypothesiser
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

##########################################################################
# Argument parser                                                        #
##########################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_path",
                    help="Path to per_mmsi folder")
parser.add_argument("-o", "--output_path",
                    help="Path to output folder")
args = parser.parse_args()
in_path = args.input_path
out_path = args.output_path

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
hypothesiser = FilteredDetectionsHypothesiser(hypothesiser, 'MMSI',
                                              match_missing=False)
associator = GlobalNearestNeighbour(hypothesiser)

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
from stonesoup.types.sets import TrackSet

# We process each MMSI file sequentially
for file_path in glob.iglob(os.path.join(in_path, r'*.csv')):

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
    writer = CSV_Writer_EE_MMSI(out_path, mmsi)

    tracks = TrackSet()  # Main set of tracks
    date = datetime.now().date()

    # We process each scan sequentially
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
                    track._metadata.update({x: detection.metadata[x] for
                                            x in static_fields})

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

        # Write data
        print("Writing....")
        writer.write(tracks,
                     detections,
                     timestamp=scan_time)

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
