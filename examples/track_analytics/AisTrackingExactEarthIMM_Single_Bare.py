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
import logging
import os
import socket
import time
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

# Create output dir, only if it doesn't exist
try:
    os.makedirs(out_path)
except FileExistsError:
    # Safety check to prevent accidentally overwriting results
    print(f"Warning, output dir '{out_path}' already exists.  "
          "Please delete or move and try again. Exiting...")
    exit(1)

# Configure logging/logger:
log_level = logging.INFO
log_format = "[%(asctime)s] %(levelname)-8s [%(filename)s:%(lineno)3i] %(message)s"
logging.basicConfig(
    level=log_level,
    format=log_format,
)
logger = logging.getLogger()

# Define log file handler, and add
log_file = os.path.join(
    out_path,
    f'{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))}.log',
)
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(log_level)
file_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(file_handler)

# Log info about this run
logger.info("Logger started...")
logger.info("Script running with arguments: %s ...", args)
logger.info("Running on %s", socket.gethostname())
logger.info("Logging to: %s", log_file)
logger.info("Reading from: %s", in_path)


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

start_time = time.time()
logger.info(f"Starting...")

discovered_files = list(glob.iglob(os.path.join(in_path, r'*.csv')))
logger.info("Discovered %d files to process...", len(discovered_files))

iteration_timestamp = time.time()

# We process each MMSI file sequentially
for i, file_path in enumerate(discovered_files, start=1):
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

        logger.debug("Measurements: %s", str(len(detections)))

        # Process static AIS
        for track in tracks:
            for detection in static_detections:
                if detection.metadata["MMSI"] == track.metadata["MMSI"]:
                    track._metadata.update({x: detection.metadata[x] for
                                            x in static_fields})
        if len(detections) != 0:
            # Perform data association
            logger.debug("Tracking.... NumTracks: %s", (str(len(tracks))))
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
                if len(track.states) > 10:
                    track.states = track.states[-10:]

        # Write data
        logger.debug("Writing....")
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

        logger.debug(
            "%s - Time: %s - Measurements: %s - Tracks: %s",
            filename, scan_time.strftime('%H:%M:%S %d/%m/%Y'),
            str(len(detections)), str(len(tracks))
        )

    # Log file processing stats
    logger.info(
        f"Processed MMSI {i:>5} ({os.path.basename(file_path)[:-4]:>9}) "
        f"in {time.time() - iteration_timestamp:>5.1f} secs"
        f" | {(time.time() - start_time)/60:>6.1f} mins elapsed "
        f" | {( ((time.time() - start_time) / i) * (len(discovered_files) - i) ) / 60:6.1f}"
        f" mins remaining (est.)..."
    )
    iteration_timestamp = time.time()

logger.info(f"Finished processing after: {(time.time() - start_time) / 60:.1f} minutes")
