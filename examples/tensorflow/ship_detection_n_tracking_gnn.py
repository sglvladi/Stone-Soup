import json
import os
import operator
import cv2
import numpy as np
from copy import copy
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import matplotlib.animation as manimation
import logging; logging.basicConfig(level=logging.DEBUG)

from stonesoup.functions import gm_reduce_single
from stonesoup.types.sensordata import ImageFrame
from stonesoup.types.state import GaussianState
from stonesoup.types.update import Update, GaussianStateUpdate
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.detection import Detection
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.initiator.simple import LinearMeasurementInitiator
from stonesoup.reader.video import VideoClipReader, FFmpegVideoStreamReader
from stonesoup.detector.tensorflow import TensorflowObjectDetector,TensorflowObjectDetectorThreaded
from stonesoup.feeder.filter import MetadataValueFilter
from stonesoup.measures import Mahalanobis
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import (
    ConstantVelocity, CombinedLinearGaussianTransitionModel)
from stonesoup.predictor.kalman import (KalmanPredictor,
    UnscentedKalmanPredictor, ExtendedKalmanPredictor)
from stonesoup.updater.kalman import (KalmanUpdater,
    UnscentedKalmanUpdater, ExtendedKalmanUpdater)
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.dataassociator.neighbour import (
    NearestNeighbour, GlobalNearestNeighbour)

from tcpsocket import TcpClient
from pid import PtzPidController
from utils import visualization_utils as vis_util
from utils import label_map_util

# PATHS
##############################################################################
# Define the video stream
MODEL_NAME = 'open_dataset_plus_faster_rcnn_coco_v876754'
PATH_TO_CKPT = 'D:/OneDrive/TensorFlow/scripts/camera_control/models/' +  MODEL_NAME + '/frozen_inference_graph.pb'
# PATH_TO_CKPT = '/home/denbridge/Documents/Tensorflow/trained-models
# PATH_TO_CKPT = 'D:/OneDrive/TensorFlow/trained-models' \
#                '/output_inference_graph_v5/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('D:/OneDrive/TensorFlow/scripts/camera_control/data', 'label_map.pbtxt')
VIDEO_PATH = "D:/OneDrive/TensorFlow/datasets/video-samples/denbridge" \
             "/video12_Trim.mp4"

# MODELS
##############################################################################
transition_model = CombinedLinearGaussianTransitionModel(
    (ConstantVelocity(0.028**2), ConstantVelocity(0.04**2),
     ConstantVelocity(0.005**2), ConstantVelocity(0.005**2)))
measurement_model = LinearGaussian(
    ndim_state=8, mapping=[0, 2, 4, 6],
    noise_covar=np.diag([0.01**2, 0.01**2, 0.05**2, 0.05**2]))

# MAIN TRACKER COMPONENTS
##############################################################################

# Filtering
predictor = UnscentedKalmanPredictor(transition_model)
updater = UnscentedKalmanUpdater(measurement_model)

# Data Association
hypothesiser = DistanceHypothesiser(predictor, updater, Mahalanobis(), 20)
associator = GlobalNearestNeighbour(hypothesiser)

# Track Management
prior_state = GaussianState(StateVector(np.zeros((8,1))),
                            CovarianceMatrix(np.diag(np.tile([0.00001**2,
                                                              0.00001**2],
                                                             4))))
initiator = LinearMeasurementInitiator(prior_state, measurement_model)
deleter = CovarianceBasedDeleter(0.1**2)

# DETECTION
##############################################################################
videoReader = VideoClipReader(VIDEO_PATH)
# videoReader = FFmpegVideoStreamReader('rtsp://admin:admin@10.36.0.158:554/videoinput_1:0/h264_1/media.stm',
#                                       input_args=[[],{'threads':1, 'fflags':'nobuffer', 'vsync':0}],
#                                       output_args=[[],{'format':'rawvideo', 'pix_fmt':'rgb24'}])
detector = TensorflowObjectDetectorThreaded(videoReader, PATH_TO_CKPT)
score_threshold = 0.4
feeder = MetadataValueFilter(detector,
                             metadata_field="score",
                             operator=operator.ge,
                             reference_value=0.4)
feeder = MetadataValueFilter(feeder,
                             metadata_field="class",
                             operator=operator.eq,
                             reference_value=1)

# CONTROL
##############################################################################
client = TcpClient('127.0.0.1', 1563)
pid_controller = PtzPidController()

# Video writing
# plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
# writer = anim.FFMpegWriter(fps=30, codec='hevc')
# FFMpegWriter = manimation.writers['ffmpeg']
# metadata = dict(title='Movie Test', artist='Matplotlib',
#                 comment='Movie support!')
# vid_writer = manimation.FFMpegWriter(fps=5)

# Load label map
NUM_CLASSES = 90
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# FUNCTIONS
##############################################################################
def affineCorrection(frame, frame_old, tracks):
    if (len(frame_old)):
        frame_old_gray = cv2.cvtColor(frame_old, cv2.COLOR_BGR2GRAY)

        # Detect feature points in previous frame
        prev_pts = cv2.goodFeaturesToTrack(frame_old_gray,
                                           maxCorners=200,
                                           qualityLevel=0.01,
                                           minDistance=30,
                                           blockSize=3)

        # Read next frame
        if not len(frame):
            return tracks

        # Convert to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(frame_old_gray, frame_gray, prev_pts, None)

        # Sanity check
        assert prev_pts.shape == curr_pts.shape

        # Filter only valid points
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        # Find transformation matrix
        m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)  # will only work with OpenCV-3 or less

        # Extract traslation
        height, width, _ = frame.shape
        affine_matrix = m[0]
        affine_matrix[0, 2] = affine_matrix[0, 2] / width
        affine_matrix[1, 2] = affine_matrix[1, 2] / height
        dx = affine_matrix[0, 2]
        dy = affine_matrix[1, 2]

        # Extract rotation angle
        da = np.arctan2(affine_matrix[1, 0], affine_matrix[0, 0])

        # Store transformation
        transform = [dx, dy, da]

        for track in tracks:
            xmin = track.state.state_vector[0, 0]
            ymin = track.state.state_vector[2, 0]
            xmax = xmin + track.state.state_vector[4, 0]
            ymax = ymin + track.state.state_vector[6, 0]
            rect_pts = np.array([[[xmin, ymin]],
                                 [[xmax, ymin]],
                                 [[xmin, ymax]],
                                 [[xmax, ymax]]], dtype=np.float32)
            affine_warp = np.vstack((affine_matrix, np.array([[0, 0, 1]])))
            rect_pts = cv2.perspectiveTransform(rect_pts, affine_warp)
            xA = rect_pts[0, 0, 0]
            yA = rect_pts[0, 0, 1]
            xB = rect_pts[3, 0, 0]
            yB = rect_pts[3, 0, 1]

            track.state.state_vector[0, 0] = xA
            track.state.state_vector[2, 0] = yA
            track.state.state_vector[4, 0] = xB - xA
            track.state.state_vector[6, 0] = yB - yA

    return tracks

def get_overlap_ratio(a, b, epsilon=1e-5):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    return area_overlap/area_a

def gen_clusters(v_matrix):
    """

    Parameters
    ----------
    v_matrix

    Returns
    -------

    """
    # Initiate parameters
    num_rows, num_cols = np.shape(v_matrix);  # Number of tracks

    # Form clusters of tracks sharing measurements
    unassoc_rows  = set()
    clusters = []

    # Iterate over all row
    for row_ind in range(num_rows):

        # Extract valid col indices
        v_col_inds = set(np.argwhere(v_matrix[row_ind, :] > 0)[:,0])

        # If there  exist valid cols
        if len(v_col_inds):

            # Check if matched measurements are members of any clusters
            num_clusters = len(clusters)
            v_matched_cluster_ind = np.zeros((num_clusters,))
            for cluster_ind in range(num_clusters):
                if any([v_ind in clusters[cluster_ind]["col_inds"]
                        for v_ind in v_col_inds ]):
                    v_matched_cluster_ind[cluster_ind] = 1

            num_matched_clusters = sum(v_matched_cluster_ind)

            # If only matched with a single cluster, join.
            if num_matched_clusters == 1:
                matched_cluster_ind = np.argwhere(v_matched_cluster_ind > 0)[0][0]
                clusters[matched_cluster_ind]["row_inds"] |= set([row_ind])
                clusters[matched_cluster_ind]["col_inds"] |= v_col_inds
            elif num_matched_clusters == 0:
                new_cluster = {"row_inds": set([row_ind]),
                              "col_inds": v_col_inds}
                clusters.append(new_cluster)
            else:
                matched_cluster_inds = np.argwhere(v_matched_cluster_ind > 0)[0]
                # Start from last cluster, joining each one with the previous
                # and removing the former.
                for matched_cluster_ind in range(num_matched_clusters - 2, -1, 0):
                    clusters[matched_cluster_inds[
                        matched_cluster_ind]]["row_inds"] |= clusters[
                        matched_cluster_inds[matched_cluster_ind + 1]]["row_inds"]
                    clusters[matched_cluster_inds[
                        matched_cluster_ind]]["col_inds"] |= clusters[
                        matched_cluster_inds[matched_cluster_ind + 1]]["col_inds"]
                    clusters[matched_cluster_inds[matched_cluster_ind + 1]] = []

                # Finally, join with associated track.
                clusters[matched_cluster_inds[
                    matched_cluster_ind]]["row_inds"] |= set([row_ind])
                clusters[matched_cluster_inds[
                    matched_cluster_ind]]["col_inds"] |= v_col_inds
        else:
            new_cluster = {"row_inds": set([row_ind]),
                          "col_inds": set([])}
            clusters.append(new_cluster)
            # clusters(end + 1) = ClusterObj;
            unassoc_rows.add(row_ind)

    return clusters, unassoc_rows

def detection_to_bbox(state_vector):
    x_min, y_min, width, height = (state_vector[0, 0],
                                   state_vector[1, 0],
                                   state_vector[2, 0],
                                   state_vector[3, 0])
    return StateVector([[x_min],
                        [y_min],
                        [x_min + width],
                        [y_min + height]])

def merge_detections(detections, threshold = 0.5):
    num_detections = len(detections)
    states = []
    detections_list = list(detections)
    for detection in detections_list:
        state_vector = detection.state_vector
        states.append(detection_to_bbox(state_vector))

    v_matrix = np.zeros((num_detections, num_detections))
    ious = np.zeros((num_detections, num_detections))
    for i, detection_i in enumerate(detections_list):
        for j, detection_j in enumerate(detections_list):
            ious[i, j] = get_overlap_ratio(states[i], states[j])
            if (ious[i, j] > threshold):
                v_matrix[i, j] = 1

    new_detections = set()
    clusters, _ = gen_clusters(v_matrix)
    for cluster in clusters:
        det_inds = list(cluster["row_inds"])

        detection = detections_list[det_inds[0]]
        state_vector = detection.state_vector
        metadata = detection.metadata
        timestamp = detection.timestamp
        for i, ind in enumerate(det_inds):
            detection = detections_list[ind]
            if i == 0:
                continue
            else:
                bbox_1 = detection_to_bbox(state_vector)
                bbox_2 = detection_to_bbox(detection.state_vector)
                if bbox_2[0][0]<bbox_1[0][0]:
                    state_vector[0][0] = bbox_2[0][0]
                if bbox_2[1][0] < bbox_1[1][0]:
                    state_vector[1][0] = bbox_2[1][0]
                if bbox_2[2][0]>bbox_1[2][0]:
                    state_vector[2][0] = bbox_2[2][0]-state_vector[0][0]
                if bbox_2[3][0]>bbox_1[3][0]:
                    state_vector[3][0] = bbox_2[3][0]-state_vector[3][0]
                #metadata["score"] += detection.metadata["score"]
        new_detection = Detection(state_vector, metadata=metadata,
                                  timestamp=timestamp)
        new_detections.add(copy(new_detection))

    return new_detections

from matplotlib.patches import Ellipse
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

def merge_tracks(tracks):
    num_tracks = len(tracks)
    track_states = []
    areas = []
    tracks = list(tracks)
    for track in tracks:
        state_vec = track.state.state_vector
        covar = track.state.covar[[0,2], :][:, [0,2]]
        x_min, y_min, width, height = (state_vec[0,0],
                                       state_vec[2,0],
                                       state_vec[4,0],
                                       state_vec[6,0])
        track_states.append(StateVector([[x_min],
                                         [y_min],
                                         [x_min+width],
                                         [y_min+height]]))
        areas.append(state_vec[4]*state_vec[6,0])

    new_tracks = set()
    v_matrix = np.zeros((num_tracks,num_tracks))
    ious = np.zeros((num_tracks, num_tracks))
    for i, track_i in enumerate(tracks):
        for j, track_j in enumerate(tracks):
            ious[i,j] = get_overlap_ratio(track_states[i], track_states[j])
            if(ious[i,j] > 0.5):
                v_matrix[i,j] = 1


    clusters, _ = gen_clusters(v_matrix)
    for cluster in clusters:
        track_inds = list(cluster["row_inds"])
        ind = np.argmax([areas[ind] for ind in track_inds ])
        ind = track_inds[ind]
        new_tracks.add(tracks[ind])

    if len(new_tracks)==0:
        a=0
    return new_tracks

def draw_detections(frame, detections):
    if len(detections):
        boxes = np.array([detection.metadata["tf_box"]
                          for detection in detections])
        classes = np.array([detection.metadata["class"]
                            for detection in detections])
        scores = np.array([detection.metadata["score"]
                           for detection in detections])
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            boxes,
            classes,
            scores,
            category_index,
            min_score_thresh=score_threshold,
            use_normalized_coordinates=True,
            line_thickness=2,
        )
    return frame

def draw_tracks(frame, tracks, converged):
    track_list = list(tracks)
    boxes = np.array(
        [track.state.state_vector[[0, 2, 4, 6]].reshape((4)) for track in track_list
         if (len([state for state in track.states if isinstance(state,
                                                                Update)]) > 10)])
    ids = [track.id for track in track_list]
    f_width, f_height, _ = frame.shape
    if converged:
        color = 'red'
    else:
        color = 'blue'
    for i,box in enumerate(boxes):
        xmin, ymin, w, h = box
        xmax = xmin + w
        ymax = ymin + h

        vis_util.draw_bounding_box_on_image_array(frame,
                                                  ymin,
                                                  xmin,
                                                  ymax,
                                                  xmax,
                                                  color=color,
                                                  thickness=2)
        cv2.putText(frame, ids[i], (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 3, cv2.LINE_AA)
    return frame

def process_pid(frame_np, tracks, feedback=None):
    converged = False
    if len(frame_np):
        boxes = np.array(
            [track.state.state_vector[[0, 2, 4, 6]].reshape((4)) for track in
             tracks
             if (len([state for state in track.states if
                      isinstance(state, Update)]) > 10)])
        areas = [track.state.state_vector[4][0] * track.state.state_vector[6][0] for
                 track in tracks
                 if (len(
                [state for state in track.states if isinstance(state, Update)]) > 10)]

        if len(boxes):
            # Select box and refactor
            ind = np.argmax(areas)
            box = boxes[ind]
            xmin, ymin, w, h = box
            xmax, ymax = (xmin + w, ymin + h)
            box = (ymin, xmin, ymax, xmax)

            # Run pid controller
            feedback = {'dt': 0.1}
            commands, _ = pid_controller.run(frame_np, box, feedback)
            print(commands)

            # Extract and serialise commands
            panSpeed, tiltSpeed = (commands["PanTilt"]["panSpeed"],
                                   commands["PanTilt"]["tiltSpeed"])
            zoomRelative = commands["Zoom"]["zoomRelative"]
            zoomSpeed = commands["Zoom"]["zoomSpeed"]

            # Pre-process commands to check convergence
            # This ensures that Pan-Tilt are allowed to converge first
            # after which the zoom will be adjusted. This helps make the
            # transformations less non-linear.
            if panSpeed + tiltSpeed == 0:
                if zoomRelative == 0:
                    converged = True
            else:
                # Control ONLY the pan-tilt
                zoomRelative = 0

            # Send command to CCTV server
            command = {"type": "command",
                       "panSpeed": panSpeed,
                       "tiltSpeed": tiltSpeed,
                       "zoom": zoomRelative}
            print("\nSending command: {}".format(json.dumps(command)))
            response = client.send_command(command)

    return converged


# MAIN
##############################################################################
# fig = plt.figure(figsize=(9, 6))
# with vid_writer.saving(fig, "denbridge_5.mp4", 100):
frameIndex = 0
tracks = set()
frame = ImageFrame([])
frame_old = ImageFrame([])
last_command_time = datetime.now()
for timestamp, detections in feeder.detections_gen():

    # Merge detections that overlap
    detections = merge_detections(detections, 0.5)

    # Read frame
    frame = videoReader.frame
    frame_np = copy(frame.pixels)

    # Correct track positions
    tracks = affineCorrection(frame.pixels, frame_old.pixels, tracks)

    # Perform data association
    associations = associator.associate(
        tracks, detections, timestamp)

    # Update tracks based on association hypotheses
    associated_detections = set()
    for track, hypothesis in associations.items():
        if hypothesis:
            state_post = updater.update(hypothesis)
            track.append(state_post)
            associated_detections.add(hypothesis.measurement)
        else:
            track.append(hypothesis.prediction)
    unassociated_detections = detections - associated_detections

    # Delete invalid tracks
    tracks -= deleter.delete_tracks(tracks)

    # Initiate new tracks
    tracks |= initiator.initiate(unassociated_detections)

    # Process PID controller
    converged = False
    if datetime.now() - last_command_time > timedelta(
            seconds=0.5):
        converged = process_pid(frame_np, tracks, None)

    frame_old = copy(frame)

    # Display output
    if frame:
        frame_np = draw_detections(frame_np, detections)
        frame_np = draw_tracks(frame_np, tracks, converged)
        # plt.clf()
        # plt.imshow(frame_np)
        # plt.pause(0.001)
        cv2.imshow("", frame_np)
        # vid_writer.grab_frame()

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
