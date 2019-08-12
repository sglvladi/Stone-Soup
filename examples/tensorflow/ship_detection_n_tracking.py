import json
import os
import operator
import cv2
import numpy as np
from copy import copy
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import matplotlib.animation as manimation
#import logging; logging.basicConfig(level=logging.DEBUG)

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
from stonesoup.hypothesiser.probability import PDAHypothesiser, IPDAHypothesiser
from stonesoup.dataassociator.probability import JPDA, JIPDA

from custom.tcpsocket import TcpClient
from custom.pid import PtzPidController2
from utils import visualization_utils as vis_util
from utils import label_map_util

# PATHS
##############################################################################
# Define the video stream
# MODEL_NAME = 'open_dataset_plus_faster_rcnn_coco_v876754'
# PATH_TO_CKPT = 'D:/OneDrive/TensorFlow/scripts/camera_control/models/' + \
#                MODEL_NAME + '/frozen_inference_graph.pb'
# # PATH_TO_CKPT = '/home/denbridge/Documents/Tensorflow/trained-models
PATH_TO_CKPT = 'D:/OneDrive/TensorFlow/trained-models' \
               '/output_inference_graph_v5/frozen_inference_graph.pb'
# # List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('D:/OneDrive/TensorFlow/scripts/camera_control/data', 'label_map.pbtxt')
VIDEO_PATH = "D:/OneDrive/TensorFlow/datasets/video-samples/denbridge" \
             "/video7_Trim.mp4"
# Define the video stream
# MODEL_NAME = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
# # PATH_TO_CKPT = 'D:/OneDrive/TensorFlow/scripts/camera_control/models/' + MODEL_NAME + '/frozen_inference_graph.pb'
# PATH_TO_CKPT = '/home/denbridge/Documents/Tensorflow/trained-models/output_inference_graph_v5/frozen_inference_graph.pb'
# #PATH_TO_CKPT = '/home/denbridge/Documents/Tensorflow/trained-models/open_dataset_plus_faster_rcnn_coco_v876754/frozen_inference_graph.pb'
# # List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('/home/denbridge/Documents/Tensorflow/trained-models/output_inference_graph_v5', 'label_map.pbtxt')
# VIDEO_PATH = "/home/denbridge/Documents/Tensorflow/datasets" \
#              "/video7_Trim.mp4"

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
hypothesiser = IPDAHypothesiser(predictor, updater, 0.001, prob_detect=0.4)
associator = JIPDA(hypothesiser, updater, 20)

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
# videoReader = FFmpegVideoStreamReader('rtsp://admin:admin@10.36.0.160:554/videoinput_1:0/h264_1/media.stm',
#                                       input_args=[[],{'threads':1, 'fflags':'nobuffer', 'vsync':0}],
#                                       output_args=[[],{'format':'rawvideo', 'pix_fmt':'rgb24'}])
detector = TensorflowObjectDetectorThreaded(videoReader, PATH_TO_CKPT)
score_threshold = 0.5
feeder = MetadataValueFilter(detector,
                             metadata_field="score",
                             operator=operator.ge,
                             reference_value=score_threshold)
feeder = MetadataValueFilter(feeder,
                             metadata_field="class",
                             operator=operator.eq,
                             reference_value=1)

# CONTROL
##############################################################################
client = TcpClient('127.0.0.1', 1563)
# (Kp,Kd,Ki)
pid_gains = ((0.01,  0.002, 0.0001),     # Pan
             (0.005, 0.001, 0.0001),     # Tilt
             (100,   10,    0))          # Zoom
# pid_gains = ((1.0,     0.2,   0.001),     # Pan
#              (0.5,   0.1,   0),     # Tilt
#              (100,   10,    0))     # Zoom
pid_controller = PtzPidController2(pid_gains)

# Video writing
# plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
# #writer = anim.FFMpegWriter(fps=30, codec='hevc')
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
from custom.functions import *
from custom.plotting import *

from skimage.util import img_as_float, img_as_ubyte
from skimage.color import rgb2gray
from skimage.feature import match_template

from utils import visualization_utils as viz_utils
def find_track(image, tracks, old_track):
    if len(image):
        gray = img_as_ubyte(rgb2gray(image))
        h, w = gray.shape
        if old_track is not None:
            (ymin, xmin, ymax, xmax) = track_to_bbox(old_track)
            if xmin<0:
                xmin = 0
            if ymin<0:
                ymin = 0
            t_box = gray[int(w* xmin):int(w * xmax), int(h * ymin):int(h * ymax)]
            result = match_template(gray, t_box)
            ij = np.unravel_index(np.argmax(result), result.shape)
            x, y = ij[::-1]
            t_w, t_h = (xmax-xmin, ymax-ymin)
            bbox = (x/h, y/w, (x/h+t_h), (y/w+t_w))
            print(bbox)
            viz_utils.draw_bounding_box_on_image_array(image,
                                                       bbox[0],
                                                       bbox[1],
                                                       bbox[2],
                                                       bbox[3],
                                                       color='yellow',
                                                       thickness=2)
    return image

def select_track(image, tracks, old_track):
    c_track = None
    # if old_track is not None:
    #     find_track(image, tracks, old_track)
    # else:

    # Select based on id
    if old_track is not None:
        t = [track for track in tracks if track.id == old_track.id]
        if len(t):
            c_track = copy(t[0])

    # Select based on max area
    if c_track is None:
        track_list = [track for track in tracks
                      if (len([state for state in track.states
                               if isinstance(state, Update)]) > 10)
                      and track.score > 0.8]
        if len(track_list) == 0:
            return None
        boxes, areas = tracks_to_bbox_areas(track_list)
        ind = np.argmax(areas)
        c_track = track_list[ind]
    return c_track

def click_select_track(image, tracks, x, y):
    num_rows, num_cols, _ = image.shape
    valid_tracks = [track for track in tracks
                    if (len([state for state in track.states
                             if isinstance(state, Update)]) > 10)
                    and (track.state.state_vector[0][0] <= x/num_cols <= track.state.state_vector[0][0]+track.state.state_vector[4][0]
                         and track.state.state_vector[2][0] <= y/num_rows <= track.state.state_vector[2][0]+track.state.state_vector[6][0])]
    global c_track
    if len(valid_tracks)>0:
        c_track = valid_tracks[0]
    else:
        c_track = None

def click_callback(event,x,y,flags,params):
    global mouseX,mouseY,clicked
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseX,mouseY = x,y
        clicked = True
        print(mouseX,mouseY)
    elif event ==cv2.EVENT_RBUTTONDBLCLK:
        command = {"type": "command",
                   "panSpeed": 0,
                   "tiltSpeed": 0,
                   "zoom": 0}
        print("Sending command: {}".format(json.dumps(command)))
        response = client.send_command(command)


def process_pid(frame_np, track, feedback=None):
    converged = False
    command = {"type": "command",
               "panSpeed":  0,
               "tiltSpeed": 0,
               "zoom":      0,
               "zoomSpeed": 0}
    if feedback is None:
        feedback = {}
    feedback['dt'] = 0.1

    if len(frame_np) and track is not None:
        # Select box and refactor
        box = track_to_bbox(track)

        # Run pid controller
        tolerance = (0.0, 0.0, 0.0)
        commands, _ = pid_controller.run(frame_np, box, feedback, tolerance)

        # Extract and serialise commands
        panSpeed = commands["Pan"]["panSpeed"]
        tiltSpeed = commands["Tilt"]["tiltSpeed"]
        zoomRelative = commands["Zoom"]["zoomRelative"]
        zoomSpeed = commands["Zoom"]["zoomSpeed"]
        zoom = commands["Zoom"]["zoom"]

        # Pre-process commands to check convergence
        # This ensures that Pan-Tilt are allowed to converge first
        # after which the zoom will be adjusted. This helps make the
        # transformations less non-linear.
        if panSpeed + tiltSpeed == 0:
            if zoomRelative == 0:
                converged = True
        else:
            # Control ONLY the pan-tilt
            zoomSpeed = 0

        # Send command to CCTV server
        command = {"type": "command",
                   "panSpeed": panSpeed,
                   "tiltSpeed": tiltSpeed,
                   "zoom": zoom,
                   "zoomSpeed": zoomSpeed}

    return command, converged

# MAIN
##############################################################################
# fig = plt.figure(figsize=(9, 6))
# with vid_writer.saving(fig, "denbridge_test_zoom2.mp4", 100):
mouseX=0
mouseY=0
clicked=False
cv2.namedWindow('image')
cv2.setMouseCallback('image',click_callback)
tracks = set()
c_track = None
frame = ImageFrame([])
frameIndex = 0
last_command = {"type": "command",
                "panSpeed": 0,
                "tiltSpeed": 0,
                "zoom": 0}
last_command_time = datetime.now()
for timestamp, detections in feeder.detections_gen():

    # Merge detections that overlap
    #detections = merge_detections(detections, 0.5)

    # Read frame
    frame_old = copy(frame)
    frame = videoReader.frame
    image = copy(frame.pixels)
    num_rows, num_cols, _ = image.shape

    # Detect the horizon in the image and disregard any detections above it
    # if (frameIndex + 20) % 20 == 0:
    #     horizon = detect_horizon(image)
    # valid_detections = set([])
    # for detection in detections:
    #     y_max = detection.state_vector[1,0] + detection.state_vector[3,0]
    #     hor_max = max([horizon[0][1], horizon[1][1]])
    #     if (y_max*num_rows>hor_max):
    #         valid_detections.add(detection)
    # detections = copy(valid_detections)

    # Correct track positions
    tracks = affineCorrection(frame.pixels, frame_old.pixels, tracks)

    # Perform data association
    associations = associator.associate(
        tracks, detections, timestamp)

    # Update tracks based on association hypotheses
    associated_detections = set()
    for track, multihypothesis in associations.items():

        # calculate each Track's state as a Gaussian Mixture of
        # its possible associations with each detection, then
        # reduce the Mixture to a single Gaussian State
        missed_hypothesis = next(hyp for hyp in multihypothesis if not hyp)
        missed_detection_weight = missed_hypothesis.weight
        posterior_states = [missed_hypothesis.prediction]
        posterior_state_weights = [missed_detection_weight]
        for hypothesis in multihypothesis:
            if hypothesis:
                posterior_states.append(
                    updater.update(hypothesis))
                posterior_state_weights.append(
                    hypothesis.probability)
                if hypothesis.weight > missed_detection_weight:
                    associated_detections.add(hypothesis.measurement)

        means = np.array([state.state_vector for state
                          in posterior_states])
        means = np.reshape(means, np.shape(means)[:-1])
        covars = np.array([state.covar for state
                           in posterior_states])
        covars = np.reshape(covars, (np.shape(covars)))
        weights = np.array([weight for weight
                            in posterior_state_weights])
        weights = np.reshape(weights, np.shape(weights))

        post_mean, post_covar = gm_reduce_single(means,
                                                 covars, weights)

        track.append(GaussianStateUpdate(
            np.array(post_mean), np.array(post_covar),
            multihypothesis,
            multihypothesis[0].measurement_prediction.timestamp))

    # Delete invalid tracks
    tracks -= delete_tracks(tracks)

    # Initiate new tracks
    unassociated_detections = detections - associated_detections
    tracks |= initiator.initiate(unassociated_detections)

    # Select centered track
    old_c_track = copy(c_track)
    # c_track = select_track(image, tracks, c_track)
    if clicked:
        print("Clicked!!")
        click_select_track(image, tracks, mouseX, mouseY)
        clicked = False


    # Process PID controller
    feedback = client.send_request()
    command, converged = process_pid(image, c_track, feedback)
    if datetime.now() - last_command_time > timedelta(seconds=0.01):
        if not last_command == command:
            print("Sending command: {}".format(json.dumps(command)))
            response = client.send_command(command)
            last_command = command
        else:
            print("Duplicate command: {}".format(json.dumps(command)))
        last_command_time = datetime.now()

    num_c_tracks = len([track for track in tracks
                        if (len([state for state in track.states if
                                 isinstance(state, Update)]) > 10)])
    print("Time: {} - Tracks: {} - Detections: {}".format(timestamp,
                                                          num_c_tracks,
                                                          len(detections)))
    frameIndex += 1

    # Display output
    if frame:
        image = draw_detections(image, detections, category_index, score_threshold)
        image = draw_tracks(image, tracks, converged, c_track)
        # if old_c_track is not None:
        #     image = find_track(image, tracks, old_c_track)
        #image = draw_horizon(image, horizon)
        # plt.clf()
        # plt.imshow(image)
        # plt.pause(0.0001)
        cv2.imshow('image', image)
        # vid_writer.grab_frame()

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break