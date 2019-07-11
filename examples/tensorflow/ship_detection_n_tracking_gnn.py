import json
import os
import operator
import cv2
import numpy as np
from copy import copy
import matplotlib
import datetime
from matplotlib import pyplot as plt
import matplotlib.animation as manimation

from stonesoup.functions import gm_reduce_single
from stonesoup.types.sensordata import ImageFrame
from stonesoup.types.state import GaussianState
from stonesoup.types.update import Update, GaussianStateUpdate
from stonesoup.types.array import StateVector, CovarianceMatrix



from stonesoup.dataassociator.probability import JPDA
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.initiator.simple import LinearMeasurementInitiator
from stonesoup.initiator.simple import SinglePointInitiator
from stonesoup.reader.video import VideoClipReader, FFmpegVideoStreamReader
from stonesoup.detector.tensorflow import TensorflowObjectDetector,TensorflowObjectDetectorThreaded
from stonesoup.feeder.filter import MetadataValueFilter
from stonesoup.tracker.simple import MultiTargetTracker

from utils import visualization_utils as vis_util
from utils import label_map_util

import logging; logging.basicConfig(level=logging.DEBUG)

from tcpsocket import TcpClient
from pid import PtzPidController
# manimation.verbose.set_level('debug')

plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
# writer = anim.FFMpegWriter(fps=30, codec='hevc')
# FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
vid_writer = manimation.FFMpegWriter(fps=5)

# MODELS
##########
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import (
    ConstantVelocity, CombinedLinearGaussianTransitionModel)
transition_model = CombinedLinearGaussianTransitionModel(
    (ConstantVelocity(0.028**2), ConstantVelocity(0.04**2),
     ConstantVelocity(0.005**2), ConstantVelocity(0.005**2)))
measurement_model = LinearGaussian(
    ndim_state=8, mapping=[0, 2, 4, 6],
    noise_covar=np.diag([0.01**2, 0.01**2, 0.05**2, 0.05**2]))

# MAIN TRACKER COMPONENTS
#########################

# Filtering
from stonesoup.predictor.kalman import (
    UnscentedKalmanPredictor, ExtendedKalmanPredictor)
from stonesoup.updater.kalman import (
    UnscentedKalmanUpdater, ExtendedKalmanUpdater)
predictor = UnscentedKalmanPredictor(transition_model)
updater = UnscentedKalmanUpdater(measurement_model)

# Data Association
from stonesoup.measures import Mahalanobis
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.dataassociator.neighbour import (
    NearestNeighbour, GlobalNearestNeighbour)
# from stonesoup.dataassociator.probability import JPDA
hypothesiser = DistanceHypothesiser(
    predictor, updater, Mahalanobis(), 20)
# hypothesiser = PDAHypothesiser(predictor, updater,
#                                clutter_spatial_density=0.1,
#                                prob_detect=0.9,
#                                prob_gate=0.9995)
# associator = JPDA(hypothesiser, updater, 20)
associator = GlobalNearestNeighbour(hypothesiser)

state_vector = StateVector(np.zeros((8,1)))
covar = CovarianceMatrix(np.diag(np.tile([0.1**2, 0.01**2],4)))
prior_state = GaussianState(state_vector, covar)
initiator = LinearMeasurementInitiator(prior_state, measurement_model)
deleter = UpdateTimeStepsDeleter(50)

# Define the video stream
MODEL_NAME = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
# PATH_TO_CKPT = 'D:/OneDrive/TensorFlow/scripts/camera_control/models/' + MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_CKPT = '/home/denbridge/Documents/Tensorflow/trained-models/output_inference_graph_v5/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/home/denbridge/Documents/Tensorflow/trained-models/output_inference_graph_v5', 'label_map.pbtxt')

# Number of classes to detect
NUM_CLASSES = 90

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Detection
VIDEO_PATH = "/home/denbridge/Documents/Tensorflow/datasets/video7_Trim.mp4"
# videoReader = VideoClipReader(VIDEO_PATH)
videoReader = FFmpegVideoStreamReader('rtsp://admin:admin@10.36.0.158:554/videoinput_1:0/h264_1/media.stm',
                                      input_args=[[],{'threads':1, 'fflags':'nobuffer', 'vsync':0}],
                                      output_args=[[],{'format':'rawvideo', 'pix_fmt':'rgb24'}])
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

client = TcpClient('127.0.0.1', 1563)
pid_controller = PtzPidController()

frameIndex = 0
tracks = set()
frame = ImageFrame([])
frame_old = ImageFrame([])
# with vid_writer.saving(fig, "writer_test7.mp4", 100):

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


def draw_tracks(frame, tracks):
    boxes = np.array(
        [track.state.state_vector[[0, 2, 4, 6]].reshape((4)) for track in tracks
         if (len([state for state in track.states if isinstance(state, Update)]) > 10)])

    for box in boxes:
        xmin, ymin, w, h = box
        xmax = xmin + w
        ymax = ymin + h
        vis_util.draw_bounding_box_on_image_array(frame,
                                                  ymin,
                                                  xmin,
                                                  ymax,
                                                  xmax,
                                                  color='blue',
                                                  thickness=2)
    return frame

fig = plt.figure(figsize=(9, 6))
i = 1

last_command = datetime.datetime.now()
with vid_writer.saving(fig, "denbridge_5.mp4", 100):
    for timestamp, detections in feeder.detections_gen():

        frame = videoReader.frame
        frame_np = copy(frame.pixels)
        tracks = affineCorrection(frame.pixels, frame_old.pixels, tracks)
        #
        frame_np = draw_detections(frame_np, detections)
        #
        # Perform data association
        associations = associator.associate(
            tracks, detections, timestamp)

        # Update tracks based on association hypotheses
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

        if len(frame_np):
            boxes = np.array(
                [track.state.state_vector[[0, 2, 4, 6]].reshape((4)) for track in tracks
                 if (len([state for state in track.states if isinstance(state, Update)]) > 10)])
            areas = [track.state.state_vector[4][0] * track.state.state_vector[6][0] for track in tracks
                     if (len([state for state in track.states if isinstance(state, Update)]) > 10)]

            if (len(boxes)):
                # areas = [box[2][0]*box[3][0] for box in boxes]
                ind = np.argmax(areas)
                box = boxes[ind]
                xmin, ymin, w, h = box
                xmax = xmin + w
                ymax = ymin + h
                color = 'red'
                feedback = {'dt': 0.1}
                box = (ymin, xmin, ymax, xmax)
                commands, _ = pid_controller.run(frame_np, box, feedback)

                print(commands)
                panSpeed, tiltSpeed = (commands["PanTilt"]["panSpeed"],
                                       commands["PanTilt"]["tiltSpeed"])
                zoomSpeed = commands["Zoom"]["zoomSpeed"]

                # print("\nSending request...")
                # response = client.send_request()
                # print(response)

                # Test sending command
                command = {"type": "command",
                           "panSpeed": panSpeed,
                           "tiltSpeed": tiltSpeed}
                if (datetime.datetime.now()-last_command> datetime.timedelta(seconds=0.5)):
                    print("\nSending command: {}".format(json.dumps(command)))
                    response = client.send_command(command)
                    last_command = datetime.datetime.now()

                # if (panSpeed + tiltSpeed == 0):
                #     if (zoomSpeed == 0):
                #         # Stop the camera
                #         if (not converged):
                #             # camera.move_http((panSpeed, tiltSpeed))
                #             camera.continuous_move(panSpeed, tiltSpeed)
                #         converged = True
                #     else:
                #         # Control the zoom
                #         camera.zoom_http(zoomSpeed)
                #         time.sleep(abs(zoomSpeed))
                #         camera.zoom_http(0)
                #         converged = False
                # else:
                #     # Control the pan-tilt
                #     # camera.move_http((panSpeed, tiltSpeed))
                #     camera.continuous_move(panSpeed, tiltSpeed)
                #     converged = False
                #
                # if converged:
                #     color = 'green'
                # # else:
                # for box in boxes:
                #     xmin, ymin, w, h = box
                #     xmax = xmin + w
                #     ymax = ymin + h
                #     vis_util.draw_bounding_box_on_image_array(frame_np,
                #                                               ymin,
                #                                               xmin,
                #                                               ymax,
                #                                               xmax,
                #                                               color=color)
            # Display output

        print(timestamp)


        frame_old = copy(frame)

        if frame:
            frame_np = draw_tracks(frame_np, tracks)
            plt.clf()
            plt.imshow(frame_np)
            plt.pause(0.001)
            # cv2.imshow("", cv2.resize(frame_np, (800, 600)))
            # vid_writer.grab_frame()

        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        #     break
        # i+=1
        # if i>100:
        #     break


a=0