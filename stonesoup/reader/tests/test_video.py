
from matplotlib import pyplot as plt
import cv2
import os
import numpy as np
from copy import copy
import ffmpeg

from stonesoup.reader.video import VideoClipReader, FFmpegVideoStreamReader
from stonesoup.detector.tensorflow import TensorflowObjectDetector, TensorflowObjectDetectorThreaded

# probe = ffmpeg.probe('rtsp://192.168.0.10:554/1/h264minor')
# video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
# width = int(video_stream['width'])
# height = int(video_stream['height'])
#
# process1 = (
#     ffmpeg
#     .input('rtsp://192.168.0.10:554/1/h264minor')
#     .output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=500)
#     .run_async(pipe_stdout=True)
# )
#
# process2 = (
#     ffmpeg
#     .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
#     .output('test.mp4', pix_fmt='yuv420p')
#     .overwrite_output()
#     .run_async(pipe_stdin=True)
# )
#
# while True:
#     in_bytes = process1.stdout.read(width * height * 3)
#     if not in_bytes:
#         break
#     in_frame = (
#         np
#         .frombuffer(in_bytes, np.uint8)
#         .reshape([height, width, 3])
#     )
#
#     process2.stdin.write(
#         in_frame
#         .astype(np.uint8)
#         .tobytes()
#     )
#
# process2.stdin.close()
# process1.wait()
# process2.wait()

# videoReader = VideoClipReader("/home/denbridge/Documents/Tensorflow/datasets/video7_Trim.mp4")
# videoReader = VideoClipReader("test.mp4")
# plt.figure()
# for time, frame in videoReader.frame_gen():
#     print(time)
#     if (frame is not None):
#         print(time)
#         cv2.imshow("", cv2.resize(frame, (800, 600)))
#
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break

videoReader = FFmpegVideoStreamReader('rtsp://admin:admin@10.36.0.158:554/videoinput_1:0/h264_1/media.stm',
                                      input_args=[[],{'threads':1, 'fflags':'nobuffer', 'vsync':0}],
                                      output_args=[[],{'format':'rawvideo', 'pix_fmt':'rgb24'}])
# for time, frame in videoReader.sensor_data_gen():
#     if frame:
#         print(time)
#         cv2.imshow("", cv2.resize(frame.pixels, (800, 600)))
#         # plt.imshow(frame)
#         # plt.pause(0.001)
#
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break

from utils import visualization_utils as vis_util
from utils import label_map_util

# Define the video stream
MODEL_NAME = 'open_dataset_plus_faster_rcnn_coco_v876754'

MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_CKPT = 'D:/OneDrive/TensorFlow/scripts/camera_control/models/' + MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_CKPT = '/home/denbridge/Documents/Tensorflow/trained-models/output_inference_graph_v5/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/home/denbridge/Documents/Tensorflow/trained-models/output_inference_graph_v5', 'label_map.pbtxt')

# Number of classes to detect
NUM_CLASSES = 90

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES,  use_display_name=True)
category_index = label_map_util.create_category_index(categories)


detector = TensorflowObjectDetectorThreaded(videoReader, PATH_TO_CKPT)
for time, detections in detector.detections_gen():
    frame = videoReader.frame
    frame_np = copy(frame.pixels)
    if len(detections):
        boxes = np.array([detection.metadata["tf_box"]
                          for detection in detections])
        classes = np.array([detection.metadata["class"]
                            for detection in detections])
        scores = np.array([detection.metadata["score"]
                           for detection in detections])
        idx = [i for (i, val) in enumerate(classes)
               if scores[i] > 0.3]
                   # and any([val == v for v in self.label_ids]))]

        # Extract valid class boxes
        classes = classes[idx, ...]
        boxes = boxes[idx, ...]
        scores = scores[idx, ...]

        vis_util.visualize_boxes_and_labels_on_image_array(
            frame_np,
            boxes,
            classes,
            scores,
            category_index,
            min_score_thresh=.2,
            use_normalized_coordinates=True,
            line_thickness=4)
        # cv2.imshow("", cv2.resize(frame_np, (800, 600)))
    if (frame is not None):
    #     print(time)
    #     cv2.imshow("", cv2.resize(frame, (800, 600)))
        plt.clf()
        plt.imshow(frame_np)
        plt.pause(0.001)
    #
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    #     break


