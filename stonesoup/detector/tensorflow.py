# -*- coding: utf-8 -*-
import threading
import numpy as np
import tensorflow as tf
from datetime import datetime

from .base import Detector
from ..types.detection import Detection
from ..types.sensordata import ImageFrame
from ..reader.file import FileReader


class TensorflowObjectDetector(FileReader, Detector):

    def __init__(self, reader, path):
        Detector.__init__(self, reader, path)
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(str(self.path), 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self._detections = []

    @property
    def detections(self):
        """The detections at the current time step.

        This is the set of detections last returned by the
        :meth:`detections_gen` generator, to allow other components, like
        metrics, to access the data.
        """
        return self._detections.copy()

    def detections_gen(self, with_frame = False):
        with self.graph.as_default():
            with tf.Session(graph=self.graph) as sess:
                for timestamp, frame in self.reader.frame_gen():

                    if not frame:
                        self._detections = {}
                        if with_frame:
                            yield timestamp, self.detections, frame.copy()
                        else:
                            yield timestamp, self.detections

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(frame.pixels, axis=0)
                    # Extract image tensor
                    image_tensor = self.graph.get_tensor_by_name(
                        'image_tensor:0')
                    # Extract detection boxes
                    boxes = self.graph.get_tensor_by_name(
                        'detection_boxes:0')
                    # Extract detection scores
                    scores = self.graph.get_tensor_by_name(
                        'detection_scores:0')
                    # Extract detection classes
                    classes = self.graph.get_tensor_by_name(
                        'detection_classes:0')
                    # Extract number of detectionsd
                    num_detections = self.graph.get_tensor_by_name(
                        'num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})

                    classes = np.squeeze(classes).astype(np.int32)
                    boxes = np.squeeze(boxes)
                    scores = np.squeeze(scores)


                    # Empty detection set
                    self._detections = set()
                    for i, box in enumerate(boxes):
                        metadata = {
                            "tf_box": boxes[i],
                            "class": classes[i],
                            "score": scores[i]
                        }
                        box_xy = np.array([[box[1]],
                                           [box[0]],
                                           [box[3] - box[1]],
                                           [box[2] - box[0]]])
                        detection = Detection(
                            state_vector=box_xy,
                            timestamp=timestamp,
                            metadata=metadata)
                        self._detections.add(detection)

                    if with_frame:
                        yield timestamp, self.detections, frame.copy()
                    else:
                        yield timestamp, self.detections

class TensorflowObjectDetectorThreaded(TensorflowObjectDetector):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tf_thread = threading.Thread(target=self.run,)
        self._thread_lock = threading.Lock()
        self._read_event = threading.Event()
        self._tf_thread.start()
        self._timestamp = None
        self._frame = ImageFrame([])

    def detections_gen(self):
        while True:
            self._thread_lock.acquire()
            if self._frame:
                self._thread_lock.release()
            else:
                self._read_event.clear()
                self._thread_lock.release()
                self._read_event.wait()
                self._read_event.clear()

            yield self._timestamp, self.detections

    def run(self):
        with self.graph.as_default():
            with tf.Session(graph=self.graph) as sess:
                for timestamp, frame in self.reader.frame_gen():

                    if not frame:
                        print("[DET] Frame empty or invalid...")
                        continue

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(frame.pixels, axis=0)
                    # Extract image tensor
                    image_tensor = self.graph.get_tensor_by_name(
                        'image_tensor:0')
                    # Extract detection boxes
                    boxes = self.graph.get_tensor_by_name(
                        'detection_boxes:0')
                    # Extract detection scores
                    scores = self.graph.get_tensor_by_name(
                        'detection_scores:0')
                    # Extract detection classes
                    classes = self.graph.get_tensor_by_name(
                        'detection_classes:0')
                    # Extract number of detectionsd
                    num_detections = self.graph.get_tensor_by_name(
                        'num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})

                    classes = np.squeeze(classes).astype(np.int32)
                    boxes = np.squeeze(boxes)
                    scores = np.squeeze(scores)

                    # Empty detection set
                    detections = set()
                    for i, box in enumerate(boxes):
                        metadata = {
                            "tf_box": boxes[i],
                            "class": classes[i],
                            "score": scores[i]
                        }
                        box_xy = np.array([[box[1]],
                                           [box[0]],
                                           [box[3] - box[1]],
                                           [box[2] - box[0]]])
                        detection = Detection(
                            state_vector=box_xy,
                            timestamp=timestamp,
                            metadata=metadata)
                        detections.add(detection)

                    # Aquire threadlock
                    self._thread_lock.acquire()

                    self._detections = detections
                    self._timestamp = timestamp
                    self._frame = frame

                    # Release lock and set read event
                    self._thread_lock.release()
                    self._read_event.set()