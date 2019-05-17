# -*- coding: utf-8 -*-
from operator import attrgetter
import numpy as np

from ..base import Property
from .base import Feeder


class MetadataReducer(Feeder):
    """Reduce detections so unique metadata value present at each time step.

    This allows to reduce detections so a single detection is returned, based
    on a particular metadata value, for example a unique identity. The most
    recent detection will be yielded for each unique metadata value at each
    time step.
    """

    metadata_field = Property(
        str,
        doc="Field used to reduce unique set of detections")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._detections = set()

    @property
    def detections(self):
        return self._detections

    def detections_gen(self):
        for time, detections in self.detector.detections_gen():
            unique_detections = set()
            sorted_detections = sorted(
                detections, key=attrgetter('timestamp'), reverse=True)
            meta_values = set()
            for detection in sorted_detections:
                meta_value = detection.metadata.get(self.metadata_field)
                if meta_value not in meta_values:
                    unique_detections.add(detection)
                    # Ignore those without meta data value
                    if meta_value is not None:
                        meta_values.add(meta_value)
                self._detections = unique_detections
            yield time, unique_detections


class BoundingBoxReducer(Feeder):
    """Reduce detections by selecting only ones that fall within a given bounding box.
    """

    bounding_box = Property(
        np.ndarray,
        doc="Bounding box used to reduce set of detections. Defined as a vector of min "
            "and max coordinates [x_min, x_max, y_min, y_max, ...].")
    mapping = Property(
        np.ndarray,
        doc="Mapping between bounding box and detection state vector coordinates. Should"
            " be specified as a vector of length `bbox_length/2`, `bbox_length` being "
            "the length of the `bounding_box` vector, whose values correspond to detection "
            "state vector indices."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._detections = set()

    @property
    def detections(self):
        return self._detections

    def detections_gen(self):
        bbox_length = int(self.bounding_box.size/2)
        for time, detections in self.detector.detections_gen():
            filtered_detections = set()
            for detection in detections:
                state_vector = detection.state_vector
                valid = 0
                for i in range(bbox_length):
                    if (state_vector[self.mapping[i]] > self.bounding_box[i*2]
                        and state_vector[self.mapping[i]] < self.bounding_box[i*2+1]):
                        valid += 1
                    else:
                        break
                if valid == bbox_length:
                    filtered_detections.add(detection)
            self._detections = filtered_detections
            yield time, filtered_detections