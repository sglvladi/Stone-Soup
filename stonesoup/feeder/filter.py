# -*- coding: utf-8 -*-
from operator import attrgetter

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


class MetadataValueFilter(MetadataReducer):
    """Reduce detections whose metadata field does not satisfy a given condition.

    This allows to reduce detections in cases where an informative metadata field
    exists (e.g. existence likelihood), that allows us to filter out unwanted
    detections.
    """

    operator = Property(object,
                        doc="A binary condition operator/function of the "
                            "form `x=op(val,ref)`, where val is the value"
                            "of the selected `metadata_field` and `ref` "
                            "is the value of the `reference_value`. The "
                            "function should return `True` to indicate "
                            "that a given detection should be filtered "
                            "through (i.e. kept) and `False` otherwise.")
    reference_value = Property(object, doc="The value which will be compared "
                                           "against the metadata value")
    keep_unmatched = Property(bool, doc="Pick whether or not to keep detections"
                                        " that don't have the selected metadata"
                                        " field. Default is False.",
                              default=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._detections = set()

    @property
    def detections(self):
        return self._detections.copy()

    def detections_gen(self):
        for time, detections in self.detector.detections_gen():
            filtered_detections = set()
            for detection in detections:
                meta_value = detection.metadata.get(self.metadata_field)
                if meta_value is None:
                    if self.keep_unmatched:
                        filtered_detections.add(detection)
                    else:
                        continue
                if self.operator(meta_value, self.reference_value):
                    filtered_detections.add(detection)

            self._detections = filtered_detections
            yield time, self.detections
