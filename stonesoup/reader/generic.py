# -*- coding: utf-8 -*-
"""Generic readers for Stone Soup.

This is a collection of generic readers for Stone Soup, allowing quick reading
of data that is in common formats.
"""

import csv
from datetime import datetime, timedelta
from typing import Sequence, Collection, Mapping

from math import modf

import numpy as np
from dateutil.parser import parse

from .base import GroundTruthReader, DetectionReader
from .file import TextFileReader
from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..types.detection import Detection
from ..types.groundtruth import GroundTruthPath, GroundTruthState


class _CSVReader(TextFileReader):
    state_vector_fields: Sequence[str] = Property(
        doc='List of columns names to be used in state vector')
    time_field: str = Property(
        doc='Name of column to be used as time field')
    time_field_format: str = Property(
        default=None, doc='Optional datetime format')
    timestamp: bool = Property(
        default=False, doc='Treat time field as a timestamp from epoch')
    metadata_fields: Collection[str] = Property(
        default=None, doc='List of columns to be saved as metadata, default all')
    csv_options: Mapping = Property(
        default={}, doc='Keyword arguments for the underlying csv reader')

    def _get_metadata(self, row):
        if self.metadata_fields is None:
            local_metadata = dict(row)
            for key in list(local_metadata):
                if key == self.time_field or key in self.state_vector_fields:
                    del local_metadata[key]
        else:
            local_metadata = {field: row[field]
                              for field in self.metadata_fields
                              if field in row}
        return local_metadata

    def _get_time(self, row):
        if self.time_field_format is not None:
            time_field_value = datetime.strptime(row[self.time_field], self.time_field_format)
        elif self.timestamp is True:
            fractional, timestamp = modf(float(row[self.time_field]))
            time_field_value = datetime.utcfromtimestamp(int(timestamp))
            time_field_value += timedelta(microseconds=fractional * 1E6)
        else:
            time_field_value = parse(row[self.time_field], ignoretz=True)
        return time_field_value


class CSVGroundTruthReader(GroundTruthReader, _CSVReader):
    """A simple reader for csv files of truth data.

    CSV file must have headers, as these are used to determine which fields
    to use to generate the ground truth state. Those states with the same ID will be put into
    a :class:`~.GroundTruthPath` in sequence, and all paths that are updated at the same time
    are yielded together, and such assumes file is in time order.

    Parameters
    ----------
    """
    path_id_field: str = Property(doc='Name of column to be used as path ID')

    @BufferedGenerator.generator_method
    def groundtruth_paths_gen(self):
        with self.path.open(encoding=self.encoding, newline='') as csv_file:
            groundtruth_dict = {}
            updated_paths = set()
            previous_time = None
            for row in csv.DictReader(csv_file, **self.csv_options):

                time = self._get_time(row)
                if previous_time is not None and previous_time != time:
                    yield previous_time, updated_paths
                    updated_paths = set()
                previous_time = time

                state = GroundTruthState(
                    np.array([[row[col_name]] for col_name in self.state_vector_fields],
                             dtype=np.float_),
                    timestamp=time,
                    metadata=self._get_metadata(row))

                id_ = row[self.path_id_field]
                if id_ not in groundtruth_dict:
                    groundtruth_dict[id_] = GroundTruthPath(id=id_)
                groundtruth_path = groundtruth_dict[id_]
                groundtruth_path.append(state)
                updated_paths.add(groundtruth_path)

            # Yield remaining
            yield previous_time, updated_paths


class CSVDetectionReader(DetectionReader, _CSVReader):
    """A simple detection reader for csv files of detections.

    CSV file must have headers, as these are used to determine which fields to use to generate
    the detection. Detections at the same time are yielded together, and such assume file is in
    time order.

    Parameters
    ----------
    """

    @BufferedGenerator.generator_method
    def detections_gen(self):
        with self.path.open(encoding=self.encoding, newline='') as csv_file:
            detections = set()
            previous_time = None
            for row in csv.DictReader(csv_file, **self.csv_options):

                time = self._get_time(row)
                if previous_time is not None and previous_time != time:
                    yield previous_time, detections
                    detections = set()
                previous_time = time

                detections.add(Detection(
                    np.array([[row[col_name]] for col_name in self.state_vector_fields],
                             dtype=np.float_),
                    timestamp=time,
                    metadata=self._get_metadata(row)))

            # Yield remaining
            yield previous_time, detections


class CSVDetectionReader_EE(CSVDetectionReader):
    """A modified CSV detection reader for the Exact Earth dataset

    This class adds a metadata field "type" to flag each detection as being
    either "static" or "dynamic"
    """

	@BufferedGenerator.generator_method
    def detections_gen(self):
        with self.path.open(encoding=self.encoding, newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            sortedlist = sorted(reader, key=lambda row: row[self.time_field],
                                reverse=False)

            for row in sortedlist:
                time_field_value = self._get_time(row)

                if self.metadata_fields is None:
                    local_metadata = dict(row)
                    copy_local_metadata = dict(local_metadata)
                    for (key, value) in copy_local_metadata.items():
                        if (key == self.time_field) or\
                                (key in self.state_vector_fields):
                            del local_metadata[key]
                else:
                    local_metadata = {field: row[field]
                                      for field in self.metadata_fields
                                      if field in row}

                # Flag dynamic/static detections
                valid = 0
                for col_name in self.state_vector_fields:
                    if row[col_name] != "":
                        valid += 1
                if valid == len(self.state_vector_fields):
                    local_metadata["type"] = "dynamic"
                    detect = Detection(np.array(
                        [[row[col_name]] for col_name in self.state_vector_fields],
                        dtype=np.float32), time_field_value,
                        metadata=local_metadata)
                    self._detections = {detect}
                elif valid == 0:
                    local_metadata["type"] = "static"
                    detect = Detection(np.array(
                        [[0]],dtype=np.float32),
                        time_field_value,
                        metadata=local_metadata)
                    self._detections = {detect}
                else:
                    self._detections = set()

                yield time_field_value, self.detections
