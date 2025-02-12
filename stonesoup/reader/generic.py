# -*- coding: utf-8 -*-
"""Generic readers for Stone Soup.

This is a collection of generic readers for Stone Soup, allowing quick reading
of data that is in common formats.
"""

import csv
import time
import threading
from queue import Queue
from datetime import datetime, timedelta
from typing import Sequence, Collection, Mapping, List

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
                             dtype=np.float64),
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
                             dtype=np.float64),
                    timestamp=time,
                    metadata=self._get_metadata(row)))

            # Yield remaining
            yield previous_time, detections


class DetectionReplayer(DetectionReader):
    """A simple detection reader that replays detections from a list of scans.

    The scans are expected to be in the format of a list of tuples, whose first element is a
    :class:`datetime.timedelta` object, representing the time elapsed since the ``start_time``,
    and the second element is a set of detections.
    """
    scans: List[tuple] = Property(
        doc='The scans to be replayed in the form of a list of tuples, whose first element is a '
            ':class:`datetime.timedelta` object, representing the time elapsed since the '
            ':attr:`start_time`, and the second element is a set of detections')
    start_time: datetime = Property(
        doc='The simulation start time. The default is ``None`` in which case it will be set to '
            'the current time when the replay is started via one of the following ways: i) on '
            'instantiation if :attr:`real_time` and :attr:`auto_start` are `True`; ii) when the '
            'object is first iterated; iii) replay is explicitly started via the :meth:`start()` '
            'method', default=None)
    real_time: bool = Property(
        doc='If set to ``True``, the reader will replay the scans in real time, according to the '
            'scan time intervals. Defaults to ``False``',
        default=False)
    buffer_size: int = Property(
        doc='The size of the buffer used to store scans. When the buffer fills up, older data is '
            'discarded in favour of more recent. Takes effect only when :attr:`real_time` is '
            '``True``. Default is 20', default=20)
    auto_start: bool = Property(
        doc='Whether to auto start the replay at object instantiation. Takes effect only when '
            ':attr:`real_time` is ``True``. Defaults to ``False``', default=False)
    sleep_interval: timedelta = Property(
        doc='Interval for thread to sleep while waiting for new data. Takes effect only when '
            ':attr:`real_time` is ``True``. Defaults to None', default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._started = False
        # Variables used in async mode
        if self.real_time:
            self._buffer = Queue(maxsize=self.buffer_size)
            # Initialise frame capture thread
            self._capture_thread = threading.Thread(target=self._capture)
            self._capture_thread.daemon = True
            self._thread_lock = threading.Lock()
            if self.sleep_interval is None:
                self.sleep_interval = timedelta(seconds=0)
            if self.auto_start:
                self.start()

    @property
    def detections(self):
        return self.current[1]

    @BufferedGenerator.generator_method
    def detections_gen(self):
        self.start()
        if self.real_time:
            yield from self._scans_gen_async()
        else:
            yield from self._scans_gen()

    def start(self):
        """ Start the replay, if not already started and set :attr:`start_time` if it is `None`.

         When :attr:`real_time` is ``True`` this method will start the thread that reads and adds
         scans to the buffer in real time.
         """
        if self.start_time is None:
            self.start_time = datetime.now()
        if self.real_time and not self._started:
            self._capture_thread.start()
        self._started = True

    def _capture(self):
        for dt, detections in self.scans:
            while datetime.now()-self.start_time < dt:
                # Adding a sleep here (even for 0 sec) allows matplotlib to plot
                # TODO: investigate why!
                time.sleep(self.sleep_interval.total_seconds())
                pass
            with self._thread_lock:
                timestamp = self.start_time + dt
                self._buffer.put((timestamp, detections))

    def _scans_gen(self):
        for dt, detections in self.scans:
            timestamp = self.start_time + dt
            for detection in detections:
                detection.timestamp = timestamp
            yield timestamp, detections

    def _scans_gen_async(self):
        while self._capture_thread.is_alive():
            scan = self._buffer.get()
            yield scan