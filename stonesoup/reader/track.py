import datetime
import threading
from copy import copy
from queue import Queue
from typing import List
import uuid
import time

import numpy as np

from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..reader.base import Reader
from ..tracker.base import Tracker
from ..types.tracklet import SensorScan, Scan


class DetectionReplayer(Reader):
    scans: List[tuple] = Property(doc='The scans to be replayed')
    run_async: bool = Property(
        doc="If set to ``True``, the reader will read tracks from the tracker asynchronously "
            "and only yield the latest set of tracks when iterated."
            "Defaults to ``False``",
        default=False)
    buffer_size: int = Property(doc='The size of the buffer used to store scans', default=20)
    start_time: datetime.datetime = Property(doc='The simulation start time', default=None)
    auto_start: bool = Property(doc='Whether to auto start the replay, when object is initialised',
                                default=False)
    sleep_interval: datetime.timedelta = Property(doc='Interval for thread to sleep while waiting '
                                                      'for new data', default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Variables used in async mode
        if self.run_async:
            self._buffer = Queue(maxsize=self.buffer_size)
            # Initialise frame capture thread
            self._capture_thread = threading.Thread(target=self._capture)
            self._capture_thread.daemon = True
            self._thread_lock = threading.Lock()
            if self.sleep_interval is None:
                self.sleep_interval = datetime.timedelta(seconds=0)
            if self.auto_start:
                if self.start_time is None:
                    self.start_time = datetime.datetime.now()
                self._capture_thread.start()

    @property
    def detections(self):
        return self.current[1]

    @BufferedGenerator.generator_method
    def scans_gen(self):
        if self.run_async:
            yield from self._scans_gen_async()
        else:
            yield from self._scans_gen()

    def start(self, start_time=False):
        if start_time:
            self.start_time = datetime.datetime.now()
        self._capture_thread.start()

    def _capture(self):
        for dt, detections in self.scans:
            while datetime.datetime.now()-self.start_time < dt:
                # Adding a sleep here (even for 0 sec) allows matplotlib to plot
                # TODO: investigate why!
                time.sleep(self.sleep_interval.total_seconds())
                pass
            with self._thread_lock:
                timestamp = self.start_time + dt
                self._buffer.put((timestamp, detections))

    def _scans_gen(self):
        for timestamp, detections in self.scans:
            yield timestamp, detections

    def _scans_gen_async(self):
        while self._capture_thread.is_alive():
            scan = self._buffer.get()
            yield scan


class TrackReader(Reader):
    tracker: Tracker = Property(doc='Tracker from which to read tracks')
    run_async: bool = Property(
        doc="If set to ``True``, the reader will read tracks from the tracker asynchronously "
            "and only yield the latest set of tracks when iterated."
            "Defaults to ``False``",
        default=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Variables used in async mode
        if self.run_async:
            self._buffer = None
            # Initialise frame capture thread
            self._capture_thread = threading.Thread(target=self._capture)
            self._capture_thread.daemon = True
            self._thread_lock = threading.Lock()
            self._capture_thread.start()

    @property
    def tracks(self):
        return self.current[1]

    @BufferedGenerator.generator_method
    def tracks_gen(self):
        if self.run_async:
            yield from self._tracks_gen_async()
        else:
            yield from self._tracks_gen()

    def _capture(self):
        for timestamp, tracks in self.tracker:
            self._thread_lock.acquire()
            self._buffer = (timestamp, tracks)
            self._thread_lock.release()

    def _tracks_gen(self):
        for timestamp, tracks in self.tracker:
            yield timestamp, tracks

    def _tracks_gen_async(self):
        while self._capture_thread.is_alive():
            if self._buffer is not None:
                self._thread_lock.acquire()
                timestamp, tracks = copy(self._buffer)
                self._buffer = None
                self._thread_lock.release()
                yield timestamp, tracks


class SensorScanReader(Reader):
    detector: Reader = Property(doc='Detector from which to read detections')
    buffer_size: int = Property(doc='The size of the buffer used to store scans', default=20)
    sensor_id: str = Property(doc='The sensor id', default=None)
    run_async: bool = Property(
        doc="If set to ``True``, the reader will read tracks from the tracker asynchronously "
            "and only yield the latest set of tracks when iterated."
            "Defaults to ``False``",
        default=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.sensor_id is None:
            self.sensor_id = str(uuid.uuid4())
        self.buffer = Queue(maxsize=self.buffer_size)

        # Variables used in async mode
        if self.run_async:
            self._buffer = None
            # Initialise frame capture thread
            self._capture_thread = threading.Thread(target=self._capture)
            self._capture_thread.daemon = True
            self._thread_lock = threading.Lock()
            self._capture_thread.start()

    @property
    def scans(self):
        return self.current[1]

    @BufferedGenerator.generator_method
    def scans_gen(self):
        if self.run_async:
            yield from self._scans_gen_async()
        else:
            yield from self._scans_gen()

    def _capture(self):
        for timestamp, detections in self.detector:
            self._thread_lock.acquire()
            scan = SensorScan(self.sensor_id, detections, timestamp=timestamp)
            self._buffer.put(scan)
            self._thread_lock.release()

    def _scans_gen(self):
        for timestamp, detections in self.detector:
            yield timestamp, SensorScan(self.sensor_id, detections, timestamp=timestamp)

    def _scans_gen_async(self):
        while self._capture_thread.is_alive():
            if self._buffer is not None:
                self._thread_lock.acquire()
                scans = []
                while not self.buffer.empty():
                    scans.append(self.buffer.get())
                self._thread_lock.release()
                yield scans


class ScanReader(Reader):
    detectors: List[SensorScanReader] = Property(doc='Detector from which to read detections')
    buffer_size: int = Property(doc='The size of the buffer used to store scans', default=20)
    sensor_id: str = Property(doc='The sensor id', default=None)
    run_async: bool = Property(
        doc="If set to ``True``, the reader will read tracks from the tracker asynchronously "
            "and only yield the latest set of tracks when iterated."
            "Defaults to ``False``",
        default=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.sensor_id is None:
            self.sensor_id = str(uuid.uuid4())
        self.buffer = Queue(maxsize=self.buffer_size)

        # Variables used in async mode
        if self.run_async:
            self._buffer = None
            # Initialise frame capture thread
            self._capture_thread = threading.Thread(target=self._capture)
            self._capture_thread.daemon = True
            self._thread_lock = threading.Lock()
            self._capture_thread.start()

    @property
    def scans(self):
        return self.current[1]

    @BufferedGenerator.generator_method
    def scans_gen(self):
        if self.run_async:
            yield from self._scans_gen_async()
        else:
            yield from self._scans_gen()

    def _capture(self):
        for sensor_scans in zip(*self.detectors):
            sensor_scans = list(sensor_scans)
            sensor_scans.sort(key= lambda x: x.timestamp)
            timestamp = sensor_scans[0].timestamp
            sensor_scans_tmp = []
            scans = []
            for sensor_scan in sensor_scans_tmp:
                if sensor_scan.timestamp != timestamp:
                    scans.append(Scan(None, timestamp, sensor_scans_tmp))
                    timestamp = sensor_scan.timestamp
                sensor_scans_tmp.append(sensor_scan)
            if len(sensor_scans_tmp):
                scans.append(Scan(None, timestamp, sensor_scans_tmp))
            self._thread_lock.acquire()
            self._buffer.put(scans)
            self._thread_lock.release()

    def _scans_gen(self):
        for data in zip(*self.detectors):
            sensor_scans = []
            for d in data:
                sensor_scans += d[1]
            sensor_scans.sort(key= lambda x: x.timestamp)
            timestamp = sensor_scans[0].timestamp
            yield timestamp, Scan(timestamp, timestamp, sensor_scans)

    def _scans_gen_async(self):
        while self._capture_thread.is_alive():
            if self._buffer is not None:
                self._thread_lock.acquire()
                scans = []
                while not self.buffer.empty():
                    scans.append(self.buffer.get()[1])
                self._thread_lock.release()
                yield scans


class ScanAggregator(Reader):
    main_reader: Reader = Property(doc='The reader that sets to clock')
    readers: List[Reader] = Property(doc='The other readers')

    def __init__(self, *args, **kwargs):
        super(ScanAggregator, self).__init__(*args, **kwargs)
        self._buffer = []
        self._reader_gens = [r.scans_gen() for r in self.readers]

    @property
    def scans(self):
        return self.current[1]

    @BufferedGenerator.generator_method
    def scans_gen(self):
        for timestamp, main_scans in self.main_reader:
            if not len(main_scans):
                continue
            scans_tmp = [scan for scan in self._buffer if scan.timestamp <= timestamp]
            self._buffer = [scan for scan in self._buffer if scan not in scans_tmp]
            for reader in self._reader_gens:
                _, scan = next(reader)
                while scan.timestamp <= timestamp:
                    scans_tmp.append(scan)
                    _, scan = next(reader)
                self._buffer.append(scan)
            # for scan in scans_tmp:
            #     scan.start_time = start_time
            scans_tmp.sort(key=lambda x: x.timestamp)
            for scan in scans_tmp:
                for detection in scan.detections:
                    detection.measurement_model.ndim_state = 8
                    detection.measurement_model.mapping = (4,6)
            scan_ts = np.array([scan.timestamp for scan in scans_tmp])
            idx = []
            if len(main_scans):
                idx = np.flatnonzero(np.logical_and(scan_ts>=main_scans[0].start_time, scan_ts<=main_scans[0].end_time))
            for i in idx:
                main_scans[0].sensor_scans.append(scans_tmp[i])
            scans_1 = [s for i, s in enumerate(scans_tmp) if i not in idx]
            if len(scans_1):
                start_time = np.min([s.timestamp for s in scans_1])
                end_time = np.max([s.timestamp for s in scans_1])
                scan = Scan(start_time, end_time, scans_1)
                scans = [scan] + main_scans
            else:
                scans = main_scans
            yield timestamp, scans
