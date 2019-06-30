# -*- coding: utf-8 -*-
from pathlib import Path

from ..base import Base, Property
from ..tracker import Tracker
from ..metricgenerator import MetricGenerator


class Writer(Base):
    """Writer base class"""


class MetricsWriter(Writer):
    """Metrics Writer base class.

    Writes out metrics to some form of storage for analysis.
    """

    metric_generator: MetricGenerator = Property(doc="Source of metric to be written out")


class TrackWriter(Writer):
    """Track Writer base class.

    Writes out tracks to some form of storage for analysis.
    """

    tracker: Tracker = Property(doc="Source of tracks to be written out")


class FileWriter(Writer):
    """Base class for file based writers."""
    path = Property(
        Path,
        doc="Path to file to be opened. Str will be converted to path.")

    def __init__(self, path, *args, **kwargs):
        if not isinstance(path, Path):
            path = Path(path)  # Ensure Path
        super().__init__(path, *args, **kwargs)
        self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if getattr(self, '_file', None):
            self._file.close()

    def __del__(self):
        self.__exit__()