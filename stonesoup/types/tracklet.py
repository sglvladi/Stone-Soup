import datetime
import uuid
from typing import Set, List

from ..base import Property
from .base import Type
from .track import Track
from .detection import Detection


class Tracklet(Track):
    pass


class SensorTracks(Type):
    """ A container object for tracks relating to a particular sensor """
    tracks: Set[Track] = Property(doc='A list of tracks', default=None)
    sensor_id: str = Property(doc='The id of the sensor', default=None)

    def __iter__(self):
        return (t for t in self.tracks)


class SensorTracklets(SensorTracks):
    """ A container object for tracklets relating to a particular sensor """
    pass


class SensorScan(Type):
    """ A wrapper around a set of detections produced by a particular sensor """
    sensor_id: str = Property(doc='The id of the sensor')
    detections: Set[Detection] = Property(doc='The detections contained in the scan')
    id: str = Property(default=None, doc="The unique scan ID")
    timestamp: datetime.datetime = Property(default=None, doc='The scan timestamp')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.id is None:
            self.id = str(uuid.uuid4())


class Scan(Type):
    """ A wrapper around a set of sensor scans within a given time interval  """
    start_time: datetime.datetime = Property(doc='The scan start time')
    end_time: datetime.datetime = Property(doc='The scan end time')
    sensor_scans: List[SensorScan] = Property(doc='The sensor scans')
    id: str = Property(default=None, doc="The unique scan ID")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.id is None:
            self.id = str(uuid.uuid4())


