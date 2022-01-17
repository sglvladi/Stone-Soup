import uuid
from typing import List


from ..base import Property
from .base import Type
from .track import Track
from .state import State


class SensorTracks(Type):
    tracks: List[Track] = Property(doc='A list of tracks', default=None)
    sensor_id: str = Property(doc='The sensor id', default=None)

    def __iter__(self):
        return (t for t in self.tracks)


class SensorTracklets(SensorTracks):
    pass


class Tracklet(Type):
    id: str = Property(doc='The tracklet id', default=None)
    priors: List[State] = Property(doc='The trackler priors', default=None)
    posteriors: List[State] = Property(doc='The tracklet posteriors', default=None)
    sensor_id: str = Property(doc='The sensor id', default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.priors = self.priors if self.priors else []
        self.posteriors = self.posteriors if self.posteriors else []


class Scan(Type):
    id: str = Property(default=None, doc="The unique track ID")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.id is None:
            self.id = str(uuid.uuid4())


class SensorScan(Type):
    id: str = Property(default=None, doc="The unique track ID")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.id is None:
            self.id = str(uuid.uuid4())
