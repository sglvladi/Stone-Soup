# -*- coding: utf-8 -*-
import datetime
from math import pi

import requests

from .base import DetectionReader
from ..base import Property
from ..models.measurement import MeasurementModel
from ..types.angle import Bearing
from ..types.detection import Detection

requests.packages.urllib3.disable_warnings()


class NelsonAISReader(DetectionReader):

    data_origin = Property(str)
    start_time = Property(datetime.datetime, default=None)
    timestep = Property(datetime.timedelta,
                        default=datetime.timedelta(minutes=1))
    url = Property(str, default='https://pepys.nelson/requests')

    _start_time_query = """query{{
      dataSources(dataType: Ais, dataOrigin: "{data_origin}"){{
        ... on AisDataSource{{
          messages(
            filter: {{
              broadcastType: Position,
            }},
            orderDirection: Ascending,
            limit: 1 ) {{
            ... on AisPositionBroadcast{{
              receivedTime
            }}
          }}
        }}
      }}
    }}"""

    _query = """query{{
      dataSources(dataType: Ais, dataOrigin: "{data_origin}"){{
        ... on AisDataSource{{
          messages(
            filter: {{
              broadcastType: Position,
              receivedTime: {{
                timeGte: "{start_time}", timeLt: "{end_time}"}}
            }},
            orderDirection: Ascending,
            limit: 100000) {{
            ... on AisPositionBroadcast{{
              mmsi
              receivedTime
              status
              statusText
              position {{
                latitude
                longitude
              }}
            }}
          }}
        }}
      }}
    }}"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._detections = set()

    @property
    def detections(self):
        return self._detections.copy()

    def detections_gen(self):
        if self.start_time is None:
            res = requests.post(
                self.url,
                json={'query': self._start_time_query.format(
                    data_origin=self.data_origin)},
                verify=False)
            res.raise_for_status()
            data = res.json()
            report = data['data']['dataSources'][0]['messages'][0]
            start_time = datetime.datetime.fromisoformat(
                report.pop('receivedTime').rstrip("Z"))
        else:
            start_time = self.start_time

        with requests.Session() as session:
            while True:
                end_time = start_time + self.timestep
                res = session.post(
                    self.url,
                    json={'query': self._query.format(
                        start_time=start_time.isoformat(),
                        end_time=end_time.isoformat(),
                        data_origin=self.data_origin,
                    )},
                    verify=False)
                res.raise_for_status()
                data = res.json()

                self._detections = set()
                messages = data['data']['dataSources'][0]['messages']
                if messages is None:
                    messages = list()
                for report in messages:
                    position = report.pop('position')
                    timestamp = datetime.datetime.fromisoformat(
                        report.pop('receivedTime').rstrip("Z"))
                    detection = Detection(
                        [[float(position['longitude'])],
                         [float(position['latitude'])]],
                        metadata=report,
                        timestamp=end_time)
                    self._detections.add(detection)
                yield end_time, self.detections
                start_time += self.timestep


class NelsonRadarReader(DetectionReader):

    data_origin = Property(str)
    measurement_model = Property(MeasurementModel)
    start_time = Property(datetime.datetime, default=None)
    timestep = Property(datetime.timedelta,
                        default=datetime.timedelta(minutes=1))
    url = Property(str, default='https://pepys.nelson/requests')

    _start_time_query = """query{{
          rad: dataSources(dataType: Radar, dataOrigin: "{data_origin}") {{
          ... on RadarTrackDataSource {{
            sensorTracks (
            orderDirection: Ascending,
            limit: 1) {{
              timeOfInformation
            }}
          }}}}}}"""

    _query = """query{{
          rad: dataSources(dataType: Radar, dataOrigin: "{data_origin}") {{
          ... on RadarTrackDataSource {{
            sensorTracks (filter: {{
              timeOfInformation: {{
                timeGte: "{start_time}", timeLt: "{end_time}"}}
            }},
            orderDirection: Ascending,
            limit: 100000) {{
              timeOfInformation
              sensorTrackID
              position {{
                ... on PolarPosition {{
                  range
                  azimuth
                }}}}
                positionCoordinateSystem {{
                  kind
                  origin
                  orientation
                }}
              }}
            }}
          }}
        }}"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._detections = set()

    @property
    def detections(self):
        return self._detections.copy()

    def detections_gen(self):
        if self.start_time is None:
            res = requests.post(
                self.url,
                json={'query': self._start_time_query.format(
                    data_origin=self.data_origin)},
                verify=False)
            res.raise_for_status()
            data = res.json()
            report = data['data']['rad'][0]['sensorTracks'][0]
            start_time = datetime.datetime.fromisoformat(
                report.pop('timeOfInformation').rstrip("Z"))
        else:
            start_time = self.start_time

        with requests.Session() as session:
            while True:
                end_time = start_time + self.timestep
                res = session.post(
                    self.url,
                    json={'query': self._query.format(
                        start_time=start_time.isoformat(),
                        end_time=end_time.isoformat(),
                        data_origin=self.data_origin,
                    )},
                    verify=False)
                res.raise_for_status()
                data = res.json()

                self._detections = set()
                messages = data['data']['rad'][0]['sensorTracks']
                if messages is None:
                    messages = list()
                for report in messages:
                    position = report.pop('position')
                    timestamp = datetime.datetime.fromisoformat(
                        report.pop('timeOfInformation').rstrip("Z"))
                    detection = Detection(
                        [[pi/2 - position['azimuth']],
                         [float(position['range'])]],
                        measurement_model=self.measurement_model,
                        metadata=report,
                        timestamp=end_time)
                    self._detections.add(detection)
                yield end_time, self.detections
                start_time += self.timestep
