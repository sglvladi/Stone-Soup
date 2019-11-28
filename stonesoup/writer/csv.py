import csv
import os
from datetime import datetime
import time as tm
import numpy as np
from copy import copy

from .base import Writer
from ..base import Property
from ..functions import mod_bearing
from ..types.prediction import StatePrediction

class CSV_Writer_NZ(Writer):
    """
    This is a CSV Writer for the NZ data, specifically configured
    to work with the "per mmsi" approach. It is expected that a new writer
    will be instantiated for each mmsi file.

    Properties
    ----------
    save_dir: str
        The directory where the output files should be saved
    filename: str
        The name of the
    """
    save_dir = Property(str, doc="Save directory")
    filename = Property(str, doc="Filename")

    def __init__(self, save_dir, mmsi, *args, **kwargs):
        super().__init__(save_dir, mmsi, *args, **kwargs)
        self.fields = ['ID', 'TrackID', 'Latitude', 'Longitude',
                       'ReceivedTime', 'DataType', 'Speed', 'Heading',
                       'LRIMOShipNo', 'MMSI', 'ShipName', 'ShipType',
                       'AdditionalInfo', 'DetectionHistory', 'CallSign',
                       'Draught', 'Destination', 'Location',
                       'StoppedProbability', 'LatitudeDot', 'LongitudeDot']

        with open(os.path.join(self.save_dir,'tracks_{}.csv'.format(
                self.filename)),
                  'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()
        with open(os.path.join(self.save_dir,'points_{}.csv'.format(
                self.filename)),
                  'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()

    def write(self, tracks, detections, timestamp=None):
        """
        Write the tracks and detections to file.

        Given a set of tracks and detections, the method will write these in two
        files:
            - {save_dir}/tracks_{mmsi}.csv: Contains all track information
                that would otherwise be written to "SS_Tracks_Live".
            - {save_dir}/points_{mmsi}.csv: Contains all point information
                that would otherwise be written to "SS_Points_Live".

        """
        # Tracks
        track_documents = []
        point_documents = []
        for track in tracks:
            # if timestamp is not None and track.last_update.timestamp != \
            #         timestamp:
            #     continue
            if isinstance(track.state, StatePrediction):
                continue
            metadata = track.metadata
            latest_position = {
                'Longitude': float(track.last_update.state_vector[0, 0]),
                'Latitude': float(track.last_update.state_vector[2, 0])}
            stopped_prob = track.state.weights[1, 0]
            speed = float(metadata['SOG (knts)']) \
                if (metadata['SOG (knts)'] and not metadata['SOG (knts)'] == 'None')\
                else 0
            heading = float(metadata['COG (deg)']) \
                if (metadata['COG (deg)'] and not metadata['COG (deg)'] == 'None') \
                else 0
            heading = mod_bearing(-np.deg2rad(heading) + np.pi/2)

            detection_history = []
            for state in track.states:
                if hasattr(state, 'hypothesis'):
                    detection_history.append(
                        state.hypothesis.measurement.id)

            # Standardise date/timestamps into both epoch, and date formats
            received_date = datetime.fromtimestamp(
                tm.mktime(timestamp.timetuple()))
            received_epoch_in_ms = self.epoch_in_ms_from_date(received_date)

            # Prepare values to insert
            doc = {
                'ID': track.id,  # TODO: confirm required as well as TrackID
                'TrackID': track.id,
                'Latitude': latest_position.get('Latitude'),
                'LatitudeDot': float(track.last_update[1, 0]),
                'Longitude': latest_position.get('Longitude'),
                'LongitudeDot': float(track.last_update[3, 0]),
                'ReceivedTime': received_epoch_in_ms,
                'ReceivedTimeDate': received_date,
                'DataType': 'fused',
                'Speed': speed,
                'Heading': heading,
                'LRIMOShipNo': int(metadata['IMO']) if ('IMO' in metadata and metadata['IMO']) else -1,
                'MMSI': int(metadata.get('MMSI')),
                'ShipName': metadata['Vessel_Name'] if 'Vessel_Name' in metadata else '',
                'ShipType': metadata['Ship_Type'] if 'Ship_Type' in metadata else '',
                'AdditionalInfo': '',
                'DetectionHistory': detection_history,
                'CallSign': metadata['Call_sign'] if 'Call_sign' in metadata else '',
                'Draught': float(metadata['Draught']) if ('Draught' in metadata and metadata['Draught']) else -1,
                'Destination': metadata['Destination'] if 'Destination' in metadata else '',
                'Location': {
                    'type': "Point",
                    'coordinates': latest_position
                },
                'StoppedProbability': stopped_prob
            }
            for field in self.new_fields:
                dicts = {field: track.metadata.get(field)}
                doc.update(dicts)
            point_documents.append(copy(doc))
            track_documents.append(copy(doc))
        # tracks_collection.insert_many(track_documents)
        with open(os.path.join(self.save_dir,'tracks_{}.csv'.format(
                self.filename)),
                  'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerows(track_documents)

        # Detections
        for detection in detections:
            metadata = detection.metadata
            position = {
                'Longitude': float(detection.state_vector[0, 0]),
                'Latitude': float(detection.state_vector[1, 0]),
            }
            speed = float(metadata['SOG (knts)']) \
                if (metadata['SOG (knts)'] and not metadata['SOG (knts)'] == 'None') \
                else 0
            heading = float(metadata['COG (deg)']) \
                if (metadata['COG (deg)'] and not metadata['COG (deg)'] == 'None') \
                else 0
            heading = mod_bearing(-np.deg2rad(heading) + np.pi / 2)

            # Standardise date/timestamps into both epoch, and date formats
            received_date = datetime.fromtimestamp(
                tm.mktime(detection.timestamp.timetuple()))
            received_epoch_in_ms = self.epoch_in_ms_from_date(received_date)

            # Prepare values to insert
            doc = {
                'ID': metadata["ID"],
                'Latitude': position.get('Latitude'),
                'Longitude': position.get('Longitude'),
                'ReceivedTime': received_epoch_in_ms,
                'ReceivedTimeDate': received_date,
                'DataType': 'self_reported',
                'Speed': speed,
                'Heading': heading,
                'LRIMOShipNo': int(metadata['IMO']) if 'IMO' in metadata else -1,
                'MMSI': int(metadata.get('MMSI')),
                'ShipName': metadata['Vessel_Name'] if 'Vessel_Name' in metadata else '',
                'ShipType': metadata['Ship_Type'] if 'Ship_Type' in metadata else '',
                'AdditionalInfo': '',
                'CallSign': metadata['Call_sign'] if 'Call_sign' in metadata else '',
                'Draught': float(metadata['Draught']) if 'Draught' in metadata else -1,
                'Destination': metadata['Destination'] if 'Destination' in metadata else '',
                'Location': {
                    'type': "Point",
                    'coordinates': position
                }
            }

            # Insert values into Mongo
            point_documents.append(doc)

        with open(os.path.join(self.save_dir,'points_{}.csv'.format(
                self.filename)),
                  'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerows(point_documents)

    @staticmethod
    def date_from_time_str(time_str):
        """Converts time string in format '2017-01-22 08:58:14.000' into a
        datetime.date object, returning None if it fails.
        """
        try:
            return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S.%f')

        except ValueError:
            return None

    @staticmethod
    def epoch_in_ms_from_date(date):
        """Converts datetime.date into an EPOCH in ms, returning None if it
        fails.
        """
        try:
            return int(date.timestamp() * 1000)

        except (AttributeError, OSError):
            # Handle date values of '9999-12-31 23:59:59.000' or None
            return None