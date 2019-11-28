# -*- coding: utf-8 -*-
import pymongo
from datetime import datetime
import time as tm
import numpy as np
from copy import copy

from .base import Writer
from ..base import Property
from ..types.prediction import Prediction
from ..functions import mod_bearing

class MongoWriter(Writer):
    """MongoDB Writer"""

    host_name = Property(str, doc="The host name or IP address")
    host_port = Property(str, doc="The host port")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = pymongo.MongoClient(self.host_name, port=self.host_port)

    def reset_collections(self, db_name, collection_names):
        """Drops named collection, and recreates indexes for each."""
        # Get list of defined collections
        db = self._client[db_name]
        collections = [db[col_name] for col_name in collection_names]

        for collection in collections:
            collection.drop()

            # Add indexes for received time and geo-location fields
            collection.create_index([('ReceivedTime', pymongo.DESCENDING)])
            collection.create_index([('Location', pymongo.GEOSPHERE)])

    def write(self, tracks, detections, db_name, collection_name, drop=False):
        # Catch server timeout error
        try:
            self._client.server_info()
        except pymongo.errors.ServerSelectionTimeoutError as err:
            # do whatever you need
            print(err)
            print("Reconnecting.....")
            self._client = pymongo.MongoClient(self.host_name, port=self.host_port)

        db = self._client[db_name]
        tracks_collection = db[collection_name[0]]
        points_collection = db[collection_name[1]]

        # Tracks
        track_documents = []
        point_documents = []
        for track in tracks:
            if isinstance(track.state, Prediction):
                continue
            metadata = track.metadata
            latest_position = {
                'Longitude': float(track.last_update.state_vector[0, 0]),
                'Latitude': float(track.last_update.state_vector[2, 0])}
            positions = [{'Longitude': float(state.state_vector[0, 0]),
                          'Latitude': float(state.state_vector[2, 0])}
                         for state in track.states]

            speed = float(metadata['SOG']) \
                if (metadata['SOG'] and not metadata['SOG'] == 'None') \
                else 0
            heading = float(metadata['Heading']) \
                if (metadata['Heading'] and not metadata['Heading'] == 'None') \
                else 0
            heading = mod_bearing(-np.deg2rad(heading) + np.pi / 2)

            detection_history = []
            for state in track.states:
                if hasattr(state, 'hypothesis'):
                    detection_history.append(
                        state.hypothesis.measurement.metadata.get('_id'))

            # Standardise date/timestamps into both epoch, and date formats
            received_date = datetime.fromtimestamp(
                tm.mktime(track.timestamp.timetuple()))
            received_epoch_in_ms = self.epoch_in_ms_from_date(received_date)

            # Prepare values to insert
            doc = {
                'ID': track.id,  # TODO: confirm required as well as TrackID
                'TrackID': track.id,
                'Latitude': latest_position.get('Latitude'),
                'Longitude': latest_position.get('Longitude'),
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
                # 'StoppedProbability': stopped_prob
            }
            point_documents.append(copy(doc))
            track_documents.append(copy(doc))

        # Write docs to Mongo
        tracks_collection.insert_many(track_documents)
        points_collection.insert_many(track_documents)

        # Detections
        for detection in detections:
            metadata = detection.metadata
            position = {
                'Longitude': float(detection.state_vector[0, 0]),
                'Latitude': float(detection.state_vector[1, 0]),
            }
            speed = float(metadata['SOG']) \
                if (metadata['SOG'] and not metadata['SOG'] == 'None') \
                else 0
            heading = float(metadata['Heading']) \
                if (metadata['Heading'] and not metadata['Heading'] == 'None') \
                else 0
            heading = mod_bearing(-np.deg2rad(heading) + np.pi / 2)

            # Standardise date/timestamps into both epoch, and date formats
            received_date = datetime.fromtimestamp(
                tm.mktime(detection.timestamp.timetuple()))
            received_epoch_in_ms = self.epoch_in_ms_from_date(received_date)

            # Prepare values to insert
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
            point_documents.append(copy(doc))

        # Insert values into Mongo
        points_collection.insert_many(point_documents)

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