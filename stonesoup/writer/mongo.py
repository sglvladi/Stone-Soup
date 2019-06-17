
# -*- coding: utf-8 -*-
import pymongo
from ..base import Property
from ..tracker import Tracker
from .base import Writer
from utm import to_latlon
from datetime import datetime
import time as tm
import numpy as np
from copy import copy
import csv

from ..functions import mod_bearing
from ..functions import cart2pol
# from ..models.measurement.nonlinear import RangeBearingGaussianToCartesian


class MongoWriter(Writer):
    """MongoDB Writer"""

    # def __init__(self):
        # with open('tracks_3.csv', 'a', newline='') as f:
        #     fields = ['ID', 'MMSI', 'LRIMOShipNo', 'Location', 'Latitude', 'Longitude',
        #               'DetectionHistory', 'ShipType', 'ShipName', 'CallSign', 'Beam', 'Draught',
        #               'Length', 'Speed', 'Heading', 'ETA', 'Destination', 'DestinationTidied',
        #               'AdditionalInfo', 'MovementDateTime', 'MovementID', 'MoveStatus', 'Time']
        #
        #     writer = csv.DictWriter(f, fieldnames=fields)
        #     writer.writeheader()

    @staticmethod
    def reset_collections(host_name, host_port, db_name, collection_names):
        """Drops named collection, and recreates indexes for each."""
        # Get list of defined collections
        client = pymongo.MongoClient(host_name, port=host_port)
        db = client[db_name]
        collections = [db[col_name] for col_name in collection_names]

        for collection in collections:
            collection.drop()

            # Add indexes for received time and geo-location fields
            collection.create_index([('ReceivedTime', pymongo.DESCENDING)])
            collection.create_index([('Location', pymongo.GEOSPHERE)])

    def write(self, tracks, detections, host_name, host_port, db_name,
              collection_name, drop=False):
        client = pymongo.MongoClient(host_name, port=host_port)
        db = client[db_name]
        tracks_collection = db[collection_name[0]]
        points_collection = db[collection_name[1]]

        # Pre-process/configure collections
        collections = [tracks_collection, points_collection]
        for collection in collections:
            if drop:
                collection.drop()

        # Tracks
        track_documents = []
        point_documents = []
        for track in tracks:
            metadata = track.metadata
            position = [track.state_vector[0, 0],
                        track.state_vector[2, 0]]
            positions = [[state.state_vector[0, 0],
                          state.state_vector[2, 0]]
                         for state in track.states]
            speed = float(metadata['Speed'])
            heading = mod_bearing(-np.deg2rad(float(metadata['Heading'])) + np.pi/2)

            detection_history = []
            for state in track.states:
                if hasattr(state, 'hypothesis'):
                    detection_history.append(state.hypothesis.measurement.metadata.get('ID'))

            # Standardise date/timestamps into both epoch, and date formats
            received_date = datetime.fromtimestamp(
                tm.mktime(track.timestamp.timetuple()))
            received_epoch_in_ms = self.epoch_in_ms_from_date(received_date)

            eta_date = self.date_from_time_str(metadata['ETA'])
            eta_epoch_in_ms = self.epoch_in_ms_from_date(eta_date)

            movement_date = self.date_from_time_str(metadata['MovementDateTime'])
            movement_epoch_in_ms = self.epoch_in_ms_from_date(movement_date)

            # Prepare values to insert
            doc = {
                'ID': track.id,  # TODO: confirm required as well as TrackID
                'TrackID': track.id,
                'Latitude': float(position[1]),
                'Longitude': float(position[0]),
                'ReceivedTime': received_epoch_in_ms,
                'ReceivedTimeDate': received_date,
                'DataType': 'fused',
                'Speed': speed,
                'Heading': heading,
                'LRIMOShipNo': int(metadata['LRIMOShipNo']),
                'MMSI': int(metadata.get('MMSI')),
                'ShipName': metadata['ShipName'],
                'ShipType': metadata['ShipType'],
                'AdditionalInfo': metadata['AdditionalInfo'],
                'DetectionHistory': detection_history,
                'CallSign': metadata['CallSign'],
                'Beam': float(metadata['Beam']),
                'Draught': float(metadata['Draught']),
                'Length': float(metadata['Length']),
                'ETA': eta_epoch_in_ms,
                'ETADate': eta_date,
                'Destination': metadata['Destination'],
                'DestinationTidied': metadata['DestinationTidied'],
                'MovementDateTime': movement_epoch_in_ms,
                'MovementDateTimeDate': movement_date,
                'MovementID': metadata['MovementID'],
                'MoveStatus': metadata['MoveStatus'],
                'Location': {
                    'type': "Point",
                    'coordinates': [float(position[0]), float(position[1])]
                }
            }
            point_documents.append(copy(doc))

            doc['LocationHistory'] = {
                'type': "MultiPoint",
                'coordinates': positions
            }
            track_documents.append(copy(doc))
        tracks_collection.insert_many(track_documents)

        # Detections
        for detection in detections:
            metadata = detection.metadata
            position = [detection.state_vector[0, 0],
                        detection.state_vector[1, 0]]
            speed = float(metadata['Speed'])
            heading = mod_bearing(
                -np.deg2rad(float(metadata['Heading'])) + np.pi / 2)

            # Standardise date/timestamps into both epoch, and date formats
            received_date = datetime.fromtimestamp(
                tm.mktime(detection.timestamp.timetuple()))
            received_epoch_in_ms = self.epoch_in_ms_from_date(received_date)

            eta_date = self.date_from_time_str(metadata['ETA'])
            eta_epoch_in_ms = self.epoch_in_ms_from_date(eta_date)

            movement_date = self.date_from_time_str(metadata['MovementDateTime'])
            movement_epoch_in_ms = self.epoch_in_ms_from_date(movement_date)

            # Prepare values to insert
            doc = {
                'ID': metadata["ID"],
                'Latitude': float(position[1]),
                'Longitude': float(position[0]),
                'ReceivedTime': received_epoch_in_ms,
                'ReceivedTimeDate': received_date,
                'DataType': 'self_reported',
                'Speed': speed,
                'Heading': heading,
                'LRIMOShipNo': int(metadata['LRIMOShipNo']),
                'MMSI': int(metadata.get('MMSI')),
                'ShipName': metadata['ShipName'],
                'ShipType': metadata['ShipType'],
                'AdditionalInfo': metadata['AdditionalInfo'],
                'CallSign': metadata['CallSign'],
                'Beam': float(metadata['Beam']),
                'Draught': float(metadata['Draught']),
                'Length': float(metadata['Length']),
                'ETA': eta_epoch_in_ms,
                'ETADate': eta_date,
                'Destination': metadata['Destination'],
                'DestinationTidied': metadata['DestinationTidied'],
                'MovementDateTime': movement_epoch_in_ms,
                'MovementDateTimeDate': movement_date,
                'MovementID': metadata['MovementID'],
                'MoveStatus': metadata['MoveStatus'],
                'Location': {
                    'type': "Point",
                    'coordinates': [float(position[0]), float(position[1])]
                }
            }
            # values_list.append(values)
            # Insert values into Mongo
            point_documents.append(doc)
        points_collection.insert_many(point_documents)


        # fields = ['ID', 'DataType', 'MMSI', 'LRIMOShipNo', 'Location',
        #           'Latitude', 'Longitude', 'DetectionHistory', 'ShipType',
        #           'ShipName', 'CallSign', 'Beam', 'Draught',
        #           'Length', 'Speed', 'Heading', 'ETA', 'Destination',
        #           'DestinationTidied', 'AdditionalInfo',
        #           'MovementDateTime', 'MovementID', 'MoveStatus', 'Time']
        # with open('tracks_3.csv', 'a', newline='') as f:
        #     writer = csv.DictWriter(f, fieldnames=fields)
        #     writer.writerows(values_list)

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
