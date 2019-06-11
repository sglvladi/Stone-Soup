
# -*- coding: utf-8 -*-
from pymongo import MongoClient
from ..base import Property
from ..tracker import Tracker
from .base import Writer
from utm import to_latlon
import time as tm
import numpy as np
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

    def write(self, tracks, detections, host_name, host_port, db_name,
              collection_name, drop=False):
        client = MongoClient(host_name, port=host_port)
        db = client[db_name]
        tracks_collection = db[collection_name[0]]
        points_collection = db[collection_name[1]]
        if drop:
            tracks_collection.drop()
            points_collection.drop()

        # Tracks
        # values_list = []
        for track in tracks:
            metadata = track.metadata
            position = [track.state_vector[0, 0],
                        track.state_vector[2, 0]]
            speed = float(metadata['Speed'])
            heading = mod_bearing(-np.deg2rad(float(metadata['Heading'])) + np.pi/2)

            detection_history = []
            for state in track.states:
                if hasattr(state, 'hypothesis'):
                    detection_history.append(state.hypothesis.measurement.metadata.get('ID'))

            # Prepare values to insert
            values = {
                'ID': track.id,
                'Latitude': float(position[1]),
                'Longitude': float(position[0]),
                'ReceivedTime': int(tm.mktime(track.timestamp.timetuple()) *
                                    1000),
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
                'ETA': metadata['ETA'],
                'Destination': metadata['Destination'],
                'DestinationTidied': metadata['DestinationTidied'],
                'MovementDateTime': metadata['MovementDateTime'],
                'MovementID': metadata['MovementID'],
                'MoveStatus': metadata['MoveStatus'],
                'Location': {
                    'type': "Point",
                    'coordinates': [float(position[1]), float(position[0])]
                }
            }
            # values_list.append(values)
            # Insert values into Mongo
            x = tracks_collection.insert_one(values).inserted_id
            x = points_collection.insert_one(values).inserted_id

        # Detections
        for detection in detections:
            metadata = detection.metadata
            position = [detection.state_vector[0, 0],
                        detection.state_vector[1, 0]]
            speed = float(metadata['Speed'])
            heading = mod_bearing(
                -np.deg2rad(float(metadata['Heading'])) + np.pi / 2)

            # Prepare values to insert
            values = {
                'ID': metadata["ID"],
                'Latitude': float(position[1]),
                'Longitude': float(position[0]),
                'ReceivedTime': int(tm.mktime(detection.timestamp.timetuple()) *
                                    1000),
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
                'ETA': metadata['ETA'],
                'Destination': metadata['Destination'],
                'DestinationTidied': metadata['DestinationTidied'],
                'MovementDateTime': metadata['MovementDateTime'],
                'MovementID': metadata['MovementID'],
                'MoveStatus': metadata['MoveStatus'],
                'Location': {
                    'type': "Point",
                    'coordinates': [float(position[1]), float(position[0])]
                }
            }
            # values_list.append(values)
            # Insert values into Mongo
            x = points_collection.insert_one(values).inserted_id


        # fields = ['ID', 'DataType', 'MMSI', 'LRIMOShipNo', 'Location',
        #           'Latitude', 'Longitude', 'DetectionHistory', 'ShipType',
        #           'ShipName', 'CallSign', 'Beam', 'Draught',
        #           'Length', 'Speed', 'Heading', 'ETA', 'Destination',
        #           'DestinationTidied', 'AdditionalInfo',
        #           'MovementDateTime', 'MovementID', 'MoveStatus', 'Time']
        # with open('tracks_3.csv', 'a', newline='') as f:
        #     writer = csv.DictWriter(f, fieldnames=fields)
        #     writer.writerows(values_list)