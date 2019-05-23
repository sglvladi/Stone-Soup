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

    def __init__(self):
        with open('tracks_3.csv', 'a', newline='') as f:
            fields = ['ID', 'MMSI', 'LRIMOShipNo', 'Location', 'Latitude', 'Longitude',
                      'DetectionHistory', 'ShipType', 'ShipName', 'CallSign', 'Beam', 'Draught',
                      'Length', 'Speed', 'Heading', 'ETA', 'Destination', 'DestinationTidied',
                      'AdditionalInfo', 'MovementDateTime', 'MovementID', 'MoveStatus', 'Time']

            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

    def write(self, tracks, host_name, host_port, db_name, collection_name, drop=False):
        # client = MongoClient(host_name, port=host_port)
        # db = client[db_name]
        # collection = db[collection_name]
        # if drop:
        #     collection.drop()

        values_list = []
        for track in tracks:
            metadata = track.metadata

            # # Determine data type
            # if 'mmsi' in metadata and 'sensorTrackID' in metadata:
            #     datatype = 'fused'
            # elif 'sensorTrackID' in metadata:
            #     datatype = 'radar'
            # else:
            #     datatype = 'self_reported'

            # Compute track latlon position and speed & heading
            # position = to_latlon(
            #     track.state_vector[0, 0],
            #     track.state_vector[2, 0],
            #     metadata['Zone_Number'],
            #     northern=metadata['Northern'])
            # speed, heading = cart2pol(track.state_vector[1, 0],
            #                           track.state_vector[3, 0])
            # heading = mod_bearing(- heading + np.pi / 2)
            position = [track.state_vector[0, 0],
                        track.state_vector[2, 0]]
            speed = float(metadata['Speed'])
            heading = mod_bearing(-np.deg2rad(float(metadata['Heading'])) + np.pi/2)

            detection_history = []
            for state in track.states:
                if hasattr(state, 'hypothesis'):
                    detection_history.append(state.hypothesis.measurement.metadata.get('ID'))
            fields = ['ID', 'MMSI', 'LRIMOShipNo', 'Location', 'Latitude', 'Longitude',
                      'DetectionHistory', 'ShipType', 'ShipName', 'CallSign', 'Beam', 'Draught',
                      'Length', 'Speed', 'Heading', 'ETA', 'Destination', 'DestinationTidied',
                      'AdditionalInfo', 'MovementDateTime', 'MovementID', 'MoveStatus', 'Time']

            # Prepare values to insert
            values = {
                'MMSI': metadata.get('MMSI'),
                'ID': track.id,
                'DetectionHistory': detection_history,
                'LRIMOShipNo': metadata['LRIMOShipNo'],
                'ShipType': metadata['ShipType'],
                'ShipName': metadata['ShipName'],
                'CallSign': metadata['CallSign'],
                'Beam': metadata['Beam'],
                'Draught': metadata['Draught'],
                'Length': metadata['Length'],
                'Latitude': position[1],
                'Longitude': position[0],
                'Speed': speed,
                'Heading': heading,
                'ETA': metadata['ETA'],
                'Destination': metadata['Destination'],
                'DestinationTidied': metadata['DestinationTidied'],
                'AdditionalInfo': metadata['AdditionalInfo'],
                'MovementDateTime': metadata['MovementDateTime'],
                'MovementID': metadata['MovementID'],
                'MoveStatus': metadata['MoveStatus'],
                'Time': int(tm.mktime(track.timestamp.timetuple()) * 1000),
                'Location': {
                    'type': "Point",
                    'coordinates': [position[1], position[0]]
                }
            }
            values_list.append(values)
            # Insert values into Mongo
            # x = collection.insert_one(values).inserted_id

        with open('tracks_3.csv', 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writerows(values_list)