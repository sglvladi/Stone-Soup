# -*- coding: utf-8 -*-
from pymongo import MongoClient
from ..base import Property
from ..tracker import Tracker
from .base import Writer
from utm import to_latlon
import time as tm

from ..types.detection import MissedDetection
from ..models.measurement.nonlinear import RangeBearingGaussianToCartesian


class MongoWriter(Writer):
    """MongoDB Writer"""

    def write(self, tracks, host_name, host_port, db_name, collection_name, drop=False):
        client = MongoClient(host_name, port=host_port)
        db = client[db_name]
        collection = db[collection_name]
        if drop:
            collection.drop()

        for track in tracks:
            # Determine data type
            if 'mmsi' in track.metadata and 'sensorTrackID' in track.metadata:
                datatype = 'fused'
            elif 'sensorTrackID' in track.metadata:
                datatype = 'radar'
            else:
                datatype = 'self_reported'

            # Compute track latlon position
            position = to_latlon(
                track.state_vector[0, 0],
                track.state_vector[2, 0],
                track.metadata['Zone_Number'],
                northern=track.metadata['Northern'])

            # Prepare values to insert
            values = {
                'MMSI': track.metadata.get('mmsi'),
                'dataType': datatype,
                'Latitude': position[0],
                'Longitude': position[1],
                'imo': track.metadata.get('imo'),
                'sensorTrackID': track.metadata.get('sensorTrackID'),
                'shipType': track.metadata.get('shipType'),
                'shipName': track.metadata.get('shipName'),
                'CallSign': track.metadata.get('callsign'),
                'Beam': track.metadata.get('beam'),
                'Draught': track.metadata.get('draught'),
                'Length': track.metadata.get('Length'),
                'Speed': float(track.metadata.get('speed')) if 'speed' in track.metadata else None,
                'heading': float(track.metadata.get('heading')) if 'heading' in track.metadata
                                                                   and track.metadata['heading'] != '' else None,
                'receivedTime': int(tm.mktime(track.timestamp.timetuple()) * 1000),
                'Location': {
                    'type': "Point",
                    'coordinates': [position[1], position[0]]
                },
                'statusText': track.metadata.get('statustext'),
                'status': float(track.metadata.get('status')) if 'status' in track.metadata
                                                                 and track.metadata["status"] != '' else None,
                'course': float(track.metadata.get('course')) if 'course' in track.metadata else None,
                'turn': float(track.metadata.get('turn')) if 'turn' in track.metadata
                                                             and track.metadata['turn'] != '' else None,
                'accuracy': int(track.metadata.get('accuracy')) if 'accuracy' in track.metadata else None,
                'raim': int(track.metadata.get('raim')) if 'raim' in track.metadata else None,
                'manoeuvre': float(track.metadata.get('manoeuvre')) if 'manoeuvre' in track.metadata
                                                                        and track.metadata['manoeuvre']!= '' else None,
                'dsc': int(track.metadata.get('dsc')) if 'dsc' in track.metadata
                                                         and track.metadata['dsc'] != '' else None,
            }

            # Insert values into Mongo
            x = collection.insert_one(values).inserted_id
