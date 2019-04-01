# -*- coding: utf-8 -*-
from pymongo import MongoClient
from ..base import Property
from ..tracker import Tracker
from .base import Writer
from utm import to_latlon
import time as tm
from ..models.measurement.nonlinear import RangeBearingGaussianToCartesian


class MongoWriter(Writer):
    """MongoDB Writer"""
    """Parameters"""
    #tracks_source = Property(Tracker)

    def write(self, tracks, host_name, host_port, db_name, collection_name, drop=False):
        # client = MongoClient()
        client = MongoClient(host_name, port=host_port)
        db = client[db_name]
        collection = db[collection_name]
        if drop:
            collection.drop()

        for track in tracks:
            try:
                measurement = track.hypothesis.measurement
            except:
                continue
            if measurement:
                # print(track.metadata)
                # print(time)
                if isinstance(measurement.measurement_model,
                              RangeBearingGaussianToCartesian):
                    position = to_latlon(
                        track.state_vector[0, 0],
                        track.state_vector[2, 0], 30,
                        # measurement.metadata['Zone_Number'],
                        northern=True)  # measurement.metadata
                    # ['Northern'])
                else:
                    position = to_latlon(
                        measurement.state_vector[0, 0],
                        measurement.state_vector[1, 0],
                        measurement.metadata['Zone_Number'],
                        northern=measurement.metadata
                        ['Northern'])

                values = {
                    'MMSI': track.metadata.get('mmsi'),
                    # if track.measurement_model==None,  # This is AIS:self_reported, radar:radar, other:fused
                    'dataType': 'radar' if isinstance(measurement.measurement_model,
                                                      RangeBearingGaussianToCartesian) else 'self_reported',
                    'Latitude': position[0],
                    'Longitude':  position[1],
                    'LRIMOShipNo': track.metadata.get('imo'),
                    'ShipType': track.metadata.get('ShipType'),
                    'ShipName': track.metadata.get('ShipName'),
                    'CallSign': track.metadata.get('CallSign'),
                    'Beam': track.metadata.get('beam'),
                    'Draught': track.metadata.get('draught'),
                    'Length': track.metadata.get('Length'),
                    'Speed': float(track.metadata.get('sog')) if 'sog' in track.metadata else None,
                    'Heading': float(track.metadata.get('heading')) if 'heading' in track.metadata else None,
                    'ETA': None,
                    'Destination': None,
                    'DestinationTidied': None,
                    'AdditionalInfo': None,
                    'MovementDateTime': None,
                    'MovementID': None,
                    'MoveStatus': None,
                    'Time': int(tm.mktime(track.timestamp.timetuple())*1000),
                    'Location': {
                        # Longitude
                        '0':    position[1],
                        # Latitude
                        '1': position[0],
                    },
                    'ETA__Date': None,
                    'MovementDateTime__Date': None,
                    'Time__Date': None,

                    # New fields
                    # 'status': float(track.metadata.get('status')),
                    # 'statusText': track.metadata.get('statustext'),
                    # 'course': float(track.metadata.get('cog')),
                    # 'turn': float(track.metadata.get('rot')),
                    # 'accuracy': int(track.metadata.get('accuracy')),
                    # 'raim': int(track.metadata.get('raim')),
                    # 'manoeuvre': float(track.metadata.get('man')),
                    # 'dsc': int(track.metadata.get('dsc')),
                    # 'all': track.metadata.get('all'),
                }

                # print(values)
                x = collection.insert_one(
                    values).inserted_id


# class MongoWriter2(Writer):
#     """MongoDB Writer"""
#     """Parameters"""
#     # tracks = Property(tracks)

#     def __init__(self, host, port, db, collection, overwite=False):
#         super().__init__(host, port, db, collection * args, **kwargs)
#         client = MongoClient()
#         client = MongoClient(host_name, port=host_port)
#         db = client[db_name]
#         collection = db[collection_name]
#         if overwite:
#             collection.drop()

#     def write(self, tracks):
#         for track in tracks:
#             measurement = track.hypothesis.measurement_model
#             if measurement:
#                 if isinstance(measurement.measurement_model,
#                               RangeBearingGaussianToCartesian):
#                     position = to_latlon(
#                         track.state_vector[0, 0],
#                         track.state_vector[3, 0],
#                         measurement.metadata['Zone_number']
#                     )
