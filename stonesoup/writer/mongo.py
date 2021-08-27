
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
from ..types.prediction import StatePrediction


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
            positions = [{'Longitude': float(state.state_vector[0, 0]),
                          'Latitude': float(state.state_vector[2, 0])}
                         for state in track.states]
            latest_position = positions[-1]

            speed = float(metadata['SOG'])
            heading = mod_bearing(-np.deg2rad(float(metadata['Heading'])) + np.pi/2)

            detection_history = []
            for state in track.states:
                if hasattr(state, 'hypothesis'):
                    detection_history.append(
                        state.hypothesis.measurement.metadata.get('_id'))

            # Standardise date/timestamps into both epoch, and date formats
            received_date = datetime.fromtimestamp(
                tm.mktime(track.timestamp.timetuple()))
            received_epoch_in_ms = self.epoch_in_ms_from_date(received_date)

            #  eta_date = self.date_from_time_str(metadata['ETA'])
            # eta_epoch_in_ms = self.epoch_in_ms_from_date(eta_date)

            # movement_date = self.date_from_time_str(metadata[
            # 'MovementDateTime'])
            # movement_epoch_in_ms = self.epoch_in_ms_from_date(movement_date)

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
                'LRIMOShipNo': int(metadata['IMO']),
                'MMSI': int(metadata.get('MMSI')),
                'ShipName': metadata['ShipName'],
                'ShipType': metadata['ShipType'],
                #'AdditionalInfo': metadata['AdditionalInfo'],
                'DetectionHistory': detection_history,
                'CallSign': metadata['CallSign'],
                'Beam': float(metadata['Beam']),
                'Draught': float(metadata['Draught']),
                'Length': float(metadata['Length']),
                # 'ETA': eta_epoch_in_ms,
                # 'ETADate': eta_date,
                'Destination': metadata['Destination'],
                'DestinationTidied': metadata['DestinationTidied'],
                # 'MovementDateTime': movement_epoch_in_ms,
                # 'MovementDateTimeDate': movement_date,
                'MovementID': metadata['MovementID'],
                'MoveStatus': metadata['MoveStatus'],
                'Location': {
                    'type': "Point",
                    'coordinates': latest_position
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
            position = {
                'Longitude': float(detection.state_vector[0, 0]),
                'Latitude': float(detection.state_vector[1, 0]),
            }
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
                'Latitude': position.get('Latitude'),
                'Longitude': position.get('Longitude'),
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
                    'coordinates': position
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


class MongoWriter_EE(MongoWriter):

    def __init__(self):
        self.main_fields = ['ID', 'TrackID', 'Latitude', 'Longitude',
                            'ReceivedTime', 'ReceivedTimeDate', 'DataType',
                            'Speed', 'Heading', 'LRIMOShipNo', 'MMSI',
                            'ShipName', 'ShipType', 'AdditionalInfo',
                            'DetectionHistory', 'CallSign', 'Draught',
                            'Destination', 'Location']
        self.new_fields = ['Message_ID', 'Repeat_indicator', 'Time',
                           'Millisecond', 'Region', 'Country', 'Base_station',
                           'Online_data', 'Group_code', 'Sequence_ID',
                           'Channel', 'Data_length', 'Call_sign', 'IMO',
                           'Dimension_to_Bow', 'Dimension_to_stern',
                           'Dimension_to_port', 'Dimension_to_starboard',
                           'AIS_version', 'Navigational_status', 'ROT',
                           'Accuracy', 'RawLongitude', 'RawLatitude', 'COG',
                           'Regional', 'Maneuver', 'RAIM_flag',
                           'Communication_flag', 'Communication_state',
                           'UTC_year', 'UTC_month', 'UTC_day', 'UTC_hour',
                           'UTC_minute', 'UTC_second', 'Fixing_device',
                           'Transmission_control', 'ETA_month', 'ETA_day',
                           'ETA_hour', 'ETA_minute', 'Sequence',
                           'Destination_ID', 'Retransmit_flag', 'Country_code',
                           'Functional_ID', 'Data', 'Destination_ID_1',
                           'Sequence_1', 'Destination_ID_2', 'Sequence_2',
                           'Destination_ID_3', 'Sequence_3',
                           'Destination_ID_4', 'Sequence_4', 'Altitude',
                           'Altitude_sensor', 'Data_terminal', 'Mode',
                           'Safety_text', 'Non-standard_bits',
                           'Name_extension', 'Name_extension_padding',
                           'Message_ID_1_1', 'Offset_1_1', 'Message_ID_1_2',
                           'Offset_1_2', 'Message_ID_2_1', 'Offset_2_1',
                           'Destination_ID_A', 'Offset_A', 'Increment_A',
                           'Destination_ID_B', 'offsetB', 'incrementB',
                           'data_msg_type', 'station_ID', 'Z_count',
                           'num_data_words', 'health', 'unit_flag',
                           'display', 'DSC', 'band', 'msg22', 'offset1',
                           'num_slots1', 'timeout1', 'Increment_1', 'Offset_2',
                           'Number_slots_2', 'Timeout_2', 'Increment_2',
                           'Offset_3', 'Number_slots_3', 'Timeout_3',
                           'Increment_3', 'Offset_4', 'Number_slots_4',
                           'Timeout_4', 'Increment_4', 'ATON_type',
                           'ATON_name', 'off_position', 'ATON_status',
                           'Virtual_ATON', 'Channel_A', 'Channel_B',
                           'Tx_Rx_mode', 'Power', 'Message_indicator',
                           'Channel_A_bandwidth', 'Channel_B_bandwidth',
                           'Transzone_size', 'Longitude_1', 'Latitude_1',
                           'Longitude_2', 'Latitude_2', 'Station_Type',
                           'Report_Interval', 'Quiet_Time', 'Part_Number',
                           'Vendor_ID', 'Mother_ship_MMSI',
                           'Destination_indicator', 'Binary_flag',
                           'GNSS_status', 'spare', 'spare2', 'spare3',
                           'spare4', 'InvalidLonLat', 'ZeroZeroLonLat',
                           'ModifiedLonLat', 'type']
        self.fields = self.main_fields + self.new_fields
        # with open('ss_tracks_live_ee2.csv', 'a', newline='') as f:
        #     writer = csv.DictWriter(f, fieldnames=self.fields)
        #     writer.writeheader()
        # with open('ss_points_live_ee2.csv', 'a', newline='') as f:
        #     writer = csv.DictWriter(f, fieldnames=self.fields)
        #     writer.writeheader()

    def write(self, tracks, detections, host_name, host_port, db_name,
              collection_name, drop=False, timestamp=None):
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
            # if timestamp is not None and track.last_update.timestamp != \
            #         timestamp:
            #     continue
            if isinstance(track.state, StatePrediction):
                continue
            metadata = track.metadata
            latest_position = {
                'Longitude': float(track.state.state_vector[0, 0]),
                'Latitude': float(track.state.state_vector[2, 0])}
            stopped_prob = track.state.weights[1, 0]
            speed = float(metadata['SOG']) \
                if (metadata['SOG'] and not metadata['SOG'] == 'None')\
                else 0
            heading = float(metadata['Heading']) \
                if (metadata['Heading'] and not metadata['Heading'] == 'None') \
                else 0
            heading = mod_bearing(-np.deg2rad(heading) + np.pi/2)

            detection_history = []
            for state in track.states:
                if hasattr(state, 'hypothesis'):
                    detection_history.append(
                        state.hypothesis.measurement.metadata.get('_id'))

            # Standardise date/timestamps into both epoch, and date formats
            received_date = datetime.fromtimestamp(
                tm.mktime(track.timestamp.timetuple()))
            received_epoch_in_ms = self.epoch_in_ms_from_date(received_date)

            # eta_date = self.date_from_time_str(metadata['ETA'])
            # eta_epoch_in_ms = self.epoch_in_ms_from_date(eta_date)

            # movement_date = self.date_from_time_str(metadata[
            # 'MovementDateTime'])
            # movement_epoch_in_ms = self.epoch_in_ms_from_date(movement_date)
            # used_fields = ['ID', 'TrackID', 'Latitude', 'Longitude', 'ReceivedTime',
            #                'ReceivedTimeDate', 'DataType', 'SOG', 'Heading',
            #                'LRIMOShipNo', 'MMSI', 'ShipName', 'ShipType',
            #                'AdditionalInfo', 'DetectionHistory', 'CallSign',
            #                'Draught', 'Destination', 'Location']
            # new_fields = [field for field in track.metadata
            #               if field not in used_fields]
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
                'LRIMOShipNo': int(float(metadata['IMO'])) if ('IMO' in metadata and metadata['IMO']) else -1,
                'MMSI': int(metadata.get('MMSI')),
                'ShipName': metadata['Vessel_Name'] if 'Vessel_Name' in metadata else '',
                'ShipType': metadata['Ship_Type'] if 'Ship_Type' in metadata else '',
                'AdditionalInfo': '',
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
        tracks_collection.insert_many(track_documents)
        # with open('ss_tracks_live_ee2.csv', 'a', newline='') as f:
        #     writer = csv.DictWriter(f, fieldnames=self.fields)
        #     writer.writerows(track_documents)

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

            # eta_date = self.date_from_time_str(metadata['ETA'])
            # eta_epoch_in_ms = self.epoch_in_ms_from_date(eta_date)
            #
            # movement_date = self.date_from_time_str(metadata['MovementDateTime'])
            # movement_epoch_in_ms = self.epoch_in_ms_from_date(movement_date)

            # Prepare values to insert

            doc = {
                # 'ID': metadata["_id"],
                'ID': metadata["ID"],
                'Latitude': position.get('Latitude'),
                'Longitude': position.get('Longitude'),
                'ReceivedTime': received_epoch_in_ms,
                'ReceivedTimeDate': received_date,
                'DataType': 'self_reported',
                'Speed': speed,
                'Heading': heading,
                'LRIMOShipNo': int(metadata['IMO']) if 'IMO' in metadata and metadata['IMO'] else -1,
                'MMSI': int(metadata.get('MMSI')),
                'ShipName': metadata['Vessel_Name'] if 'Vessel_Name' in metadata else '',
                'ShipType': metadata['Ship_Type'] if 'Ship_Type' in metadata else '',
                'AdditionalInfo': '',
                'CallSign': metadata['Call_sign'] if 'Call_sign' in metadata else '',
                'Draught': float(metadata['Draught']) if 'Draught' in metadata and metadata['Draught'] else -1,
                'Destination': metadata['Destination'] if 'Destination' in metadata else '',
                'Location': {
                    'type': "Point",
                    'coordinates': position
                }
            }
            for field in self.new_fields:
                dicts = {field: detection.metadata.get(field)}
                doc.update(dicts)
            # values_list.append(values)
            # Insert values into Mongo
            point_documents.append(doc)
        points_collection.insert_many(point_documents)
        # with open('ss_points_live_ee2.csv', 'a', newline='') as f:
        #     writer = csv.DictWriter(f, fieldnames=self.fields)
        #     writer.writerows(point_documents)
