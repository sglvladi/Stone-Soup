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

class CSV_Writer_EE_MMSI(Writer):
    """
    This is a CSV Writer for the Exact Earth data, specifically configured
    to work with the "per mmsi" approach. It is expected that a new writer
    will be instantiated for each mmsi file.

    Properties
    ----------
    save_dir: str
        The directory where the output files should be saved
    mmsi: str
        The name of the
    """
    save_dir = Property(str, doc="Save directory")
    mmsi = Property(str, doc="MMSI")

    def __init__(self, save_dir, mmsi, *args, **kwargs):
        super().__init__(save_dir, mmsi, *args, **kwargs)
        self.main_fields = ['ID', 'TrackID', 'Latitude', 'Longitude',
                            'ReceivedTime', 'ReceivedTimeDate', 'DataType',
                            'Speed', 'Heading', 'LRIMOShipNo', 'MMSI',
                            'ShipName', 'ShipType', 'AdditionalInfo',
                            'DetectionHistory', 'CallSign', 'Draught',
                            'Destination', 'Location','StoppedProbability']
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
        with open(os.path.join(self.save_dir,'tracks_{}.csv'.format(
                self.mmsi)),
                  'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()
        with open(os.path.join(self.save_dir,'points_{}.csv'.format(
                self.mmsi)),
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
                        state.hypothesis.measurement.metadata.get('ID'))

            # Standardise date/timestamps into both epoch, and date formats
            received_date = datetime.fromtimestamp(
                tm.mktime(timestamp.timetuple()))
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
                self.mmsi)),
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
            for field in self.new_fields:
                dicts = {field: detection.metadata.get(field)}
                doc.update(dicts)

            # Insert values into Mongo
            point_documents.append(doc)

        with open(os.path.join(self.save_dir,'points_{}.csv'.format(
                self.mmsi)),
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