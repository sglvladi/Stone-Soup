import numpy as np
import utm
import warnings
from collections import OrderedDict
from typing import Dict, Union
import datetime

from geographiclib.geodesic import Geodesic


#from apt.visualisation.target_localisation.bistatic_equation import bistatic # need to install latest version of APT - Acoustic Processing toolkit - search on Dstl WikiD
#from apt.geo.bistatic import bistatic_geo_coords
#from apt.geo.geo import geo_pos_from_delta_pos

from ._bistatic_util import bistatic_geo_coords, geo_pos_from_delta_pos


#from stanagio.STANAG_MSG import STANAG_MSG, STANAG_MSG_SET
#from stanagio import STANAG_MSG_UTIL

from niag_reader.STANAG_MSG import STANAG_MSG, STANAG_MSG_SET
from niag_reader import STANAG_MSG_UTIL





# Stone Soup imports:
from stonesoup.base import Property
from stonesoup.measures import Euclidean
from stonesoup.functions import mod_bearing
from stonesoup.types.state import State
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange, CartesianToBearingRangeBias
from stonesoup.reader.base import DetectionReader
from stonesoup.reader.generic import _CSVReader
from stonesoup.tracker.simple import Tracker
from stonesoup.types.detection import Detection
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.update import GaussianMixtureUpdate
from stonesoup.types.groundtruth import GroundTruthState









class STANAGContactReader(DetectionReader, _CSVReader):
    """A reader for STANAG 4585 binary files containing the LCAS sonar contacts.

    This reads in Message 1: Basic Situational Awareness, Message 2: Transmission and message 4: Contacts

    Parameters
    ----------
    """

    snr_threshold: float = Property(default=0, doc='SNR value used to threshold the contacts')
    berr: float = Property(default=0.005, doc='bearing error used in measurement noise matrix')
    rerr: float = Property(default=200, doc='range error used in measurement noise matrix')
    reference_lat: float = Property(default=42.9, doc='Reference Lat for cartesian conversion')
    reference_lon: float = Property(default=10.0, doc='Reference Long for cartesian conversion')
    refDatetime: float = Property(default = datetime.datetime(2018,11,19,13,00,00), doc='Initial reference time for STANAG message which is seconds past the hour') # in STANAG message, time is specified as seconds past the hour. so can use a reference datetime to provide a full datetime object
    endianness: float = Property(default = 0, doc='State endianness of input files. 1 = Big Endian, 0 = Little Endian')
    stanag_msg_directory: str = Property(default=r'data/20240701-STANAG_TrackerTest', doc='Filepath of STANAG Message Directory')
    
    # Truth data
    truth_time: float = Property(default=np.array((np.nan)), doc = 'Truth positional information time steps')
    truth_TX: float = Property(default=np.array([(np.nan,np.nan)]), doc = 'Truth TX positional information')
    truth_RX: float = Property(default=np.array([(np.nan,np.nan)]), doc = 'Truth RX positional information')

    # Toggle for actions on first iteration only
    extractReferenceData: int = Property(default = 1, doc = 'Toggle to extract reference data from contact. Starts as true then made false once data extracted')

    with_bias: bool = Property(default=False, doc='Flag to indicate if the reader should include bias in the measurement model')


    def read_stanag_files(self, rx_plat_id_select=None, truth_tx_plat_unit=(1,1), truth_rx_plat_unit=(1,1), config_subfolder='TrialMercury', meta_header_name=None):
        self.stanag_msg_set = STANAG_MSG_SET.from_directory(str(self.stanag_msg_directory), endianness=self.endianness,
                                                            config_subfolder=config_subfolder, meta_header_name=meta_header_name)
        self.sm02_stanag_msgs = self.stanag_msg_set.search_msg_headers({'MessageTypeIdentifier':'==2'})
        self.sm04_stanag_msgs = self.stanag_msg_set.search_msg_headers({'MessageTypeIdentifier':'==4'})
        self.sm01_stanag_msgs = self.stanag_msg_set.search_msg_headers({'MessageTypeIdentifier':'==1'})

        self.detection_dicts = []


        for sm04_msg_time, sm04_msg in self.sm04_stanag_msgs:            
            sm02_msg_time, sm02_msg = STANAG_MSG_UTIL.get_corresponding_tx_message(self.stanag_msg_set, sm04_msg, sm04_msg_time)
            assumed_sm04_time = STANAG_MSG_UTIL.guess_time_from_hundredths(sm04_msg_time, 100 * sm04_msg.msg_header[
                'DirectBlastPropagationTime'])
            
            if sm02_msg_time is None or sm02_msg is None:
                print(f"Failed to find a corresponding SM02 message for {sm04_msg_time} {assumed_sm04_time} {sm04_msg.msg_header['DirectBlastPropagationTime']}")                
                continue
            

            if rx_plat_id_select is not None and sm04_msg.msg_header['PlatformID_RXUnit'] != rx_plat_id_select:
                continue

            tx_lat, tx_lon = sm02_msg.msg_header['UnitLat'], sm02_msg.msg_header['UnitLong']

            if self.extractReferenceData == 1:
                # Extract a reference position on first time through only
                self.reference_lat = tx_lat
                self.reference_lon = tx_lon
                self.extractReferenceData = 0

            rx_ref_lat, rx_ref_lon = sm04_msg.msg_header['ReferencePos_RX_Lat'], sm04_msg.msg_header['ReferencePos_RX_Long']
            rx_depth = sm04_msg.msg_header['UnitDepth_RX'] if sm04_msg.msg_header['UnitDepth_RX'] < 10000 else 0
            tx_depth = sm02_msg.msg_header['UnitDepth'] if sm02_msg.msg_header['UnitDepth'] < 10000 else rx_depth  # Deal with 'unknown' depths
            depth_hypo = None
            depth_hypo = rx_depth if depth_hypo is None else depth_hypo
            tx_event_time = STANAG_MSG_UTIL.guess_time_from_hundredths(sm02_msg_time, 100*sm02_msg.msg_header['TransmitEventTime'])
            print(f"Processing Pings from Tx time : {tx_event_time}  {len(self.detection_dicts)}  from SM04 message {sm04_msg_time}")
            direct_blast_detect_time = tx_event_time + datetime.timedelta(seconds=sm04_msg.msg_header['DirectBlastPropagationTime'])

            for sm04_detection_subblock in sm04_msg.data_blocks['CONTACT_BLOCK'].block_data:
                if sm04_detection_subblock['ContactSNR'] < self.snr_threshold: # skip low snr detections
                    continue

                rx_lat, rx_lon = geo_pos_from_delta_pos((rx_ref_lat, rx_ref_lon),
                                                            delta_north_metres=sm04_detection_subblock['DeltaPosNorth_RX'],
                                                            delta_east_metres=sm04_detection_subblock['DeltaPosEast_RX'])


                ref_pos_solver = Geodesic.WGS84.Inverse(self.reference_lat, self.reference_lon, rx_lat, rx_lon)
                tx_ref_pos_solver = Geodesic.WGS84.Inverse(self.reference_lat, self.reference_lon, tx_lat, tx_lon)
                rx_delta_north = ref_pos_solver['s12'] * np.cos(np.deg2rad(ref_pos_solver['azi1']))
                rx_delta_east = ref_pos_solver['s12'] * np.sin(np.deg2rad(ref_pos_solver['azi1']))
                tx_delta_north = tx_ref_pos_solver['s12'] * np.cos(np.deg2rad(tx_ref_pos_solver['azi1']))
                tx_delta_east = tx_ref_pos_solver['s12'] * np.sin(np.deg2rad(tx_ref_pos_solver['azi1']))
                

                echo_detect_time = tx_event_time + datetime.timedelta(seconds=sm04_detection_subblock['ContactEchoDetectionTime'])
                delta_time = sm04_detection_subblock['ContactEchoDetectionTime'] - sm04_msg.msg_header['DirectBlastPropagationTime']
                ref_sound_speed = 1500.0

                rel_brgs = [sm04_detection_subblock['FirstRelBearingContactAtRX']]
                
                if bool(sm04_detection_subblock['Ambiguity']):
                    rel_brgs += [sm04_detection_subblock['SecondRelBearingContactAtRX']]


                bistatic_ranges = bistatic_geo_coords(tx_pos_geo=(tx_lat, tx_lon, tx_depth),
                                                   rx_pos_geo=(rx_lat, rx_lon, rx_depth),
                                                   ref_pos=(rx_lat, rx_lon, rx_depth),
                                                   delta_time=delta_time,
                                                   rel_brg_deg=rel_brgs,
                                                   rx_hdg=sm04_msg.msg_header['UnitCourse_RX'],
                                                   use_conical_beams=False,
                                                   ref_sound_spd=ref_sound_speed, depth_hypo=depth_hypo,
                                                   range_only=True)
                
                #bistatic_ranges = [bistatic_ranges] if isinstance(bistatic_ranges, float) else bistatic_ranges
                for brg, bistat_range in zip(rel_brgs, bistatic_ranges):                    
                    self.detection_dicts.append({'time':tx_event_time, 'rel_brg':brg, 
                                        'ContactRXRange':bistat_range,
                                        'rx_delta_n':rx_delta_north, 'rx_delta_e':rx_delta_east, 'rx_hdg': sm04_msg.msg_header['UnitCourse_RX'],
                                        'tx_delta_n':tx_delta_north, 'tx_delta_e':tx_delta_east,})

        self.detection_dicts.sort(key=lambda d: d['time'])
        
        return


    def get_stanag_ground_truth_from_SM01(self, target_plat_unit_id=(4,1)):

        self.ground_truth = []
        for sm01_msg_time, sm01_msg in self.sm01_stanag_msgs:
            plat_id, unit_id = sm01_msg.msg_header['PlatformID'], sm01_msg.msg_header['UnitID']
            if (plat_id, unit_id) != target_plat_unit_id:
                continue
            time = STANAG_MSG_UTIL.guess_time_from_hundredths(sm01_msg_time, 100*sm01_msg.msg_header['TimeStamp'])# + timedelta(minutes=17.5)
            
            lat, lon, crse, speed = sm01_msg.msg_header['UnitLat'], sm01_msg.msg_header['UnitLong'], sm01_msg.msg_header['UnitCourse'], sm01_msg.msg_header['UnitSpeed']
            
            pos_solver = Geodesic.WGS84.Inverse(self.reference_lat, self.reference_lon, lat, lon)
            cart_x = pos_solver['s12']*np.sin(np.deg2rad(pos_solver['azi1']))
            cart_y = pos_solver['s12']*np.cos(np.deg2rad(pos_solver['azi1']))
            cart_x_dot = 0.514444 * speed * np.sin(np.deg2rad(crse)) #kts to m/s
            cart_y_dot = 0.514444 * speed * np.cos(np.deg2rad(crse)) #kts to m/s

            state_vector = StateVector([cart_x, cart_x_dot, cart_y, cart_y_dot])
            
            if (plat_id, unit_id) == target_plat_unit_id:
                self.ground_truth.append(GroundTruthState(state_vector=state_vector, timestamp=time + datetime.timedelta(seconds=0)))
        return



    @BufferedGenerator.generator_method
    def detections_gen(self):
        previous_time = None
        detections = set()

        for detection_dict in self.detection_dicts:
            if previous_time is not None and previous_time != detection_dict['time']:
                yield previous_time, detections
                detections = set()

            previous_time = detection_dict['time']
            
            self.truth_time = np.vstack((self.truth_time, detection_dict['time']))
            tx_pos_cart = [detection_dict['tx_delta_e'], detection_dict['tx_delta_n']]
            rx_pos_cart = [detection_dict['rx_delta_e'], detection_dict['rx_delta_n']]
            self.truth_TX = np.vstack((self.truth_TX, tx_pos_cart))
            self.truth_RX = np.vstack((self.truth_RX, rx_pos_cart))

            # TODO Use errors from STANAG messages - for now, use universal bearing and range errors
            cov = np.zeros((2,2))                           
            cov[0,0] = self.berr
            cov[1,1] = self.rerr

            #cov = get_measurement_covariance(self.berr, self.rerr, detection_dict['rel_brg'])
            alliance_heading = mod_bearing(-1* (np.deg2rad(detection_dict['rx_hdg']) - np.pi/2))  # convert to Stone soup angle convention
            alliance_e = detection_dict['rx_delta_e']
            alliance_n = detection_dict['rx_delta_n']
            rel_brg_stone_soup = mod_bearing(-1*(np.deg2rad(detection_dict['rel_brg'])))
            

            if not self.with_bias:
                measurement_model = CartesianToBearingRange(
                    mapping=(0, 2),
                    noise_covar=CovarianceMatrix(cov),
                    rotation_offset=StateVector([[float(0)], [float(0)], [alliance_heading]]),
                    translation_offset=StateVector([[detection_dict['rx_delta_e']], [detection_dict['rx_delta_n']]]),
                    ndim_state=6
                )
            else:
                measurement_model = CartesianToBearingRangeBias(
                    mapping=(0, 2),
                    noise_covar=CovarianceMatrix(cov),
                    rotation_offset=StateVector([[float(0)], [float(0)], [alliance_heading]]),
                    translation_offset=StateVector([[detection_dict['rx_delta_e']], [detection_dict['rx_delta_n']]]),
                    ndim_state=6
                )

            detection = Detection(np.array([rel_brg_stone_soup, detection_dict['ContactRXRange']]),
                        timestamp=detection_dict['time'],
                        measurement_model=measurement_model,
                        #metadata=self._get_metadata(row_df))
                        metadata=None)
            detections.add(detection)
    

        # Yield remaining
        yield previous_time, detections



if __name__ == "__main__":
    stanag_folder = r'data/20240902_TrackerTest/MSDFE_Scenario_Four'
    contacts_reader = STANAGContactReader(stanag_folder,
                                       state_vector_fields=("RelBearing", "RX2contact_range"),
                                       time_field = None,
                                       snr_threshold=12,
                                       berr = 10,
                                       rerr = 123,
                                       refDatetime = datetime.datetime(2023,1,1,0,0,0),
                                       endianness = 0,
                                       stanag_msg_directory=stanag_folder)
    contacts_reader.read_stanag_files(rx_plat_id_select=1)
