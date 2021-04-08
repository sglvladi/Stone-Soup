from scipy.io import loadmat
from scipy.linalg import block_diag
from datetime import datetime, timedelta
from math import modf

import numpy as np
from dateutil.parser import parse

from stonesoup.types.array import StateVector, CovarianceMatrix
from ..base import Property
from ..types.detection import Detection
from .base import DetectionReader
from .file import TextFileReader
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.wrapper.matlab import MatlabWrapper

class ElintDetectionReaderMatlab(DetectionReader, MatlabWrapper):

    num_targets: float = Property(doc='Number of targets to simulate')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_targets = float(self.num_targets)
        # a, b, c, d = self.matlab_engine.simulateDenseTargets_LV(1000, nargout=4)
        self.matlab_engine.rng(1, 'twister')
        self._sensordata, self._colors, self._truthdata, self._time_indices = \
            self._read_data(*self.matlab_engine.simulateDenseTargets_LV(self.num_targets, nargout=4))
        self._sensdata = []
        for sensor in self._sensordata:
            self._sensdata.append({key: value for key, value in sensor.items() if key != 'meas'})
        a=2

    @BufferedGenerator.generator_method
    def detections_gen(self):
        self.num_scans = len(self._time_indices['sensor'])
        time_init = parse(self._sensordata[self._time_indices['sensor'][0][0]-1]['meas']['times'][self._time_indices['line'][0][0]-1])
        current_time = 0

        for meas_num in range(self.num_scans):
            sensor_idx = int(self._time_indices['sensor'][meas_num][0])
            line_idx = int(self._time_indices['line'][meas_num][0])

            this_meas = self.get_measurement(sensor_idx, line_idx)

            this_meas_time = parse(this_meas['time'][0])
            this_meas['time_seconds'] = float((this_meas_time-time_init).seconds)
            new_time = this_meas['time_seconds']
            dt_seconds = new_time - current_time
            metadata = this_meas
            metadata['dt_seconds'] = float(dt_seconds)
            metadata['sensor_idx'] = sensor_idx
            metadata['meas_num'] = meas_num+1
            detection = Detection(StateVector(this_meas['pos']), timestamp=this_meas_time, metadata=metadata)

            current_time = new_time

            yield this_meas_time, {detection}
        # n_timestamps = len(self._scans.transmit_times)
        # R = CovarianceMatrix(np.diag(np.concatenate((self._params.thetaErrorSDs,
        #                                              self._params.psiErrorSDs,
        #                                              self._params.timeErrorSDs)) ** 2))
        #
        # # Estimate the times each pulse hit the target
        # timedelay1 = np.stack(self._scans.received_times, axis=0)[:, 0] - self._scans.transmit_times.to_numpy()
        # hit_times_est = self._scans.transmit_times.to_numpy() + timedelay1 / 2
        #
        # # Deal with measurements in the order they were estimated to hit the target
        # measOrder = np.argsort(hit_times_est)
        #
        # timestamp_init = datetime.now()
        #
        # for k in range(n_timestamps):
        #     measidx = measOrder[k]
        #     state_vector = self._get_meas(self._scans, measidx)
        #     hittime = hit_times_est[measidx]
        #     timestamp = timestamp_init + timedelta(seconds=hittime)
        #
        #     # Position of sensor 1 at transmit time
        #     sensor1_pos_trans = self._scans.sensor1_xyz[measidx][:, np.newaxis]
        #
        #     # Positions of sensors at receive times
        #     sensor1_pos_rec = self._scans.sensor1_xyz_rec[measidx][:, np.newaxis]
        #     sensor2_pos_rec = self._scans.sensor2_xyz_rec[measidx][:, np.newaxis]
        #
        #     model = SimpleBlueMeasurementModel(ndim_state=12, mapping=[0, 2, 4],
        #                                        noise_covar=R, sensor1_pos_rec=sensor1_pos_rec,
        #                                        sensor1_pos_trans=sensor1_pos_trans, sensor2_pos_rec=sensor2_pos_rec)
        #
        #     detection = Detection(state_vector, timestamp=timestamp, measurement_model=model)
        #
        #     yield timestamp, {detection}

    # def _get_meas(self, data, k):
        # theta = np.array([Elevation(t) for t in data.received_theta[k]])
        # psi = np.array([Bearing(t) for t in data.received_psi[k]])
        # times = data.received_times[k] - data.transmit_times[k]
        # z = StateVector(np.concatenate((theta, psi, times)))
        # return z

    @staticmethod
    def _read_data(sensor_data, colors, truth, time_indices):

        for key in time_indices:
            time_indices[key] = np.array(time_indices[key]).astype(int)

        return sensor_data, colors, truth, time_indices
        # scans_dict['transmit_times'] = list(np.array(measdata['transmit_times']).ravel().astype(float).T)
        # scans_dict['received_times'] = list(np.array(measdata['received_times']).astype(float).T)
        # scans_dict['sensor1_xyz'] = list(np.array(measdata['sensor1_xyz']).astype(float).T)
        # scans_dict['sensor2_xyz'] = list(np.array(measdata['sensor2_xyz']).astype(float).T)
        # scans_dict['sensor1_xyz_rec'] = list(np.array(measdata['sensor1_xyz_rec']).astype(float).T)
        # scans_dict['sensor2_xyz_rec'] = list(np.array(measdata['sensor2_xyz_rec']).astype(float).T)
        # scans_dict['received_theta'] = list(np.array(measdata['received_theta']).astype(float).T)
        # scans_dict['received_psi'] = list(np.array(measdata['received_psi']).astype(float).T)
        # scans_df = pd.DataFrame(scans_dict)
        #
        # params_dict = dict()
        # params_dict['timeErrorSDs'] = np.array(params['timeErrorSDs']).ravel().astype(float)
        # params_dict['thetaErrorSDs'] = np.array(params['thetaErrorSDs']).ravel().astype(float)
        # params_dict['psiErrorSDs'] = np.array(params['psiErrorSDs']).ravel().astype(float)
        # params_df = pd.DataFrame(params_dict)
        #
        # truth_dict = dict()
        # truth_dict['hit_times'] = np.array(truthdata['hit_times']).astype(float).ravel().tolist()
        # truth_dict['target_pos'] = np.array(truthdata['target_xyz']).astype(float).T.tolist()
        # truth_dict['target_pos_hit'] = np.array(truthdata['target_xyz_hit']).astype(float).T.tolist()
        # truth_df = pd.DataFrame(truth_dict)

        # return scans_df, truth_df, params_df

    def get_measurement(self, sensor_idx, line_idx):
        sensor_idx = sensor_idx-1
        line_idx = line_idx-1
        sensordata = self._sensordata[sensor_idx]

        # Get kinematic measurement information
        meas_pos = np.atleast_2d(sensordata['meas']['coords'][line_idx]).T
        H = sensordata['sensor']['H']
        if 'R' in sensordata['sensor']:
            R = sensordata['sensor']['R']
        else:
            # If R not defined in sensor, calculate it from smaj, smin, orient
            meas_semimajor = sensordata['meas']['semimajorSD'][line_idx]
            meas_semiminor = sensordata['meas']['semiminorSD'][line_idx]
            meas_orientDeg = sensordata['meas']['orientation'][line_idx]
            R = self.matlab_engine.getLonLatR(self.matlab_array(meas_pos[[0,1], :]), meas_semimajor, meas_semiminor, meas_orientDeg)
            a = 2

        # Get colour information
        meas_colour, colours_defined = self._get_colour_likelihood(line_idx, sensordata)

        # Get MMSI if defined
        if 'mmsi' in sensordata['meas']:
            mmsi = sensordata['meas']['mmsi'][line_idx]
        else:
            mmsi = ''

        meas = {
            'time': [sensordata['meas']['times'][line_idx]],
            'pos': self.matlab_array(meas_pos),
            'colour': self.matlab_array(meas_colour),
            'coloursDefined': self.matlab_array(colours_defined),
            'mmsi': mmsi,
            'H': H,
            'R': R
        }
        # a = self.matlab_engine.getMeasurementData_LV(self._sensordata[sensor_idx], line_idx+1, self._colors)
        return meas
        # return self.matlab_engine.getMeasurementData_LV(self._sensordata[sensor_idx], line_idx+1, self._colors)

    def _get_colour_likelihood(self, line_idx, sensordata):
        colour_names = [colour['name'] for colour in self._colors]
        colours_defined = [i for i, colour in enumerate(colour_names) if colour in sensordata['meas']]

        num_colours_def = len(colours_defined)
        meas = np.zeros((num_colours_def, 1))
        for i in range(num_colours_def):
            thisc = colours_defined[i]
            meas[i, 0] = sensordata['meas'][colour_names[thisc]][line_idx][0]

        idx = ~np.isnan(meas).ravel()
        if len(meas):
            meas = meas[idx, :]
        defined = np.array([float(colour+1) for i, colour in enumerate(colours_defined) if idx[i]])
        return meas, defined
