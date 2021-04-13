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
        self.matlab_engine.rng(1, 'twister')
        self._sensordata, self._colors, self._truthdata, self._time_indices = \
            self._read_data(*self.matlab_engine.simulateDenseTargets_LV(self.num_targets, nargout=4))
        # Sensor data without measurements
        self._sensdata = []
        for sensor in self._sensordata:
            self._sensdata.append({key: value for key, value in sensor.items() if key != 'meas'})

    @BufferedGenerator.generator_method
    def detections_gen(self):
        self.num_scans = len(self._time_indices['sensor'])
        time_init = parse(self._sensordata[self._time_indices['sensor'][0][0]-1]['meas']['times'][self._time_indices['line'][0][0]-1])
        current_time = 0

        for meas_num in range(self.num_scans):
            sensor_idx = int(self._time_indices['sensor'][meas_num][0])
            line_idx = int(self._time_indices['line'][meas_num][0])

            this_meas = self._get_measurement(sensor_idx, line_idx)

            this_meas_time = parse(this_meas['time'][0])
            this_meas['time_seconds'] = float((this_meas_time-time_init).seconds)
            new_time = this_meas['time_seconds']
            dt_seconds = new_time - current_time

            # Detection metadata
            metadata = this_meas
            metadata['dt_seconds'] = float(dt_seconds)
            metadata['sensor_idx'] = sensor_idx
            metadata['meas_num'] = meas_num+1

            detection = Detection(StateVector(this_meas['pos']), timestamp=this_meas_time, metadata=metadata)

            current_time = new_time

            yield this_meas_time, {detection}

    @staticmethod
    def _read_data(sensor_data, colors, truth, time_indices):

        for key in time_indices:
            time_indices[key] = np.array(time_indices[key]).astype(int)

        return sensor_data, colors, truth, time_indices

    def _get_measurement(self, sensor_idx, line_idx):
        """ Equivalent to getMeasurementData_LV(). Doing this here is a lot faster, since we avoid passing the
        sensordata to MATLAB. """
        # Adjust indexing
        sensor_idx = sensor_idx-1
        line_idx = line_idx-1

        # Get current sensor's data
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
        return meas

    def _get_colour_likelihood(self, line_idx, sensordata):
        """ Equivalent to getColourLikelihood() in MATLAB. """
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
