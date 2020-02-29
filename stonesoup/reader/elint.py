# -*- coding: utf-8 -*-
"""Generic readers for Stone Soup.

This is a collection of generic readers for Stone Soup, allowing quick reading
of data that is in common formats.
"""

from scipy.io import loadmat
from scipy.linalg import block_diag
from datetime import datetime, timedelta
from math import modf

import numpy as np
from dateutil.parser import parse

from stonesoup.types.array import CovarianceMatrix
from ..base import Property
from ..types.detection import Detection
from .base import DetectionReader
from .file import TextFileReader
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.models.measurement.linear import LinearGaussian


class BasicELINTDetectionReader(DetectionReader, TextFileReader):
    """A custom detection reader for elint detections.

    Parameters
    ----------
    """

    time_field_format = Property(
        str, default=None, doc='Optional datetime format')
    timestamp = Property(
        bool, default=False, doc='Treat time field as a timestamp from epoch')
    metadata_fields = Property(
        [str], default=None, doc='List of columns to be saved as metadata, '
                                 'default all')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @BufferedGenerator.generator_method
    def detections_gen(self):
        wp = loadmat(self.path)
        elintdata = wp['elintdata'][0,0]
        coords = elintdata['coords'][:,0:2].T
        timestamps = elintdata['timestr']
        covars = elintdata['covars']

        n_timestamps = len(timestamps)

        for i in range(n_timestamps):

            # Process timestamp
            if self.time_field_format is not None:
                time_field_value = datetime.strptime(
                    timestamps[i], self.time_field_format)
            elif self.timestamp is True:
                fractional, timestamp = modf(float(timestamps[i]))
                time_field_value = datetime.utcfromtimestamp(
                    int(timestamp))
                time_field_value += timedelta(microseconds=fractional * 1E6)
            else:
                time_field_value = parse(timestamps[i])

            # Process metadata (TODO)
            # if self.metadata_fields is None:
            #     local_metadata = dict(row)
            #     copy_local_metadata = dict(local_metadata)
            #     for (key, value) in copy_local_metadata.items():
            #         if (key == self.time_field) or \
            #                 (key in self.state_vector_fields):
            #             del local_metadata[key]
            # else:
            #     local_metadata = {field: row[field]
            #                       for field in self.metadata_fields
            #                       if field in row}
            local_metadata = dict() #REMOVE

            # Create detection
            R = covars[:, :, i]
            model = LinearGaussian(ndim_state=4, mapping=[0, 2],
                                   noise_covar=R)
            detect = Detection(np.array(coords[:, i], dtype=np.float32),
                               time_field_value,
                               measurement_model=model,
                               metadata=local_metadata)
            yield time_field_value, {detect}


class AisElintDetectionReader(DetectionReader, TextFileReader):
    """A custom detection reader for elint and ais detections.

    Parameters
    ----------
    """

    sensors = Property(dict())
    time_field_format = Property(
        str, default=None, doc='Optional datetime format')
    timestamp = Property(
        bool, default=False, doc='Treat time field as a timestamp from epoch')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @BufferedGenerator.generator_method
    def detections_gen(self):
        # Load data from mat file
        wp = loadmat(self.path)
        elintdata = wp['elintdata'][0,0]
        aisdata = wp['aisdata'][0, 0]
        measdata = [aisdata, elintdata]
        timeIndices = wp['timeIndices'][0, 0]
        timeIndicesSensor = timeIndices['sensor'].ravel()
        timeIndicesLine = timeIndices['line'].ravel()

        # Generate detections
        for (sensor_idx, line_idx) in zip(timeIndicesSensor, timeIndicesLine):
            yield self._extract_measurement(sensor_idx-1, measdata, line_idx-1)

    def _extract_measurement(self, sensor_idx, measdata, line_idx):
        colornames = ['colour1', 'colour2', 'colour3', 'colour4', 'colour5', 'colour6']
        colordim = len(colornames)

        # Extract base measurement and timestamp
        meas = measdata[sensor_idx]['coords'][line_idx, 0:2]
        timestamp = measdata[sensor_idx]['timestr'][line_idx]

        # Process timestamp
        if self.time_field_format is not None:
            time_field_value = datetime.strptime(
                timestamp, self.time_field_format)
        elif self.timestamp is True:
            fractional, timestamp = modf(float(timestamp))
            time_field_value = datetime.utcfromtimestamp(
                int(timestamp))
            time_field_value += timedelta(microseconds=fractional * 1E6)
        else:
            time_field_value = parse(timestamp)

        # Sensor base detection generation
        if sensor_idx == 0:
            # AIS
            R = self.sensors[sensor_idx]['RLonLat']
            model = LinearGaussian(ndim_state=4+colordim, mapping=[0, 2],
                                   noise_covar=R)
            metadata = {'sensor': self.sensors[sensor_idx]}
            detect = Detection(np.array(meas, dtype=np.float32),
                               time_field_value,
                               measurement_model=model,
                               metadata=metadata)
        else:
            # ELINT
            meas_color = self._get_colour_meas(measdata[sensor_idx], line_idx, colornames)
            meas = np.concatenate((meas, meas_color))
            R = measdata[sensor_idx]['covars'][:, :, line_idx]
            R_col = np.diag(self.sensors[sensor_idx]['colour_error_sd']**2)
            R = block_diag(R, R_col)
            model = LinearGaussian(ndim_state=4 + colordim, mapping=[0, 2, 4, 5, 6, 7, 8, 9],
                                   noise_covar=R)
            metadata = {'sensor': self.sensors[sensor_idx]}
            detect = Detection(np.array(meas, dtype=np.float32),
                               time_field_value,
                               measurement_model=model,
                               metadata=metadata)
        return time_field_value, {detect}


    def _get_colour_meas(self, data, line_idx, colournames):
        colourdim = len(colournames)
        meas_colour = np.empty((colourdim,))
        meas_colour[:] = np.NaN
        for i in range(colourdim):
            if colournames[i] in data.data.format:
                meas_colour[i] = data[colournames[i]][line_idx]
        return meas_colour