import numpy as np
from scipy.io import loadmat
from datetime import datetime, timedelta
from typing import Sequence, Collection, Mapping

from stonesoup.base import Property
from stonesoup.reader import DetectionReader
from stonesoup.reader.file import TextFileReader
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.types.array import StateVector
from stonesoup.types.detection import Detection
from stonesoup.types.angle import Bearing, Elevation
from stonesoup.models.measurement.blue import SimpleBlueMeasurementModel


class BasicBlueDetectionReader(DetectionReader, TextFileReader):
    time_field_format: str = Property(default=None, doc='Optional datetime format')
    timestamp: bool = Property(default=False, doc='Treat time field as a timestamp from epoch')
    metadata_fields: Sequence[str] = Property(default=None, doc='List of columns to be saved as metadata, default all')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @BufferedGenerator.generator_method
    def detections_gen(self):
        data = self._read_mat(self.path)

        n_timestamps = len(data.transmit_times)
        R = np.diag(np.concatenate((data.thetaErrorSDs, data.psiErrorSDs, data.timeErrorSDs))**2)

        # Estimate the times each pulse hit the target
        timedelay1 = data.received_times[0,:] - data.transmit_times
        hit_times_est = data.transmit_times + timedelay1/2

        # Deal with measurements in the order they were estimated to hit the target
        measOrder = np.argsort(hit_times_est)

        timestamp_init = datetime.now()

        for k in range(n_timestamps):
            measidx = measOrder[k]
            state_vector = self._get_meas(data, measidx)
            hittime = hit_times_est[measidx]
            timestamp = timestamp_init + timedelta(seconds=hittime)

            # Position of sensor 1 at transmit time
            sensor1_pos_trans = data.sensor1_xyz[:, measidx]

            # Positions of sensors at receive times
            sensor1_pos_rec = data.sensor1_xyz_rec[:, measidx]
            sensor2_pos_rec = data.sensor2_xyz_rec[:, measidx]

            model = SimpleBlueMeasurementModel(ndim_state=12, mapping=[0, 2, 4],
                                               noise_covar=R, sensor1_pos_rec=sensor1_pos_rec,
                                               sensor1_pos_trans=sensor1_pos_trans, sensor2_pos_rec=sensor2_pos_rec)

            detection = Detection(state_vector, timestamp=timestamp, measurement_model=model)

            yield timestamp, {detection}

    def _get_meas(self, data, k):
        theta = np.array([Elevation(t) for t in data.received_theta[:, k]])
        psi = np.array([Bearing(t) for t in data.received_psi[:, k]])
        times = data.received_times[:, k] - data.transmit_times[k]
        z = StateVector(np.concatenate((theta, psi, times)))
        return z

    @staticmethod
    def _read_mat(path):
        wp = loadmat(path)
        class MeasData:
            def __init__(self, wp):
                measdata = wp['measdata'][0, 0]
                self.transmit_times = measdata['transmit_times'].ravel().astype(float)
                self.received_times = measdata['received_times'].astype(float)
                self.sensor1_xyz = measdata['sensor1_xyz'].astype(float)
                self.sensor2_xyz = measdata['sensor1_xyz'].astype(float)
                self.sensor1_xyz_rec = measdata['sensor1_xyz_rec'].astype(float)
                self.sensor2_xyz_rec = measdata['sensor1_xyz_rec'].astype(float)
                self.received_theta = measdata['received_theta'].astype(float)
                self.received_psi = measdata['received_psi'].astype(float)
                self.timeErrorSDs = measdata['timeErrorSDs'].ravel().astype(float)
                self.thetaErrorSDs = measdata['thetaErrorSDs'].ravel().astype(float)
                self.psiErrorSDs = measdata['psiErrorSDs'].ravel().astype(float)
        return MeasData(wp)