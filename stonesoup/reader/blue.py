import numpy as np
import pandas as pd

from scipy.io import loadmat
from datetime import datetime, timedelta
from typing import Sequence, Collection, Mapping

from stonesoup.base import Property
from stonesoup.reader import DetectionReader
from stonesoup.reader.file import TextFileReader
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.detection import Detection
from stonesoup.types.angle import Bearing, Elevation
from stonesoup.models.measurement.blue import SimpleBlueMeasurementModel, SimpleBlueMeasurementModel2
from stonesoup.wrapper.matlab import MatlabWrapper


class BlueDetectionReaderMatlab(DetectionReader, MatlabWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scans, self._truthdata, self._params = self._read_data(*self.matlab_engine.simulateData(nargout=3))

    @BufferedGenerator.generator_method
    def detections_gen(self):

        n_timestamps = len(self._scans.transmit_times)
        R = CovarianceMatrix(np.diag(np.concatenate((self._params.thetaErrorSDs,
                                                     self._params.psiErrorSDs,
                                                     self._params.timeErrorSDs)) ** 2))

        # Estimate the times each pulse hit the target
        timedelay1 = np.stack(self._scans.received_times, axis=0)[:, 0] - self._scans.transmit_times.to_numpy()
        hit_times_est = self._scans.transmit_times.to_numpy() + timedelay1 / 2

        # Deal with measurements in the order they were estimated to hit the target
        measOrder = np.argsort(hit_times_est)

        timestamp_init = datetime.now()

        for k in range(n_timestamps):
            measidx = measOrder[k]
            state_vector = self._get_meas(self._scans, measidx)
            hittime = hit_times_est[measidx]
            timestamp = timestamp_init + timedelta(seconds=hittime)

            # Position of sensor 1 at transmit time
            sensor1_pos_trans = self._scans.sensor1_xyz[measidx][:, np.newaxis]

            # Positions of sensors at receive times
            sensor1_pos_rec = self._scans.sensor1_xyz_rec[measidx][:, np.newaxis]
            sensor2_pos_rec = self._scans.sensor2_xyz_rec[measidx][:, np.newaxis]

            model = SimpleBlueMeasurementModel(ndim_state=12, mapping=[0, 2, 4],
                                               noise_covar=R, sensor1_pos_rec=sensor1_pos_rec,
                                               sensor1_pos_trans=sensor1_pos_trans, sensor2_pos_rec=sensor2_pos_rec)

            detection = Detection(state_vector, timestamp=timestamp, measurement_model=model)

            yield timestamp, {detection}

    def _get_meas(self, data, k):
        theta = np.array([Elevation(t) for t in data.received_theta[k]])
        psi = np.array([Bearing(t) for t in data.received_psi[k]])
        times = data.received_times[k] - data.transmit_times[k]
        z = StateVector(np.concatenate((theta, psi, times)))
        return z

    @staticmethod
    def _read_data(measdata, truthdata, params):
        scans_dict = dict()
        scans_dict['transmit_times'] = list(np.array(measdata['transmit_times']).ravel().astype(float).T)
        scans_dict['received_times'] = list(np.array(measdata['received_times']).astype(float).T)
        scans_dict['sensor1_xyz'] = list(np.array(measdata['sensor1_xyz']).astype(float).T)
        scans_dict['sensor2_xyz'] = list(np.array(measdata['sensor2_xyz']).astype(float).T)
        scans_dict['sensor1_xyz_rec'] = list(np.array(measdata['sensor1_xyz_rec']).astype(float).T)
        scans_dict['sensor2_xyz_rec'] = list(np.array(measdata['sensor2_xyz_rec']).astype(float).T)
        scans_dict['received_theta'] = list(np.array(measdata['received_theta']).astype(float).T)
        scans_dict['received_psi'] = list(np.array(measdata['received_psi']).astype(float).T)
        scans_df = pd.DataFrame(scans_dict)

        params_dict = dict()
        params_dict['timeErrorSDs'] = np.array(params['timeErrorSDs']).ravel().astype(float)
        params_dict['thetaErrorSDs'] = np.array(params['thetaErrorSDs']).ravel().astype(float)
        params_dict['psiErrorSDs'] = np.array(params['psiErrorSDs']).ravel().astype(float)
        params_df = pd.DataFrame(params_dict)

        truth_dict = dict()
        truth_dict['hit_times'] = np.array(truthdata['hit_times']).astype(float).ravel().tolist()
        truth_dict['target_pos'] = np.array(truthdata['target_xyz']).astype(float).T.tolist()
        truth_dict['target_pos_hit'] = np.array(truthdata['target_xyz_hit']).astype(float).T.tolist()
        truth_df = pd.DataFrame(truth_dict)

        return scans_df, truth_df, params_df


class BlueDetectionReaderFile(DetectionReader, TextFileReader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scans, self._truthdata, self._params = self._read_data(self.path)

    @BufferedGenerator.generator_method
    def detections_gen(self):
        n_timestamps = len(self._scans.transmit_times)
        R = CovarianceMatrix(np.diag(np.concatenate((self._params.thetaErrorSDs,
                                                     self._params.psiErrorSDs,
                                                     self._params.timeErrorSDs)) ** 2))

        # Estimate the times each pulse hit the target
        timedelay1 = np.stack(self._scans.received_times, axis=0)[:, 0] - self._scans.transmit_times.to_numpy()
        hit_times_est = self._scans.transmit_times.to_numpy() + timedelay1 / 2

        # Deal with measurements in the order they were estimated to hit the target
        measOrder = np.argsort(hit_times_est)

        timestamp_init = datetime.now()

        for k in range(n_timestamps):
            measidx = measOrder[k]
            state_vector = self._get_meas(self._scans, measidx)
            hittime = hit_times_est[measidx]
            timestamp = timestamp_init + timedelta(seconds=hittime)

            # Position of sensor 1 at transmit time
            sensor1_pos_trans = self._scans.sensor1_xyz[measidx][:, np.newaxis]

            # Positions of sensors at receive times
            sensor1_pos_rec = self._scans.sensor1_xyz_rec[measidx][:, np.newaxis]
            sensor2_pos_rec = self._scans.sensor2_xyz_rec[measidx][:, np.newaxis]

            model = SimpleBlueMeasurementModel(ndim_state=12, mapping=[0, 2, 4],
                                               noise_covar=R, sensor1_pos_rec=sensor1_pos_rec,
                                               sensor1_pos_trans=sensor1_pos_trans, sensor2_pos_rec=sensor2_pos_rec)

            detection = Detection(state_vector, timestamp=timestamp, measurement_model=model)

            yield timestamp, {detection}

    def _get_meas(self, data, k):
        theta = np.array([Elevation(t) for t in data.received_theta[k]])
        psi = np.array([Bearing(t) for t in data.received_psi[k]])
        times = data.received_times[k] - data.transmit_times[k]
        z = StateVector(np.concatenate((theta, psi, times)))
        return z

    @staticmethod
    def _read_data(path):
        wp = loadmat(path)

        measdata = wp['measdata'][0, 0]
        scans_dict = dict()
        scans_dict['transmit_times'] = list(measdata['transmit_times'].ravel().astype(float).T)
        scans_dict['received_times'] = list(measdata['received_times'].astype(float).T)
        scans_dict['sensor1_xyz'] = list(measdata['sensor1_xyz'].astype(float).T)
        scans_dict['sensor2_xyz'] = list(measdata['sensor2_xyz'].astype(float).T)
        scans_dict['sensor1_xyz_rec'] = list(measdata['sensor1_xyz_rec'].astype(float).T)
        scans_dict['sensor2_xyz_rec'] = list(measdata['sensor2_xyz_rec'].astype(float).T)
        scans_dict['received_theta'] = list(measdata['received_theta'].astype(float).T)
        scans_dict['received_psi'] = list(measdata['received_psi'].astype(float).T)
        scans_df = pd.DataFrame(scans_dict)

        params = wp['params'][0, 0]
        params_dict = dict()
        params_dict['timeErrorSDs'] = params['timeErrorSDs'].ravel().astype(float)
        params_dict['thetaErrorSDs'] = params['thetaErrorSDs'].ravel().astype(float)
        params_dict['psiErrorSDs'] = params['psiErrorSDs'].ravel().astype(float)
        params_df = pd.DataFrame(params_dict)

        truthdata = wp['truthdata'][0, 0]
        truth_dict = dict()
        truth_dict['hit_times'] = truthdata['hit_times'].astype(float).ravel().tolist()
        truth_dict['target_pos'] = truthdata['target_xyz'].astype(float).T.tolist()
        truth_dict['target_pos_hit'] = truthdata['target_xyz_hit'].astype(float).T.tolist()
        truth_df = pd.DataFrame(truth_dict)

        return scans_df, truth_df, params_df


class BlueMultiDetectionReaderFile(DetectionReader, TextFileReader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scans, self._truthdata, self._params = self._read_data(self.path)

    @BufferedGenerator.generator_method
    def detections_gen(self):
        n_timestamps = len(self._scans.transmit_times)
        R = CovarianceMatrix(np.diag(np.concatenate((self._params.thetaErrorSDs,
                                                     self._params.psiErrorSDs,
                                                     self._params.timeErrorSDs)) ** 2))

        timestamp_init = datetime.now()

        for k in range(n_timestamps):
            received_times = self._scans.received_times[k].ravel()
            transmit_time = self._scans.transmit_times[k]

            # Estimate the times each pulse hit the target
            timedelays = received_times - transmit_time
            hittimes = transmit_time + timedelays/2

            detections = set()
            num_targets = int(len(hittimes)/2)
            for i in range(num_targets):
                hittime = hittimes[2*i]
                state_vector = self._get_meas(k, i)

                timestamp = timestamp_init + timedelta(seconds=hittime)

                # Position of sensor 1 at transmit time
                sensor1_pos_trans = self._scans.sensor_trans_pos[k].astype(float)

                # Positions of sensors at receive times
                sensor1_pos_rec = self._scans.sensor_rec_pos[k].astype(float)[:, 2*i][:, np.newaxis]
                sensor2_pos_rec = self._scans.sensor_rec_pos[k].astype(float)[:, 2*i+1][:, np.newaxis]

                model = SimpleBlueMeasurementModel(ndim_state=12, mapping=[0, 2, 4],
                                                   noise_covar=R, sensor1_pos_rec=sensor1_pos_rec,
                                                   sensor1_pos_trans=sensor1_pos_trans, sensor2_pos_rec=sensor2_pos_rec)

                detection = Detection(state_vector, timestamp=timestamp, measurement_model=model)
                detections.add(detection)

            yield timestamp, detections

    def _get_meas(self, k, i):
        theta = np.array([Elevation(t) for t in self._scans.received_theta[k].ravel()[2*i:2*i+2]])
        psi = np.array([Bearing(t) for t in self._scans.received_psi[k].ravel()[2*i:2*i+2]])
        times = self._scans.received_times[k].ravel()[2*i:2*i+2] - self._scans.transmit_times[k]
        z = StateVector(np.concatenate((theta, psi, times)))
        return z

    @staticmethod
    def _read_mat(path):
        wp = loadmat(path)
        class MeasData2:
            def __init__(self, wp):
                measdata = wp['scans']
                params = wp['params'][0,0]
                self.transmit_times = measdata[0, :]['transmit_time'].astype(float)
                self.received_times = np.squeeze(np.dstack(tuple(measdata[0, :]['received_times'])))
                self.sensor_trans_pos = np.squeeze(np.dstack(tuple(measdata[0, :]['sensor_trans_pos'])))
                self.sensor_rec_pos = np.dstack(tuple(measdata[0, :]['sensor_rec_pos']))
                # self.sensor_rec_pos = np.reshape(self.sensor_rec_pos, (3, 2, -1), 'F')
                self.received_theta = np.squeeze(np.dstack(tuple(measdata[0, :]['received_theta'])))
                # self.received_theta = np.reshape(self.received_theta, (2, -1),'F')
                self.received_psi = np.squeeze(np.dstack(tuple(measdata[0, :]['received_psi'])))
                # self.received_psi = np.reshape(self.received_psi, (2, -1),'F')
                self.timeErrorSDs = params['timeErrorSDs'].ravel().astype(float)
                self.thetaErrorSDs = params['thetaErrorSDs'].ravel().astype(float)
                self.psiErrorSDs = params['psiErrorSDs'].ravel().astype(float)

        class TruthData2:
            def __init__(self, truthdata):
                self.hit_times = truthdata['hit_times'].ravel().astype(float)
                self.target_xyz = truthdata['target_xyz'].astype(float)
                self.target_xyz_hit = truthdata['target_xyz_hit'].astype(float)

        def get_truth(wp):
            truthdata = wp['truthdata']
            truths = []
            for i in range(truthdata.size):
                truths.append(TruthData2(truthdata[0, i]))
            return truths

        return MeasData2(wp), get_truth(wp)

    @staticmethod
    def _read_data(path):
        wp = loadmat(path)

        measdata = wp['scans']
        params = wp['params'][0, 0]
        scans_dict = dict()
        scans_dict['transmit_times'] = measdata[0, :]['transmit_time'].astype(float)
        scans_dict['received_times'] = measdata[0, :]['received_times'].tolist()
        scans_dict['sensor_trans_pos'] = measdata[0, :]['sensor_trans_pos'].tolist()
        scans_dict['sensor_rec_pos'] = measdata[0, :]['sensor_rec_pos'].tolist()
        scans_dict['received_theta'] = measdata[0, :]['received_theta'].tolist()
        scans_dict['received_psi'] = measdata[0, :]['received_psi'].tolist()
        scans_df = pd.DataFrame(scans_dict)

        params_dict = dict()
        params_dict['timeErrorSDs'] = params['timeErrorSDs'].ravel().astype(float)
        params_dict['thetaErrorSDs'] = params['thetaErrorSDs'].ravel().astype(float)
        params_dict['psiErrorSDs'] = params['psiErrorSDs'].ravel().astype(float)
        params_df = pd.DataFrame(params_dict)

        truth_df = []
        truthdata = wp['truthdata']
        truth_dict = dict()
        truth_dict['hit_times'] = truthdata[0, :]['hit_times'].tolist()
        truth_dict['target_pos'] = truthdata[0, :]['target_xyz'].tolist()
        truth_dict['target_pos_hit'] = truthdata[0, :]['target_xyz_hit'].tolist()
        truth_df = pd.DataFrame(truth_dict)

        return scans_df, truth_df, params_df

    @staticmethod
    def _inteleave_array_cols(arr, num_repeats):
        arrays = [arr for _ in range(num_repeats)]
        shape = (arr.shape[0], num_repeats * arr.shape[1])
        interleaved_array = np.dstack(arrays).reshape(shape)
        return interleaved_array

class BlueMultiDetectionReaderFile2(DetectionReader, TextFileReader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scans, self._truthdata, self._params = self._read_data(self.path)

    @BufferedGenerator.generator_method
    def detections_gen(self):
        n_timestamps = len(self._scans.transmit_times)
        # R = CovarianceMatrix(np.diag(np.concatenate((self._params.thetaErrorSDs,
        #                                              self._params.psiErrorSDs,
        #                                              self._params.timeErrorSDs)) ** 2))

        timestamp_init = datetime.now()
        for i in range(self._truthdata.hit_times.shape[0]):
            self._truthdata.hit_times[i] = self._truthdata.hit_times[i] * timedelta(seconds=1) + timestamp_init

        for k in range(n_timestamps):
            received_times = self._scans.received_times[k].ravel()
            transmit_time = self._scans.transmit_times[k]

            # Estimate the times each pulse hit the target
            timedelays = received_times - transmit_time
            hittimes = transmit_time + timedelays[0::2] / 2
            target_order = np.argsort(hittimes)

            for i in target_order:
                hittime = hittimes[i]
                state_vectors = self._get_meas(k, i)

                timestamp = timestamp_init + timedelta(seconds=hittime)

                # Position of sensor 1 at transmit time
                sensor_pos_trans = self._scans.sensor_trans_pos[k].astype(float)

                # Positions of sensors at receive times
                sensor_pos_rec = [self._scans.sensor_rec_pos[k].astype(float)[:, 2 * i][:, np.newaxis],
                                  self._scans.sensor_rec_pos[k].astype(float)[:, 2 * i + 1][:, np.newaxis]]

                for sensor_id, state_vector in enumerate(state_vectors):
                    R = CovarianceMatrix(np.diag((self._params.thetaErrorSDs[sensor_id],
                                                  self._params.psiErrorSDs[sensor_id],
                                                  self._params.timeErrorSDs[sensor_id])) ** 2)

                    model = SimpleBlueMeasurementModel2(ndim_state=12, mapping=[0, 2, 4],
                                                        noise_covar=R, sensor_pos_rec=sensor_pos_rec[sensor_id],
                                                        sensor_pos_trans=sensor_pos_trans, sensor_id=sensor_id)

                    detection = Detection(state_vector, timestamp=timestamp, measurement_model=model)
                    yield timestamp, {detection}

    def _get_meas(self, k, i):
        theta = np.array([Elevation(t) for t in self._scans.received_theta[k].ravel()[2 * i:2 * i + 2]])
        psi = np.array([Bearing(t) for t in self._scans.received_psi[k].ravel()[2 * i:2 * i + 2]])
        times = self._scans.received_times[k].ravel()[2 * i:2 * i + 2] - self._scans.transmit_times[k]
        state_vectors = []
        for sv in zip(theta, psi, times):
            state_vectors.append(StateVector(sv))
        return state_vectors

    @staticmethod
    def _read_mat(path):
        wp = loadmat(path)

        class MeasData2:
            def __init__(self, wp):
                measdata = wp['scans']
                params = wp['params'][0, 0]
                self.transmit_times = measdata[0, :]['transmit_time'].astype(float)
                self.received_times = np.squeeze(np.dstack(tuple(measdata[0, :]['received_times'])))
                self.sensor_trans_pos = np.squeeze(np.dstack(tuple(measdata[0, :]['sensor_trans_pos'])))
                self.sensor_rec_pos = np.dstack(tuple(measdata[0, :]['sensor_rec_pos']))
                # self.sensor_rec_pos = np.reshape(self.sensor_rec_pos, (3, 2, -1), 'F')
                self.received_theta = np.squeeze(np.dstack(tuple(measdata[0, :]['received_theta'])))
                # self.received_theta = np.reshape(self.received_theta, (2, -1),'F')
                self.received_psi = np.squeeze(np.dstack(tuple(measdata[0, :]['received_psi'])))
                # self.received_psi = np.reshape(self.received_psi, (2, -1),'F')
                self.timeErrorSDs = params['timeErrorSDs'].ravel().astype(float)
                self.thetaErrorSDs = params['thetaErrorSDs'].ravel().astype(float)
                self.psiErrorSDs = params['psiErrorSDs'].ravel().astype(float)

        class TruthData2:
            def __init__(self, truthdata):
                self.hit_times = truthdata['hit_times'].ravel().astype(float)
                self.target_xyz = truthdata['target_xyz'].astype(float)
                self.target_xyz_hit = truthdata['target_xyz_hit'].astype(float)

        def get_truth(wp):
            truthdata = wp['truthdata']
            truths = []
            for i in range(truthdata.size):
                truths.append(TruthData2(truthdata[0, i]))
            return truths

        return MeasData2(wp), get_truth(wp)

    @staticmethod
    def _read_data(path):
        wp = loadmat(path)

        measdata = wp['scans']
        params = wp['params'][0, 0]
        scans_dict = dict()
        scans_dict['transmit_times'] = measdata[0, :]['transmit_time'].astype(float)
        scans_dict['received_times'] = measdata[0, :]['received_times'].tolist()
        scans_dict['sensor_trans_pos'] = measdata[0, :]['sensor_trans_pos'].tolist()
        scans_dict['sensor_rec_pos'] = measdata[0, :]['sensor_rec_pos'].tolist()
        scans_dict['received_theta'] = measdata[0, :]['received_theta'].tolist()
        scans_dict['received_psi'] = measdata[0, :]['received_psi'].tolist()
        scans_df = pd.DataFrame(scans_dict)

        params_dict = dict()
        params_dict['timeErrorSDs'] = params['timeErrorSDs'].ravel().astype(float)
        params_dict['thetaErrorSDs'] = params['thetaErrorSDs'].ravel().astype(float)
        params_dict['psiErrorSDs'] = params['psiErrorSDs'].ravel().astype(float)
        params_df = pd.DataFrame(params_dict)

        truth_df = []
        truthdata = wp['truthdata']
        truth_dict = dict()
        truth_dict['hit_times'] = truthdata[0, :]['hit_times'].tolist()
        truth_dict['target_pos'] = truthdata[0, :]['target_xyz'].tolist()
        truth_dict['target_pos_hit'] = truthdata[0, :]['target_xyz_hit'].tolist()
        truth_df = pd.DataFrame(truth_dict)

        return scans_df, truth_df, params_df

    @staticmethod
    def _inteleave_array_cols(arr, num_repeats):
        arrays = [arr for _ in range(num_repeats)]
        shape = (arr.shape[0], num_repeats * arr.shape[1])
        interleaved_array = np.dstack(arrays).reshape(shape)
        return interleaved_array