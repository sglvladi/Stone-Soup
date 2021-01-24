import numpy as np
import scipy.io as sio

from stonesoup.base import Property
from stonesoup.initiator import Initiator
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track
from stonesoup.wrapper.matlab import MatlabWrapper
from stonesoup.types.update import GaussianStateUpdate
from stonesoup.types.array import StateVector, CovarianceMatrix


class SimpleBlueInitiator(Initiator, MatlabWrapper):

    with_bias: bool = Property(default=True)

    def initiate(self, detections, **kwargs):
        tracks = set()

        for detection in detections:

            model = detection.measurement_model

            prior = dict()
            prior['xyz_min'] = self.matlab_array(np.array([[-10000.],[-10000.],[0.]])) # np.array([[-10000.],[-10000.],[0.]])
            prior['xyz_max'] =  self.matlab_array(np.array([[0.],[10000.],[4000.]])) # np.array([[0.],[10000.],[4000.]])
            prior['vel_xyz_sd'] = self.matlab_array(np.array([10., 10., 1.])) # np.array([10., 10., 1.])
            prior['el_bias_sd'] = 0.1
            prior['az_bias_sd'] = 0.01
            prior['dt_bias_sd'] = 0.2

            meas = self.matlab_array(detection.state_vector)
            sensor1_xyz_trans = self.matlab_array(model.sensor1_pos_trans)
            sensor1_xyz_rec = self.matlab_array(model.sensor1_pos_rec)
            sensor2_xyz_rec = self.matlab_array(model.sensor2_pos_rec)
            R = self.matlab_array(model.covar())

            # my_struct = {
            #     'prior': prior,
            #     'meas': detection.state_vector.astype(float),
            #     'sensor1_xyz_trans': model.sensor1_pos_trans,
            #     'sensor1_xyz_rec': model.sensor1_pos_rec,
            #     'sensor2_xyz_rec': model.sensor2_pos_rec,
            #     'R': model.covar()
            # }
            # sio.savemat(r'C:\Users\sglvladi\OneDrive\Workspace\PostDoc\CADMURI\MATLAB\BLUE\Simulator\TestData\test_data.mat', my_struct)

            state_mean, state_cov = self.matlab_engine.getInitialStateDistribution(
                prior, meas, sensor1_xyz_trans, sensor1_xyz_rec, sensor2_xyz_rec, R, nargout=2)
            state_mean = StateVector(np.array(state_mean))
            state_cov = CovarianceMatrix(np.array(state_cov))
            if not self.with_bias:
                state_mean = state_mean[:6]
                state_cov = state_cov[:6, :6]

            state = GaussianStateUpdate(state_mean, state_cov, SingleHypothesis(None, detection),
                                        timestamp=detection.timestamp)
            tracks.add(Track(state))
        return tracks


class SingleMeasurementBlueInitiator(Initiator, MatlabWrapper):

    with_bias: bool = Property(default=True)

    def initiate(self, detections, **kwargs):
        tracks = set()
        for detection in detections:

            model = detection.measurement_model

            prior = dict()
            prior['xyz_min'] = self.matlab_array(np.array([[0.],[-10000.],[0.]]))
            prior['xyz_max'] = self.matlab_array(np.array([[20000.],[10000.],[4000.]]))
            prior['vel_xyz_sd'] = self.matlab_array(np.array([10., 10., 1.]))
            prior['el_bias_sd'] = 0.1
            prior['az_bias_sd'] = 0.01
            prior['dt_bias_sd'] = 0.2

            meas = self.matlab_array(detection.state_vector)
            sensor_xyz_trans = self.matlab_array(model.sensor_pos_trans)
            sensor_xyz_rec = self.matlab_array(model.sensor_pos_rec)
            sensor_rec_id = model.sensor_id + 1
            R = self.matlab_array(model.covar())

            state_mean, state_cov = self.matlab_engine.getInitDistributionSingle(
                prior, meas, sensor_xyz_trans, sensor_xyz_rec, R, sensor_rec_id, nargout=2)
            state_mean = StateVector(np.array(state_mean))
            state_cov = CovarianceMatrix(np.array(state_cov))
            if not self.with_bias:
                state_mean = state_mean[:6]
                state_cov = state_cov[:6, :6]

            state = GaussianStateUpdate(state_mean, state_cov, SingleHypothesis(None, detection),
                                        timestamp=detection.timestamp)
            tracks.add(Track(state))
        return tracks