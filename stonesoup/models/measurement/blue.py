from stonesoup.models.base import NonLinearModel, GaussianModel
from stonesoup.models.measurement import MeasurementModel
from stonesoup.base import Property
from stonesoup.types.angle import Elevation, Bearing
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.functions import cart2sphere

class SimpleBlueMeasurementModel(MeasurementModel, NonLinearModel, GaussianModel):
    noise_covar: CovarianceMatrix = Property(doc="Noise covariance")
    sensor1_pos_trans: StateVector = Property(doc='Position of sensor 1 at transmit time', default=StateVector([0,0,0]))
    sensor1_pos_rec: StateVector = Property(doc='Position of sensor 1 at receive time', default=StateVector([0,0,0]))
    sensor2_pos_rec: StateVector = Property(doc='Position of sensor 2 at receive time', default=StateVector([0,0,0]))
    with_bias: bool = Property(default=True)

    @property
    def ndim_meas(self) -> int:
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions
        """

        return 6

    def covar(self, **kwargs) -> CovarianceMatrix:
        """Returns the measurement model noise covariance matrix.

        Returns
        -------
        :class:`~.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_meas`, :py:attr:`~ndim_meas`)
            The measurement noise covariance.
        """

        return self.noise_covar

    def function(self, state, noise=False, **kwargs):

        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs()
            else:
                noise = 0

        xyz = state.state_vector[[0, 2, 4], :]
        if self.with_bias:
            bias = state.state_vector[6:, :]
        else:
            bias = StateVector([0, 0, 0, 0, 0, 0])
        diff = xyz - self.sensor1_pos_trans
        diff1 = self.sensor1_pos_rec - xyz
        diff2 = self.sensor2_pos_rec - xyz

        # Get distance from transmitter to target
        rT, _, _ = cart2sphere(*diff.ravel())

        # Get distance, el, az from target to receiving sensors
        r1, psi1, theta1 = cart2sphere(*diff1.ravel())
        r2, psi2, theta2 = cart2sphere(*diff2.ravel())

        # Get unbiased el, az, time delay
        c = 1.4933e+03
        predmeas = StateVector([Elevation(theta1), Elevation(theta2),
                                Bearing(psi1), Bearing(psi2),
                                (rT+r1)/c, (rT+r2)/c]) + bias

        return predmeas + noise


class SimpleBlueMeasurementModel2(MeasurementModel, NonLinearModel, GaussianModel):
    noise_covar: CovarianceMatrix = Property(doc="Noise covariance")
    sensor_pos_trans: StateVector = Property(doc='Position of sensor at transmit time', default=StateVector([0,0,0]))
    sensor_pos_rec: StateVector = Property(doc='Position of sensor at receive time', default=StateVector([0,0,0]))
    sensor_id: int = Property(doc='Sensor id', default=-1)
    with_bias: bool = Property(default=True)

    @property
    def ndim_meas(self) -> int:
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions
        """

        return 3

    def covar(self, **kwargs) -> CovarianceMatrix:
        """Returns the measurement model noise covariance matrix.

        Returns
        -------
        :class:`~.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_meas`, :py:attr:`~ndim_meas`)
            The measurement noise covariance.
        """

        return self.noise_covar

    def function(self, state, noise=False, **kwargs):

        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs()
            else:
                noise = 0

        xyz = state.state_vector[[0, 2, 4], :]
        if self.with_bias:
            bias = state.state_vector[6+self.sensor_id::2, :]
        else:
            bias = StateVector([0, 0, 0])
        diff = xyz - self.sensor_pos_trans
        diff1 = self.sensor_pos_rec - xyz

        # Get distance from transmitter to target
        rT, _, _ = cart2sphere(*diff.ravel())

        # Get distance, el, az from target to receiving sensors
        r1, psi1, theta1 = cart2sphere(*diff1.ravel())

        # Get unbiased el, az, time delay
        c = 1.4933e+03
        predmeas = StateVector([Elevation(theta1),
                                Bearing(psi1),
                                (rT+r1)/c]) + bias

        return predmeas + noise