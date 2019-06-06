# -*- coding: utf-8 -*-
import copy
import numpy as np

from .base import Sensor
from ..base import Property
from ..models.measurement.nonlinear import BearingGaussianToCartesian
from ..types.array import CovarianceMatrix
from ..types.detection import Detection, Clutter
from ..types.state import State, StateVector
from ..types.numeric import Probability
from ..types.angle import Bearing


class SonarJustBearing(Sensor):
    """A very simple sonar sensor that generates measurements of targets, using
    a :class:`~.BearingGaussianToCartesian` model, relative to its position.

    Note
    ----
    The current implementation of this class assumes a 3D Cartesian plane.

    """

    position = Property(StateVector,
                        doc="The sonar position on a 3D Cartesian plane,\
                             expressed as a 3x1 array of Cartesian coordinates\
                             in the order :math:`x,y,z`")
    orientation = Property(
        StateVector,
        doc="A 3x1 array of angles (rad), specifying the sonar orientation in \
            terms of the counter-clockwise rotation around each Cartesian \
            axis in the order :math:`x,y,z`. The rotation angles are positive \
            if the rotation is in the counter-clockwise direction when viewed \
            by an observer looking along the respective rotation axis, \
            towards the origin")
    ndim_state = Property(
        int,
        doc="Number of state dimensions. This is utilised by (and follows in\
            format) the underlying :class:`~.BearingGaussianToCartesian`\
            model")
    mapping = Property(
        [np.array], doc="Mapping between the targets state space and the\
                        sensors measurement capability")
    noise_covar = Property(CovarianceMatrix,
                           doc="The sensor noise covariance matrix. This is \
                                utilised by (and follows in format) the \
                                underlying \
                                :class:`~.BearingGaussianToCartesian`\
                                model")
    detection_probability = Property(Probability, default=1.0)
    clutter_rate = Property(float, default=0.0)

    def __init__(self, position, orientation, ndim_state, mapping, noise_covar,
                 *args, **kwargs):
        measurement_model = BearingGaussianToCartesian(
            ndim_state=ndim_state,
            mapping=mapping,
            noise_covar=noise_covar,
            translation_offset=position,
            rotation_offset=orientation)

        super().__init__(position, orientation, ndim_state, mapping,
                         noise_covar, *args, measurement_model, **kwargs)

    def set_position(self, position):
        self.position = position
        self.measurement_model.translation_offset = position

    def set_orientation(self, orientation):
        self.orientation = orientation
        self.measurement_model.rotation_offset = orientation

    def gen_measurement(self, ground_truth, noise=None, **kwargs):
        """Generate a measurement for a given state

        Parameters
        ----------
        ground_truth : :class:`~.State`
            A ground-truth state

        Returns
        -------
        :class:`~.Detection`
            A measurement generated from the given state. The timestamp of the\
            measurement is set equal to that of the provided state.
        """

        detection = None
        if np.random.rand() < self.detection_probability:
            # Set rotation offset of underlying measurement model
            rot_offset = \
                StateVector(
                    [[self.orientation[0, 0]],
                     [self.orientation[1, 0]],
                     [self.orientation[2, 0]]])
            self.measurement_model.rotation_offset = rot_offset

            measurement_vector = self.measurement_model.function(
                ground_truth.state_vector, noise=0, **kwargs)

            if (noise is None):
                measurement_noise = self.measurement_model.rvs()
            else:
                measurement_noise = noise

            model_copy = copy.copy(self.measurement_model)
            measurement_vector += measurement_noise  # Add noise
            detection = Detection(measurement_vector,
                                  measurement_model=model_copy,
                                  timestamp=ground_truth.timestamp)
        return detection

    def gen_clutter(self, timestamp, **kwargs):
        """Generate clutter

        Returns
        -------
        set
            A set of clutter measurements
        """
        rot_offset = \
            StateVector(
                [[self.orientation[0, 0]],
                 [self.orientation[1, 0]],
                 [self.orientation[2, 0]]])
        self.measurement_model.rotation_offset = rot_offset

        clutter = set()
        # generate clutter
        for _ in range(np.random.poisson(self.clutter_rate)):
            bearing = -np.pi+2*np.pi*np.random.rand()
            detection = Clutter(
                state_vector=np.array([[Bearing(bearing)]]),
                measurement_model=copy.copy(self.measurement_model),
                timestamp=timestamp)
            clutter.add(detection)
        return clutter