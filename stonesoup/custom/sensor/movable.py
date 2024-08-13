import datetime
from typing import Union, List, Set

import numpy as np
import geopy.distance
from shapely import Point

from stonesoup.base import Property
from stonesoup.custom.functions import geodesic_point_buffer, \
    cover_rectangle_with_minimum_overlapping_circles
from stonesoup.custom.sensor.action.location import LocationActionGenerator
from stonesoup.models.clutter import ClutterModel
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.sensor.action import ActionGenerator
from stonesoup.sensor.actionable import ActionableProperty
from stonesoup.sensor.sensor import Sensor
from stonesoup.types.array import CovarianceMatrix, StateVector
from stonesoup.types.detection import TrueDetection
from stonesoup.types.groundtruth import GroundTruthState


class MovableUAVCamera(Sensor):
    """A movable UAV camera sensor."""

    ndim_state: int = Property(
        doc="Number of state dimensions. This is utilised by (and follows in\
                    format) the underlying :class:`~.CartesianToElevationBearing`\
                    model")
    mapping: np.ndarray = Property(
        doc="Mapping between the targets state space and the sensors\
                    measurement capability")
    noise_covar: CovarianceMatrix = Property(
        doc="The sensor noise covariance matrix. This is utilised by\
                    (and follow in format) the underlying \
                    :class:`~.CartesianToElevationBearing` model")
    fov_radius: Union[float, List[float]] = Property(
        doc="The detection field of view radius of the sensor")
    clutter_model: ClutterModel = Property(
        default=None,
        doc="An optional clutter generator that adds a set of simulated "
            ":class:`Clutter` objects to the measurements at each time step. "
            "The clutter is simulated according to the provided distribution.")
    location: StateVector = ActionableProperty(
        doc="The sensor location. Defaults to zero",
        default=None,
        generator_cls=LocationActionGenerator
    )
    limits: dict = Property(
        doc="The sensor min max location",
        default=None
    )
    fov_in_km: bool = Property(
        doc="Whether the FOV radius is in kilo-meters or degrees",
        default=True)
    rfis: List = Property(
        doc="The RFIs in the scene",
        default=None
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._footprint = None
        if self.rfis is None:
            self.rfis = []

    @location.setter
    def location(self, value):
        self._property_location = value
        if not self.movement_controller:
            return
        new_position = self.movement_controller.position.copy()
        new_position[0] = value[0]
        new_position[1] = value[1]
        self.movement_controller.position = new_position

    @property
    def measurement_model(self):
        return LinearGaussian(
            ndim_state=self.ndim_state,
            mapping=self.mapping,
            noise_covar=self.noise_covar)

    @property
    def footprint(self):
        if self._footprint is None:
            if self.fov_in_km:
                self._footprint = geodesic_point_buffer(*np.flip(self.position[0:2]),
                                                        self.fov_radius)
            else:
                self._footprint = Point(self.position[0:2]).buffer(self.fov_radius)
        return self._footprint

    def act(self, timestamp: datetime.datetime):
        super().act(timestamp)
        if self.fov_in_km:
            self._footprint = geodesic_point_buffer(*np.flip(self.position[0:2]), self.fov_radius)
        else:
            self._footprint = Point(self.position[0:2]).buffer(self.fov_radius)

    def measure(self, ground_truths: Set[GroundTruthState], noise: Union[np.ndarray, bool] = True,
                **kwargs) -> Set[TrueDetection]:

        detections = set()
        measurement_model = self.measurement_model

        for truth in ground_truths:
            # Transform state to measurement space and generate random noise
            measurement_vector = measurement_model.function(truth, noise=noise, **kwargs)

            if self.fov_in_km:
                # distance = geopy.distance.distance(np.flip(self.position[0:2]),
                #                                    np.flip(measurement_vector[0:2])).km
                if not self._footprint.contains(Point(measurement_vector[0:2])):
                    continue
            else:
                # Normalise measurement vector relative to sensor position
                norm_measurement_vector = measurement_vector.astype(float) - self.position.astype(
                    float)
                distance = np.linalg.norm(norm_measurement_vector[0:2])

                # Do not measure if state not in FOV
                if distance > self.fov_radius:
                    continue

            detection = TrueDetection(measurement_vector,
                                      measurement_model=measurement_model,
                                      timestamp=truth.timestamp,
                                      groundtruth_path=truth)
            detections.add(detection)

        # Generate clutter at this time step
        if self.clutter_model is not None:
            self.clutter_model.measurement_model = measurement_model
            clutter = self.clutter_model.function(ground_truths)
            detections |= clutter

        return detections

    def _default_action(self, name, property_, timestamp):
        """Returns the default action of the action generator associated with the property
        (assumes the property is an :class:`~.ActionableProperty`)."""
        generator = self._get_generator(name, property_, timestamp, self.timestamp)
        return generator.default_action

    def actions(self, timestamp: datetime.datetime, start_timestamp: datetime.datetime = None
                ) -> Set[ActionGenerator]:
        """Method to return a set of action generators available up to a provided timestamp.

        A generator is returned for each actionable property that the sensor has.

        Parameters
        ----------
        timestamp: datetime.datetime
            Time of action finish.
        start_timestamp: datetime.datetime, optional
            Time of action start.

        Returns
        -------
        : set of :class:`~.ActionGenerator`
            Set of action generators, that describe the bounds of each action space.
        """

        if not self.validate_timestamp():
            self.timestamp = timestamp

        if start_timestamp is None:
            start_timestamp = self.timestamp

        started_rfis = [rfi for rfi in self.rfis if rfi.status == "started"]
        rois = [roi for rfi in started_rfis for roi in rfi.region_of_interest]
        possible_locations = []
        footprint = self.footprint
        # Get min max lat lon of the footprint
        min_lon, min_lat, max_lon, max_lat = footprint.bounds
        # NOTE: This is an approximation of asset fov in lat/long degrees (1 degree = 111km)
        # asset_fov_ll = self.fov_radius / 111
        asset_fov_ll = min((max_lat - min_lat), (max_lon - min_lon))
        for roi in rois:
            x1 = roi.corners[0].longitude
            y1 = roi.corners[0].latitude
            x2 = roi.corners[1].longitude
            y2 = roi.corners[1].latitude
            # For each roi, find the minimum number of overlapping circles required to cover it
            possible_locations += cover_rectangle_with_minimum_overlapping_circles(
                x1, y1, x2, y2, asset_fov_ll
            )
        possible_locations = [StateVector([loc[0], loc[1]]) for loc in possible_locations]
        generators = set()
        for name, property_ in self._actionable_properties.items():
            generators.add(
                self._get_generator(name, property_, timestamp, start_timestamp, possible_locations)
            )

        # generators = {self._get_generator(name, property_, timestamp, start_timestamp, rois)
        #               for name, property_ in self._actionable_properties.items()}

        return generators

    def _get_generator(self, name, prop, timestamp, start_timestamp, possible_values=None):
        """Returns the action generator associated with the """
        kwargs = {'owner': self, 'attribute': name, 'start_time': start_timestamp,
                  'end_time': timestamp, 'possible_values': possible_values}
        if self.resolutions and name in self.resolutions.keys():
            kwargs['resolution'] = self.resolutions[name]
        if self.limits and name in self.limits.keys():
            kwargs['limits'] = self.limits[name]
        generator = prop.generator_cls(**kwargs)
        return generator