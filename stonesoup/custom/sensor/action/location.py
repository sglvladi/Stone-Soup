import itertools

import numpy as np
from typing import Iterator, List

from stonesoup.custom.functions import get_nearest
from stonesoup.types.array import StateVector

from stonesoup.base import Property

from stonesoup.sensor.action import Action, RealNumberActionGenerator


class ChangeLocationAction(Action):
    def act(self, current_time, timestamp, init_value):
        """Assumes that duration keeps within the action end time

        Parameters
        ----------
        current_time: datetime.datetime
            Current time
        timestamp: datetime.datetime
            Modification of attribute ends at this time stamp
        init_value: Any
            Current value of the dwell centre

        Returns
        -------
        Any
            The new value of the dwell centre"""

        if timestamp >= self.end_time:
            return self.target_value  # target direction
        else:
            return init_value  # same direction


class LocationActionGenerator(RealNumberActionGenerator):
    """Generates possible actions for changing the dwell centre of a sensor in a given
        time period."""

    owner: object = Property(doc="Object with `timestamp`, `rpm` (revolutions per minute) and "
                                 "dwell-centre attributes")
    resolution: float = Property(default=10, doc="Resolution of action space")
    limits: StateVector = Property(doc="Min and max values of the action space",
                                   default=StateVector([-100, 100]))
    possible_values: List = Property(doc="List of possible values for the action space",
                                     default=None)
    _action_cls = ChangeLocationAction

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.possible_values is not None:
            self.possible_values = sorted(self.possible_values)

    @property
    def default_action(self):
        return self._action_cls(generator=self,
                                end_time=self.end_time,
                                target_value=self.current_value)

    def __call__(self, resolution=None, epsilon=None):
        """
        Parameters
        ----------
        resolution : float
            Resolution of yielded action target values
        epsilon: float
            Epsilon value for action target values

        Returns
        -------
        :class:`.Action`
            Action with target value
        """
        if resolution is not None:
            self.resolution = resolution
        if epsilon is not None:
            self.epsilon = epsilon

    @property
    def initial_value(self):
        return self.current_value

    @property
    def min(self):
        # Pan can rotate freely, while tilt is limited to +/- 90 degrees
        return self.limits[0]

    @property
    def max(self):
        # Pan can rotate freely, while tilt is limited to +/- 90 degrees
        return self.limits[1]

    def __contains__(self, item):

        if isinstance(item, self._action_cls):
            item = item.target_value

        possible_values = self._get_possible_values()
        possible_values = np.append(possible_values, self.current_value)
        possible_values.sort()
        return possible_values[0] <= item <= possible_values[-1]

    def __iter__(self) -> Iterator[ChangeLocationAction]:
        """Returns all possible ChangePanTiltAction types"""
        possible_values = self._get_possible_values()

        yield self.default_action
        for angle in possible_values:
            if angle == self.current_value:
                continue
            yield self._action_cls(generator=self,
                                   end_time=self.end_time,
                                   target_value=angle)

    def action_from_value(self, value):
        if value not in self:
            return None
        possible_values = self._get_possible_values()
        possible_values = np.append(possible_values, self.current_value)
        angle = get_nearest(possible_values, value)
        return self._action_cls(generator=self,
                                end_time=self.end_time,
                                target_value=angle)

    def _get_possible_values(self):
        if self.possible_values is not None:
            return np.array(self.possible_values)
        else:
            return np.arange(self.min, self.max + self.resolution, self.resolution, dtype=float)