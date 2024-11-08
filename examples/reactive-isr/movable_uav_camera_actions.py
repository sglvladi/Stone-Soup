import itertools
from datetime import datetime, timedelta
from uuid import uuid4

import numpy as np

from reactive_isr_core.data import RFI, GeoRegion, GeoLocation, PriorityOverTime
from stonesoup.custom.sensor.movable import MovableUAVCamera
from stonesoup.types.angle import Angle
from stonesoup.types.array import StateVector
from stonesoup.types.state import State

# Set a query time
timestamp = datetime.now()

# The camera is initially positioned
position = StateVector([-2.976389, 53.385070, 100.])

rfi = RFI(
    id=uuid4(),
    task_type='count',
    region_of_interest=[
        GeoRegion(corners=[GeoLocation(latitude=53.420079, longitude=-2.878155, altitude=0),
                           GeoLocation(latitude=53.420705, longitude=-2.876979, altitude=0)]),
    ],
    start_time=timestamp,
    end_time=timestamp + timedelta(seconds=20),
    priority_over_time=PriorityOverTime(
        priority=[1, 1],
        timescale=[timestamp, timestamp + timedelta(seconds=20)]
    ),
    targets=[],
    status='started'
)

# Create a camera object
sensor = MovableUAVCamera(ndim_state=6, mapping=[0, 2, 4],
                          noise_covar=np.diag([0.05, 0.05, 0.05]),
                          location=position[0:2],
                          position=position,
                          fov_radius=100,
                          fov_in_km=False,
                          max_speed=14,  # ~50 km/h
                          constrain_speed=True,
                          rfis=[rfi])
sensor.movement_controller.states[0].timestamp = timestamp

# Set a query time
timestamp += timedelta(seconds=20)

# Calling sensor.actions() will return a set of action generators. Each action generator is an
# object that contains all the actions that can be performed by the sensor at a given time. In this
# case, the sensor has two actionable properties: X and Y location. Hence, the result of
# sensor.actions() is a set of two action generators: one for moving on the X-axis and one for
# moving on the Y-axis.
action_generators = sensor.actions(timestamp)

# Let's look at the action generators
# The first action generator is for the X location. We can extract the action generator by
# searching for the action generator that controls the 'location_x' property. So, the following
# line of code simply filters the action generators that control 'location_x' (the for-if
# statement) and then selects the first action generator (since there is only one), via the next()
# statement.
action_generator = next(ag for ag in action_generators if ag.attribute == 'location')

# We can now look at the actions that can be performed by the action generators. The action
# generators provide a Python "iterator" interface. This means that we can iterate over the action
# generators to get the actions that can be performed (e.g. with a "for" loop). Instead, we can
# also use the list() function to get a list of all the actions that can be performed.
possible_actions = list(action_generator)

# Each action has a "target_value" property that specifies the value that the property will be
# set to if the action is performed. The following line of code prints the target values of the
# 10th action for pan and tilt.
print(possible_actions[1].target_value)

# Let us now select the 10th action combination and task the sensor to perform the action.
chosen_action_combination = possible_actions[1]
sensor.add_actions([chosen_action_combination])
sensor.act(timestamp)

# We can also create a custom action combination. For example, we can move the camera to the
# location (0, 10, 100) by generating an action that sets the X location to 0 and an action that
# sets the Y location to 10. We can then combine these two actions into a single action combination
# and task the sensor to perform the action.
custom_action = action_generator.action_from_value([0, 10])   # Action that sets the location to (0, 10)
custom_action_combination = (custom_action,)
sensor.add_actions(custom_action_combination)
sensor.act(timestamp)

# The statement below is just an extra statement to allow us to breakpoint the code and inspect
# the possible actions.
end = True
