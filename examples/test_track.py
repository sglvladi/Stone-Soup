import datetime
import numpy as np

from stonesoup.types.track import Track
from stonesoup.types.state import State

tracks = {Track(states=[State(np.array([[1]]),
                             timestamp=datetime.datetime.now()),
                        # State(np.array([[1]]),
                        #       timestamp=datetime.datetime.now())
                        ])}
