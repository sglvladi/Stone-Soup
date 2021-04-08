import numpy as np

from stonesoup.base import Property
from stonesoup.tracker.base import Tracker
from stonesoup.detector.base import Detector
from stonesoup.wrapper.matlab import MatlabWrapper
from stonesoup.buffered_generator import BufferedGenerator

class ElintTracker(Tracker, MatlabWrapper):
    detector: Detector = Property(doc='Detector')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set up prior
        speed_metres_sd = 10.
        self._prior = {'posMin': self.matlab_array([-84., 34.]),
                       'posMax': self.matlab_array([9., 62.]),
                       'speedMetresSD': speed_metres_sd}
        # Set up transition model
        mins2sec = 60
        hours2sec = 60 * mins2sec
        days2sec = 24 * hours2sec
        stationary_speed = speed_metres_sd
        q_metres = .1
        self._tx_model = {
            'isOrnsteinUhlenbeck': True,
            'q_metres': q_metres,
            'K': q_metres/(2*stationary_speed**2),
            'deathRate': 1/(4*days2sec)
        }

        # Set up params
        self._params ={
            'killProbThresh': .01,
            'measHistLen': 2.,
            'compLogProbThresh': float(np.log(1e-3)),
            'trackstooutput': 'mostlikely',
            'outputMostLikelyComp': True,
            'estNumUnconfirmedTargets': self.detector.num_targets,
            'truth': self.detector._truthdata
        }

    @BufferedGenerator.generator_method
    def tracks_gen(self):
        tracks = set()
        tracks_matlab = []
        next_track_id = 1

        for time, detections in self.detector:
            detection = detections.pop()

            this_meas = dict()
            invalid_keys = ['time_seconds', 'dt_seconds', 'sensor_idx', 'meas_num']

            for key in detection.metadata:
                if key in invalid_keys:
                    continue
                this_meas[key] = detection.metadata[key]
            this_meas['time'] = detection.metadata['time_seconds']
            dt_seconds = detection.metadata['dt_seconds']
            sensor_idx = detection.metadata['sensor_idx']
            meas_num = detection.metadata['meas_num']

            tracks_matlab, next_track_id = \
                self.matlab_engine.tracker_single_python(tracks_matlab, this_meas, dt_seconds,
                                                         self._tx_model, sensor_idx, self.detector._sensdata,
                                                         self._prior, self.detector._colors, self._params, meas_num,
                                                         self.detector.num_scans, next_track_id, nargout=2)
            ASD=2
            # print(len(tracks_matlab))
            # print(next_track_id)
            for i in range(len(tracks_matlab)):
                for j in range(len(tracks_matlab[i]['mmsis'])):
                    if len(tracks_matlab[i]['mmsis'][j]) == 0:
                        tracks_matlab[i]['mmsis'][j] = ''
                        a = 2
