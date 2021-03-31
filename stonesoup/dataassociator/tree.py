# -*- coding: utf-8 -*-
import datetime
from collections import defaultdict
from operator import attrgetter

import numpy as np
import scipy as sp
import rtree
from scipy.spatial import KDTree

from stonesoup.types.angle import Longitude, Latitude
from stonesoup.types.detection import MissedDetection
from stonesoup.types.multihypothesis import MultipleHypothesis
from .base import DataAssociator
from ..base import Property
from ..models.base import LinearModel
from ..models.measurement import MeasurementModel
from ..predictor import Predictor
from ..types.update import Update
from ..updater import Updater


# import concurrent.futures
# from joblib import Parallel, delayed
# import multiprocessing
# def chunks(l,n):
#     """Yield successive n-sized chunks from l."""
#     c = []
#     j = []
#     x = 0
#     m = np.max([int(np.ceil(float(len(l))/float(n))),1])
#     for i in range(0, len(l), m):
#         c.append(l[i:i + m])
#         j.append([k for k in range(x, x + len(c[-1]))])
#         x += len(c[-1])
#     return c, j
#
# def f(args):
#     hyp = args[0]
#     tracks = args[1]
#     t_inds = args[2]
#     detections = args[3]
#     time = args[4]
#     return [{t_ind: hyp.hypothesise(tracks[i], detections,time)}
#             for i, t_ind in enumerate(t_inds)]
#
# def f2(args):
#     hyp = args[0]
#     track = args[1]
#     detections = args[2]
#     time = args[3]
#     return {track: hyp.hypothesise(track, detections,time)}

class DetectionKDTreeMixIn(DataAssociator):
    """Detection kd-tree based mixin

    Construct a kd-tree from detections and then use a :class:`~.Predictor` and
    :class:`~.Updater` to get prediction of track in measurement space. This is
    then queried against the kd-tree, and only matching detections are passed
    to the :attr:`hypothesiser`.

    Notes
    -----
    This is only suitable where measurements are in same space as each other
    and at the same timestamp.
    """
    predictor = Property(
        Predictor,
        doc="Predict tracks to detection times")
    updater = Property(
        Updater,
        doc="Updater used to get measurement prediction")
    number_of_neighbours = Property(
        int, default=None,
        doc="Number of neighbours to find. Default `None`, which means all "
            "points within the :attr:`max_distance` are returned.")
    max_distance = Property(
        float, default=np.inf,
        doc="Max distance to return points. Default `inf`")

    def generate_hypotheses(self, tracks, detections, time, **kwargs):
        # No need for tree here.
        if not tracks:
            return set()
        if not detections:
            return {track: self.hypothesiser.hypothesise(
                track, detections, time, **kwargs)
                for track in tracks}

        detections_list = list(detections)
        tree = KDTree(
            np.vstack([detection.state_vector[:, 0]
                       for detection in detections_list]))

        track_detections = defaultdict(set)
        for track in tracks:
            prediction = self.predictor.predict(track.state, time)
            meas_pred = self.updater.predict_measurement(prediction)

            if self.number_of_neighbours is None:
                indexes = tree.query_ball_point(
                    meas_pred.state_vector.ravel(),
                    r=self.max_distance)
            else:
                _, indexes = tree.query(
                    meas_pred.state_vector.ravel(),
                    k=self.number_of_neighbours,
                    distance_upper_bound=self.max_distance)

            for index in np.atleast_1d(indexes):
                if index != len(detections_list):
                    track_detections[track].add(detections_list[index])

        return {track: self.hypothesiser.hypothesise(
            track, track_detections[track], time, **kwargs)
            for track in tracks}


class TPRTreeMixIn(DataAssociator):
    measurement_model = Property(MeasurementModel)
    horizon_time = Property(datetime.timedelta)
    vel_mapping = Property(np.ndarray, default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.vel_mapping is None:
            self.pos_mapping = self.measurement_model.mapping[::2]
            self.vel_mapping = self.measurement_model.mapping[1::2]
        else:
            self.pos_mapping = self.measurement_model.mapping

        # Create tree
        tree_property = rtree.index.Property(
            type=rtree.index.RT_TPRTree,
            tpr_horizon=self.horizon_time.total_seconds(),
            dimension=len(self.pos_mapping))
        self._tree = rtree.index.RtreeContainer(properties=tree_property)
        self._coords = dict()

    def _track_tree_coordinates(self, track):
        state_vector =track.state_vector[self.pos_mapping, :]

        state_delta = 10 * np.sqrt(
            np.diag(track.covar)[self.pos_mapping].reshape(-1, 1))
        meas_delta = 10 * np.sqrt(np.diag(self.measurement_model.covar()).reshape(-1, 1))
        vel_vector = track.state_vector[self.vel_mapping, :]
        vel_delta = 10 * np.sqrt(
            np.diag(track.covar)[self.vel_mapping].reshape(-1, 1))

        min_pos = (state_vector - state_delta - meas_delta).ravel()
        max_pos = (state_vector + state_delta + meas_delta).ravel()
        min_vel = (vel_vector - vel_delta).ravel()
        max_vel = (vel_vector + vel_delta).ravel()

        return ((*min_pos, *max_pos), (*min_vel, *max_vel),
                track.timestamp.timestamp())

    def generate_hypotheses(self, tracks, detections, time, **kwargs):
        # No need for tree here.
        if not tracks:
            return dict()

        c_time = None
        # Tree management
        print("Tree management")
        for track in sorted(tracks.union(self._tree),
                            key=attrgetter('timestamp')):

            if c_time is None:
                c_time = track.timestamp

            if track not in self._tree:
                self._coords[track] = self._track_tree_coordinates(track)
                self._tree.insert(track, self._coords[track])

            elif track not in tracks:
                c_time = track.timestamp
                coords = self._coords[track][:-1] \
                         + ((self._coords[track][-1] - 1e-3, c_time.timestamp()),)
                self._tree.delete(track, coords)
                del self._coords[track]
            elif isinstance(track.state, Update):
                c_time = track.timestamp
                coords = self._coords[track][:-1] \
                    + ((self._coords[track][-1]-1e-3, c_time.timestamp()),)
                self._tree.delete(track, coords)
                self._coords[track] = self._track_tree_coordinates(track)
                self._tree.insert(track, self._coords[track])

        # Detection gating
        print("Detection gating management")
        track_detections = defaultdict(set)
        for detection in sorted(detections, key=attrgetter('timestamp')):
            if detection.measurement_model is not None:
                model = detection.measurement_model
            else:
                model = self.measurement_model

            if isinstance(model, LinearModel):
                model_matrix = model.matrix(**kwargs)
                inv_model_matrix = sp.linalg.pinv(model_matrix)
                state_meas = (inv_model_matrix
                              @ detection.state_vector)[self.pos_mapping, :]
            else:
                state_meas = model.inverse_function(
                    detection.state_vector, **kwargs)[self.pos_mapping, :]

            det_time = detection.timestamp.timestamp()
            intersected_tracks = self._tree.intersection((
                (*state_meas.ravel(), *state_meas.ravel()),
                (0, 0)*len(self.pos_mapping),
                (det_time, det_time + 1e-3)))
            for track in intersected_tracks:
                track_detections[track].add(detection)

        print("Hypothesising")

        misdet = MissedDetection(timestamp=time)
        mult = MultipleHypothesis()
        return {track: self.hypothesiser.hypothesise(
            track, track_detections[track], time, missed_detection=misdet, mult=mult, **kwargs)
            for track in tracks}


class LongLatTPRTreeMixIn(TPRTreeMixIn):

    offset = Property(np.array,
                      doc='2D vector specifying the offset. Default is ``np.array([[Longitude(0)], [Latitude(0)]])``.',
                      default=np.array([[Longitude(0)], [Latitude(0)]]))
    window_delta = Property(float,
                            doc='Confidence window added to detection windows when checking for +-180 overlaps. '
                                'Default is 0.2rad',
                            default=0.2)

    def _track_tree_coordinates(self, track):
        state_vector = np.array(track.state_vector[self.pos_mapping, :], dtype=np.float32) \
                       + np.array(self.offset, dtype=np.float32)

        state_delta = 3 * np.sqrt(
            np.diag(track.covar)[self.pos_mapping].reshape(-1, 1))
        meas_delta = 3 * np.sqrt(np.diag(self.measurement_model.covar()).reshape(-1, 1))
        vel_vector = track.state_vector[self.vel_mapping, :]
        vel_delta = 3 * np.sqrt(
            np.diag(track.covar)[self.vel_mapping].reshape(-1, 1))

        min_pos = (state_vector - state_delta - meas_delta).ravel()
        max_pos = (state_vector + state_delta + meas_delta).ravel()
        min_vel = (vel_vector - vel_delta).ravel()
        max_vel = (vel_vector + vel_delta).ravel()

        return ((*min_pos, *max_pos), (*min_vel, *max_vel),
                track.timestamp.timestamp())

    def _get_min_max(self):
        minlons = [self._coords[key][0][0] for key in self._coords]
        maxlons = np.array([self._coords[key][0][2] for key in self._coords])
        minlon = np.amin(minlons)
        maxlon = np.amax(maxlons)
        return minlon, maxlon

    def generate_hypotheses(self, tracks, detections, time, **kwargs):
        # No need for tree here.
        if not tracks:
            return dict()

        c_time = None
        # Tree management
        print("Tree management")
        for track in sorted(tracks.union(self._tree),
                            key=attrgetter('timestamp')):

            if c_time is None:
                c_time = track.timestamp

            if track not in self._tree:
                self._coords[track] = self._track_tree_coordinates(track)
                self._tree.insert(track, self._coords[track])
                c_time = track.timestamp
            elif track not in tracks:
                c_time = track.timestamp
                coords = self._coords[track][:-1] \
                    + ((self._coords[track][-1], c_time.timestamp()),)
                self._tree.delete(track, coords)
                del self._coords[track]
            elif isinstance(track.state, Update):
                c_time = track.timestamp
                if self._coords[track][-1] - c_time.timestamp() >= 0:
                    coords = self._coords[track][:-1] \
                        + ((self._coords[track][-1]-1e-3, c_time.timestamp()),)
                else:
                    coords = self._coords[track][:-1] \
                        + ((self._coords[track][-1], c_time.timestamp()),)
                self._tree.delete(track, coords)
                self._coords[track] = self._track_tree_coordinates(track)
                self._tree.insert(track, self._coords[track])

        # Detection gating
        print("Detection gating management")
        track_detections = defaultdict(set)
        minlon, maxlon = self._get_min_max()
        for detection in sorted(detections, key=attrgetter('timestamp')):
            if detection.measurement_model is not None:
                model = detection.measurement_model
            else:
                model = self.measurement_model

            if isinstance(model, LinearModel):
                model_matrix = model.matrix(**kwargs)
                inv_model_matrix = sp.linalg.pinv(model_matrix)
                state_meas = (inv_model_matrix
                              @ detection.state_vector)[self.pos_mapping, :]+self.offset
            else:
                state_meas = model.inverse_function(
                    detection.state_vector, **kwargs)[self.pos_mapping, :]+self.offset

            det_time = detection.timestamp.timestamp()
            intersected_tracks = self._tree.intersection((
                (*state_meas.ravel(), *state_meas.ravel()),
                (0, 0)*len(self.pos_mapping),
                (det_time, det_time + 1e-3)))
            for track in intersected_tracks:
                track_detections[track].add(detection)

            # Check for overlapps
            # Case 1: Track boxes overlap -180 lon line and measurement falls within this region
            if minlon < -np.pi and float(state_meas[0]) - 2*np.pi + self.window_delta > minlon:
                state_meas[0] = float(state_meas[0]) - 2*np.pi
                c_intersected_tracks = self._tree.intersection((
                    (*state_meas.ravel(), *state_meas.ravel()),
                    (0, 0) * len(self.pos_mapping),
                    (det_time, det_time + 1e-3)))
                for track in c_intersected_tracks:
                    track_detections[track].add(detection)
            # Case 2: Track boxes overlap +180 lon line and measurement falls within this region
            elif maxlon > np.pi and float(state_meas[0]) + 2*np.pi - self.window_delta < maxlon:
                state_meas[0] = float(state_meas[0]) + 2 * np.pi
                c_intersected_tracks = self._tree.intersection((
                    (*state_meas.ravel(), *state_meas.ravel()),
                    (0, 0) * len(self.pos_mapping),
                    (det_time, det_time + 1e-3)))
                for track in c_intersected_tracks:
                    track_detections[track].add(detection)

        print("Hypothesising")

        misdet = MissedDetection(timestamp=time)
        mult = MultipleHypothesis()
        return {track: self.hypothesiser.hypothesise(
            track, track_detections[track], time, missed_detection=misdet, mult=mult, **kwargs)
            for track in tracks}
