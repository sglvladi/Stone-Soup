import cv2
import matplotlib
matplotlib.use( 'tkagg' )
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib import pyplot as plt

from utils import visualization_utils as vis_util

from stonesoup.types.update import Update

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
        """
        Plots an `nstd` sigma error ellipse based on the specified covariance
        matrix (`cov`). Additional keyword arguments are passed on to the
        ellipse patch artist.
        Parameters
        ----------
            cov : The 2x2 covariance matrix to base the ellipse on
            pos : The location of the center of the ellipse. Expects a 2-element
                sequence of [x0, y0].
            nstd : The radius of the ellipse in numbers of standard deviations.
                Defaults to 2 standard deviations.
            ax : The axis that the ellipse will be plotted on. Defaults to the
                current axis.
            Additional keyword arguments are pass on to the ellipse patch.
        Returns
        -------
            A matplotlib ellipse artist
        """

        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:, order]

        if ax is None:
            ax = plt.gca()

        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        # Width and height are "full" widths, not radius
        width, height = 2 * nstd * np.sqrt(vals)
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

        ax.add_artist(ellip)
        return ellip

def track_to_bbox(track):
    box = tuple(track.state.state_vector[[0, 2, 4, 6]].reshape((4)))
    xmin, ymin, w, h = box
    xmax, ymax = (xmin + w, ymin + h)
    return (ymin, xmin, ymax, xmax)

def tracks_to_bbox_areas(tracks):
    boxes = [track_to_bbox(track) for track in tracks]
    areas = [abs(box[2]-box[0]) * abs(box[3]-box[1]) for box in boxes]
    return boxes, areas

def draw_detections(frame, detections, category_index, score_threshold):
    if len(detections):
        boxes = np.array([detection.metadata["raw_box"]
                          for detection in detections])
        classes = np.array([detection.metadata["class"]
                            for detection in detections])
        scores = np.array([detection.metadata["score"]
                           for detection in detections])
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            boxes,
            classes,
            scores,
            category_index,
            min_score_thresh=score_threshold,
            use_normalized_coordinates=True,
            line_thickness=2,
        )
    return frame

def draw_track(frame, track, color='red'):
    (ymin, xmin, ymax, xmax) = track_to_bbox(track)
    vis_util.draw_bounding_box_on_image_array(frame,
                                              ymin,
                                              xmin,
                                              ymax,
                                              xmax,
                                              color=color,
                                              thickness=2)
    return frame

def draw_tracks(frame, tracks, converged, c_track=None):
    track_list = list([track for track in tracks
                       if (len([state for state in track.states
                                if isinstance(state, Update)]) > 10)])
    ids = [track.id for track in track_list]
    f_width, f_height, _ = frame.shape
    if converged:
        vis_util.draw_bounding_box_on_image_array(frame,
                                                  0,0,1,1,
                                                  color='green',
                                                  thickness=20)
    for track in track_list:
        if c_track is not None and track.id == c_track.id:
            color = 'red'
            #print(c_track.id)
        else:
            color = 'blue'

        frame = draw_track(frame, track, color)
        # frame_t = cv2.putText(frame, ids[i], (0, 0),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (255, 255, 255), 3, cv2.LINE_AA)
        # frame = copy(frame_t)
    return frame

def draw_horizon(image, horizon):
    return cv2.line(image, horizon[0], horizon[1], (0, 0, 255), 2, cv2.LINE_4)