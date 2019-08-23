import cv2
import numpy as np
from copy import copy

from stonesoup.types.state import StateVector
from stonesoup.types.detection import Detection

def affineCorrection(frame, frame_old, tracks):
    if (len(frame_old)):
        frame_old_gray = cv2.cvtColor(frame_old, cv2.COLOR_BGR2GRAY)

        # Detect feature points in previous frame
        prev_pts = cv2.goodFeaturesToTrack(frame_old_gray,
                                           maxCorners=200,
                                           qualityLevel=0.01,
                                           minDistance=30,
                                           blockSize=3)

        # Read next frame
        if not len(frame):
            return tracks

        # Convert to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(frame_old_gray, frame_gray, prev_pts, None)

        # Sanity check
        assert prev_pts.shape == curr_pts.shape

        # Filter only valid points
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        # Find transformation matrix
        m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)  # will only work with OpenCV-3 or less

        # Extract traslation
        height, width, _ = frame.shape
        affine_matrix = m[0]
        affine_matrix[0, 2] = affine_matrix[0, 2] / width
        affine_matrix[1, 2] = affine_matrix[1, 2] / height
        dx = affine_matrix[0, 2]
        dy = affine_matrix[1, 2]

        # Extract rotation angle
        da = np.arctan2(affine_matrix[1, 0], affine_matrix[0, 0])

        # Store transformation
        transform = [dx, dy, da]

        for track in tracks:
            xmin = track.state.state_vector[0, 0]
            ymin = track.state.state_vector[2, 0]
            xmax = xmin + track.state.state_vector[4, 0]
            ymax = ymin + track.state.state_vector[6, 0]
            rect_pts = np.array([[[xmin, ymin]],
                                 [[xmax, ymin]],
                                 [[xmin, ymax]],
                                 [[xmax, ymax]]], dtype=np.float32)
            affine_warp = np.vstack((affine_matrix, np.array([[0, 0, 1]])))
            rect_pts = cv2.perspectiveTransform(rect_pts, affine_warp)
            xA = rect_pts[0, 0, 0]
            yA = rect_pts[0, 0, 1]
            xB = rect_pts[3, 0, 0]
            yB = rect_pts[3, 0, 1]

            track.state.state_vector[0, 0] = xA
            track.state.state_vector[2, 0] = yA
            track.state.state_vector[4, 0] = xB - xA
            track.state.state_vector[6, 0] = yB - yA

    return tracks

def get_overlap_ratio(a, b, epsilon=1e-5):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    return area_overlap/area_a

def gen_clusters(v_matrix):
    """

    Parameters
    ----------
    v_matrix

    Returns
    -------

    """
    # Initiate parameters
    num_rows, num_cols = np.shape(v_matrix);  # Number of tracks

    # Form clusters of tracks sharing measurements
    unassoc_rows  = set()
    clusters = []

    # Iterate over all row
    for row_ind in range(num_rows):

        # Extract valid col indices
        v_col_inds = set(np.argwhere(v_matrix[row_ind, :] > 0)[:,0])

        # If there  exist valid cols
        if len(v_col_inds):

            # Check if matched measurements are members of any clusters
            num_clusters = len(clusters)
            v_matched_cluster_ind = np.zeros((num_clusters,))
            for cluster_ind in range(num_clusters):
                if any([v_ind in clusters[cluster_ind]["col_inds"]
                        for v_ind in v_col_inds ]):
                    v_matched_cluster_ind[cluster_ind] = 1

            num_matched_clusters = int(sum(v_matched_cluster_ind))

            # If only matched with a single cluster, join.
            if num_matched_clusters == 1:
                matched_cluster_ind = np.argwhere(v_matched_cluster_ind > 0)[0][0]
                clusters[matched_cluster_ind]["row_inds"] |= set([row_ind])
                clusters[matched_cluster_ind]["col_inds"] |= v_col_inds
            elif num_matched_clusters == 0:
                new_cluster = {"row_inds": set([row_ind]),
                              "col_inds": v_col_inds}
                clusters.append(new_cluster)
            else:
                matched_cluster_inds = np.argwhere(v_matched_cluster_ind > 0)[0]
                # Start from last cluster, joining each one with the previous
                # and removing the former.
                for matched_cluster_ind in range(num_matched_clusters - 2, 0, -1):
                    clusters[matched_cluster_inds[
                        matched_cluster_ind]]["row_inds"] |= clusters[
                        matched_cluster_inds[matched_cluster_ind + 1]]["row_inds"]
                    clusters[matched_cluster_inds[
                        matched_cluster_ind]]["col_inds"] |= clusters[
                        matched_cluster_inds[matched_cluster_ind + 1]]["col_inds"]
                    clusters[matched_cluster_inds[matched_cluster_ind + 1]] = []

                # Finally, join with associated track.
                clusters[matched_cluster_inds[0]]["row_inds"] |= set([row_ind])
                clusters[matched_cluster_inds[0]]["col_inds"] |= v_col_inds
        else:
            new_cluster = {"row_inds": set([row_ind]),
                          "col_inds": set([])}
            clusters.append(new_cluster)
            # clusters(end + 1) = ClusterObj;
            unassoc_rows.add(row_ind)

    return clusters, unassoc_rows

def detection_to_bbox(state_vector):
    x_min, y_min, width, height = (state_vector[0, 0],
                                   state_vector[1, 0],
                                   state_vector[2, 0],
                                   state_vector[3, 0])
    return StateVector([[x_min],
                        [y_min],
                        [x_min + width],
                        [y_min + height]])

def merge_detections(detections, threshold = 0.5):
    num_detections = len(detections)
    states = []
    detections_list = list(detections)
    for detection in detections_list:
        state_vector = detection.state_vector
        states.append(detection_to_bbox(state_vector))

    v_matrix = np.zeros((num_detections, num_detections))
    ious = np.zeros((num_detections, num_detections))
    for i, detection_i in enumerate(detections_list):
        for j, detection_j in enumerate(detections_list):
            ious[i, j] = get_overlap_ratio(states[i], states[j])
            if (ious[i, j] > threshold):
                v_matrix[i, j] = 1

    new_detections = set()
    clusters, _ = gen_clusters(v_matrix)
    for cluster in clusters:
        det_inds = list(cluster["row_inds"])

        detection = detections_list[det_inds[0]]
        state_vector = detection.state_vector
        metadata = detection.metadata
        timestamp = detection.timestamp
        for i, ind in enumerate(det_inds):
            detection = detections_list[ind]
            if i == 0:
                continue
            else:
                bbox_1 = detection_to_bbox(state_vector)
                bbox_2 = detection_to_bbox(detection.state_vector)
                if bbox_2[0][0]<bbox_1[0][0]:
                    state_vector[0][0] = bbox_2[0][0]
                if bbox_2[1][0] < bbox_1[1][0]:
                    state_vector[1][0] = bbox_2[1][0]
                if bbox_2[2][0]>bbox_1[2][0]:
                    state_vector[2][0] = bbox_2[2][0]-state_vector[0][0]
                if bbox_2[3][0]>bbox_1[3][0]:
                    state_vector[3][0] = bbox_2[3][0]-state_vector[3][0]
                #metadata["score"] += detection.metadata["score"]
        new_detection = Detection(state_vector, metadata=metadata,
                                  timestamp=timestamp)
        new_detections.add(copy(new_detection))

    return new_detections

def merge_tracks(tracks):
    num_tracks = len(tracks)
    track_states = []
    areas = []
    tracks = list(tracks)
    for track in tracks:
        state_vec = track.state.state_vector
        covar = track.state.covar[[0,2], :][:, [0,2]]
        x_min, y_min, width, height = (state_vec[0,0],
                                       state_vec[2,0],
                                       state_vec[4,0],
                                       state_vec[6,0])
        track_states.append(StateVector([[x_min],
                                         [y_min],
                                         [x_min+width],
                                         [y_min+height]]))
        areas.append(state_vec[4]*state_vec[6,0])

    new_tracks = set()
    v_matrix = np.zeros((num_tracks,num_tracks))
    ious = np.zeros((num_tracks, num_tracks))
    for i, track_i in enumerate(tracks):
        for j, track_j in enumerate(tracks):
            ious[i,j] = get_overlap_ratio(track_states[i], track_states[j])
            if(ious[i,j] > 0.5):
                v_matrix[i,j] = 1


    clusters, _ = gen_clusters(v_matrix)
    for cluster in clusters:
        track_inds = list(cluster["row_inds"])
        ind = np.argmax([areas[ind] for ind in track_inds ])
        ind = track_inds[ind]
        new_tracks.add(tracks[ind])

    if len(new_tracks)==0:
        a=0
    return new_tracks

def delete_tracks(tracks, threshold = 0.1):
    deleted_tracks = set([])
    for track in tracks:
        if track.score<threshold:
            deleted_tracks.add(track)

    return deleted_tracks