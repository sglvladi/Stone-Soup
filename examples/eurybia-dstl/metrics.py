from copy import deepcopy

from stonesoup.metricgenerator.manager import SimpleManager
from stonesoup.metricgenerator.ospametric import OSPAMetric, GOSPAMetric
from stonesoup.measures import Euclidean
from stonesoup.metricgenerator.tracktotruthmetrics import SIAPMetrics
from stonesoup.dataassociator.tracktotrack import TrackToTruth


def prepare_tracks(dataset, gnd, tracks):
    if dataset in ['Sim1', 'Real']:
        associator = TrackToTruth(association_threshold=1000, measure=Euclidean([0, 2]))
        assoc = associator.associate_tracks(tracks, gnd)
        tracks = {assoc.associations.pop().objects[0]}
        return tracks
    else:
        return tracks

def prepare_tracks_fuse(dataset, fuse_timestamps, all_gnd, fused_tracks, local_tracks):
    if dataset in ['Sim1', 'Real']:
        associator_fuse = TrackToTruth(association_threshold=1000, measure=Euclidean([4, 6], [0, 2]))
        fuse_gnd = deepcopy(all_gnd)
        for gnd in fuse_gnd:
            gnd.states = [state for state in gnd.states if state.timestamp in fuse_timestamps]
        assoc = associator_fuse.associate_tracks(fused_tracks, fuse_gnd)
        fused_tracks = {assoc.associations.pop().objects[0]}
        associator_local = TrackToTruth(association_threshold=1000, measure=Euclidean([0, 2]))
        filtered_tracklets = dict()
        for key, tracks in local_tracks.items():
            tracks_tmp = deepcopy(tracks)
            for track in tracks_tmp:
                track.states = [state for state in track.states if state.timestamp in fuse_timestamps]
            assoc = associator_local.associate_tracks(tracks_tmp, fuse_gnd)
            filtered_tracklets[key] = {assoc.associations.pop().objects[0]}
        return fused_tracks, filtered_tracklets
    else:
        return fused_tracks, local_tracks


def gen_metric(metric_type, gnd, tracks):
    if metric_type in ['OSPA', 'GOSPA']:
        metric_cls = OSPAMetric if metric_type == 'OSPA' else GOSPAMetric
        metric_gen = metric_cls(c=1000, p=1, measure=Euclidean([0, 2]))
        metric = metric_gen.compute_over_time(*metric_gen.extract_states(tracks, True),
                                              *metric_gen.extract_states(gnd, True))

    elif metric_type == 'SIAP':
        metric_gen = SIAPMetrics(position_measure=Euclidean((0, 2)),
                                 velocity_measure=Euclidean((1, 3)))

        # The SIAP Metrics requires a way to associate tracks to truth, so we'll use a Track to Truth
        # associator, which uses Euclidean distance measure by default.
        associator = TrackToTruth(association_threshold=1000, measure=Euclidean([0, 2]))
        metric_manager = SimpleManager([metric_gen], associator=associator)
        metric_manager.add_data(
            gnd, tracks, overwrite=True,  # Don't overwrite, instead add above as additional data
        )
        metric = metric_manager.generate_metrics()
    else:
        raise ValueError('Invalid metric type')
    return metric


def gen_metric_fuse(metric_type, fuse_timestamps, all_gnd, fused_tracks, local_tracks):
    """Helper function to generate metrics for both fusion and local trackers.

    Parameters
    ----------
    metric_type : str
        The type of metric to generate. Either 'OSPA', 'GOSPA', or 'SIAP'.
    fuse_timestamps : list
        List of timestamps to generate metrics for.
    all_gnd : list
        List of ground truths.
    fused_tracks : list
        List of fused tracks.
    local_tracks : dict
        Dictionary of local tracks. The keys are the local tracker indices, and the values are the
        list of tracks.
    """

    metrics = dict()

    fuse_gnd = deepcopy(all_gnd)
    for gnd in fuse_gnd:
        gnd.states = [state for state in gnd.states if state.timestamp in fuse_timestamps]

    if metric_type in ['OSPA', 'GOSPA']:
        metric_cls = OSPAMetric if metric_type == 'OSPA' else GOSPAMetric
        metric_gen_fuse = metric_cls(c=1000, p=1, measure=Euclidean([4, 6], [0, 2]))
        metric_gen_local = metric_cls(c=1000, p=1, measure=Euclidean([0, 2]))
        metrics['fuse'] = metric_gen_fuse.compute_over_time(*metric_gen_fuse.extract_states(fused_tracks, True),
                                                            *metric_gen_fuse.extract_states(fuse_gnd, True))

        for i, tracks in local_tracks.items():
            tracks_tmp = deepcopy(tracks)
            for track in tracks_tmp:
                track.states = [state for state in track.states if state.timestamp in fuse_timestamps]
            metrics[f'local_{i}'] = metric_gen_local.compute_over_time(
                *metric_gen_local.extract_states(tracks_tmp, True),
                *metric_gen_local.extract_states(fuse_gnd, True))
    elif metric_type == 'SIAP':
        metric_gen_fuse = SIAPMetrics(position_measure=Euclidean([0, 2], [4, 6]),
                                      velocity_measure=Euclidean([1, 3], [5, 7]))
        metric_gen_local = SIAPMetrics(position_measure=Euclidean((0, 2)),
                                       velocity_measure=Euclidean((1, 3)))

        # The SIAP Metrics requires a way to associate tracks to truth, so we'll use a Track to Truth
        # associator, which uses Euclidean distance measure by default.
        associator_fuse = TrackToTruth(association_threshold=1000, measure=Euclidean([4, 6], [0, 2]))

        metric_manager_fuse = SimpleManager([metric_gen_fuse],
                                            associator=associator_fuse)


        metric_manager_fuse.add_data(
            fuse_gnd, fused_tracks, overwrite=True,  # Don't overwrite, instead add above as additional data
        )
        metrics['fuse'] = metric_manager_fuse.generate_metrics()

        for i, tracks in local_tracks.items():
            associator_local = TrackToTruth(association_threshold=1000, measure=Euclidean([0, 2]))
            metric_manager_local = SimpleManager([metric_gen_local],
                                                 associator=associator_local)
            tracks_tmp = deepcopy(tracks)
            for track in tracks_tmp:
                track.states = [state for state in track.states if state.timestamp in fuse_timestamps]
            metric_manager_local.add_data(
                fuse_gnd, tracks_tmp, overwrite=True,  # Don't overwrite, instead add above as additional data
            )
            metrics[f'local_{i}'] = metric_manager_local.generate_metrics()
    else:
        raise ValueError('Invalid metric type')

    return metrics
