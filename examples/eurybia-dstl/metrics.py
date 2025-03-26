from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from stonesoup.metricgenerator.ospametric import OSPAMetric, GOSPAMetric
from stonesoup.measures import Euclidean

def gen_metric_fuse(metric_type, fuse_timestamps, all_gnd, fused_tracks, local_tracks):
    metrics = dict()

    if metric_type == 'OSPA':
        metric_gen_fuse = OSPAMetric(c=1000, p=1, measure=Euclidean([4, 6], [0, 2]))
        metric_gen_local = OSPAMetric(c=1000, p=1, measure=Euclidean([0, 2]))
    elif metric_type == 'GOSPA':
        metric_gen_fuse = GOSPAMetric(c=1000, p=1, measure=Euclidean([4, 6], [0, 2]))
        metric_gen_local = GOSPAMetric(c=1000, p=1, measure=Euclidean([0, 2]))
    else:
        raise Exception()

    fuse_gnd = deepcopy(all_gnd)
    for gnd in fuse_gnd:
        gnd.states = [state for state in gnd.states if state.timestamp in fuse_timestamps]
    metrics['fuse'] = metric_gen_fuse.compute_over_time(*metric_gen_fuse.extract_states(fused_tracks, True),
                                                        *metric_gen_fuse.extract_states(fuse_gnd, True))

    for i, tracks in local_tracks.items():
        metrics[f'local_{i}'] = metric_gen_local.compute_over_time(*metric_gen_local.extract_states(tracks, True),
                                                                   *metric_gen_local.extract_states(all_gnd, True))
    return metrics
