# -*- coding: utf-8 -*-
import datetime
import operator

from ..filter import MetadataReducer, MetadataValueFilter


def test_metadata_reducer(detector):
    feeder = MetadataReducer(detector, metadata_field="colour")

    multi_none = False
    for time, detections in feeder.detections_gen():
        all_colours = [detection.metadata.get('colour')
                       for detection in detections]
        if not multi_none:
            multi_none = len(
                [colour for colour in all_colours if colour is None]) > 1

        colours = [colour for colour in all_colours if colour is not None]
        assert len(colours) == len(set(colours))

        assert "red" in colours
        assert "blue" in colours
        if time < datetime.datetime(2019, 4, 1, 14, 0, 2):
            assert "green" in colours
        else:
            assert "green" not in colours

        assert all(time == detection.timestamp for detection in detections)

    assert multi_none


def test_metadata_value_filter(detector):
    feeder = MetadataValueFilter(detector,
                                 metadata_field="score",
                                 operator=operator.le,
                                 reference_value=0.1)


    nones = False
    for time, detections in feeder.detections_gen():
        all_scores = [detection.metadata.get('score')
                       for detection in detections]
        print(all_scores)
        nones = nones | (len([score for score in all_scores if score is None]) > 1)

        scores = [score for score in all_scores if score is not None]
        assert len(scores) == len(set(scores))

        assert 0 not in scores
        if time < datetime.datetime(2019, 4, 1, 14, 0, 2):
            assert all([score<=0.5 for score in scores])

        assert all(time == detection.timestamp for detection in detections)

    assert not nones