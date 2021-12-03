# -*- coding: utf-8 -*-
from typing import Sequence

import numpy as np

from ..base import Property
from ..reader import GroundTruthReader
from .base import DetectionSimulator
from ..buffered_generator import BufferedGenerator
from ..platform import Platform
from ..types.numeric import Probability
from ..types.detection import Clutter
from ..types.groundtruth import GroundTruthState


class PlatformDetectionSimulator(DetectionSimulator):
    """A simple platform detection simulator.

    Processes ground truth data and generates :class:`~.Detection` data
    according to a list of platforms by calling each sensor in these platforms.

    """
    groundtruth: GroundTruthReader = Property(
        doc='Source of ground truth tracks used to generate detections for.')
    platforms: Sequence[Platform] = Property(
        doc='List of platforms in :class:`~.Platform` to generate sensor detections from.')

    @BufferedGenerator.generator_method
    def detections_gen(self):
        for time, truths in self.groundtruth:
            for platform in self.platforms:
                platform.move(time)
            for platform in self.platforms:
                for sensor in platform.sensors:
                    truths_to_be_measured = truths.union(self.platforms) - {platform}
                    detections = sensor.measure(truths_to_be_measured)
                    yield time, detections


class PlatformTargetDetectionSimulator(PlatformDetectionSimulator):
    """A simple platform detection simulator.

    Processes ground truth data and generates :class:`~.Detection` data
    according to a list of platforms by calling each sensor in these platforms.

    """
    targets: Sequence[Platform] = Property(
        doc='List of target platforms to be detected'
    )
    meas_range: np.ndarray = Property(default=None)
    detection_probability: Probability = Property(default=0.9)
    clutter_rate: float = Property(default=2.0)

    @BufferedGenerator.generator_method
    def detections_gen(self):
        for time, truths in self.groundtruth:
            for platform in self.platforms:
                platform.move(time)
            for platform in self.targets:
                platform.move(time)
            for platform in self.platforms:
                for sensor in platform.sensors:
                    # True detections
                    truths_to_be_measured = truths.union(self.targets)
                    detections = sensor.measure(truths_to_be_measured)
                    # Clutter
                    gnd_clutter = []
                    for _ in range(np.random.poisson(self.clutter_rate)):
                        sv = np.random.rand(len(self.meas_range), 1) * np.diff(self.meas_range) + self.meas_range[:, :1]
                        sv.resize((4,1), refcheck=False)
                        sv = sv[[0, 2, 1, 3],:]
                        gnd_clutter.append(GroundTruthState(sv, time))
                    detections |= sensor.measure(gnd_clutter)
                    yield time, detections
