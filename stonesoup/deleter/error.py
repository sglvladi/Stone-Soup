# -*- coding: utf-8 -*-
"""Contains collection of error based deleters"""
from typing import Sequence

import numpy as np

from ..base import Property
from .base import Deleter
from ..functions import gauss2sigma, unscented_transform
from ..types.update import Update


class CovarianceBasedDeleter(Deleter):
    """ Track deleter based on covariance matrix size.

    Deletes tracks whose state covariance matrix (more specifically its trace)
    exceeds a given threshold.
    """

    covar_trace_thresh: float = Property(doc="Covariance matrix trace threshold")
    mapping: Sequence[int] = Property(default=None,
                                      doc="Track state vector indices whose corresponding "
                                          "covariances' sum is to be considered. Defaults to"
                                          "None, whereby the entire track covariance trace is "
                                          "considered.")

    def check_for_deletion(self, track, **kwargs):
        """Check if a given track should be deleted

        A track is flagged for deletion if the trace of its state covariance
        matrix is higher than :py:attr:`~covar_trace_thresh`.

        Parameters
        ----------
        track : Track
            A track object to be checked for deletion.

        Returns
        -------
        bool
            `True` if track should be deleted, `False` otherwise.
        """

        diagonals = np.diag(track.state.covar)
        if self.mapping:
            track_covar_trace = np.sum(diagonals[self.mapping])
        else:
            track_covar_trace = np.sum(diagonals)

        if track_covar_trace > self.covar_trace_thresh:
            return True
        return False


class MeasurementCovarianceBasedDeleter(Deleter):
    """ Track deleter based on covariance matrix in measurement space.

    Deletes tracks whose measurement covariance matrix values exceed a given threshold.
    """

    diag_covar_thresh: Sequence[float] = Property(doc="Covariance matrix diagonal threshold")
    use_ut: bool = Property(doc="Whether to use unscented transform to calculate measurement "
                                "matrix. Defaults to False, in which case jacobian will be used",
                            default=False)
    mapping: Sequence[int] = Property(default=None,
                                      doc="Track state vector indices whose corresponding "
                                          "covariances' sum is to be considered. Defaults to"
                                          "None, whereby the entire track covariance trace is "
                                          "considered.")

    def check_for_deletion(self, track, **kwargs):
        """Check if a given track should be deleted

        A track is flagged for deletion if any element of its measurement covariance
        matrix is higher than :py:attr:`~diag_covar_thresh`.

        Parameters
        ----------
        track : Track
            A track object to be checked for deletion.

        Returns
        -------
        bool
            `True` if track should be deleted, `False` otherwise.
        """

        covar = self._get_covar(track)
        diagonals = np.diag(covar)
        # Compare against thresholds
        for val, thresh in zip(diagonals, self.diag_covar_thresh):
            if val > thresh:
                return True
        return False

    def _get_covar(self, track):
        # Get the measurement model used in the last update
        measurement_model = self._find_last_meas_model(track)
        if self.use_ut:
            sigma_points, mean_weights, covar_weights = \
                gauss2sigma(track.state, 0.5, 2, 0)
            _, covar, _, _, _, _ = \
                unscented_transform(sigma_points, mean_weights, covar_weights,
                                    measurement_model.function)
        else:
            # Find its jacobian
            h = measurement_model.jacobian(track.state)
            # Transform covariance
            covar = h @ track.state.covar @ h.T

        return covar

    def _find_last_meas_model(self, track):
        # Get the last update
        last_update = next((state for state in reversed(track) if isinstance(state, Update)), None)
        hypothesis = last_update.hypothesis
        try:
            # Single hypothesis objects
            return hypothesis.measurement.measurement_model
        except AttributeError:
            # Multi hypothesis
            return next((hyp.measurement.measurement_model for hyp in hypothesis if hyp), None)
