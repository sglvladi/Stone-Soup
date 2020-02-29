from datetime import timedelta

from ..base import Property
from ..types.update import Update
from .base import Deleter


class ELINTDeleter(Deleter):
    """Update Time based deleter

    Identify tracks for deletion which an :class:`~.Update` has occurred in
    last :attr:`time_steps_since_update`.
    """

    deleteThreshold = Property(
        float, doc="")

    def check_for_deletion(self, track, **kwargs):
        """Delete track without update with measurements within time steps

        Parameters
        ----------
        track : Track
            Track to check for deletion

        Returns
        -------
        bool
            `True` if track has an :class:`~.Update` with measurements within
            time steps; `False` otherwise.
        """
        if track.metadata["existence"]["value"]<self.deleteThreshold:
            return True
        return False
