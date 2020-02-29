# -*- coding: utf-8 -*-
import uuid

from ..base import Property
from .multihypothesis import MultipleHypothesis
from .state import State, StateMutableSequence
from .update import Update


class Track(StateMutableSequence):
    """Track type

    A :class:`~.StateMutableSequence` representing a track.
    """

    states = Property(
        [State],
        default=None,
        doc="The initial states of the track. Default `None` which initialises"
            "with empty list.")

    id = Property(
        str,
        default=None,
        doc="The unique track ID")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_update = None
        for state in reversed(self.states):
            if isinstance(state, Update):
                self._last_update = state
                break
        self._metadata = {}
        for state in self.states:
            self._update_metadata_from_state(state)

        if self.id is None:
            self.id = str(uuid.uuid4())

    def __setitem__(self, index, value):
        # Update metadata
        self._update_metadata_from_state(value)
        if isinstance(value, Update):
            self._last_update = value
        return super().__setitem__(index, value)

    def insert(self, index, value):
        # Update metadata
        self._update_metadata_from_state(value)
        if isinstance(value, Update):
            self._last_update = value
        return self.states.insert(index, value)

    @property
    def last_update(self):
        return self._last_update

    @property
    def metadata(self):
        """Returns metadata associated with a track.

        Parameters
        ----------
        None

        Returns
        -------
        : :class:`dict` of variable size
            All metadata associate with this track.
        """

        return self._metadata

    @metadata.setter
    def metadata(self, x):
        self._metadata.update(x)

    def _update_metadata_from_state(self, state):
        """ Extract and update track metadata, given a state

        Parameters
        ----------
        state: State
            A state object from which to extract metadata. Metadata can only
            be extracted from Update (or subclassed) objects. Calling this
            method with a non-Update (subclass) object will NOT return an
            error, but will have no effect on the metadata.

        """

        if isinstance(state, Update):
            if isinstance(state.hypothesis, MultipleHypothesis):
                # Sort and iterate through multiple hypotheses such that most
                # likely hypothesis comes last. This ensures that metadata
                # from all hypotheses are retained, but more likely
                # hypotheses will over-write the metadata set by less likely
                # ones.
                for hypothesis in sorted(state.hypothesis, reverse=True):
                    if hypothesis \
                            and hypothesis.measurement.metadata is not None:
                        self._metadata.update(hypothesis.measurement.metadata)
            else:
                hypothesis = state.hypothesis
                if hypothesis and hypothesis.measurement.metadata is not None:
                    self._metadata.update(hypothesis.measurement.metadata)
