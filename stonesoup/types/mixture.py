from collections.abc import Sized, Iterable, Container

from .base import Type
from ..base import Property
from .state import TaggedWeightedGaussianState, WeightedGaussianState


class GaussianMixtureState(Type, Sized, Iterable, Container):
    """
    Gaussian Mixture State type

    Represents the target space through a Gaussian Mxture. Individual Gaussian
    components are contained in a :class:`list` of
    :class:`WeightedGaussianState`.
    """

    components = Property(
        [WeightedGaussianState],
        default=[],
        doc="""The initial list of :class:`WeightedGaussianState` components.
        Default `None` which initialises with empty list.""")

    current_tag = Property(
        int,
        default=2,
        doc="""The tag number that should be assigned for the latest
        component""")

    extraction_threshold = Property(
        float,
        default=0.5,
        doc="""The threshold to used to extract "active" states
        from the Gaussian Mixture""")

    def __init__(self, components=[], *args, **kwargs):
        super().__init__(components, *args, **kwargs)
        if not isinstance(components, list):
            components = [components]
        if len(components) > 0:
            if any(not isinstance(component, WeightedGaussianState)
                    for component in components):
                raise ValueError("Cannot form GaussianMixtureState out of "
                                 "non-WeightedGaussianState inputs!")
        self.components = components.copy()
        # Check for existing GaussianMixtureState so unique tag scheme can be
        # preserved
        if self.components:
            if any(isinstance(component, TaggedWeightedGaussianState)
                    for component in self.components):
                self.current_tag = max(
                    self.current_tag,
                    max([component.tag for component in self.components]) + 1
                )

    def __contains__(self, index):
        # check if 'components' contains any WeightedGaussianState
        # matching 'index'
        if isinstance(index, WeightedGaussianState):
            return index in self.components
        else:
            raise ValueError("Index must be WeightedGaussianState")

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.components):
            component = self.components[self.index]
            self.index += 1
            return component
        else:
            raise StopIteration

    def __getitem__(self, index):
        # retrieve WeightedGaussianState by array index
        if isinstance(index, int):
            return self.components[index]
        else:
            raise ValueError("Index must be int")

    def __setitem__(self, index, value):
        return self.components.__setitem__(index, value)

    def __len__(self):
        return len(self.components)

    def append(self, component):
        return self.components.append(component)

    def extract_states(self):
        extracted_states = []
        for component in self:
            if component.weight > self.extraction_threshold:
                extracted_states.append(component)
        return extracted_states
