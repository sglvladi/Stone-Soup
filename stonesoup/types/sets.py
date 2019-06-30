from scipy.spatial import cKDTree

from collections.abc import MutableSet

from .base import Type
from ..base import Property
from .track import Track

class BaseSet(MutableSet, Type):
    """Base StoneSoup set class"""

    items = Property({Track}, doc="Set of underlying items", default=set())

    def __init__(self, items=None, *args, **kwargs):
        if items is None:
            items = set()
        else:
            items = set(items)
        super().__init__(items, *args, **kwargs)

    def __len__(self):
        return self.items.__len__()

    def __contains__(self, item):
        return self.items.__contains__(item)

    def __iter__(self):
        for item in self.items.__iter__():
            yield item

    def add(self, other):
        self.items.add(other)

    def discard(self, other):
        self.items.discard(other)

    def __getattr__(self, item):
        if item.startswith("_"):
            # Don't proxy special/private attributes to `state`
            raise AttributeError(
                "{!r} object has no attribute {!r}".format(
                    type(self).__name__, item))
        else:
            return getattr(self.items, item)


class TrackSet(BaseSet):
    """ A mutable set container of StoneSoup items """


# class KdSet(BaseSet):
#     """ Class implementation for a set that is indexed by a kd-tree """
#
#     _item_list = []
#     _data = []
#
#     @property
#     def kdtree(self):
#         self._item_list = list(self.items)
#         self._data = []
#         for item in self._item_list:
#             self._data.append(item.)


## TODO: Remove code below
# class Nuset(set):
#
#     @classmethod
#     def _wrap_methods(cls, names):
#         def wrap_method_closure(name):
#             def inner(self, *args):
#                 result = getattr(super(cls, self), name)(*args)
#                 if isinstance(result, set) and not hasattr(result, 'foo'):
#                     result = cls(result, foo=self.foo)
#                 return result
#
#             inner.fn_name = name
#             setattr(cls, name, inner)
#
#         for name in names:
#             wrap_method_closure(name)
#
#
# Nuset._wrap_methods(['__ror__', 'difference_update', '__isub__',
#                      'symmetric_difference', '__rsub__', '__and__', '__rand__',
#                      'intersection',
#                      'difference', '__iand__', 'union', '__ixor__',
#                      'symmetric_difference_update', '__or__', 'copy',
#                      '__rxor__',
#                      'intersection_update', '__xor__', '__ior__', '__sub__',
#                      ])
