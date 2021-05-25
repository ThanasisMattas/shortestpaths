# priorityq.py is part of ShortestPaths
#
# ShortestPaths is free software; you may redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version. You should have received a copy of the GNU General Pu-
# blic License along with this program. If not, see
# <https://www.gnu.org/licenses/>.
#
# (C) 2020 Athanasios Mattas
# ==========================================================================
"""Implementation of the priority queue data structure."""

import copy
import heapq
import itertools
from typing import Hashable, Iterable, Union


class PriorityQueue:
  """Implements the priority queue, using the heapq module.

  The basic features not supported by the heapq module but introduced here are
  getting, setting and deleting an entry anywhere in the priority queue, using
  as query key a unique entry_id. This is needed when, for example, an applica-
  tion involves updating the priority value of the entries.

  PriorityQueue uses an entry_finder dictionairy, pointing to each entry in the
  queue. When setting or deleting occurs, it marks the entry as REMOVED and, in
  case of setting, it pushes a new entry. It does not delete the entry or up-
  date its priority on the spot, because that would break the heapq.

  The _counter variable serves as a tie-breaker when multiple entries have the
  same priority and the entry_id (or object) cannot be used to prioritize with.
  An entry_count value will be inserted after the cost values and before the
  entry attributes:

            [cost_1, cost_2, ..., entry_count, entry_attrs, entry_id]

  More info:
  https://docs.python.org/3/library/heapq.html#priority-queue-implementation-notes

  TODO:
    entry_attrs, entry_id become a small wrapper class (or use @dataclass)

  Args:
    data (list): Holds the cost values to prioritize with, any entry attributes
                 and the entry_id. The hierarchy of the prioritization complies
                 with the sequence of the costs provided. (Defaults to None)
                 Format:
                   [[cost_1, cost_2, ..., entry_attrs, entry_id],]
                   cost_1, ... (sortable) : the values to prioritize with
                   entry_attrs (any)      : (optional) attrs of each entry
                   entry_id (hashable)    : the unique id of each entry
  """
  # placeholder for a removed entry
  _REMOVED = "<removed-entry>"

  def __init__(self, data=None):
    self._heapq = []
    heapq.heapify(self._heapq)
    self._entry_finder = {}
    self._counter = itertools.count()
    if data is None:
      pass
    else:
      # heapify performs in linear time, so is iterating through the data, and
      # this way the entry_finder dictionary can be constructed, pointing to
      # the entries of the queue.
      for entry in data:
        entry_count = next(self._counter)
        entry.insert(-2, entry_count)
        heapq.heappush(self._heapq, entry)
        self._entry_finder[entry[-1]] = entry

  def __copy__(self):
    new_obj = PriorityQueue()
    new_obj._heapq = self._heapq.copy()
    new_obj._entry_finder = self._entry_finder.copy()
    new_obj._counter = copy.copy(self._counter)
    return new_obj

  def __deepcopy__(self, *args, **kargs):
    new_obj = PriorityQueue()
    for entry in self._heapq:
      if entry[-1] != self._REMOVED:
        entry_copy = entry.copy()
        new_obj._heapq.append(entry_copy)
        new_obj._entry_finder[entry_copy[-1]] = entry_copy
    heapq.heapify(new_obj._heapq)
    new_obj._counter = copy.copy(self._counter)
    return new_obj

  def __len__(self):
    return len(self._entry_finder)

  def __bool__(self):
    if self._entry_finder:
      return True
    return False

  def __getitem__(self, entry_id):
    entry = self._entry_finder[entry_id].copy()
    # remove the counter
    del entry[-3]
    return entry

  def __delitem__(self, entry_id: Union[Hashable, Iterable]):
    """Marks an existing entry as REMOVED and removes it from _entry_finder.

    Raises:
      KeyError : if entry_id is not found
    """
    try:
      for id in entry_id:
        entry = self._entry_finder.pop(id)
        entry[-1] = self._REMOVED
    except TypeError:
      entry = self._entry_finder.pop(entry_id)
      entry[-1] = self._REMOVED

  def __setitem__(self, entry_id, entry):
    """Adds a new entry or updates an existing one.

    In case the entry already exists, it is marked as REMOVED and the updated
    entry is pushded to the queue and the entry_finder dictionary.

    Example-doctest:
    >>> pr_queue = PriorityQueue([[4, "node_1"], [5, "node_2"], [2, "node_3]])
    >>> pr_queue["node_1"]
    4
    >>> pr_queue["node_3"] = 120
    >>> pr_queue["node_3"]
    120
    >>> pr_queue.pop_low()
    [4, "node_1"]
    """
    if entry_id in self._entry_finder:
      del self[entry_id]
    entry_count = next(self._counter)
    entry.insert(-2, entry_count)
    self._entry_finder[entry_id] = entry
    heapq.heappush(self._heapq, entry)

  def add(self, entry):
    self[entry[-1]] = entry

  def __contains__(self, entry_id):
    if entry_id in self._entry_finder:
      return True
    return False

  def empty(self):
    if self._entry_finder:
      return False
    return True

  def pop_low(self):
    """Pop the lowest weighted node. Raise KeyError if empty."""
    while self._heapq:
      entry = heapq.heappop(self._heapq)
      if entry[-1] != self._REMOVED:
        del self._entry_finder[entry[-1]]
        del entry[-3]
        return entry
    raise KeyError("Trying to pop from an empty PriorityQueue.")

  def pop_high(self):
    raise NotImplementedError()

  def peek(self):
    entry = self._heapq[0]
    while entry[-1] == self._REMOVED:
      heapq.heappop(self._heapq)
      entry = self._heapq[0]
    entry = entry.copy()
    del entry[-3]
    return entry

  def __iter__(self):
    heapq_ = [x.copy() for x in self._heapq]
    heapq.heapify(heapq_)
    for _ in range(len(heapq_)):
      entry = heapq.heappop(heapq_)
      if entry[-1] in self._entry_finder:
        # remove the counter
        del entry[-3]
        yield entry

  def clear(self):
    self._heapq.clear()
    self._entry_finder.clear()

  def keys(self):
    return list(self._entry_finder.keys())

  def relax_priority(self, entry):
    if entry[-1] in self._entry_finder:
      if entry[0] < self._entry_finder[entry[-1]][0]:
        self[entry[-1]] = entry
    else:
      raise KeyError({entry[-1]})
