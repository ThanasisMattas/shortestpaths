# priorityq.py is part of PathPlanning
#
# PathPlanning is free software; you may redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version. You should have received a copy of the GNU
# General Public License along with this program. If not, see
# <https://www.gnu.org/licenses/>.
#
# (C) 2020 Athanasios Mattas
# =======================================================================
"""Implementation of the priority queue data structure."""

import heapq
import itertools


class PriorityQueue:
  """Implements the priority queue, using the heapq module.

  The basic features, not supported by the heapq module and introduced here,
  are getting, setting and deleting an entry anywhere in the priority queue,
  using as query key a unique entry_id. This is needed when, for example, an
  application involves updating the priority value of the entries.

  PriorityQueue uses an entry_finder dictionairy, pointing to each entry in the
  queue. When setting or deleting occurs, it marks the entry as REMOVED and, in
  case of setting, it pushes a new entry. It does not delete the entry or
  update its priority on the spot, because that would break the heapq.

  The _counter variable serves as a tie-breaker when multiple entries have the
  same priority and the entry_id (or object) cannot be used to prioritize with.
  A count value will be inserted after the cost values and before the entry
  attributes:
             [cost_1, cost_2, ..., count, entry_attrs, entry_id]

  More info:
  https://docs.python.org/3/library/heapq.html#priority-queue-implementation-notes
  """
  _heapq = []
  _entry_finder = {}
  _REMOVED = "<removed-entry>"     # placeholder for a removed entry
  _counter = itertools.count()

  def __init__(self, data=None):
    """Constructs the PriorityQueue object.

    Args:
      data (list): Holds the cost values to prioritize with and the entry_id.
                   The hierarchy of the prioritization complies with the
                   sequence of the costs provided. (Defaults to None)
                   Format:
                   [
                     [cost_1, cost_2, ..., entry_attrs, entry_id],
                     [cost_1, cost_2, ..., entry_attrs, entry_id],
                     ...
                   ]
        cost_1, ... (any sortable type) : the values to prioritize with
        entry_attrs (list or obj)       : (optional) attrs of each entry
        entry_id (any hashable type)    : the unique id of each entry

    TODO:
      entry_attrs, entry_id become a small wrapper class (or use @dataclass)
    """
    # heapify performs in linear time, so is iterating through the data, and
    # this way the entry_finder dictionary can be constructed, pointing to the
    # entries of the queue.
    if data is not None:
      for entry in data:
        count = next(self._counter)
        entry.insert(-2, count)
        heapq.heappush(self._heapq, entry)
        self._entry_finder[entry[-1]] = entry
    else:
      pass

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

  def __delitem__(self, entry_id):
    """Mark an existing entry as REMOVED. Raise KeyError if not found."""
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
    count = next(self._counter)
    entry.insert(-2, count)
    self._entry_finder[entry_id] = entry
    heapq.heappush(self._heapq, entry)

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
      if entry[-1] is not self._REMOVED:
        del self._entry_finder[entry[-1]]
        # remove the counter
        del entry[-3]
        return entry
    raise KeyError("trying to pop from an empty priority queue")

  def pop_high(self):
    pass

  def __iter__(self):
    heapq_ = self._heapq.copy()
    for _ in range(len(heapq_)):
      entry = heapq.heappop(heapq_)
      if entry[-1] in self._entry_finder:
        # remove the counter
        del entry[-3]
        yield entry

  def clear(self):
    while self._heapq:
      heapq.heappop(self._heapq)
    self._entry_finder.clear()
