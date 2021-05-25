# priorityq_test.py is part of ShortestPaths
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
"""Houses all the tests for the priorityq module."""

import copy
from operator import itemgetter

import pytest

from shortestpaths.priorityq import PriorityQueue


class TestPriorityQueue():

  def setup_method(self):
    self.data = [[4, 5, 1],
                 [3, 1, 5],
                 [2, 6, 3],
                 [1, 8, 9],
                 [8, 3, 12],
                 [5, 11, 10]]
    self.pq = PriorityQueue([entry.copy() for entry in self.data])
    self.length = len(self.data)

  def teardown_method(self):
    self.pq.clear()
    self.data.clear()

  def test_instantiation(self):
    pq1 = PriorityQueue(copy.deepcopy(self.data[:4]))
    pq2 = copy.deepcopy(pq1)
    del pq2[1]  # 4th priority
    assert len(self.pq) == self.length
    assert len(pq1) == 4
    assert len(pq2) == 3

    data = sorted(self.data, key=itemgetter(0, 1, 2))

    counter = 0
    for i, entry in enumerate(self.pq):
      counter += 1
      assert entry[-1] == data[i][-1]
    assert counter == self.length

    counter = 0
    for i, entry in enumerate(pq1):
      counter += 1
      assert entry[-1] == data[i][-1]
    assert counter == 4

    counter = 0
    for i, entry in enumerate(pq2):
      counter += 1
      assert entry[-1] == data[i][-1]
    assert counter == 3

  def test_len(self):
    assert len(self.pq) == self.length
    del self.pq[9]
    assert len(self.pq) == self.length - 1
    self.pq.clear()
    assert len(self.pq) == 0

  def test_bool(self):
    assert self.pq
    self.pq.clear()
    assert not self.pq

  @pytest.mark.parametrize(
    "entry_id, entry_expected",
    [(10, [5, 11, 10]), (12, [8, 3, 12]), (3, [2, 6, 3])]
  )
  def test_getitem(self, entry_id, entry_expected):
    assert self.pq[entry_id] == entry_expected
    assert len(self.pq) == self.length

  def test_delitem__single_item(self):
    # delete a single item
    del self.pq[5]
    assert len(self.pq) == self.length - 1
    with pytest.raises(KeyError) as ke:
      print(self.pq[5])

  def test_delitem__collection_of_items(self):
    items = [5, 3, 12]
    del self.pq[items]
    assert len(self.pq) == self.length - 3
    for item in items:
      with pytest.raises(KeyError) as ke:
        print(self.pq[item])

  @pytest.mark.parametrize(
    "entry_id, entry, length_diff",
    [(12, [6, 2, 12], 0), (6, [1, 1, 6], 1)]
  )
  def test_setitem(self, entry_id, entry, length_diff):
    self.pq[entry_id] = entry.copy()
    assert self.pq[entry_id] == entry
    assert len(self.pq) == self.length + length_diff

  @pytest.mark.parametrize(
    "entry, length_diff",
    [([6, 2, 12], 0), ([1, 1, 6], 1)]
  )
  def test_add(self, entry, length_diff):
    self.pq.add(entry.copy())
    assert self.pq[entry[-1]] == entry
    assert len(self.pq) == self.length + length_diff

  def test_contains(self):
    assert 5 in self.pq
    assert 9 in self.pq
    assert 15 not in self.pq
    assert 0 not in self.pq
    assert len(self.pq) == self.length

  def test_empty(self):
    assert not self.pq.empty()
    self.pq.clear()
    assert self.pq.empty()

  def test_pop_low(self):
    entry = self.pq.pop_low()
    assert entry == [1, 8, 9]
    assert len(self.pq) == self.length - 1
    while self.pq:
      self.pq.pop_low()
    assert self.pq.empty()
    with pytest.raises(KeyError) as ke:
      self.pq.pop_low()

  def test_peek(self):
    entry = self.pq.peek()
    assert entry == [1, 8, 9]
    assert entry == self.pq[9]
    assert len(self.pq) == self.length
    self.pq.clear()
    with pytest.raises(IndexError) as ke:
      self.pq.peek()

  def test_iter(self):
    data = iter(sorted(self.data, key=itemgetter(0, 1, 2)))
    for entry in self.pq:
      assert entry == next(data)

  def test_clear(self):
    assert self.pq
    self.pq.clear()
    assert not self.pq
    assert len(self.pq) == 0

  def test_keys(self):
    pq_keys = self.pq.keys()
    raw_data_keys = list(zip(*self.data))[-1]
    assert len(pq_keys) == len(raw_data_keys)
    # symmetric difference
    diff = set(pq_keys) ^ set(raw_data_keys)
    assert not diff

  def test_relax_priority(self):
    self.pq.relax_priority([3, 5, 1])
    assert self.pq[1] == [3, 5, 1]
    self.pq.relax_priority([4, 5, 1])
    assert self.pq[1] == [3, 5, 1]
    with pytest.raises(KeyError) as ke:
      self.pq.relax_priority([4, 5, 99])
    assert len(self.pq) == self.length
