# PathPlanning is free software; you may redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version. You should have received a copy of the GNU
# General Public License along with this program. If not, see
# <https://www.gnu.org/licenses/>.
"""
info:
    file        :  priorityq_test.py
    author      :  Thanasis Mattas
    license     :  GNU General Public License v3
    description :  Houses all the tests for the priorityq module
"""

from operator import itemgetter

import pytest

from pathplanning.priorityq import PriorityQueue


class TestPriorityQueue():

  def setup_method(self):
    self.data = [[4, 5, 1],
                 [3, 1, 5],
                 [2, 6, 3],
                 [1, 8, 9],
                 [8, 3, 12],
                 [5, 11, 10]]
    self.pq = PriorityQueue(self.data)
    self.length = 6

  def teardown_method(self):
    self.pq.clear()
    self.data.clear()

  def test_len(self):
    assert len(self.pq) == self.length

  def test_bool(self):
    assert self.pq

  @pytest.mark.parametrize(
    "entry_id,entry_expected",
    [(10, [5, 11, 10]), (12, [8, 3, 12]), (3, [2, 6, 3])]
  )
  def test_getitem(self, entry_id, entry_expected):
    assert self.pq[entry_id] == entry_expected
    assert len(self.pq) == self.length

  def test_delitem(self):
    del self.pq[5]
    assert len(self.pq) == self.length - 1
    with pytest.raises(KeyError) as ke:
      assert self.pq[5] == [3, 1, 5]

  def test_setitem(self):
    self.pq[12] = [6, 2, 12]
    assert self.pq[12] == [6, 2, 12]
    assert len(self.pq) == self.length

  def test_contains(self):
    five = False
    if 5 in self.pq:
      five = True
    assert self.pq[5] == [3, 1, 5]
    assert five

  def test_empty(self):
    assert not self.pq.empty()

  def test_pop_low(self):
    entry = self.pq.pop_low()
    assert entry == [1, 8, 9]
    assert len(self.pq) == self.length - 1

  def test_iter(self):
    data = iter(sorted(self.data, key=itemgetter(0, 1, 2)))
    for entry in self.pq:
      assert entry == next(data)

  def test_clear(self):
    assert self.pq
    self.pq.clear()
    assert not self.pq
