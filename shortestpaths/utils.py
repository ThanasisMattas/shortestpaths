# utils.py is part of ShortestPaths
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
"""Houses some utility functions."""

import ast
import copy
from datetime import timedelta
from functools import wraps
import heapq
from inspect import currentframe
from operator import itemgetter
from os.path import basename, realpath
from timeit import default_timer as timer
from typing import Iterable
import warnings

import click


def time_this(f):
    """function timer decorator

    - Uses wraps to preserve the metadata of the decorated function
      (__name__ and __doc__)
    - prints wall and CPU (user+sys) time

    usage:
      @time_this
      def a_func(): pass

    Args:
        f(funtion)      : the function to be decorated

    Returns:
        wrap (callable) : returns the result of the decorated function
    """
    assert callable(f)

    @wraps(f)
    def wrap(*args, **kwargs):
      start = timer()
      result = f(*args, **kwargs)
      end = timer()
      duration = timedelta(seconds=round(end - start, 3))
      print(f"{f.__name__:-<30}{duration}"[:45])
      return result
    return wrap


class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:  # noqa: E722
            raise click.BadParameter(value)


def check_nodal_connection(nodes: Iterable,
                           adj_list: list,
                           disconnected_nodes: Iterable) -> Iterable:
  """Checks for connection status of important nodes, i.e. source node and,
  in case of a disconnected important node, it replaces it with the closest
  neighbor.
  """
  for i, node in enumerate(nodes):
    if node in disconnected_nodes:
      input(f"Node <{node}> is disconnected. Setting it to its closest "
            "first neighbor. Press ENTER to continue...")
      node_neighbors = sorted(adj_list[node], key=itemgetter(1))
      for neighbor in node_neighbors:
        if neighbor[1] not in disconnected_nodes:
          nodes[i] = neighbor[1]
          break
  return nodes


def deb_trace(msg=None, condition=None):
  """Plants a debugging trace on the line.

  Usage:

  29 deb_trace()
  # prints: <file_name>::<func_name>::29

   7 deb_trace("A descriptive message")
  # prints: <file_name>::<func_name>::7:: A descriptive message

  45 deb_trace("Another descriptive message", <False condition>)
  # prints nothing

  Args:
    msg (str)        : An optional msg to be printed after the trace info
    condition (bool) : if (condition is None) or (condition) : print trace
                       (defauls to None)
  """
  if (condition is None) or (condition):
    # NOTE: f_back is key to move one frame back to the call-stack.
    #       It can be chained to get more frames or even the full call-stack.
    print(f"{basename(realpath(currentframe().f_back.f_code.co_filename))}::"
          f"{currentframe().f_back.f_code.co_name}::"
          f"{currentframe().f_back.f_lineno}"
          + (f":: {msg}" if msg else ''))


def print_heap(h):
  h_copy = copy.deepcopy(h)
  while h_copy:
    entry = heapq.heappop(h_copy)
    if hasattr(entry, "__len__"):
      for i in entry:
        print(i)
      if h_copy:
        print()
    else:
      print(entry)


def path_cum_hop_weights(path, adj_list):
  """Finds the cumulative hop weights of a path."""
  cum_hop_weights = [0]
  for i, u in enumerate(path[:-1]):
    # Find the edge weight.
    for v, uv_weight in adj_list[u]:
      if v == path[i + 1]:
        cum_hop_weights.append(uv_weight + cum_hop_weights[-1])
        break
    if len(cum_hop_weights) < i + 2:
      raise Exception(f"Path: {path} is disconnected at {path[i: i + 2]}")
  return cum_hop_weights


def verify_paths(paths_data):
  """Verifies that each path has at least a nodes-list and a cost."""
  invalid_paths_idxs = []
  offset = 0
  for i in range(len(paths_data)):
    if ((not hasattr(paths_data[i - offset], "__len__"))
            and (paths_data[i - offset] is not None)):
      invalid_paths_idxs.append(i)
      del paths_data[i - offset]
      offset += 1
  if invalid_paths_idxs:
      warnings.warn(f"Invalid paths indexes: {invalid_paths_idxs}")


def str_to_type(type_str):
  types = {
    "int": int,
    "float": float,
    "str": str
  }
  try:
    return types[type_str]
  except KeyError:
    return None
