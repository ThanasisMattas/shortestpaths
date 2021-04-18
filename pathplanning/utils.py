# utils.py is part of PathPlanning
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
"""Houses some utility functions."""

import ast
from datetime import timedelta
from functools import wraps
from operator import itemgetter
import time
from timeit import default_timer as timer
from typing import Iterable
import warnings

import click


def print_duration(start, end, process, time_type=None):
    """Prints the duration of a process.

    Args:
      start (float)      : the starting timestamp in seconds
      end (float)        : the ending timestamp in seconds
      process (string)   : the process name
      time_type (string) : i.e. wall-clock, CPU or sys time (defaults to None)
    """
    if time_type is None:
      prefix = f"{process} time"
    else:
      prefix = f"{process:-<24}{time_type} time"
    duration = timedelta(seconds=end - start)
    print(f"{prefix:-<40}{duration}"[:51])


def time_this(wall_clock=None):
    """function timer decorator

    - Uses wraps to preserve the metadata of the decorated function
      (__name__ and __doc__)
    - prints wall and CPU (user+sys) time

    usage:
      @time_this
      def a_func(): pass

      @time_this()  # wall_clock=False
      def a_func(): pass

      @time_this(wall_clock=True)
      def a_func(): pass

    Args:
        f(funtion)      : the function to be decorated

    Returns:
        wrap (callable) : returns the result of the decorated function
    """
    def inner_decorator(f):
      if not callable(f):
        raise Exception(f"{f} is not a callable and, thus, it cannot be "
                        f"decorated with @time_this.")

      @wraps(f)
      def wrapper(*args, **kwargs):
        using_wall_clock = False
        if callable(wall_clock):
          # Meaning that @time_this is used without ()
          pass
        else:
          if wall_clock:
            using_wall_clock = True
            start_wall = timer()
        start_user_plus_sys = time.process_time()
        result = f(*args, **kwargs)
        end_user_plus_sys = time.process_time()
        if using_wall_clock:
          end_wall = timer()
          print_duration(start_wall,
                         end_wall,
                         f.__name__,
                         "Wall")
        print_duration(start_user_plus_sys,
                       end_user_plus_sys,
                       f.__name__,
                       "CPU")
        return result
      return wrapper

    if callable(wall_clock):
      return inner_decorator(wall_clock)
    return inner_decorator


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


def extract_path(visited, source, sink, with_hop_weights=False):
  """Extracts the shortest-path from a Dijkstra's algorithm output.

  Dijkstra's algorithm saves the shortest path cost for each node of the graph,
  as well as its previous node on the path, so as to retrieve the path by
  jumping through previous nodes, until the source node.

  Args:
    visited (2D list)        : each entry is a 2-list:
                               [path_cost, prev_node_id]
    source, sink (hashable)  : the ids of source and sink nodes
    with_hop_weights (bool)  : - True : returns just the node_id's
                               - False: returns 2-lists:
                                        (node_id, hop_cost)

  Returns:
    path (list)              : if with_hop_weights:
                                 each entry is a 2-list,
                                 [node_id, edge-cost]
                               else:
                                 list of the consecutive nodes in the path
  """
  if with_hop_weights:
    path = [[sink, visited[sink][0]]]
    node = sink
    while node != source:
      prev_node = visited[node][1]
      prev_node_cost = visited[prev_node][0]
      # The corresponding costs are path-costs. In order to get the hop-cost, we
      # have to offset with the path-cost of the previous node in the path.
      path[-1][1] -= prev_node_cost
      path.append([prev_node, prev_node_cost])
      if node == prev_node:
        # Some node/edge failures may disconnect the graph. This can be dete-
        # cted because at initialization prev_node_id is set to node_id. In
        # that case, a warning is printed and we move to the next path, if any.
        warnings.warn(f"The source ({source}) is not connected to the sink"
                      f" ({sink}).")
        return []
      node = prev_node
  else:
    path = [sink]
    node = sink
    while node != source:
      prev_node = visited[node][1]
      path.append(prev_node)
      if node == prev_node:
        warnings.warn(f"The source ({source}) is not connected to the sink"
                      f" ({sink}).")
        return []
      node = prev_node

  path.reverse()
  return path
