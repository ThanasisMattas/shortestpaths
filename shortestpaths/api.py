# api.py is part of ShortestPaths
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
"""Provides dot delimited symbols to the shortestpaths API (sp.func)."""

from typing import Hashable, Union

import networkx as nx

from shortestpaths import core, graph


def _make_config(G: Union[nx.Graph, nx.DiGraph, list],
                 s: Hashable,
                 t: Hashable,
                 problem: str,
                 method: str,
                 **kwargs):
  """Generates the configuration of the problem.

  Args:
    G (nx.Graph, nx.DiGraph or list)
                       : the graph
                         adj_list format: [{(head, cost),},]
    s, t (hashable)    : source, target
    method (str)       : options:
                           'y' (Yen    + unidirectional Dijkstra)
                           'l' (Lawler + unidirectional Dijkstra)
                           'b' (Lawler + bidirectional Dijkstra)
                           'd' (DP     + bidirectional Dijkstra)
    kwargs:
      failing (str)    : 'nodes' or 'edges' [default: 'edges']
      online (bool)    : When online, the path up until the failure is kept as
                         it is (the algorithm is getting informed upon meeting
                         the failed node or edge), whereas when not online, a
                         new search starts from the source, ignoring the parent
                         path (the algorithm is a priori informed about the
                         failure). [default: False]

  Returns:
    mode (dict)        : the configuration of the problem
    init_config (dict) : kwargs for dijkstra_init()
    decoder (dict)     : maps the graph nodes with 1-n integers, to be used as
                         indexes of the adj_list
  """
  if problem == "k-shortest-paths":
    methods = {
        'y': (True, False, False, False),
        'l': (False, True, False, False),
        'b': (False, False, True, False),
        'd': (False, False, True, True)
    }
    yen, lawler, bidirectional, dynamic = methods[method]
  elif problem == "replacement-paths":
    methods = {
        'u': (False, False),
        'b': (True, False),
        'd': (True, True)
    }
    bidirectional, dynamic = methods[method]
  else:
    raise ValueError("Expected problems: ['k-shortest-paths',"
                     f" 'replacement-paths']. Instead got: {problem}")

  if not isinstance(G, list):
    adj_list, encoder, decoder = graph.nx_to_adj_list(G)
    s = encoder[s]
    t = encoder[t]

  if bidirectional:
    adj_list_reverse = graph.adj_list_reversed(adj_list)
  else:
    adj_list_reverse = None

  init_config = {
      "adj_list": adj_list,
      "adj_list_reverse": adj_list_reverse,
      "source": s,
      "sink": t
  }
  mode = {
      "bidirectional": bidirectional,
      "parallel": False,
      "dynamic": dynamic,
      "failing": kwargs.pop("failing", "edges"),
      "online": kwargs.pop("online", True),
      "verbose": 1,
  }
  if problem == "k-shortest-paths":
    mode.update({"yen_": yen, "lawler": lawler})

  return mode, init_config, decoder


def k_shortest_paths(G: Union[nx.Graph, nx.DiGraph, list],
                     s: Hashable,
                     t: Hashable,
                     k: int,
                     method: str = 'd'):
  """Generates the k-shortest s->t loopless paths.

  Args:
    G (nx.Graph, nx.DiGraph or list) : the graph
                                       adj_list format: [{(head, cost),},]
    s, t (hashable)                  : source, target
    k (int)                          : number of paths
    method (str)                     : options:
                                         'y' (Yen    + unidirectional Dijkstra)
                                         'l' (Lawler + unidirectional Dijkstra)
                                         'b' (Lawler + bidirectional Dijkstra)
                                         'd' (DP     + bidirectional Dijkstra)
                                       [default: 'd']

  Returns:
    k_paths (list)                   : [(path: list, cost: float),]
  """
  mode, init_config, decoder = _make_config(G,
                                            s,
                                            t,
                                            "k-shortest-paths",
                                            method)

  k_paths = core.k_shortest_paths(k, mode, init_config)

  if decoder:
    k_paths = graph.decode_path_nodes(k_paths, decoder)

  return k_paths


def replacement_paths(G: Union[nx.Graph, nx.DiGraph, list],
                      s: Hashable,
                      t: Hashable,
                      failing: str = "edges",
                      online: bool = False,
                      method: str = 'd'):
  """Generates the replacement paths.

  Args:
    G (nx.Graph, nx.DiGraph or list)
                    : the graph
                      adj_list format: [{(head, cost),},]
    s, t (hashable) : source, target
    failing (str)   : 'nodes' or 'edges' [default: 'edges']
    online (bool)   : When online, the path up until the failure is kept as it
                      is (the algorithm is getting informed upon meeting the
                      failed node or edge), whereas when not online, a new
                      search starts from the source, ignoring the parent-path
                      (the algorithm is a priori informed about the failure).
                      [default: False]
    method (str)    : options:
                        'y' (Yen    + unidirectional Dijkstra)
                        'l' (Lawler + unidirectional Dijkstra)
                        'b' (Lawler + bidirectional Dijkstra)
                        'd' (DP     + bidirectional Dijkstra)
                      [default: 'd']

  Returns:
    k_paths (list)  : [(path: list, cost: float),]
  """
  mode, init_config, decoder = _make_config(G,
                                            s,
                                            t,
                                            "replacement-paths",
                                            method,
                                            failing=failing,
                                            online=online)

  r_paths = core.replacement_paths(mode, init_config)

  if decoder:
    r_paths = graph.decode_path_nodes(r_paths, decoder)

  return r_paths
