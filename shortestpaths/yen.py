# yen.py is part of ShortestPaths
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
"""Implementation of Yen's k-shortests paths algorithm with improvements.

  Improvements (see Brander-Sinclair 1996):
    - Not searching for deviation paths already found. (Lawler 1972)
    - Using a heap instead of list (Yen's B), to store candidate paths.
    - If at least K - k candidates with the same cost as the (k - 1)th path
      were already found, append them to the k_paths list (Yen's A) and return.
"""

import heapq

from shortestpaths import dijkstra


def fail_found_spur_edges(adj_list,
                          spur_node,
                          spur_node_path_idx,
                          base_path,
                          k_paths,
                          adj_list_reverse=None,
                          head=None):
  """Failes the edges having spur-node as tail, for each of the K - k found
  paths, that have the same root-path with the prospect-path."""
  # {head: (head, edge_cost)}
  failed_edges = dict()
  # {invertd_tail (head): (inverted_head (tail), edge_cost)}
  failed_inverted_edges = dict()
  for j in k_paths:
    if j[0][:spur_node_path_idx + 1] == base_path[:spur_node_path_idx + 1]:
      failed_edges[j[0][spur_node_path_idx + 1]] = None
      failed_inverted_edges[j[0][spur_node_path_idx + 1]] = None

  # Don't disconnect the failed edge yet, because it will be disconnected
  # in the subsequent loop.
  if head:
    del failed_edges[head]
    del failed_inverted_edges[head]

  for v, uv_weight in adj_list[spur_node]:
    if v in failed_edges.keys():
      failed_edges[v] = (v, uv_weight)
      failed_inverted_edges[v] = (spur_node, uv_weight)
  for v, edge in failed_edges.items():
    adj_list[spur_node].remove(edge)

  if adj_list_reverse:
    for u, edge in failed_inverted_edges.items():
      adj_list_reverse[u].remove(edge)
  return failed_edges, failed_inverted_edges


def reconnect_spur_edges(spur_node,
                         adj_list,
                         failed_edges,
                         adj_list_reverse=None,
                         failed_inverted_edges=None):
  for _, edge in failed_edges.items():
    adj_list[spur_node].add(edge)
  failed_edges.clear()
  if adj_list_reverse:
    for u, edge in failed_inverted_edges.items():
      adj_list_reverse[u].add(edge)
    failed_inverted_edges.clear()


def push_prospect(path,
                  path_cost,
                  cum_hop_weights,
                  spur_node_idx,
                  meeting_edge_head,
                  K,
                  k,
                  prospects):
  if ((len(prospects) < K - k)
          or (path_cost < heapq.nsmallest(K - k, prospects)[-1][0])):
    # Check if the prospect is already found.
    prospect_already_found = False
    for prospect in prospects:
      if (prospect[0] == path_cost) and (prospect[1] == path):
        prospect_already_found = True
        break
    if not prospect_already_found:
      heapq.heappush(
        prospects,
        [path_cost, path, cum_hop_weights, spur_node_idx, meeting_edge_head]
      )
  return prospects


def update_prospects(k,
                     K,
                     k_paths,
                     mode,
                     init_config,
                     parent_spur_node_idx,
                     prospects,
                     cum_hop_weights):
  """Finds the deviation-paths of the (k-1)-th path and updates the prospects
  heap.

  - Replacement-paths search starts form the parent_spur_node_idx, as suggested
    by Lawler.
  - <k_paths> and <prospects> are Yen's a and B, respectively.
  """
  # This counter is used to print the spur-paths search difference produced
  # by Lawler's modification.
  spur_counter = 0
  last_path = k_paths[-1][0]
  if not mode.get("lawler"):
    parent_spur_node_idx = 0

  # Construct the deviation paths of the last found shortest path.
  # (u is the spur node)
  for i, u in enumerate(last_path[parent_spur_node_idx: -1]):
    spur_counter += 1

    u_idx = i + parent_spur_node_idx
    # Fail the (i, i + 1) edges of the found k - 1 shortest paths.
    failed_edges, _ = fail_found_spur_edges(init_config["adj_list"],
                                            u,
                                            u_idx,
                                            last_path,
                                            k_paths)
    # Fail the root-path nodes from the to_visit PriorityQueue. Note that the
    # PriorityQueue should have been deepcopied.
    dijkstra_config, _ = dijkstra.dijkstra_init(**init_config)
    del dijkstra_config["to_visit"][last_path[:u_idx]]

    # Set the spur-node as source and initialize its cost to root-path-cost.
    dijkstra_config["to_visit"][u] = [cum_hop_weights[u_idx], u, u]

    spur, prospect_cost, spur_weights, _ = dijkstra.unidirectional_dijkstra(
      dijkstra_config,
      mode
    )

    if spur:
      prospect = last_path[:u_idx] + spur
      prospect_hop_weights = cum_hop_weights[:u_idx] + spur_weights

      prospects = push_prospect(prospect,
                                prospect_cost,
                                prospect_hop_weights,
                                u_idx,
                                None,
                                K,
                                k,
                                prospects)

    # Restore the failed edges.
    reconnect_spur_edges(u, init_config["adj_list"], failed_edges)

  if mode.get("verbose", 0) >= 2:
    print(f"k: {k + 1:{len(str(K))}}   spur paths: {spur_counter}")
  return prospects
