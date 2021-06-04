#!/usr/env/bin/ python3
"""
python state_retrieval_vis.py --save-graph -s 0 -l 45 -e 5 --online 100
"""
import click

from shortestpaths import (core,
                           dijkstra,
                           graph_generator,
                           post_processing)
from shortestpaths.utils import print_heap  # noqa: F401

def nodes_retrieved(failed_idx,
                    failing,
                    tapes,
                    base_path,
                    source,
                    sink):

  to_visit, visited, discovered_forward = tapes[0][
      dijkstra._state_idx(failed_idx,
                          tapes[0],
                          base_path,
                          "forward",
                          "edges")
  ]
  to_visit_reverse, visited_reverse, discovered_forward = tapes[1][
      dijkstra._state_idx(failed_idx + int(failing == "edges"),
                          tapes[1],
                          base_path,
                          "reverse",
                          "edges")
  ]
  visited_nodes_forward = post_processing.visited_nodes(visited, source)
  visited_nodes_reverse = post_processing.visited_nodes(visited_reverse, sink)
  return visited_nodes_forward, visited_nodes_reverse


def visited_after_retrieved(s, e, online):
  """c = 0.25, gradient = 0.7, p_0 = 0.7"""
  #############################################################################
  if s == 0:
    visited_after_retrieval_forward_me = {9, 19, 21, 22, 25, 26, 28, 30, 35, 37, 38, 48}
    visited_after_retrieval_reverse_me = {56, 57, 63, 64, 67, 69, 71, 76, 77, 81, 8}

    if e == 4:
      if online:
        visited_after_retrieval_forward = {63}
        visited_after_retrieval_reverse = {75}
      else:
        visited_after_retrieval_forward = {9, 19, 21, 22, 25, 26, 28, 30, 34, 35, 37, 38, 39, 48, 56}
        visited_after_retrieval_reverse = {65, 68, 70, 75, 78, 80, 82, 87, 91, 93, 95, 51, 59, 62}
    elif e == 2:
      visited_after_retrieval_forward = {4, 8, 19, 22, 24, 27, 29, 32, 37, 40}
      visited_after_retrieval_reverse = {35, 56, 57, 64, 67, 69, 71, 76, 77, 82}
    elif (e == 1) and (online):
      visited_after_retrieval_forward = {35, 37, 22, 23, 27, 13}
      visited_after_retrieval_reverse = {69, 71, 76, 77, 82, 56, 57}
    elif (e == 5) and (online):
      visited_after_retrieval_forward = {67, 51, 70, 57, 62}
      visited_after_retrieval_reverse = {98, 86, 72, 92, 79}

    else:
      visited_after_retrieval_forward = visited_after_retrieval_reverse = set()
  #############################################################################
  elif s == 20:
    visited_after_retrieval_forward_me = {5, 20, 23, 26, 29, 32, 35, 37, 41, 50, 56}
    visited_after_retrieval_reverse_me = {38, 48, 52, 54, 55, 61, 66, 69, 87, 91}

    if e == 1:
      visited_after_retrieval_forward = {4, 7, 8, 9, 10, 12, 13, 15, 17, 18, 19, 21, 24, 25, 27, 28, 30, 31, 39, 46}
      visited_after_retrieval_reverse = {26, 38, 42, 45, 48, 49, 52, 54, 55, 56, 61, 63, 65, 66, 69, 75, 87, 91, 96}
    elif e == 0:
      visited_after_retrieval_forward = {4, 6, 7, 8, 9, 10, 12, 15, 16, 17, 18, 21, 25, 28, 31, 35, 39, 46}
      visited_after_retrieval_reverse = {26, 38, 42, 45, 48, 49, 52, 54, 55, 56, 61, 63, 65, 66, 69, 75, 87, 91, 96}
    elif e == 2:
      visited_after_retrieval_forward = {4, 8, 9, 12, 13, 15, 17, 18, 19, 25, 27, 28, 30, 31, 32, 34, 35, 39, 46}
      visited_after_retrieval_reverse = {26, 38, 42, 45, 48, 49, 52, 54, 55, 56, 61, 63, 65, 66, 69, 75, 87, 91, 96}
    elif e == 4:
      visited_after_retrieval_reverse = {5, 20, 23, 26, 29, 32, 33, 35, 37, 41, 42, 49, 50, 55, 56, 63}
      visited_after_retrieval_forward = {38, 52, 54, 57, 59, 66, 67, 69, 72, 73, 80, 86, 87, 91, 92, 95}
    else:
      visited_after_retrieval_forward = visited_after_retrieval_reverse = set()
  #############################################################################
  else:
    visited_after_retrieval_forward_me = visited_after_retrieval_reverse_me = set()
    visited_after_retrieval_forward = visited_after_retrieval_reverse = set()
  #############################################################################

  visited_after_retrieval_me = visited_after_retrieval_forward_me.union(
      visited_after_retrieval_reverse_me
  )
  visited_after_retrieval = visited_after_retrieval_forward.union(
      visited_after_retrieval_reverse
  )

  return visited_after_retrieval_me, visited_after_retrieval


@click.command()
@click.argument('n', type=click.INT)
@click.option('-e', "--tail_idx", type=click.INT, default=4, show_default=True,
              help=("the position at the path of the in query failed edge,"
                    " starting from 0"))
@click.option('-s', "--seed", "random_seed", default=0, show_default=True)
@click.option('-l', "--layout-seed", default=0, show_default=True)
@click.option('-c', "--center-portion", default=0.25, show_default=True)
@click.option('-g', "--gradient", default=0.7, show_default=True)
@click.option('-p', "--p_0", default=0.7, show_default=True)
@click.option('-o', "--online", is_flag=True)
@click.option("--show-graph", is_flag=True)
@click.option("--save-graph", is_flag=True)
def main(n,
         tail_idx,
         random_seed,
         layout_seed,
         center_portion,
         gradient,
         p_0,
         online,
         show_graph,
         save_graph):
  directed = False
  weights_on = "edges-and-nodes"
  failing = "edges"
  source = 1
  sink = n

  adj_list, G = graph_generator.random_graph(n=n,
                                             directed=directed,
                                             weights_on=weights_on,
                                             random_seed=random_seed,
                                             center_portion=center_portion,
                                             gradient=gradient,
                                             p_0=p_0)
  adj_list_reverse = graph_generator.adj_list_reversed(adj_list)
  init_config = {
      "adj_list": adj_list,
      "adj_list_reverse": adj_list_reverse,
      "source": source,
      "sink": sink
  }
  mode = {
      "bidirectional": True,
      "dynamic": True,
      "failing": failing,
      "online": online
  }
  path_data, tapes = core.first_shortest_path(mode, init_config)
  if online:
    r_paths, visited_reverse_me, visited_reverse_query, failed_head = \
        core.replacement_paths(mode, init_config)
  else:
    r_paths = core.replacement_paths(mode, init_config)

  paths_data_me = [path_data]
  paths_data = [path_data]
  base_path = path_data[0]
  meeting_edge_head = path_data[4]
  meeting_edge_head_idx = path_data[0].index(meeting_edge_head)
  query_tail = base_path[tail_idx]

  num_paths = 0
  for path in r_paths[1:]:
    if path[3][1] == meeting_edge_head:
      paths_data_me.append(path)
      num_paths += 1
    if path[3][0] == query_tail:
      paths_data.append(path)
      num_paths += 1
    if num_paths == 2:
      break

  if online:
    # Get the state that corresponds to the meeting edge.
    visited_nodes_reverse_me = post_processing.visited_nodes(
        visited_reverse_me,
        sink
    )
    # Get the state that corresponds to the query edge.
    if tail_idx > meeting_edge_head_idx:
      visited_nodes_reverse = post_processing.visited_nodes(
          visited_reverse_query,
          sink
      )
    else:
      visited_nodes_reverse = visited_nodes_reverse_me
    visited_nodes_forward_me = visited_nodes_forward = set()
  else:
    # Get the state that corresponds to the meeting edge.
    visited_nodes_forward_me, visited_nodes_reverse_me = nodes_retrieved(
        meeting_edge_head_idx - 1,
        failing,
        tapes,
        base_path,
        source,
        sink
    )
    # Get the state that corresponds to the query edge.
    visited_nodes_forward, visited_nodes_reverse = nodes_retrieved(
        tail_idx,
        failing,
        tapes,
        base_path,
        source,
        sink
    )

  visited_after_retrieval_me, visited_after_retrieval = \
      visited_after_retrieved(random_seed, tail_idx, online)

  # 1. State retrieval vis for the meeting edge
  #    (all the visited nodes are retrieved)
  if not online:
    post_processing.state_retrieval_vis(
        G,
        paths_data_me,
        visited_nodes_forward_me,
        visited_nodes_reverse_me,
        visited_nodes_forward_me,
        visited_nodes_reverse_me,
        visited_after_retrieval_me,
        (base_path[meeting_edge_head_idx - 1], meeting_edge_head),
        mode,
        random_seed,
        layout_seed,
        show_graph,
        save_graph
    )

  # 2. State retrieval vis for the in query edge
  #    (all the visited nodes are retrieved)
  post_processing.state_retrieval_vis(
      G,
      paths_data,
      visited_nodes_forward_me,
      visited_nodes_reverse_me,
      visited_nodes_forward,
      visited_nodes_reverse,
      visited_after_retrieval,
      (base_path[meeting_edge_head_idx - 1], meeting_edge_head),
      mode,
      random_seed,
      layout_seed,
      show_graph,
      save_graph
  )


if __name__ == '__main__':
  main()
