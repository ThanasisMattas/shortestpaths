#!/usr/bin/env python3
#
# __main__.py is part of ShortestPaths
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
"""Produces a ShortestPaths demo."""

import click
import networkx as nx

from shortestpaths import (core,  # noqa F401
                           graph_generator,
                           io,
                           post_processing,
                           utils)


@click.group(invoke_without_command=True)
@click.pass_context
@click.option("--path", type=click.STRING, default=None, show_default=True,
              help=("The NetworkX-file path to read the graph from. If not"
                    " provided, a random graph of n nodes will be generated.\n"
                    "Supported formats:\n"
                    "[.adjlist, .edgelist, .gexf, .gml, .gpickle]\n"
                    "Note that .adjlist does not include weights."))
@click.option("--nodetype", default="int", show_default=True,
              help="convert nodes to this type")
@click.option("--comments", type=click.STRING, default='#', show_default=True,
              help="marker for comment lines")
@click.option("--delimiter", type=click.STRING, default=' ', show_default=True,
              help="Separator for node labels. The default is whitespace.")
@click.option("--encoding", type=click.STRING,
              default="utf-8", show_default=True)
@click.option('-s', "--source", default=None, show_default=True,
              help="If a graph is not provided, the source defaults to node 1")
@click.option('-t', "--target", default=None, show_default=True,
              help="If a graph is not provided, the target defaults to node n")
@click.option('-n', type=click.INT, default=100, show_default=True,
              help="number of nodes (used when path is None)")
@click.option("--weighted/--no-weighted", default=True, show_default=True)
@click.option("--directed", is_flag=True)
@click.option("--weights-on", default="edges-and-nodes", show_default=True,
              type=click.Choice(["edges", "nodes", "edges-and-nodes"],
                                case_sensitive=False))
@click.option("--max-edge-weight", default=1000, show_default=True)
@click.option("--max-node-weight", default=50, show_default=True)
@click.option('-k', 'K', type=click.INT, default=1, show_default=True,
              help="number of shortest paths to be generated")
@click.option('-y', "--yen", is_flag=True)
@click.option('-l', "--lawler", is_flag=True)
@click.option('-b', "--bidirectional", is_flag=True,
              help="bidirectional shortest path search")
@click.option('-p', "--parallel", is_flag=True,
              help="whether to use multiprocessing or not")
@click.option('-d', "--dynamic", is_flag=True,
              help="whether to use dynamic programming or not")
@click.option("--seed", "random_seed", type=click.INT,
              default=None, show_default=True,
              help="If provided, a fixed random graph will be generated.")
@click.option("--layout-seed", type=click.INT, default=1, show_default=True,
              help="Fixes the random initialization of the spring_layout.")
@click.option("--show-graph", is_flag=True, help="plots up to 8 paths")
@click.option("--save-graph", is_flag=True, help="format: png")
@click.option('-v', "--verbose", count=True)
# @utils.time_this
def main(ctx,
         path,
         nodetype,
         comments,
         delimiter,
         encoding,
         source,
         target,
         n,
         weighted,
         directed,
         weights_on,
         max_edge_weight,
         max_node_weight,
         K,
         yen,
         lawler,
         bidirectional,
         parallel,
         dynamic,
         random_seed,
         layout_seed,
         show_graph,
         save_graph,
         verbose):

  # 1. Preprocessing
  if path is None:
    decoder = None
    source = int(source) if source else 1
    target = int(target) if target else n
    adj_list, G = graph_generator.random_graph(
        n=n,
        weighted=weighted,
        directed=directed,
        weights_on=weights_on,
        max_edge_weight=max_edge_weight,
        max_node_weight=max_node_weight,
        random_seed=random_seed
    )
  else:
    if (source is None) or (target is None):
      raise Exception("Both source and target sould be defined via the -s and"
                      " -t options.")
    nodetype = utils.str_to_type(nodetype)
    read_graph_config = {
        "path": path,
        "nodetype": nodetype,
        "comments": comments,
        "delimiter": delimiter,
        "encoding": encoding
    }
    if directed:
      read_graph_config["create_using"] = nx.DiGraph
    else:
      read_graph_config["create_using"] = nx.Graph

    adj_list, G, encoder, decoder = io.read_graph(read_graph_config, weighted)
    source = encoder[nodetype(source)]
    target = encoder[nodetype(target)]

  if dynamic:
    bidirectional = True
  if bidirectional:
    adj_list_reverse = graph_generator.adj_list_reversed(adj_list)
  else:
    adj_list_reverse = None

  init_config = {
      "adj_list": adj_list,
      "adj_list_reverse": adj_list_reverse,
      "source": source,
      "sink": target
  }
  mode = {
      "bidirectional": bidirectional,
      "parallel": parallel,
      "dynamic": dynamic,
      "failing": "edges",
      "online": True,
      "verbose": verbose
  }
  ctx_config = {
      "init_config": init_config,
      "mode": mode,
      "G": G,
      "layout_seed": layout_seed,
      "show_graph": show_graph,
      "save_graph": save_graph,
      "decoder": decoder
  }
  if ctx.invoked_subcommand is not None:
    if ctx.invoked_subcommand == "dynamic_graph_demo":
      ctx_config["random_seed"] = random_seed
    ctx.ensure_object(dict)
    ctx.obj.update(ctx_config)
    return

  mode.update({"yen_": yen, "lawler": lawler})

  # 2. Paths generation
  k_paths = core.k_shortest_paths(K, mode, init_config)

  if decoder:
    k_paths = graph_generator.decode_path_nodes(k_paths, decoder)

  # 3. Post-processing
  if verbose:
    post_processing.print_paths(k_paths)
  if save_graph or show_graph:
    ctx_config.pop("init_config")
    ctx_config.pop("decoder")
    post_processing.plot_paths(paths_data=k_paths, **ctx_config)


@main.command()
@click.pass_context
@click.option('-f', "--failing", default="nodes", show_default=True,
              type=click.Choice(["edges", "nodes"], case_sensitive=False),
              help="Setting what to fail, path edges or path nodes, in order"
                   " to produce the replacement paths.")
@click.option("--online", is_flag=True,
              help=("When online, the path up until the failure is kept as it"
                    " is (the algorithm is getting informed upon meeting the"
                    " failed node or edge), whereas when not online, a new"
                    " search starts from the source, ignoring the parent-path"
                    " (the algorithm is a priori informed about the failure)"))
@utils.time_this
def replacement_paths(ctx, failing, online):
  """CLI command for the replacement paths

  Args:
    ctx(click.core.Context) : has obj dict with the parameters of the group
    failing (str)           : "edges" or "nodes" (CLI option)
    online (bool)           : When online, the path up until the failure is
                              kept as it is (the algorithm is getting informed
                              upon meeting the failed node or edge), whereas
                              when not online, a new search starts from the
                              source, ignoring the parent-path (the algorithm
                              is a priori informed about the failure).
                              (CLI option)
  """
  ctx.obj["mode"].update({"failing": failing, "online": online})
  r_paths = core.replacement_paths(ctx.obj["mode"], ctx.obj.pop("init_config"))

  decoder = ctx.obj.pop("decoder", None)
  if decoder:
    r_paths = graph_generator.decode_path_nodes(r_paths, decoder)

  if ctx.obj["mode"]["verbose"]:
    post_processing.print_paths(r_paths, ctx.obj["mode"]["failing"])
  if ctx.obj["save_graph"] or ctx.obj["show_graph"]:
    post_processing.plot_paths(paths_data=r_paths, **ctx.obj)


@main.command()
@click.pass_context
def dynamic_graph_demo(ctx):
  raise NotImplementedError()


if __name__ == "__main__":
  main(obj={})
