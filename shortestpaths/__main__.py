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
"""Produces a ShortestPaths demo on a random Erdős-Rényi graph.

Usage: shortestpaths [OPTIONS] N COMMAND [OPTIONS]

Examples:
shortestpaths 100
shortestpaths -s 0 --layout-seed 1 -k 5 200
shortestpaths 200 replacement-paths --failing edges
shortestpaths -b -p 300 replacement-paths -f nodes
shortestpaths -d 300 replacement-paths -f nodes
"""

import click

from shortestpaths import core, graph_generator, post_processing, utils  # noqa F401


@click.group(invoke_without_command=True)
@click.pass_context
@click.argument("n", type=click.INT)
@click.option("--unweighted", "weighted", is_flag=True,
              default=True, show_default="weighted")
@click.option("--weights-on", default="edges-and-nodes", show_default=True,
              type=click.Choice(["edges", "nodes", "edges-and-nodes"],
                                case_sensitive=False))
@click.option("--max-edge-weight", default=1000, show_default=True)
@click.option("--max-node-weight", default=50, show_default=True)
@click.option('-k', 'K', type=click.INT, default=1, show_default=True,
              help="number of shortest paths to be generated")
@click.option('-y', "--yen", is_flag=True)
@click.option('-l', "--lawler", is_flag=True)
# @click.option("--disconnected-nodes",
#               cls=PythonLiteralOption, default="[]", show_default=True,
#               help=("Usage:\n\n"
#                     "--disconnected-nodes '[id_1, id_2, ...]'"
#                     "\n\n"
#                     "An alternative path will be constructed, disregarding"
#                     " the disconnected nodes."))
@click.option('-b', "--bidirectional", is_flag=True,
              help="bidirectional shortest path search (uses 2 processes)")
@click.option('-p', "--parallel", is_flag=True,
              help="whether to use multiprocessing or not")
@click.option('-d', "--dynamic", is_flag=True,
              help="whether to use dynamic programming or not")
@click.option('-s', "--seed", "random_seed", type=click.INT,
              default=4, show_default=True,
              help="If provided, a fixed random graph will be generated.")
@click.option("--layout-seed", type=click.INT, default=1, show_default=True,
              help="Fixes the random initialization of the spirng_layout.")
@click.option("--show-graph", is_flag=True)
@click.option("--save-graph", is_flag=True)
@click.option('-v', "--verbose", count=True)
# @utils.time_this
def main(ctx,
         n,
         weighted,
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
  adj_list, G = graph_generator.random_graph(n=n,
                                             weighted=weighted,
                                             weights_on=weights_on,
                                             max_edge_weight=max_edge_weight,
                                             max_node_weight=max_node_weight,
                                             random_seed=random_seed)

  if dynamic:
    bidirectional = True

  if ctx.invoked_subcommand is not None:
    # populate ctx.obj
    ctx_config = {
      "adj_list": adj_list,
      "G": G,
      "n": n,
      "bidirectional": bidirectional,
      "parallel": parallel,
      "dynamic": dynamic,
      "layout_seed": layout_seed,
      "show_graph": show_graph,
      "save_graph": save_graph,
      "verbose": verbose
    }
    if ctx.invoked_subcommand == "dynamic_graph_demo":
      ctx_config["random_seed"] = random_seed
    ctx.obj.update(ctx_config)
    return

  # 2. Paths generation
  k_paths = core.k_shortest_paths(adj_list=adj_list,
                                  source=1,
                                  sink=n,
                                  K=K,
                                  bidirectional=bidirectional,
                                  parallel=parallel,
                                  dynamic=dynamic,
                                  yen_=yen,
                                  lawler=lawler,
                                  verbose=verbose)

  # 3. Post-processing
  if verbose:
    post_processing.print_paths(k_paths)
  if save_graph or show_graph:
    post_processing.plot_graph(G,
                               k_paths,
                               None,
                               save_graph,
                               show_graph,
                               layout_seed=layout_seed)


@main.command()
@click.pass_context
@click.option('-f', "--failing", default="nodes", show_default=True,
              type=click.Choice(["edges", "nodes"], case_sensitive=False),
              help="Setting what to fail, path edges or path nodes, in order"
                   " to produce the replacement paths.")
@click.option("--online", is_flag=True,
              help="Each replacement path is forced to contain the root sub-"
                   "path, till the previous node of the failed node/edge.")
@utils.time_this
def replacement_paths(ctx, failing, online):
  """CLI command for the replacement paths

  Args:
    ctx(click.core.Context) : has obj dict with the parameters of the group
    failing (str)           : "edges" or "nodes" (CLI option)
  """
  verbose = ctx.obj.pop("verbose", 0)

  r_paths = core.replacement_paths(
    ctx.obj.pop("adj_list"),
    n=ctx.obj.get("n", 100),
    source=1,
    sink=ctx.obj.pop("n", 100),
    failing=failing,
    bidirectional=ctx.obj.pop("bidirectional", False),
    parallel=ctx.obj.pop("parallel", False),
    dynamic=ctx.obj.pop("dynamic", False),
    online=online,
    verbose=verbose
  )

  if verbose:
    post_processing.print_paths(r_paths, failing)
  if ctx.obj["save_graph"] or ctx.obj["show_graph"]:
    post_processing.plot_graph(paths_data=r_paths,
                               failing=failing,
                               **ctx.obj)


@main.command()
@click.pass_context
def dynamic_graph_demo(ctx):
  raise NotImplementedError()


if __name__ == "__main__":
  main(obj={})
