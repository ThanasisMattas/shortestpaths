import os
import subprocess
from pathlib import Path
import sys

cwd = Path(os.getcwd())
home_dir = str(cwd.parent)
sys.path.insert(0, home_dir)

import click

from shortestpaths.utils import time_this


def _clear(tool, dijkstra_options):
  """Clears output files from pervious runs.

  usage:
    $ python shortestpaths_profiler --tool=<tool> --clear
  """
  for mode in dijkstra_options.keys():
    input(f"Clearing {tool} output. Press ENTER to proceed...")
    outputs = [f"{mode}_{tool}.txt", f"{mode}_{tool}.log", "cpu_profile.txt"]
    for output in outputs:
      path_to_output = os.path.join(os.getcwd(), output)
      if os.path.exists(path_to_output):
        os.remove(path_to_output)


@click.command()
@click.option("--tool", default="callgrind", show_default=True,
              type=click.Choice(["callgrind", "massif"], case_sensitive=False))
@click.option("--num-graphs", type=click.INT, default=50, show_default=True,)
@click.option("--num-nodes", type=click.INT, default=500, show_default=True)
@click.option("--clear/--no-clear", default=False, show_default=True)
@time_this
def main(tool, num_graphs, n, clear):
  NUM_GRAPHS = num_graphs
  N = n

  dijkstra_options = {"original": "--adapted-path --no-saving-states",
                      "adaptive": "--adapted-path --saving-states"}
  if tool == "callgrind":
    dijkstra_options["reference"] = "--no-adapted-path --saving-states"

  tool_options = {"callgrind": "--branch-sim=yes --cache-sim=yes",
                  "massif": "--max-snapshots=500"}

  if clear:
    _clear(tool, dijkstra_options)
    return

  for mode, options in dijkstra_options.items():
    print(f"mode: {mode}")
    cmd = (
      f"valgrind --tool={tool} {tool_options[tool]} "
      f"--{tool}-out-file={mode}_{tool}.txt "
      f"--log-file={mode}_{tool}.log "
      f"python shortestpaths_reps.py {options} {NUM_GRAPHS} {N}"
    )
    subprocess.run(cmd.split())
    print()

  if tool == "callgrind":
    cpu_cycles = dict()

    for mode in dijkstra_options.keys():
      output = f"{mode}_callgrind.txt"
      with open(output, 'r') as fr:
        # Costs are sorted by events count, so we have to read the event labels,
        # too, to match them accordingly.
        for ln in fr:
          if ln.startswith("events:"):
            events = ln.strip()[8:].split()
          if ln.startswith("summary:"):
            summary = ln.strip()[9:].split()
            break

      costs = dict()
      for i in range(len(events)):
        costs[events[i]] = int(summary[i])

      Ir = costs["Ir"]
      L1m = costs["I1mr"] + costs["D1mr"] + costs["D1mw"]
      L3m = costs["ILmr"] + costs["DLmr"] + costs["DLmw"]
      Bm = costs["Bim"] + costs["Bcm"]

      cc = Ir + 10 * L1m + 100 * L3m + 20 * Bm
      cpu_cycles[os.path.splitext(output)[0].replace("_callgrind", '')] = cc
      # os.remove(output)

    net_cc_adaptive = cpu_cycles["adaptive"] - cpu_cycles["reference"]
    net_cc_original = cpu_cycles["original"] - cpu_cycles["reference"]
    optimization_gain = round(
      (net_cc_original - net_cc_adaptive) / net_cc_original * 100
    )

    with open("cpu_profile.txt", 'w') as fw:
      fw.write(
        "Performance experiment for the Adaptive Dijkstra's algorithm\n"
        "------------------------------------------------------------\n"
        "\n"
        f"#graphs : {NUM_GRAPHS}\n"
        f"#nodes  : {N}\n"
        "\n"
        f"Dijkstra's algorithm          : {net_cc_original} cycles\n"
        f"Adaptive Dijkstra's algorithm : {net_cc_adaptive} cycles\n"
        f"Performance gain              : {optimization_gain}%\n"
      )


if __name__ == '__main__':
  main()
