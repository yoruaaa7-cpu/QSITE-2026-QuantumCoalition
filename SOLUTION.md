# Yahye Abdullahi's computational track solution

**Code:** [`solution.py`](./solution.py) · **Challenge:** [Computational Track handout](./Computational%20Track/README.md)

This is my Python submission for the QSITE 2026 Quantum Coalition computational track. The original challenge, benchmark programs, hardware graph, and scorer came from the [challenge organizers](https://github.com/benmcdonough20/QSITE-2026-QuantumCoalition). My contribution in this fork is the `solution.py` placement and routing implementation.

## Problem and interface

A program contains logical one-qubit and two-qubit operations. The hardware graph lists which physical qubits can interact directly. `solve(program, hardware_graph)` returns:

1. An injective mapping from each used logical qubit to a physical qubit.
2. An ordered list of operations on physical qubits, with adjacent-qubit `SWAP` operations inserted where needed.

The scorer checks that every two-qubit operation and SWAP follows a hardware edge, that the mapping stays consistent, and that the original logical operations remain in order. It scores each benchmark as `swap_count + 0.5 × depth`, where the supplied scorer calculates depth.

## Approach in `solution.py`

1. **Measure interactions.** Count pairs of logical qubits that interact and calculate shortest-path distances between physical qubits.
2. **Choose an initial placement.** Prefer higher-degree, central physical qubits for frequently interacting logical qubits. Improve the mapping with multiple seeded simulated-annealing trials that exchange placements or move into free physical positions. Earlier gates receive more weight in the placement cost.
3. **Route each two-qubit gate.** When its qubits are not adjacent, evaluate candidate SWAPs along hardware edges. Prefer SWAPs that reduce the current distance and improve distances for upcoming two-qubit gates, then update both placement maps.
4. **Preserve program order.** Emit each input operation on its current physical qubits. The implementation does not create a separate scheduler; the supplied scorer determines parallel depth from the routed operations.

This is a heuristic: it does not guarantee the minimum number of SWAPs or the best possible score. The seeded search makes runs repeatable for the same input ordering, and the solution assumes a connected hardware graph with enough physical nodes for the used logical qubits.

## Run the supplied benchmarks

From the repository root, install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) and use the environment specified by the original computational track:

```bash
uv sync --project "Computational Track"
uv run --project "Computational Track" python
```

Then, in that Python session:

```python
import sys
sys.path.insert(0, "Computational Track")

from starter_kit.benchmarks import BENCHMARKS
from starter_kit.hardware import build_hardware_graph
from starter_kit.scorer import score_summary
from solution import solve

graph = build_hardware_graph()
for name, program in BENCHMARKS.items():
    placement, routed = solve(program, graph)
    result = score_summary(program, graph, placement, routed)
    print(name, "valid:", result["valid"], "swaps:", result["swap_count"],
          "depth:", result["depth"], "score:", result["score"])
```

The six benchmark programs and validation logic above are provided in the organizers' `Computational Track/starter_kit/`.
