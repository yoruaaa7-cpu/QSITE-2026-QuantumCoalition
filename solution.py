from __future__ import annotations

import collections
import math
import random

import networkx as nx


def _used_logical_qubits(program: list[tuple]) -> list[int]:
    return sorted({q for op in program for q in op[1:]})


def _all_pairs_distances(graph: nx.Graph) -> dict[int, dict[int, int]]:
    return {u: dict(lengths) for u, lengths in nx.all_pairs_shortest_path_length(graph)}


def _interaction_weights(program: list[tuple]) -> collections.Counter:
    weights = collections.Counter()
    for op in program:
        if op[0] == "2Q":
            a, b = sorted(op[1:])
            weights[(a, b)] += 1.0
    return weights


def _placement_cost(
    placement: dict[int, int],
    program: list[tuple],
    distances: dict[int, dict[int, int]],
) -> float:
    """
    Penalize logical interactions that are far apart.

    Earlier gates get a little extra weight because poor early placement can
    cause cascading SWAPs that hurt later routing as well.
    """
    cost = 0.0
    for i, op in enumerate(program):
        if op[0] != "2Q":
            continue

        _, a, b = op
        front_weight = 1.0 + 1.5 * math.exp(-i / 8.0)
        cost += front_weight * (distances[placement[a]][placement[b]] - 1)

    return cost


def _optimized_placement(
    program: list[tuple],
    hardware_graph: nx.Graph,
    distances: dict[int, dict[int, int]],
    trials: int = 5,
    steps: int = 2500,
) -> dict[int, int]:
    """
    Deterministic multi-start simulated annealing placement.

    We first favor high-degree / central hardware nodes for highly interacting
    logical qubits, then improve the mapping through swaps and moves.
    """
    logical_qubits = _used_logical_qubits(program)

    seed = 0xC0FFEE + len(program) * 131 + sum(logical_qubits) * 17
    rng = random.Random(seed)

    interaction_weights = _interaction_weights(program)

    logical_degree = collections.Counter()
    for (a, b), weight in interaction_weights.items():
        logical_degree[a] += weight
        logical_degree[b] += weight

    physical_nodes = list(hardware_graph.nodes)

    # Favor high hardware degree, then graph-central nodes.
    physical_rank = sorted(
        physical_nodes,
        key=lambda p: (
            -hardware_graph.degree[p],
            sum(distances[p].values()),
            p,
        ),
    )

    logical_rank = sorted(
        logical_qubits,
        key=lambda q: (-logical_degree[q], q),
    )

    best_placement = None
    best_cost = float("inf")

    for trial in range(trials):
        if trial == 0:
            chosen = physical_rank[: len(logical_qubits)]
            placement = {q: p for q, p in zip(logical_rank, chosen)}
        else:
            chosen = rng.sample(physical_nodes, len(logical_qubits))
            placement = {q: p for q, p in zip(logical_qubits, chosen)}

        current_cost = _placement_cost(placement, program, distances)

        for step in range(steps):
            q = rng.choice(logical_qubits)
            occupied = set(placement.values())

            # Usually swap two logical qubits; sometimes move into an empty
            # physical node if the hardware has spare qubits.
            if rng.random() < 0.7 and len(logical_qubits) > 1:
                q2 = rng.choice([x for x in logical_qubits if x != q])
                candidate = placement.copy()
                candidate[q], candidate[q2] = candidate[q2], candidate[q]
            else:
                empty_nodes = [p for p in physical_nodes if p not in occupied]
                if not empty_nodes:
                    continue
                candidate = placement.copy()
                candidate[q] = rng.choice(empty_nodes)

            candidate_cost = _placement_cost(candidate, program, distances)

            temperature = 3.0 * (1.0 - step / steps) + 0.02

            accept = candidate_cost < current_cost
            if not accept:
                delta = current_cost - candidate_cost
                accept = rng.random() < math.exp(delta / temperature)

            if accept:
                placement = candidate
                current_cost = candidate_cost

            if current_cost < best_cost:
                best_cost = current_cost
                best_placement = placement.copy()

    assert best_placement is not None
    return best_placement


def _route_with_lookahead(
    program: list[tuple],
    hardware_graph: nx.Graph,
    initial_placement: dict[int, int],
    distances: dict[int, dict[int, int]],
    lookahead: int = 10,
) -> list[tuple]:
    """
    Route each required interaction using a SABRE-inspired look-ahead heuristic.

    For each required SWAP, try every hardware edge and score the virtual result:
      * strongly prioritize making the current gate adjacent;
      * also prefer SWAPs that improve the next few gates;
      * mildly prefer touching one of the current gate's qubits.

    The program's original logical operation order is never changed.
    """
    placement = initial_placement.copy()
    physical_to_logical = {
        physical: logical for logical, physical in placement.items()
    }

    routed_program: list[tuple] = []

    for index, op in enumerate(program):
        kind = op[0]

        if kind == "1Q":
            logical = op[1]
            routed_program.append(("1Q", placement[logical]))
            continue

        _, logical_a, logical_b = op

        while not hardware_graph.has_edge(
            placement[logical_a], placement[logical_b]
        ):
            physical_a = placement[logical_a]
            physical_b = placement[logical_b]
            current_distance = distances[physical_a][physical_b]

            future_two_qubit_ops = []
            for future_op in program[index:]:
                if future_op[0] == "2Q":
                    future_two_qubit_ops.append(future_op)
                    if len(future_two_qubit_ops) >= lookahead:
                        break

            best_edge = None
            best_score = float("inf")

            for left, right in hardware_graph.edges:
                left_logical = physical_to_logical.get(left)
                right_logical = physical_to_logical.get(right)

                def virtual_position(logical: int) -> int:
                    physical = placement[logical]

                    if left_logical is not None and logical == left_logical:
                        return right
                    if right_logical is not None and logical == right_logical:
                        return left

                    return physical

                new_current_distance = distances[
                    virtual_position(logical_a)
                ][
                    virtual_position(logical_b)
                ]

                # Avoid obvious moves that make the immediate gate strictly worse.
                if new_current_distance >= current_distance + 1:
                    continue

                heuristic = 4.0 * new_current_distance

                for offset, future_op in enumerate(
                    future_two_qubit_ops[1:], start=1
                ):
                    _, u, v = future_op
                    heuristic += (
                        (0.8 ** offset)
                        * distances[virtual_position(u)][virtual_position(v)]
                    )

                # Small tie-break preference for directly moving a current-gate qubit.
                if (
                    left in (physical_a, physical_b)
                    or right in (physical_a, physical_b)
                ):
                    heuristic -= 0.15

                if heuristic < best_score:
                    best_score = heuristic
                    best_edge = (left, right)

            # Safety fallback: always make progress toward the current interaction.
            if best_edge is None:
                path = nx.shortest_path(
                    hardware_graph, physical_a, physical_b
                )
                best_edge = (path[0], path[1])

            left, right = best_edge
            left_logical = physical_to_logical.get(left)
            right_logical = physical_to_logical.get(right)

            routed_program.append(("SWAP", left, right))

            physical_to_logical[left], physical_to_logical[right] = (
                right_logical,
                left_logical,
            )

            if left_logical is not None:
                placement[left_logical] = right
            if right_logical is not None:
                placement[right_logical] = left

        routed_program.append(
            (
                "2Q",
                placement[logical_a],
                placement[logical_b],
            )
        )

    return routed_program


def solve(
    program: list[tuple],
    hardware_graph: nx.Graph,
) -> tuple[dict[int, int], list[tuple]]:
    """
    QSITE 2026 Quantum Coalition - Computational Track solution.

    Returns:
        initial_placement:
            logical qubit -> physical qubit

        routed_program:
            operations expressed on physical qubits, with SWAPs inserted.
    """
    distances = _all_pairs_distances(hardware_graph)

    initial_placement = _optimized_placement(
        program,
        hardware_graph,
        distances,
    )

    routed_program = _route_with_lookahead(
        program,
        hardware_graph,
        initial_placement,
        distances,
    )

    return initial_placement, routed_program
