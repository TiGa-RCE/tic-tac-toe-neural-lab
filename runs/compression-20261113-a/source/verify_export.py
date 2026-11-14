"""Independent exported-network qualification (Python standard library only).

This file intentionally imports neither the training code nor its solver.
Run it through the user's memory-run launcher because it loads and evaluates
a model.  Exhaustive decision accuracy and no-loss-from-empty are distinct
acceptance conditions; both are reported without conflating them.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path

LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)
EMPTY = (0,) * 9


def outcome(board):
    """Absolute result: X=+1, O=-1, draw=0, unfinished=None."""
    totals = [sum(board[index] for index in line) for line in LINES]
    if 3 in totals:
        return 1
    if -3 in totals:
        return -1
    return None if 0 in board else 0


def successor(board, index):
    next_mark = 1 if sum(board) == 0 else -1
    cells = list(board)
    cells[index] = next_mark
    return tuple(cells)


def census():
    """Generate legal states with an independent breadth-first traversal."""
    seen = {EMPTY}
    pending = [EMPTY]
    cursor = 0
    while cursor < len(pending):
        board = pending[cursor]
        cursor += 1
        if outcome(board) is not None:
            continue
        for index, cell in enumerate(board):
            if cell == 0:
                child = successor(board, index)
                if child not in seen:
                    seen.add(child)
                    pending.append(child)
    return pending


@lru_cache(maxsize=None)
def exact_x_value(board):
    terminal = outcome(board)
    if terminal is not None:
        return terminal
    values = [exact_x_value(successor(board, index))
              for index, cell in enumerate(board) if cell == 0]
    return (max if sum(board) == 0 else min)(values)


def exact_moves(board):
    target = exact_x_value(board)
    return [index for index, cell in enumerate(board)
            if cell == 0 and exact_x_value(successor(board, index)) == target]


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_model(model):
    architecture = model.get("architecture")
    if (not isinstance(architecture, list) or len(architecture) < 2
            or architecture[0] not in (9, 18, 27) or architecture[-1] != 9
            or any(type(size) is not int or not 1 <= size <= 1024 for size in architecture)):
        raise ValueError("Expected a bounded 9/18/27-input, 9-output dense architecture")
    encoding = model.get('encoding', {9:'relative-signed',18:'relative-two-channel',27:'relative-one-hot-interleaved'}[architecture[0]])
    widths = {'relative-signed':9,'relative-two-channel':18,'relative-one-hot-interleaved':27}
    if encoding not in widths or widths[encoding] != architecture[0]:
        raise ValueError('Encoding metadata does not match the input width')
    layers = model.get("layers")
    if not isinstance(layers, list) or len(layers) != len(architecture) - 1:
        raise ValueError("Layer count does not match architecture")
    for index, layer in enumerate(layers):
        n_input, n_output = architecture[index:index + 2]
        weights, bias = layer.get("weights"), layer.get("bias")
        if (not isinstance(weights, list) or len(weights) != n_input
                or any(not isinstance(row, list) or len(row) != n_output
                       or any(not _finite_number(value) for value in row)
                       for row in weights)):
            raise ValueError(f"Layer {index} weights have invalid dimensions or nonfinite values")
        if (not isinstance(bias, list) or len(bias) != n_output
                or any(not _finite_number(value) for value in bias)):
            raise ValueError(f"Layer {index} bias has invalid dimensions or nonfinite values")
        expected_activation = "linear" if index == len(layers) - 1 else "tanh"
        if layer.get("activation") != expected_activation:
            raise ValueError(f"Layer {index} must use {expected_activation} activation")
    return architecture


def encode(board, encoding='relative-one-hot-interleaved'):
    """Cell-major [own, opponent, empty] one-hot encoding, next-player view."""
    next_mark = 1 if sum(board) == 0 else -1
    if encoding == 'relative-signed':
        return [float(cell*next_mark) for cell in board]
    if encoding == 'relative-two-channel':
        return [value for cell in board for value in (float(cell==next_mark),float(cell==-next_mark))]
    if encoding != 'relative-one-hot-interleaved':
        raise ValueError('Unrecognized board encoding')
    values = []
    for cell in board:
        relative = cell * next_mark
        values.extend((float(relative == 1), float(relative == -1), float(relative == 0)))
    return values


def predict(model, board):
    encoding=model.get('encoding',{9:'relative-signed',18:'relative-two-channel',27:'relative-one-hot-interleaved'}[model['architecture'][0]])
    values = encode(board,encoding)
    for layer in model["layers"]:
        weights, bias = layer["weights"], layer["bias"]
        values = [bias[j] + sum(value * weights[i][j] for i, value in enumerate(values))
                  for j in range(len(bias))]
        if layer["activation"] == "tanh":
            values = [math.tanh(value) for value in values]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Inference produced a nonfinite output")
    return values


def qualify(model, include_fixtures=False):
    architecture = validate_model(model)
    states = census()
    if len(states) != 5478:
        raise AssertionError(f"Independent reachable census is {len(states)}, expected 5478")
    decisions = [board for board in states if outcome(board) is None]
    if len(decisions) != 4520:
        raise AssertionError(f"Independent decision census is {len(decisions)}, expected 4520")
    choices = {}
    mistakes = []
    fixtures = []
    optimal_count = 0
    for board in decisions:
        logits = predict(model, board)
        move = max((index for index, cell in enumerate(board) if cell == 0),
                   key=lambda index: logits[index])
        optimal = exact_moves(board)
        choices[board] = move
        if move in optimal:
            optimal_count += 1
        elif len(mistakes) < 10:
            mistakes.append({"board": list(board), "move": move, "optimal_moves": optimal,
                             "predicted_scores": logits})
        if include_fixtures:
            fixtures.append({"board": list(board), "logits": logits, "move": move,
                             "optimal_moves": optimal})

    def worst_case(neural_mark):
        @lru_cache(maxsize=None)
        def visit(board):
            result = outcome(board)
            if result is not None:
                return result * neural_mark, ()
            mark = 1 if sum(board) == 0 else -1
            if mark == neural_mark:
                move = choices[board]
                value, continuation = visit(successor(board, move))
                return value, (move,) + continuation
            branches = [(visit(successor(board, move)), move)
                        for move, cell in enumerate(board) if cell == 0]
            (value, continuation), move = min(branches, key=lambda item: item[0][0])
            return value, (move,) + continuation

        value, path = visit(EMPTY)
        return {"value": value, "result": {-1: "loss", 0: "draw", 1: "win"}[value],
                "worst_case_moves": list(path), "visited_states": visit.cache_info().currsize}

    as_x, as_o = worst_case(1), worst_case(-1)
    report = {
        "schema_version": 1,
        "verifier": "independent-standard-library-export-verifier",
        "run_id": model.get("run_id"),
        "checkpoint": model.get("checkpoint"),
        "architecture": architecture,
        "parameter_count": sum((a + 1) * b for a, b in zip(architecture, architecture[1:])),
        "encoding": model.get('encoding', {9:'relative-signed',18:'relative-two-channel',27:'relative-one-hot-interleaved'}[architecture[0]]),
        "tie_break": "lowest legal cell index",
        "reachable_count": len(states),
        "total": len(decisions),
        "optimal_count": optimal_count,
        "optimal_rate": optimal_count / len(decisions),
        "mistake_count": len(decisions) - optimal_count,
        "mistakes": mistakes,
        "perfect_policy": optimal_count == len(decisions),
        "as_x": as_x,
        "as_o": as_o,
        "no_loss_from_empty": as_x["value"] >= 0 and as_o["value"] >= 0,
        "scope": "All reachable decision boards; both sides against every possible opponent continuation from empty",
    }
    if include_fixtures:
        report["fixtures"] = fixtures
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-fixtures", action="store_true")
    parser.add_argument("--expected", type=Path,
                        help="Optional trainer evaluation JSON for count comparison only")
    args = parser.parse_args()
    with args.model.open() as handle:
        model = json.load(handle)
    report = qualify(model, args.full_fixtures)
    if args.expected:
        with args.expected.open() as handle:
            expected = json.load(handle)
        expected_count = expected.get("optimal_count")
        expected_total = expected.get("total")
        report["comparison"] = {
            "expected_optimal_count": expected_count,
            "expected_total": expected_total,
            "counts_match": report["optimal_count"] == expected_count and report["total"] == expected_total,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "fixtures"}, allow_nan=False))


if __name__ == "__main__":
    main()
