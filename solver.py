"""Exact teacher for ordinary 3 x 3 tic-tac-toe.

Boards contain 0 (empty), 1 (X), and -1 (O), with X moving first.
Values are from the perspective of the player whose turn is next.  The
teacher returns every optimal move, avoiding arbitrary tie-breaking labels.
"""

from functools import lru_cache

Board = tuple[int, ...]
EMPTY: Board = (0,) * 9
WIN_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)


def _board(board) -> Board:
    """Normalize and reject malformed board representations."""
    try:
        result = tuple(board)
    except TypeError as exc:
        raise ValueError("A board must contain nine cells") from exc
    if len(result) != 9 or any(type(cell) is not int or cell not in (-1, 0, 1) for cell in result):
        raise ValueError("A board must contain nine integer cells in {-1, 0, 1}")
    return result


def _winner(board: Board) -> int | None:
    for a, b, c in WIN_LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return None if 0 in board else 0


def winner(board: Board) -> int | None:
    """Return 1/-1 for a winner, 0 for a full-board draw, or None."""
    return _winner(_board(board))


def current_player(board: Board) -> int:
    """Return the next player; terminal boards retain the same convention."""
    board = _board(board)
    difference = board.count(1) - board.count(-1)
    if difference not in (0, 1):
        raise ValueError("X and O counts do not describe alternating turns")
    return 1 if difference == 0 else -1


@lru_cache(maxsize=1)
def _states() -> frozenset[Board]:
    seen: set[Board] = set()

    def visit(board: Board, player: int):
        if board in seen:
            return
        seen.add(board)
        if _winner(board) is not None:
            return
        for move, cell in enumerate(board):
            if cell == 0:
                visit(board[:move] + (player,) + board[move + 1:], -player)

    visit(EMPTY, 1)
    return frozenset(seen)


def validate_board(board: Board) -> Board:
    """Require reachability with alternating turns and no play after a win."""
    result = _board(board)
    if result not in _states():
        raise ValueError("Board is not reachable under tic-tac-toe rules")
    return result


def enumerate_states() -> list[Board]:
    """All 5,478 reachable states, in a stable row-major lexical order."""
    return sorted(_states())


def legal_moves(board: Board) -> list[int]:
    board = validate_board(board)
    if _winner(board) is not None:
        return []
    return [move for move, cell in enumerate(board) if cell == 0]


def play(board: Board, move: int) -> Board:
    """Apply a legal move, rejecting unreachable boards and terminal play."""
    board = validate_board(board)
    if type(move) is not int or not 0 <= move < 9:
        raise ValueError("Move must be an integer from 0 through 8")
    if _winner(board) is not None:
        raise ValueError("The game is already over")
    if board[move] != 0:
        raise ValueError("That cell is occupied")
    player = 1 if board.count(1) == board.count(-1) else -1
    return board[:move] + (player,) + board[move + 1:]


@lru_cache(maxsize=None)
def _solve(board: Board) -> tuple[int, tuple[int, ...]]:
    outcome = _winner(board)
    if outcome is not None:
        if outcome == 0:
            return 0, ()
        player = 1 if board.count(1) == board.count(-1) else -1
        return (1 if outcome == player else -1), ()
    player = 1 if board.count(1) == board.count(-1) else -1
    choices = []
    for move, cell in enumerate(board):
        if cell == 0:
            child = board[:move] + (player,) + board[move + 1:]
            value = -_solve(child)[0]
            choices.append((move, value))
    best = max(value for _, value in choices)
    return best, tuple(move for move, value in choices if value == best)


def solve(board: Board) -> tuple[int, tuple[int, ...]]:
    """Exact minimax value and every optimal move, in cell order."""
    return _solve(validate_board(board))


def action_values(board: Board) -> tuple[int | None, ...]:
    """Exact per-action values; occupied cells/terminal actions are None."""
    board = validate_board(board)
    if _winner(board) is not None:
        return (None,) * 9
    player = 1 if board.count(1) == board.count(-1) else -1
    return tuple(
        None if cell else -_solve(board[:move] + (player,) + board[move + 1:])[0]
        for move, cell in enumerate(board)
    )


def symmetries(board: Board) -> tuple[Board, ...]:
    """Eight D4 transforms: four rotations, then reflected rotations.

    Duplicate transforms are retained for symmetric boards.  This helper
    accepts any well-formed board, including action masks represented as
    -1/0/1 cells; reachability is not required for geometric operations.
    """
    board = _board(board)
    transformed = []
    for reflect in (False, True):
        for rotations in range(4):
            cells = [0] * 9
            for index, cell in enumerate(board):
                row, col = divmod(index, 3)
                if reflect:
                    col = 2 - col
                for _ in range(rotations):
                    row, col = col, 2 - row
                cells[row * 3 + col] = cell
            transformed.append(tuple(cells))
    return tuple(transformed)


def canonical_key(board: Board) -> str:
    """Stable orbit key for symmetry-disjoint dataset splits."""
    canonical = min(symmetries(board))
    return "".join(str(cell + 1) for cell in canonical)
