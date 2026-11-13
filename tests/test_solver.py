import unittest

from solver import (
    EMPTY, WIN_LINES, action_values, canonical_key, current_player,
    enumerate_states, legal_moves, play, solve, symmetries, winner,
)


class SolverTests(unittest.TestCase):
    def test_reachable_census_and_terminal_cutoff(self):
        states = enumerate_states()
        self.assertEqual(len(states), 5478)
        self.assertEqual(len(set(states)), 5478)
        terminals = [board for board in states if winner(board) is not None]
        self.assertEqual(len(terminals), 958)
        self.assertEqual(sum(winner(board) == 1 for board in terminals), 626)
        self.assertEqual(sum(winner(board) == -1 for board in terminals), 316)
        self.assertEqual(sum(winner(board) == 0 for board in terminals), 16)
        for board in terminals:
            self.assertEqual(legal_moves(board), [])
            self.assertEqual(solve(board)[1], ())
            with self.assertRaises(ValueError):
                play(board, next((i for i, cell in enumerate(board) if not cell), 0))

    def test_empty_game_is_draw_and_every_opening_is_optimal(self):
        self.assertEqual(solve(EMPTY), (0, tuple(range(9))))

    def test_take_win_and_block_immediate_loss(self):
        win = (1, 1, 0, -1, -1, 0, 0, 0, 0)
        self.assertEqual(solve(win), (1, (2,)))
        block = (1, 1, 0, 0, -1, 0, 0, 0, 0)
        self.assertEqual(current_player(block), -1)
        self.assertEqual(solve(block), (0, (2,)))
        self.assertEqual(action_values(block)[2], 0)
        self.assertTrue(all(value == -1 for i, value in enumerate(action_values(block))
                            if i != 2 and value is not None))

    def test_terminal_value_uses_next_player_perspective(self):
        terminal = (1, 1, 1, -1, -1, 0, 0, 0, 0)
        self.assertEqual(winner(terminal), 1)
        self.assertEqual(current_player(terminal), -1)
        self.assertEqual(solve(terminal), (-1, ()))
        self.assertEqual(action_values(terminal), (None,) * 9)

    def test_illegal_play_and_unreachable_input(self):
        with self.assertRaises(ValueError):
            play(EMPTY, -1)
        with self.assertRaises(ValueError):
            play(EMPTY, True)
        with self.assertRaises(ValueError):
            play(play(EMPTY, 0), 0)
        # Both players have lines, so this board cannot occur in a real game.
        impossible = (1, 1, 1, -1, -1, -1, 0, 0, 0)
        with self.assertRaises(ValueError):
            solve(impossible)
        with self.assertRaises(ValueError):
            play(impossible, 6)
        with self.assertRaises(ValueError):
            winner((0,) * 8)

    def test_all_states_minimax_against_independent_absolute_x_solver(self):
        # Independent utility convention and max/min recurrence.  This
        # deliberately avoids solve, action_values, play, and winner.
        memo = {}

        def absolute(board):
            if board in memo:
                return memo[board]
            lines = [board[a] + board[b] + board[c] for a, b, c in WIN_LINES]
            if 3 in lines:
                result = 1
            elif -3 in lines:
                result = -1
            elif 0 not in board:
                result = 0
            else:
                x_turn = sum(board) == 0
                results = []
                for index, cell in enumerate(board):
                    if cell == 0:
                        child = list(board)
                        child[index] = 1 if x_turn else -1
                        results.append(absolute(tuple(child)))
                result = max(results) if x_turn else min(results)
            memo[board] = result
            return result

        for board in enumerate_states():
            expected = absolute(board) * current_player(board)
            actual, moves = solve(board)
            self.assertEqual(actual, expected)
            if winner(board) is None:
                legal = legal_moves(board)
                scores = action_values(board)
                self.assertEqual(set(moves), {move for move in legal if scores[move] == expected})

    def test_symmetry_orbits_do_not_change_teacher_values(self):
        states = enumerate_states()
        self.assertEqual(len({canonical_key(board) for board in states}), 765)
        for board in states:
            key = canonical_key(board)
            value = solve(board)[0]
            for transformed in symmetries(board):
                self.assertEqual(canonical_key(transformed), key)
                self.assertEqual(solve(transformed)[0], value)


if __name__ == "__main__":
    unittest.main()
