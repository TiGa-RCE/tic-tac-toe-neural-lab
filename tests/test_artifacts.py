"""Audit persisted experiment boundaries without loading model tensors.

The latest completed run with a test-consumption receipt is audited.  Earlier
development runs lacking that receipt are explicitly skipped, never silently
treated as qualified.  These checks read JSON and hash files only; they do
not import training code, instantiate networks, or load NPZ checkpoints.
"""

from datetime import datetime
from hashlib import sha256
import itertools
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def independent_orbit_key(board):
    """Geometric D4 orbit calculated without the project's solver."""
    variants = []
    for mirror in (False, True):
        for turn in range(4):
            variant = [0] * 9
            for source, value in enumerate(board):
                x, y = source % 3, source // 3
                if mirror:
                    x = 2 - x
                for _ in range(turn):
                    x, y = 2 - y, x
                variant[y * 3 + x] = value
            variants.append(tuple(variant))
    return ''.join(str(cell + 1) for cell in min(variants))


class ExperimentArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        candidates = []
        for run in (ROOT / 'runs').glob('*'):
            status_path = run / 'status.json'
            receipt_path = run / 'test-consumption.json'
            if not run.is_dir() or not status_path.is_file() or not receipt_path.is_file():
                continue
            status = read_json(status_path)
            if status.get('status') == 'completed':
                candidates.append((receipt_path.stat().st_mtime_ns, run))
        if not candidates:
            raise unittest.SkipTest('No completed run with a test-consumption receipt is available')
        cls.run_dir = max(candidates)[1]
        cls.status = read_json(cls.run_dir / 'status.json')
        cls.protocol = read_json(cls.run_dir / 'protocol-before-training.json')
        cls.corpus = read_json(cls.run_dir / 'corpus.json')
        cls.parts = {name: set(indices) for name, indices in cls.corpus['partitions'].items()}

    def test_four_partitions_are_disjoint_geometric_orbits(self):
        boards = self.corpus['boards']
        self.assertEqual(len(boards), 4520)
        self.assertEqual(len({tuple(board) for board in boards}), 4520)
        self.assertEqual(set(self.parts), {'train', 'meta', 'validation', 'test'})
        self.assertEqual(set.union(*self.parts.values()), set(range(4520)))
        for name, indices in self.corpus['partitions'].items():
            with self.subTest(partition=name):
                self.assertEqual(len(indices), len(set(indices)))
                self.assertTrue(indices)
                self.assertTrue(all(type(index) is int and 0 <= index < 4520 for index in indices))
        for (a, left), (b, right) in itertools.combinations(self.parts.items(), 2):
            self.assertFalse(left & right, f'Partitions {a} and {b} share state indices')
        keys = [independent_orbit_key(board) for board in boards]
        self.assertEqual(keys, self.corpus['orbit_keys'])
        self.assertEqual(len(set(keys)), 627)
        orbit_sets = {name: {keys[index] for index in indices} for name, indices in self.parts.items()}
        for (a, left), (b, right) in itertools.combinations(orbit_sets.items(), 2):
            self.assertFalse(left & right, f'Partitions {a} and {b} share a D4 orbit')
        membership = self.corpus['orbit_membership']
        self.assertEqual(set(membership), set(keys))
        for name, orbits in orbit_sets.items():
            self.assertTrue(all(membership[key] == name for key in orbits))

    def assert_train_schedule(self, rows, updates):
        self.assertEqual(len(rows), updates)
        for row_index, row in enumerate(rows):
            self.assertEqual(len(row), 128, f'Wrong batch width at update {row_index}')
            self.assertTrue(all(type(index) is int and index in self.parts['train'] for index in row),
                            f'Non-TRAIN index at update {row_index}')

    def test_literal_founder_and_nine_phase_schedules_use_train_only(self):
        self.assertEqual(self.protocol['founder_updates'], 1024)
        self.assertEqual(self.protocol['branch_phases'], 3)
        self.assertEqual(self.protocol['updates_per_phase'], 512)
        self.assertEqual(self.protocol['batch_size'], 128)
        self.assertEqual(self.protocol['branch_seeds'], [11, 23, 37])
        self.assert_train_schedule(read_json(self.run_dir / 'founder-schedule.json'), 1024)
        expected = {f's{seed}-p{phase}-schedule.json' for seed in (11, 23, 37) for phase in (1, 2, 3)}
        actual = {path.name for path in self.run_dir.glob('s*-p*-schedule.json')}
        self.assertEqual(actual, expected)
        for filename in sorted(expected):
            with self.subTest(schedule=filename):
                schedule = read_json(self.run_dir / filename)
                self.assert_train_schedule(schedule['indices'], 512)
                self.assertEqual(set(schedule['native_pool']), self.parts['train'])
                self.assertEqual(len(schedule['native_pool']), len(schedule['multiplicity']))
                self.assertTrue(all(type(n) is int and n > 0 for n in schedule['multiplicity']))
                self.assertTrue(all(type(index) is int and index in self.parts['train']
                                    for index in schedule['target_errors']))

    def test_native_traces_stop_at_partition_boundaries(self):
        expected = {f's{seed}-p{phase}-native.json' for seed in (11, 23, 37) for phase in (1, 2, 3)}
        self.assertEqual({path.name for path in self.run_dir.glob('s*-p*-native.json')}, expected)
        for filename in sorted(expected):
            with self.subTest(trace=filename):
                records = read_json(self.run_dir / filename)
                self.assertTrue(records, 'Native collection must preserve its observed actions')
                for record in records:
                    index, move = record['state'], record['move']
                    self.assertIs(type(index), int)
                    self.assertIn(index, self.parts['train'])
                    self.assertIs(record['retained_for_training'], True)
                    self.assertIs(type(move), int)
                    self.assertIn(move, range(9))
                    self.assertEqual(self.corpus['boards'][index][move], 0)

    def test_selection_and_test_consumption_are_hash_bound_and_ordered(self):
        selection_path = self.run_dir / 'selection-before-test.json'
        selection = read_json(selection_path)
        receipt = read_json(self.run_dir / 'test-consumption.json')
        selected = selection['selected']['checkpoint']
        self.assertEqual(selection['model_sha256'], digest(self.run_dir / f'{selected}.json'))
        self.assertEqual(selection['model_sha256'], digest(self.run_dir / 'model.json'))
        self.assertEqual(selection['protocol_sha256'], digest(self.run_dir / 'protocol-before-training.json'))
        self.assertIs(selection['test_consumed'], False)
        self.assertEqual(receipt['selection_sha256'], digest(selection_path))
        self.assertEqual(receipt['evaluation_file'], 'test-evaluation.json')
        evaluation_path = self.run_dir / receipt['evaluation_file']
        self.assertEqual(receipt['evaluation_sha256'], digest(evaluation_path))
        self.assertIs(receipt['retuned_after_test'], False)
        self.assertIs(receipt['fresh_independent_campaign'], False)
        sealed_at = datetime.fromisoformat(selection['sealed_at'])
        consumed_at = datetime.fromisoformat(receipt['consumed_at'])
        self.assertIsNotNone(sealed_at.tzinfo)
        self.assertIsNotNone(consumed_at.tzinfo)
        self.assertLess(sealed_at, consumed_at)
        immutable_evaluation = read_json(evaluation_path)
        self.assertEqual(immutable_evaluation['selected_checkpoint'], selected)
        self.assertEqual(immutable_evaluation['test']['count'], len(self.parts['test']))
        self.assertEqual(immutable_evaluation['exhaustive']['count'], 4520)
        self.assertEqual(self.status['selection']['checkpoint'], selected)
        self.assertEqual(self.status['model_version'], selected)

    def test_corpus_and_every_source_snapshot_match_pretraining_protocol(self):
        self.assertEqual(self.protocol['corpus_sha256'], digest(self.run_dir / 'corpus.json'))
        source_hashes = self.protocol['source_hashes']
        self.assertTrue(source_hashes)
        self.assertTrue({'train.py', 'solver.py', 'verify_export.py'} <= set(source_hashes))
        snapshot = self.run_dir / 'source'
        actual = {str(path.relative_to(snapshot)) for path in snapshot.rglob('*') if path.is_file()}
        self.assertEqual(actual, set(source_hashes))
        for relative, expected in source_hashes.items():
            with self.subTest(source=relative):
                path = Path(relative)
                self.assertFalse(path.is_absolute())
                self.assertNotIn('..', path.parts)
                self.assertEqual(digest(snapshot / path), expected)


if __name__ == '__main__':
    unittest.main()
