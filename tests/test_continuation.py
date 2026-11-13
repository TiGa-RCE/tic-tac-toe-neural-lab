"""Read-only receipts for the finite-domain continuation (no model inference)."""

from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ContinuationArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runs = []
        for receipt in (ROOT / 'runs').glob('*/qualification-consumption.json'):
            status_path = receipt.parent / 'status.json'
            if status_path.exists():
                status = read(status_path)
                if status.get('training_mode') == 'finite-domain-continuation' and status.get('status') == 'completed':
                    runs.append(receipt.parent)
        if not runs:
            raise unittest.SkipTest('No completed finite-domain continuation receipt yet')
        cls.run_dir = max(runs, key=lambda path: path.stat().st_mtime_ns)
        cls.protocol = read(cls.run_dir / 'protocol-before-training.json')
        cls.corpus = read(cls.run_dir / 'corpus.json')

    def test_complete_domain_is_explicitly_train_without_heldout_claim(self):
        p, corpus = self.protocol, self.corpus
        self.assertEqual(p['training_mode'], 'finite-domain-continuation')
        self.assertTrue(p['all_decision_states_promoted_to_train'])
        self.assertTrue(p['previous_test_labels_used_for_training'])
        self.assertFalse(p['held_out_generalization_claim'])
        self.assertEqual(p['architecture'], [27, 64, 64, 9])
        self.assertEqual(p['parameters'], 6537)
        self.assertEqual(len(corpus['boards']), 4520)
        self.assertEqual(len({tuple(board) for board in corpus['boards']}), 4520)
        self.assertEqual(corpus['partitions']['train'], list(range(4520)))
        for name in ('meta', 'validation', 'test'):
            self.assertEqual(corpus['partitions'][name], [])
        self.assertEqual(set(corpus['orbit_membership'].values()), {'train'})

    def test_native_multiplicity_and_literal_schedules_are_reconstructible(self):
        p = self.protocol
        self.assertEqual(p['branch_seeds'], [11, 23, 37])
        self.assertEqual(p['updates_per_phase'], 512)
        self.assertEqual(p['batch_size'], 128)
        self.assertEqual(p['gradient_clip'], 1)
        self.assertEqual(p['retention_coefficient'], .1)
        self.assertEqual(p['temperature'], 2)
        counts = {seed: Counter({index: 1 for index in range(4520)}) for seed in p['branch_seeds']}
        paths = list(self.run_dir.glob('s*-p*-schedule.json'))
        self.assertGreaterEqual(len(paths), 3)
        self.assertEqual(len(paths) % 3, 0)
        self.assertLessEqual(len(paths), p['max_phases'] * 3)
        order = lambda path: tuple(map(int, re.match(r's(\d+)-p(\d+)-schedule.json', path.name).groups()))
        for path in sorted(paths, key=lambda path: (order(path)[1], order(path)[0])):
            seed, phase = order(path)
            with self.subTest(seed=seed, phase=phase):
                trace = read(path.with_name(f's{seed}-p{phase}-native.json'))
                for entry in trace:
                    index = entry['state']
                    self.assertTrue(0 <= index < 4520)
                    self.assertIs(entry['retained_for_training'], True)
                    self.assertEqual(self.corpus['boards'][index][entry['move']], 0)
                    counts[seed][index] += 1
                schedule = read(path)
                self.assertEqual(schedule['native_pool'], list(range(4520)))
                self.assertEqual(schedule['multiplicity'], [counts[seed][index] for index in range(4520)])
                rows = schedule['indices']
                self.assertEqual(len(rows), 512)
                targets = set(schedule['target_errors']) | set(schedule['critical'])
                self.assertTrue(all(0 <= value < 4520 for value in targets))
                for index, row in enumerate(rows):
                    self.assertEqual(len(row), 128)
                    self.assertTrue(all(type(value) is int and 0 <= value < 4520 for value in row))
                    if targets and index < 32:
                        self.assertTrue(set(row[:32]) <= targets)

    def test_parent_corpus_and_source_snapshots_are_hash_bound(self):
        p = self.protocol
        parent = ROOT / 'runs' / p['parent_run'] / p['parent_checkpoint']
        self.assertEqual(sha(parent), p['parent_sha256'])
        self.assertEqual(sha(self.run_dir / 'corpus.json'), p['corpus_sha256'])
        for filename, expected in p['source_hashes'].items():
            with self.subTest(filename=filename):
                self.assertEqual(sha(self.run_dir / 'source' / filename), expected)

    def test_selected_neural_export_and_qualification_receipts_match(self):
        seal_path = self.run_dir / 'selection-before-qualification.json'
        receipt = read(self.run_dir / 'qualification-consumption.json')
        seal = read(seal_path)
        checkpoint_path = self.run_dir / f'{seal["selected"]["checkpoint"]}.json'
        self.assertEqual(sha(checkpoint_path), seal['model_sha256'])
        self.assertEqual(sha(self.run_dir / 'model.json'), seal['model_sha256'])
        self.assertEqual(sha(self.run_dir / 'protocol-before-training.json'), seal['protocol_sha256'])
        self.assertEqual(sha(seal_path), receipt['selection_sha256'])
        self.assertEqual(sha(self.run_dir / 'independent-verification.json'), receipt['independent_sha256'])
        self.assertEqual(sha(self.run_dir / 'portable-verification.json'), receipt['portable_sha256'])
        self.assertLessEqual(datetime.fromisoformat(seal['frozen_at']), datetime.fromisoformat(receipt['consumed_at']))
        self.assertFalse(seal['independent_qualification_consumed'])
        self.assertTrue(receipt['independent_no_loss'])
        self.assertTrue(receipt['independent_perfect_policy'])

    def test_independent_full_domain_and_both_side_verdicts(self):
        independent = read(self.run_dir / 'independent-verification.json')
        self.assertEqual(independent['reachable_count'], 5478)
        self.assertEqual(independent['total'], 4520)
        self.assertEqual(independent['optimal_count'], 4520)
        self.assertEqual(independent['mistake_count'], 0)
        self.assertTrue(independent['perfect_policy'])
        self.assertTrue(independent['no_loss_from_empty'])
        for side in ('as_x', 'as_o'):
            self.assertEqual(independent[side]['value'], 0)
            self.assertEqual(independent[side]['result'], 'draw')
        fixtures = independent['fixtures']
        self.assertEqual(len(fixtures), 4520)
        self.assertEqual({tuple(row['board']) for row in fixtures}, {tuple(board) for board in self.corpus['boards']})
        for row in fixtures:
            self.assertIn(row['move'], row['optimal_moves'])
            self.assertEqual(row['board'][row['move']], 0)

    def test_browser_parity_uses_fixed_tolerance_and_no_action_mismatch(self):
        portable = read(self.run_dir / 'portable-verification.json')
        self.assertEqual(portable['positions'], 4520)
        self.assertEqual(portable['action_mismatches'], 0)
        self.assertEqual(portable['optimal_count'], 4520)
        self.assertEqual(portable['tolerance'], 1e-9)
        self.assertLessEqual(portable['max_logit_error'], portable['tolerance'])
        self.assertTrue(portable['passed'])
        evaluation = read(self.run_dir / 'evaluation.json')
        self.assertTrue(evaluation['no_solver_fallback'])
        self.assertGreater(evaluation['minimum_optimal_vs_suboptimal_logit_margin'], 2e-9)


if __name__ == '__main__':
    unittest.main()
