"""Read-only acceptance checks for compact policies certified optimal everywhere.

These tests inspect frozen data and use the independent symbolic oracle. They
do not load a numerical neural backend or perform learned-network inference.
"""
import hashlib
from datetime import datetime
import copy
import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import build_site

from verify_export import census, exact_moves, outcome, validate_model


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CompactPerfectPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_dirs = []
        for receipt in (ROOT / 'runs').glob('*/independent-verification.json'):
            folder = receipt.parent
            if not (folder / 'model.json').exists() or not (folder / 'status.json').exists():
                continue
            status = read(folder / 'status.json')
            # Select by the declared artifact family, not by the proof fields
            # these tests must independently check.
            if (status.get('training_mode') == 'finite-domain-perfect-export'
                    and status.get('status') == 'completed'):
                cls.run_dirs.append(folder)
        if not cls.run_dirs:
            raise unittest.SkipTest('No completed compact all-board-perfect policy yet')
        cls.expected_boards = {board for board in census() if outcome(board) is None}
        if len(cls.expected_boards) != 4520:
            raise AssertionError('Independent census did not yield 4,520 decision boards')

    def test_independent_verdict_requires_every_position_optimal(self):
        for folder in self.run_dirs:
            with self.subTest(run=folder.name):
                report = read(folder / 'independent-verification.json')
                self.assertEqual(report['total'], 4520)
                self.assertEqual(report['optimal_count'], 4520)
                self.assertEqual(report['optimal_rate'], 1.)
                self.assertEqual(report['mistake_count'], 0)
                self.assertEqual(report['mistakes'], [])
                self.assertIs(report['perfect_policy'], True)
                self.assertIs(report['no_loss_from_empty'], True)
                self.assertGreaterEqual(report['as_x']['value'], 0)
                self.assertGreaterEqual(report['as_o']['value'], 0)

    def test_full_fixtures_cover_the_domain_with_independently_optimal_moves(self):
        for folder in self.run_dirs:
            with self.subTest(run=folder.name):
                fixtures = read(folder / 'independent-verification.json')['fixtures']
                self.assertEqual(len(fixtures), 4520)
                boards = [tuple(row['board']) for row in fixtures]
                self.assertEqual(len(set(boards)), 4520)
                self.assertEqual(set(boards), self.expected_boards)
                for row, board in zip(fixtures, boards):
                    self.assertEqual(row['optimal_moves'], exact_moves(board))
                    self.assertIn(row['move'], row['optimal_moves'])
                    self.assertEqual(board[row['move']], 0)

    def test_browser_pass_is_exact_in_actions_and_keeps_fixed_tolerance(self):
        for folder in self.run_dirs:
            with self.subTest(run=folder.name):
                portable = read(folder / 'portable-verification.json')
                self.assertIs(portable['passed'], True)
                self.assertEqual(portable['positions'], 4520)
                self.assertEqual(portable['optimal_count'], 4520)
                self.assertEqual(portable['action_mismatches'], 0)
                self.assertEqual(portable['tolerance'], 1e-9)
                self.assertLessEqual(portable['max_logit_error'], portable['tolerance'])

    def test_parameter_count_includes_every_stored_learned_coefficient(self):
        for folder in self.run_dirs:
            with self.subTest(run=folder.name):
                model = read(folder / 'model.json')
                validate_model(model)
                count = sum(sum(len(row) for row in layer['weights']) + len(layer['bias'])
                            for layer in model['layers'])
                self.assertEqual(model['parameter_count'], count)
                self.assertLess(count, 6537)
                self.assertEqual(read(folder / 'independent-verification.json')['parameter_count'], count)
                status = read(folder / 'status.json')
                self.assertEqual(status['parameter_count'], count)
                self.assertEqual(status['compression']['selected_parameters'], count)
                self.assertEqual(status['compression']['selected_checkpoint'], model['checkpoint'])

    def test_declared_binary_coefficient_payload_matches_export(self):
        types = {'float16': 'e', 'float32': 'f', 'float64': 'd'}
        for folder in self.run_dirs:
            binary = folder / 'coefficients.bin'
            if not binary.exists():
                continue
            with self.subTest(run=folder.name):
                model = read(folder / 'model.json')
                kind = model['coefficient_precision']
                self.assertIn(kind, types)
                values = [value for layer in model['layers']
                          for value in [v for row in layer['weights'] for v in row] + layer['bias']]
                expected = struct.pack('<' + types[kind] * len(values), *values)
                self.assertEqual(binary.read_bytes(), expected)

    def test_qualification_is_bound_to_frozen_model_source_and_transform(self):
        for folder in self.run_dirs:
            with self.subTest(run=folder.name):
                protocol_path = folder / 'transformation-before-qualification.json'
                protocol = read(protocol_path)
                receipt = read(folder / 'qualification-consumption.json')
                source = ROOT / 'runs' / protocol['source_run']
                self.assertEqual(source.resolve().parent, (ROOT / 'runs').resolve())
                self.assertEqual(sha(source / protocol['source_model']), protocol['source_model_sha256'])
                self.assertEqual(sha(source / 'perfect-protocol.json'), protocol['source_protocol_sha256'])
                for filename, expected in protocol['source_hashes'].items():
                    self.assertEqual(sha(folder / 'source' / filename), expected)
                self.assertEqual(sha(protocol_path), receipt['transformation_sha256'])
                self.assertEqual(sha(folder / 'model.json'), receipt['model_sha256'])
                self.assertEqual(sha(folder / 'independent-verification.json'), receipt['independent_sha256'])
                self.assertEqual(sha(folder / 'portable-verification.json'), receipt['portable_sha256'])
                self.assertLessEqual(datetime.fromisoformat(protocol['registered_at']),
                                     datetime.fromisoformat(receipt['consumed_at']))
                self.assertIn('not a proof of global minimality', protocol['minimality_scope'])
                seal = read(source / 'selection-before-qualification.json')
                selection = seal['selected']
                self.assertTrue(selection['perfect_policy'])
                self.assertEqual(selection['optimal_count'], 4520)
                self.assertEqual(protocol['source_model'], selection['checkpoint'] + '.json')
                self.assertEqual(seal['model_sha256'], protocol['source_model_sha256'])
                self.assertEqual(seal['protocol_sha256'], protocol['source_protocol_sha256'])
                self.assertLessEqual(datetime.fromisoformat(seal['frozen_at']),
                                     datetime.fromisoformat(protocol['registered_at']))

    def test_first_successful_precision_is_selected_with_no_relaxed_gate(self):
        for folder in self.run_dirs:
            with self.subTest(run=folder.name):
                protocol = read(folder / 'transformation-before-qualification.json')
                attempts = read(folder / 'transformation-attempts.json')
                model = read(folder / 'model.json')
                self.assertTrue(attempts)
                self.assertEqual([row['precision'] for row in attempts],
                                 protocol['precision_order'][:len(attempts)])
                qualified = [row for row in attempts if row['perfect_policy']
                             and row['optimal_count'] == 4520
                             and row['no_loss_from_empty'] and row['portable']['passed']]
                self.assertEqual(len(qualified), 1)
                selected = qualified[0]
                self.assertIs(selected, attempts[-1])
                self.assertEqual(selected['checkpoint'], model['checkpoint'])
                self.assertEqual(selected['precision'], model['coefficient_precision'])
                self.assertEqual(selected['model_sha256'], sha(folder / 'model.json'))
                self.assertEqual(selected['portable']['tolerance'], 1e-9)
                self.assertEqual(selected['portable']['action_mismatches'], 0)

    def test_public_perfect_choice_refuses_incomplete_optimality_or_browser_proof(self):
        # Mutate small temporary receipts, never the frozen run or public site.
        baseline = ROOT / 'runs/compression-20261113-selected'
        selected = self.run_dirs[0]
        base_files = {name: read(baseline / name) for name in (
            'model.json', 'status.json', 'evaluation.json',
            'independent-verification.json', 'portable-verification.json')}
        perfect_files = {name: read(selected / name) for name in (
            'model.json', 'independent-verification.json', 'portable-verification.json')}
        for files in (base_files, perfect_files):
            files['independent-verification.json'].pop('fixtures', None)
        cases = [
            ('independent-verification.json', 'perfect_policy', False),
            ('independent-verification.json', 'optimal_count', 4519),
            ('independent-verification.json', 'total', 4519),
            ('portable-verification.json', 'action_mismatches', 1),
            ('portable-verification.json', 'optimal_count', 4519),
        ]
        work = ROOT.parents[1] / 'work'
        with tempfile.TemporaryDirectory(dir=work, prefix='perfect-catalog-test-') as temp:
            project = Path(temp)
            (project / 'web').mkdir()
            published = project / 'runs/published'
            candidate = project / 'runs/candidate'
            published.mkdir(parents=True)
            candidate.mkdir()
            for name, value in base_files.items():
                (published / name).write_text(json.dumps(value))
            (project / 'runs/perfect-optimal-selection.json').write_text(
                json.dumps({'run_id': 'candidate'}))
            with patch.object(build_site, 'ROOT', project), contextlib.redirect_stdout(io.StringIO()):
                for filename, field, value in cases:
                    with self.subTest(receipt=filename, field=field):
                        altered = copy.deepcopy(perfect_files)
                        altered[filename][field] = value
                        for name, document in altered.items():
                            (candidate / name).write_text(json.dumps(document))
                        with self.assertRaisesRegex(ValueError, 'compact perfect choice'):
                            build_site.build(published, project / 'site',
                                             'https://example.invalid/repo', 'https://example.invalid')

                # A registry cannot evade its full-domain gate by aliasing the
                # already published, non-losing but imperfect compact model.
                (project / 'runs/perfect-optimal-selection.json').write_text(
                    json.dumps({'run_id': 'published'}))
                with self.assertRaisesRegex(ValueError, 'compact perfect choice'):
                    build_site.build(published, project / 'site',
                                     'https://example.invalid/repo', 'https://example.invalid')


class HistoricalCompactChampionTests(unittest.TestCase):
    def test_original_38_coefficient_model_is_preserved_byte_for_byte(self):
        path = ROOT / 'runs/compression-20261113-selected/model.json'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         '9b3b98f4b995c0c434bdd248ca2042844485379d0f999751b7e0f40296277632')


if __name__ == '__main__':
    unittest.main()
