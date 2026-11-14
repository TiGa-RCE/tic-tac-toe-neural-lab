"""Compression contracts and frozen receipts; no learned-model inference."""

import copy
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from verify_export import census, encode, validate_model
from compress_train import D4, np
from build_site import build


ROOT = Path(__file__).resolve().parents[1]


class CompressionEncodingTests(unittest.TestCase):
    def test_raw_relative_encodings_have_the_same_cell_information(self):
        for board in census():
            turn = 1 if sum(board) == 0 else -1
            signed = encode(board, 'relative-signed')
            paired = encode(board, 'relative-two-channel')
            legacy = encode(board, 'relative-one-hot-interleaved')
            self.assertEqual(len(signed), 9)
            self.assertEqual(len(paired), 18)
            self.assertEqual(len(legacy), 27)
            for index, cell in enumerate(board):
                self.assertEqual(signed[index], cell * turn)
                own, opponent = paired[2*index:2*index+2]
                self.assertEqual(signed[index], own-opponent)
                self.assertEqual(legacy[3*index:3*index+3], [own, opponent, 1-own-opponent])

    def test_encoding_uses_the_next_player_perspective(self):
        # X has already opened, so the next player is O.
        board = (1, 0, 0, 0, 0, 0, 0, 0, 0)
        self.assertEqual(encode(board, 'relative-signed'), [-1., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.assertEqual(encode(board, 'relative-two-channel')[:4], [0., 1., 0., 0.])
        self.assertEqual(encode(board)[:6], [0., 1., 0., 0., 0., 1.])
        with self.assertRaises(ValueError):
            encode(board, 'unregistered-feature-map')

    def test_model_shape_and_encoding_contract_reject_mismatches(self):
        # Structural fixtures only: these arrays are never loaded into or run as a network.
        for width, encoding in ((9, 'relative-signed'), (18, 'relative-two-channel'),
                                (27, 'relative-one-hot-interleaved')):
            model = {'architecture': [width, 2, 9], 'encoding': encoding, 'layers': [
                {'weights': [[0., 0.] for _ in range(width)], 'bias': [0., 0.], 'activation': 'tanh'},
                {'weights': [[0.] * 9 for _ in range(2)], 'bias': [0.] * 9, 'activation': 'linear'},
            ]}
            self.assertEqual(validate_model(model), [width, 2, 9])
            for other in ('relative-signed', 'relative-two-channel', 'relative-one-hot-interleaved', 'unknown'):
                if other == encoding:
                    continue
                bad = copy.deepcopy(model)
                bad['encoding'] = other
                with self.assertRaises(ValueError):
                    validate_model(bad)
            bad = copy.deepcopy(model)
            bad['layers'][0]['weights'][0][0] = float('nan')
            with self.assertRaises(ValueError):
                validate_model(bad)
            bad = copy.deepcopy(model)
            bad['layers'][1]['activation'] = 'tanh'
            with self.assertRaises(ValueError):
                validate_model(bad)
            d4 = copy.deepcopy(model)
            d4['symmetry_average'] = 'D4'
            self.assertEqual(validate_model(d4), [width, 2, 9])
            d4['symmetry_average'] = 'unregistered-group'
            with self.assertRaises(ValueError):
                validate_model(d4)

    def test_d4_coordinate_maps_form_the_eight_square_symmetries(self):
        maps = [tuple(permutation) for permutation in D4]
        self.assertEqual(len(maps), 8)
        self.assertEqual(len(set(maps)), 8)
        self.assertEqual(maps[0], tuple(range(9)))
        self.assertEqual(maps[1], (2, 5, 8, 1, 4, 7, 0, 3, 6))
        self.assertEqual(maps[4], (2, 1, 0, 5, 4, 3, 8, 7, 6))
        lines = {frozenset(line) for line in (
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))}
        for permutation in maps:
            self.assertEqual(set(permutation), set(range(9)))
            self.assertEqual(permutation[4], 4)
            self.assertEqual({frozenset(permutation[i] for i in line) for line in lines}, lines)
            for other in maps:
                self.assertIn(tuple(permutation[other[i]] for i in range(9)), maps)
        # Scatter input old→new and gather score new→old are inverse operations.
        cells = list(range(9))
        for permutation in maps:
            transformed = [None] * 9
            for old, new in enumerate(permutation):
                transformed[new] = cells[old]
            self.assertEqual([transformed[permutation[old]] for old in range(9)], cells)

    def test_shared_bias_reference_readout_contract_rejects_invalid_formats(self):
        # Only validate structure; none of these synthetic coefficients are inferred.
        model = {'architecture': [9, 2, 8], 'encoding': 'relative-signed',
                 'reference_logit_zero': True, 'symmetry_average': 'D4',
                 'output_bias_orbits': 'D4-corner-zero', 'logit_round_decimals': 12,
                 'layers': [
                     {'weights': [[0., 0.] for _ in range(9)], 'bias': [0., 0.], 'activation': 'tanh'},
                     {'weights': [[0.] * 8 for _ in range(2)], 'bias': [0., 0.], 'activation': 'linear'}]}
        self.assertEqual(validate_model(model), [9, 2, 8])
        for key, value in [('reference_logit_zero', False), ('reference_logit_zero', 'true'),
                           ('symmetry_average', None), ('output_bias_orbits', 'unregistered-groups'),
                           ('logit_round_decimals', -1), ('logit_round_decimals', 13),
                           ('logit_round_decimals', True), ('logit_round_decimals', 12.)]:
            with self.subTest(key=key, value=value):
                bad = copy.deepcopy(model)
                bad[key] = value
                with self.assertRaises(ValueError):
                    validate_model(bad)
        bad = copy.deepcopy(model)
        bad['layers'][-1]['bias'] = [0.] * 8
        with self.assertRaises(ValueError):
            validate_model(bad)
        bad = copy.deepcopy(model)
        bad['architecture'][-1] = 9
        bad['reference_logit_zero'] = False
        bad['layers'][-1]['weights'] = [[0.] * 9 for _ in range(2)]
        with self.assertRaises(ValueError):
            validate_model(bad)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CompressionArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = []
        for protocol in (ROOT / 'runs').glob('*/compression-protocol.json'):
            status_path = protocol.parent / 'status.json'
            if status_path.exists() and read(status_path).get('status') == 'completed':
                cls.runs.append(protocol.parent)
        if not cls.runs:
            raise unittest.SkipTest('No completed compression run yet')

    def test_selected_size_counts_every_stored_dense_weight_and_bias(self):
        for run_dir in self.runs:
            with self.subTest(run=run_dir.name):
                seal = read(run_dir / 'selection-before-qualification.json')
                status = read(run_dir / 'status.json')
                model = read(run_dir / 'model.json')
                architecture = validate_model(model)
                count = sum((a+1)*b for a, b in zip(architecture, architecture[1:]))
                stored = sum(sum(len(row) for row in layer['weights']) + len(layer['bias'])
                             for layer in model['layers'])
                self.assertEqual(stored, count)
                self.assertEqual(model['parameter_count'], count)
                self.assertEqual(status['parameter_count'], count)
                self.assertEqual(seal['selected']['parameters'], count)
                self.assertEqual(status['compression']['selected_parameters'], count)
                self.assertEqual(model.get('forward_passes_per_move', 1),
                                 8 if model.get('symmetry_average') == 'D4' else 1)
                self.assertLess(count, status['compression']['baseline_parameters'])

    def test_failed_portable_qualifications_cannot_build_a_public_baseline(self):
        for receipt in (ROOT / 'runs').glob('*/portable-verification.json'):
            portable = read(receipt)
            if portable.get('passed') is not False:
                continue
            run_dir = receipt.parent
            status_path = run_dir / 'status.json'
            if not status_path.exists():
                continue
            with self.subTest(run=run_dir.name):
                self.assertNotEqual(read(status_path)['status'], 'completed')
                with tempfile.TemporaryDirectory(dir=ROOT.parent.parent / 'work') as temp:
                    target = Path(temp) / 'unqualified-site'
                    with self.assertRaises(ValueError):
                        build(run_dir, target, 'https://example.invalid/repo', 'https://example.invalid')
                    self.assertFalse(target.exists())

    def test_selected_model_is_smallest_recorded_success_not_a_global_claim(self):
        for run_dir in self.runs:
            with self.subTest(run=run_dir.name):
                protocol = read(run_dir / 'compression-protocol.json')
                trials = read(run_dir / 'search-trials.json')
                selected = read(run_dir / 'selection-before-qualification.json')['selected']
                successes = []
                for trial in trials:
                    self.assertIn(trial['architecture'], protocol['architectures'])
                    self.assertIn(trial['seed'], protocol['seeds'])
                    for phase in trial['phases']:
                        self.assertEqual(phase['architecture'], trial['architecture'])
                        self.assertEqual(phase['parameters'], trial['parameters'])
                        self.assertEqual(phase['no_loss_from_empty'], phase['as_x'] >= 0 and phase['as_o'] >= 0)
                        if phase['no_loss_from_empty']:
                            successes.append(phase)
                self.assertTrue(successes)
                self.assertEqual(selected['parameters'], min(row['parameters'] for row in successes))
                self.assertIn(selected['checkpoint'], {row['checkpoint'] for row in successes})
                self.assertIn('bounded', protocol['minimality_claim'].lower())
                self.assertIn('not an impossibility proof', protocol['minimality_claim'].lower())

    def test_source_and_frozen_selection_qualification_hashes_match(self):
        for run_dir in self.runs:
            with self.subTest(run=run_dir.name):
                protocol_path = run_dir / 'compression-protocol.json'
                protocol = read(protocol_path)
                seal_path = run_dir / 'selection-before-qualification.json'
                seal = read(seal_path)
                receipt = read(run_dir / 'qualification-consumption.json')
                self.assertEqual(sha(run_dir / 'corpus.json'), protocol['corpus_sha256'])
                self.assertEqual(sha(ROOT / 'runs/20261113T233235Z-7e5be6/model.json'), protocol['teacher_sha256'])
                for filename, expected in protocol['source_hashes'].items():
                    self.assertEqual(sha(run_dir / 'source' / filename), expected)
                self.assertEqual(sha(protocol_path), seal['protocol_sha256'])
                self.assertEqual(sha(run_dir / 'model.json'), seal['model_sha256'])
                self.assertEqual(sha(run_dir / (seal['selected']['checkpoint'] + '.json')), seal['model_sha256'])
                self.assertEqual(sha(seal_path), receipt['selection_sha256'])
                self.assertEqual(sha(run_dir / 'independent-verification.json'), receipt['independent_sha256'])
                self.assertEqual(sha(run_dir / 'portable-verification.json'), receipt['portable_sha256'])
                self.assertLessEqual(datetime.fromisoformat(seal['frozen_at']),
                                     datetime.fromisoformat(receipt['consumed_at']))

    def test_literal_schedule_shapes_ranges_and_recorded_weighting(self):
        # These NPZs contain sampled integer indices and weighting receipts, not model parameters.
        for run_dir in self.runs:
            corpus = read(run_dir / 'corpus.json')
            trials = read(run_dir / 'search-trials.json')
            expected = {phase['checkpoint'] for trial in trials for phase in trial['phases']}
            paths = list(run_dir.glob('*-schedule.npz'))
            self.assertEqual({path.name.removesuffix('-schedule.npz') for path in paths}, expected)
            for path in paths:
                with self.subTest(run=run_dir.name, schedule=path.name):
                    with np.load(path, allow_pickle=False) as receipt:
                        indices = receipt['indices']
                        native = receipt['native']
                        critical = receipt['critical']
                        weights = receipt['multiplicity']
                        self.assertEqual(indices.shape, (512, 128))
                        self.assertTrue(np.issubdtype(indices.dtype, np.integer))
                        self.assertTrue(bool(((indices >= 0) & (indices < 4520)).all()))
                        self.assertEqual(native.ndim, 2)
                        self.assertEqual(native.shape[1], 2)
                        self.assertTrue(bool(((native[:, 0] >= 0) & (native[:, 0] < 4520)).all()))
                        self.assertTrue(bool(((native[:, 1] >= 0) & (native[:, 1] < 9)).all()))
                        for state, move in native:
                            self.assertEqual(corpus['boards'][int(state)][int(move)], 0)
                        self.assertTrue(bool(((critical >= 0) & (critical < 4520)).all()))
                        self.assertEqual(len(set(map(int, critical))), len(critical))
                        self.assertEqual(weights.shape, (4520,))
                        self.assertTrue(bool(np.isfinite(weights).all()))
                        self.assertTrue(bool((weights >= 1).all()))
                        recorded = Counter(map(int, native[:, 0]))
                        remainder = weights.copy() - 1
                        for state, count in recorded.items():
                            remainder[state] -= count
                        remainder[critical.astype(int)] -= 100
                        self.assertTrue(bool((remainder >= 0).all()))
                        self.assertTrue(bool((remainder % 20 == 0).all()))
                        if len(critical):
                            self.assertTrue(set(map(int, indices[:32, :32].flat)) <= set(map(int, critical)))

    def test_independent_no_loss_and_browser_parity_are_separate_from_full_optimality(self):
        for run_dir in self.runs:
            with self.subTest(run=run_dir.name):
                independent = read(run_dir / 'independent-verification.json')
                selected = read(run_dir / 'selection-before-qualification.json')['selected']
                portable = read(run_dir / 'portable-verification.json')
                self.assertEqual(independent['reachable_count'], 5478)
                self.assertEqual(independent['total'], 4520)
                self.assertEqual(independent['optimal_count'], selected['optimal_count'])
                self.assertEqual(independent['perfect_policy'], independent['optimal_count'] == 4520)
                self.assertTrue(independent['no_loss_from_empty'])
                for side in ('as_x', 'as_o'):
                    self.assertEqual(independent[side]['value'], 0)
                    self.assertEqual(independent[side]['result'], 'draw')
                fixtures = independent['fixtures']
                self.assertEqual(len(fixtures), 4520)
                self.assertEqual(sum(row['move'] in row['optimal_moves'] for row in fixtures),
                                 independent['optimal_count'])
                self.assertTrue(all(row['board'][row['move']] == 0 for row in fixtures))
                self.assertEqual(portable['positions'], 4520)
                self.assertEqual(portable['action_mismatches'], 0)
                self.assertEqual(portable['optimal_count'], independent['optimal_count'])
                self.assertEqual(portable['tolerance'], 1e-9)
                self.assertLessEqual(portable['max_logit_error'], portable['tolerance'])
                self.assertTrue(portable['passed'])
                self.assertTrue(read(run_dir / 'evaluation.json')['no_solver_fallback'])


class DerivedCompressionArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = []
        for protocol in (ROOT / 'runs').glob('*/transformation-before-qualification.json'):
            path = protocol.parent / 'status.json'
            source = ROOT / 'runs' / read(protocol)['source_run']
            # This class certifies the historical compression derivation.
            # Full-domain perfect exports have a separate protocol and tests.
            if (path.exists() and read(path).get('status') == 'completed'
                    and (source / 'compression-protocol.json').exists()):
                cls.runs.append(protocol.parent)
        if not cls.runs:
            raise unittest.SkipTest('No independently qualified derived compression export yet')

    def test_derived_export_binds_source_lineage_frozen_model_and_qualification(self):
        for run_dir in self.runs:
            with self.subTest(run=run_dir.name):
                protocol_path = run_dir / 'transformation-before-qualification.json'
                protocol = read(protocol_path)
                source = ROOT / 'runs' / protocol['source_run']
                receipt = read(run_dir / 'qualification-consumption.json')
                self.assertEqual(sha(source / protocol['source_model']), protocol['source_model_sha256'])
                self.assertEqual(sha(source / 'compression-protocol.json'), protocol['source_protocol_sha256'])
                self.assertEqual(sha(protocol_path), receipt['transformation_sha256'])
                self.assertEqual(sha(run_dir / 'model.json'), receipt['model_sha256'])
                self.assertEqual(sha(run_dir / 'independent-verification.json'), receipt['independent_sha256'])
                self.assertEqual(sha(run_dir / 'portable-verification.json'), receipt['portable_sha256'])
                self.assertLessEqual(datetime.fromisoformat(protocol['registered_at']),
                                     datetime.fromisoformat(receipt['consumed_at']))
                self.assertEqual(protocol['precision_order'], ['float16', 'float32', 'float64'])
                self.assertIn('floor', protocol['policy_score_rule'])
                self.assertIn('lowest legal cell index', protocol['policy_score_rule'])
                # Derivation must not rewrite the original failed strict comparison.
                old_portable = read(source / 'portable-verification.json')
                self.assertFalse(old_portable['passed'])
                self.assertGreater(old_portable['action_mismatches'], 0)
                self.assertEqual(read(source / 'status.json')['status'], 'failed')

    def test_stored_coefficients_and_binary16_payload_match_exactly(self):
        for run_dir in self.runs:
            with self.subTest(run=run_dir.name):
                model = read(run_dir / 'model.json')
                self.assertEqual(validate_model(model), [9, 2, 8])
                self.assertTrue(model['reference_logit_zero'])
                self.assertEqual(model['symmetry_average'], 'D4')
                self.assertEqual(model['output_bias_orbits'], 'D4-corner-zero')
                self.assertEqual(model['logit_round_decimals'], 12)
                self.assertEqual(model['coefficient_precision'], 'float16')
                self.assertEqual(model['forward_passes_per_move'], 8)
                coefficients = []
                for layer in model['layers']:
                    coefficients.extend(value for row in layer['weights'] for value in row)
                    coefficients.extend(layer['bias'])
                self.assertEqual(len(coefficients), 38)
                self.assertEqual(model['parameter_count'], 38)
                self.assertEqual([len(layer['bias']) for layer in model['layers']], [2, 2])
                payload = (run_dir / 'coefficients.bin').read_bytes()
                self.assertEqual(len(payload), 76)
                self.assertEqual(payload, struct.pack('<38e', *coefficients))
                self.assertEqual(list(struct.unpack('<38e', payload)), coefficients)
                status = read(run_dir / 'status.json')
                self.assertEqual(status['parameter_count'], 38)
                self.assertNotIn('error', status)
                self.assertEqual(status['compression']['coefficient_payload_bytes'], 76)

    def test_exact_selected_policy_is_no_loss_and_strictly_portable(self):
        for run_dir in self.runs:
            with self.subTest(run=run_dir.name):
                model = read(run_dir / 'model.json')
                independent = read(run_dir / 'independent-verification.json')
                portable = read(run_dir / 'portable-verification.json')
                attempts = read(run_dir / 'transformation-attempts.json')
                selected = next(row for row in attempts if row['checkpoint'] == model['checkpoint'])
                self.assertEqual(selected['model_sha256'], sha(run_dir / 'model.json'))
                self.assertTrue(selected['no_loss_from_empty'])
                self.assertEqual(selected['parameters'], 38)
                self.assertEqual(selected['precision'], 'float16')
                self.assertEqual(independent['parameter_count'], 38)
                self.assertEqual(independent['total'], 4520)
                self.assertEqual(independent['optimal_count'], selected['optimal_count'])
                self.assertFalse(independent['perfect_policy'])
                self.assertTrue(independent['no_loss_from_empty'])
                self.assertEqual(independent['forward_passes_per_move'], 8)
                for side in ('as_x', 'as_o'):
                    self.assertEqual(independent[side]['result'], 'draw')
                    self.assertEqual(independent[side]['value'], 0)
                self.assertEqual(portable['positions'], 4520)
                self.assertEqual(portable['action_mismatches'], 0)
                self.assertEqual(portable['optimal_count'], independent['optimal_count'])
                self.assertEqual(portable['tolerance'], 1e-9)
                self.assertLessEqual(portable['max_logit_error'], portable['tolerance'])
                self.assertTrue(portable['passed'])
                self.assertTrue(read(run_dir / 'evaluation.json')['no_solver_fallback'])
if __name__ == '__main__':
    unittest.main()
