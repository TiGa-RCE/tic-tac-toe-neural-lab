"""Sparse coefficient codec and frozen qualification evidence, without NN inference."""
import copy
import contextlib
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import build_site

from verify_export import (census, exact_moves, outcome, stored_coefficient,
                           validate_model)

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slots(model):
    return [value for layer in model['layers']
            for value in [v for row in layer['weights'] for v in row]+layer['bias']]


def synthetic_model():
    # Pure invented constants exercise shape/codec validation, never a forward
    # pass or a learned model. Two slots are structural, so 18 are numeric.
    model = {'architecture': [9, 1, 8], 'encoding': 'relative-signed',
             'symmetry_average': 'D4', 'reference_logit_zero': True,
             'output_bias_orbits': 'D4-corner-zero',
             'coefficient_storage': 'sparse-fixed-v1', 'parameter_count': 18,
             'layers': [
                 {'weights': [[1.] for _ in range(9)], 'bias': [1.], 'activation': 'tanh'},
                 {'weights': [[1.]*8], 'bias': [1., 1.], 'activation': 'linear'},
             ]}
    model['layers'][0]['weights'][0][0] = None
    model['layers'][1]['weights'][0][0] = 'fixed:+1'
    return model


class SparseCodecTests(unittest.TestCase):
    def test_only_explicit_sparse_storage_accepts_zero_and_unit_markers(self):
        for value, expected in ((None, 0.), ('fixed:+1', 1.), ('fixed:-1', -1.)):
            self.assertEqual(stored_coefficient(value, 'sparse-fixed-v1'), expected)
            with self.assertRaises(ValueError):
                stored_coefficient(value)
        for value in (0., 1., -2.5):
            self.assertEqual(stored_coefficient(value), value)
            self.assertEqual(stored_coefficient(value, 'sparse-fixed-v1'), value)
        for value in (True, False, float('inf'), float('nan'), 'fixed:0',
                      'fixed:+2', 'fixed:1', '1', {}, []):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                stored_coefficient(value, 'sparse-fixed-v1')
        with self.assertRaises(ValueError):
            stored_coefficient(0., 'sparse-fixed-v2')

    def test_validation_counts_numeric_values_and_rejects_bad_schema_or_count(self):
        model = synthetic_model()
        self.assertEqual(validate_model(model), [9, 1, 8])
        for count in (17, 19, None, '18'):
            broken = copy.deepcopy(model)
            broken['parameter_count'] = count
            with self.subTest(count=count), self.assertRaises(ValueError):
                validate_model(broken)
        broken = copy.deepcopy(model)
        broken.pop('coefficient_storage')
        with self.assertRaises(ValueError):
            validate_model(broken)
        broken = copy.deepcopy(model)
        broken['coefficient_storage'] = 'unknown'
        with self.assertRaises(ValueError):
            validate_model(broken)
        for value in ('fixed:+2', True, float('inf'), {}):
            broken = copy.deepcopy(model)
            broken['layers'][0]['weights'][0][0] = value
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                validate_model(broken)

    def test_browser_constant_decoder_matches_the_independent_codec(self):
        if shutil.which('node') is None:
            self.skipTest('Node is required for the real browser coefficient decoder')
        script = r"""
import assert from 'node:assert/strict';
import {coefficientValue} from './web/network.js';
for(const [v,want] of [[null,0],['fixed:+1',1],['fixed:-1',-1],[0,0],[2.5,2.5]])
  assert.equal(coefficientValue(v,'sparse-fixed-v1'),want);
for(const v of [null,'fixed:+1','fixed:-1']) assert.throws(()=>coefficientValue(v));
for(const v of [true,false,Infinity,NaN,'fixed:+2','fixed:0','1',{},[]])
  assert.throws(()=>coefficientValue(v,'sparse-fixed-v1'));
assert.throws(()=>coefficientValue(0,'unknown'));
console.log(JSON.stringify({passed:true,inference_performed:false}));
"""
        result = subprocess.run(['node', '--input-type=module', '-e', script],
                                cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertTrue(json.loads(result.stdout)['passed'])


class SparsePerfectArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = []
        for path in (ROOT/'runs').glob('*/status.json'):
            state = read(path)
            if (state.get('status') == 'completed' and
                    state.get('training_mode') == 'finite-domain-sparse-perfect-export'):
                cls.runs.append(path.parent)
        if not cls.runs:
            raise unittest.SkipTest('No completed sparse perfect export yet')

    def test_every_decision_fixture_is_independently_optimal(self):
        expected = {board for board in census() if outcome(board) is None}
        self.assertEqual(len(expected), 4520)
        for folder in self.runs:
            with self.subTest(run=folder.name):
                report = read(folder/'independent-verification.json')
                self.assertEqual(report['optimal_count'], 4520)
                self.assertEqual(report['total'], 4520)
                self.assertIs(report['perfect_policy'], True)
                self.assertIs(report['no_loss_from_empty'], True)
                self.assertEqual(report['mistake_count'], 0)
                self.assertEqual(report['mistakes'], [])
                self.assertGreaterEqual(report['as_x']['value'], 0)
                self.assertGreaterEqual(report['as_o']['value'], 0)
                fixtures = report['fixtures']
                self.assertEqual(len(fixtures), 4520)
                self.assertEqual({tuple(row['board']) for row in fixtures}, expected)
                for row in fixtures:
                    board = tuple(row['board'])
                    self.assertEqual(row['optimal_moves'], exact_moves(board))
                    self.assertIn(row['move'], row['optimal_moves'])

    def test_stored_numeric_count_payload_and_structural_disclosure(self):
        kind = {'float16': 'e', 'float32': 'f', 'float64': 'd'}
        for folder in self.runs:
            with self.subTest(run=folder.name):
                model = read(folder/'model.json')
                validate_model(model)
                self.assertEqual(model['coefficient_storage'], 'sparse-fixed-v1')
                values = slots(model)
                learned = [v for v in values if type(v) in (int, float)]
                self.assertEqual(model['parameter_count'], len(learned))
                self.assertLess(len(learned), 74)
                self.assertEqual(model['dense_coefficient_slots'], len(values))
                self.assertEqual(model['structural_zero_slots'], sum(v is None for v in values))
                self.assertEqual(model['fixed_unit_slots'], sum(type(v) is str for v in values))
                self.assertEqual(len(values), len(learned)+model['structural_zero_slots']+model['fixed_unit_slots'])
                if model.get('output_scale_normalized'):
                    self.assertEqual(model['fixed_unit_slots'], 1)
                    self.assertTrue(any(type(v) is str for v in
                                        sum(model['layers'][-1]['weights'], [])+model['layers'][-1]['bias']))
                self.assertNotIn('learned_parameter_mask', model)
                self.assertNotIn('search_only_masked_dense', model)
                packed = struct.pack('<'+kind[model['coefficient_precision']]*len(learned), *learned)
                self.assertEqual((folder/'coefficients.bin').read_bytes(), packed)
                self.assertEqual(model['coefficient_payload_bytes'], len(packed))
                self.assertEqual(read(folder/'independent-verification.json')['parameter_count'], len(learned))
                evaluation = read(folder/'evaluation.json')
                self.assertEqual(evaluation['coefficient_payload_bytes'], len(packed))
                self.assertIs(evaluation['no_solver_fallback'], True)
                self.assertEqual(read(folder/'status.json')['parameter_count'], len(learned))

    def test_frozen_transform_and_source_lineage_match_qualification(self):
        for folder in self.runs:
            with self.subTest(run=folder.name):
                protocol = read(folder/'transformation-before-qualification.json')
                receipt = read(folder/'qualification-consumption.json')
                source = ROOT/'runs'/protocol['source_run']
                self.assertEqual(source.resolve().parent, (ROOT/'runs').resolve())
                self.assertEqual(sha(source/protocol['source_model']), protocol['source_model_sha256'])
                if protocol.get('source_selection_sha256'):
                    self.assertEqual(sha(source/protocol['source_selection_file']),
                                     protocol['source_selection_sha256'])
                # Source searches have different protocol names. Verify any
                # explicit binding without pretending one family owns all runs.
                self.assertTrue(protocol['source_protocols'])
                for name, expected in protocol['source_protocols'].items():
                    self.assertEqual(sha(source/name), expected)
                for name, expected in protocol['source_hashes'].items():
                    self.assertEqual(sha(folder/'source'/name), expected)
                self.assertEqual(sha(folder/'transformation-before-qualification.json'), receipt['transformation_sha256'])
                self.assertEqual(sha(folder/'model.json'), receipt['model_sha256'])
                self.assertEqual(sha(folder/'independent-verification.json'), receipt['independent_sha256'])
                self.assertEqual(sha(folder/'portable-verification.json'), receipt['portable_sha256'])
                self.assertLessEqual(datetime.fromisoformat(protocol['registered_at']),
                                     datetime.fromisoformat(receipt['consumed_at']))

    def test_unchanged_browser_action_gate_and_first_passing_precision(self):
        for folder in self.runs:
            with self.subTest(run=folder.name):
                parity = read(folder/'portable-verification.json')
                self.assertIs(parity['passed'], True)
                self.assertEqual(parity['positions'], 4520)
                self.assertEqual(parity['optimal_count'], 4520)
                self.assertEqual(parity['action_mismatches'], 0)
                self.assertEqual(parity['tolerance'], 1e-9)
                self.assertLessEqual(parity['max_logit_error'], 1e-9)
                protocol = read(folder/'transformation-before-qualification.json')
                attempts = read(folder/'transformation-attempts.json')
                model = read(folder/'model.json')
                self.assertEqual([r['precision'] for r in attempts],
                                 protocol['precision_order'][:len(attempts)])
                accepted = [r for r in attempts if r['perfect_policy']
                            and r['optimal_count'] == 4520 and r['no_loss_from_empty']
                            and r['portable']['passed'] and r['portable']['action_mismatches'] == 0]
                self.assertEqual(len(accepted), 1)
                self.assertIs(accepted[0], attempts[-1])
                self.assertEqual(accepted[0]['model_sha256'], sha(folder/'model.json'))
                self.assertEqual(accepted[0]['checkpoint'], model['checkpoint'])

    def test_recorded_worker_freezes_structural_coefficients_when_present(self):
        for folder in self.runs:
            path = folder/'browser-worker-verification.json'
            if not path.exists():
                continue
            with self.subTest(run=folder.name):
                report = read(path)
                self.assertIs(report['passed'], True)
                self.assertEqual(report['steps'], 1536)
                self.assertEqual(report['optimal_count'], 4520)
                self.assertEqual(report['minimum_optimal'], 4520)
                self.assertIs(report['structural_coefficients_preserved'], True)
                self.assertEqual(report['worker_source_sha256'],
                                 sha(folder/'source/web/training-worker.js'))
                self.assertEqual(report['network_source_sha256'],
                                 sha(folder/'source/web/network.js'))

    def test_sparse_catalog_refuses_incomplete_optimality_and_browser_receipts(self):
        baseline = ROOT/'runs/compression-20261113-selected'
        selected = min(self.runs, key=lambda p: read(p/'model.json')['parameter_count'])
        published_docs = {name: read(baseline/name) for name in (
            'model.json', 'status.json', 'evaluation.json',
            'independent-verification.json', 'portable-verification.json')}
        sparse_docs = {name: read(selected/name) for name in (
            'model.json', 'independent-verification.json', 'portable-verification.json')}
        for docs in (published_docs, sparse_docs):
            docs['independent-verification.json'].pop('fixtures', None)
        cases = [
            ('independent-verification.json', 'perfect_policy', False),
            ('independent-verification.json', 'optimal_count', 4519),
            ('independent-verification.json', 'total', 4519),
            ('independent-verification.json', 'no_loss_from_empty', False),
            ('portable-verification.json', 'passed', False),
            ('portable-verification.json', 'action_mismatches', 1),
            ('portable-verification.json', 'optimal_count', 4519),
        ]
        with tempfile.TemporaryDirectory(dir=ROOT.parent, prefix='sparse-catalog-') as temp:
            project = Path(temp)
            (project/'web').mkdir()
            published = project/'runs/published'
            sparse = project/'runs/sparse'
            published.mkdir(parents=True)
            sparse.mkdir()
            for name, document in published_docs.items():
                (published/name).write_text(json.dumps(document))
            registry = project/'runs/sparse-perfect-selection.json'
            registry.write_text(json.dumps({'run_id': 'sparse'}))
            with patch.object(build_site, 'ROOT', project), contextlib.redirect_stdout(io.StringIO()):
                # A good receipt reaches the actual catalog as a separate
                # choice. No forward pass or mocked build seam is involved.
                for name, document in sparse_docs.items():
                    (sparse/name).write_text(json.dumps(document))
                build_site.build(published, project/'site',
                                 'https://example.invalid/repo', 'https://example.invalid')
                index = read(project/'site/data/networks.json')
                item = next(row for row in index['networks'] if row['id'] == 'perfect-sparse')
                self.assertEqual(item['parameters'], sparse_docs['model.json']['parameter_count'])
                self.assertEqual(item['optimal_count'], 4520)
                self.assertEqual(item['coefficient_storage'], 'sparse-fixed-v1')
                self.assertEqual(item['structural_zero_slots'], sparse_docs['model.json']['structural_zero_slots'])
                self.assertEqual(item['fixed_unit_slots'], sparse_docs['model.json']['fixed_unit_slots'])
                for filename, key, value in cases:
                    with self.subTest(receipt=filename, field=key):
                        changed = copy.deepcopy(sparse_docs)
                        changed[filename][key] = value
                        for name, document in changed.items():
                            (sparse/name).write_text(json.dumps(document))
                        with self.assertRaisesRegex(ValueError, 'sparse perfect choice'):
                            build_site.build(published, project/'site',
                                             'https://example.invalid/repo', 'https://example.invalid')
                # Reusing the already seen compact SHA must still meet the
                # stricter perfect gate, before deduplication can hide it.
                registry.write_text(json.dumps({'run_id': 'published'}))
                with self.assertRaisesRegex(ValueError, 'sparse perfect choice'):
                    build_site.build(published, project/'site',
                                     'https://example.invalid/repo', 'https://example.invalid')
                for name in ('..', '../published', '.', ''):
                    registry.write_text(json.dumps({'run_id': name}))
                    with self.subTest(registry=name), self.assertRaisesRegex(ValueError, 'Sparse model registry'):
                        build_site.build(published, project/'site',
                                         'https://example.invalid/repo', 'https://example.invalid')
                registry.write_text(json.dumps({'run_id': 'missing'}))
                with self.assertRaisesRegex(ValueError, 'registered sparse perfect model is missing'):
                    build_site.build(published, project/'site',
                                     'https://example.invalid/repo', 'https://example.invalid')


class HistoricalSparseScopeTests(unittest.TestCase):
    def test_existing_74_53_and_38_models_are_unchanged(self):
        for name, expected in (
            ('perfect-optimal-74', 'f2126224b52abde726908257a1faa918bf84247601f799f31876c64e2443f9f7'),
            ('perfect-sparse-53', '22af9dcaabff1439a0c11723fc7eb32e1767989789c58e2ae053960fc2a5301c'),
            ('compression-20261113-selected', '9b3b98f4b995c0c434bdd248ca2042844485379d0f999751b7e0f40296277632'),
        ):
            with self.subTest(run=name):
                self.assertEqual(sha(ROOT/'runs'/name/'model.json'), expected)


if __name__ == '__main__':
    unittest.main()
