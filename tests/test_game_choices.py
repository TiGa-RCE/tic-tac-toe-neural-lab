"""Game-choice contracts using frozen metadata and the actual hosted adapter.

No learned-network forward pass, model backend, browser, or training is run.
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import build_site

ROOT = Path(__file__).resolve().parents[1]


class GameChoiceTests(unittest.TestCase):
    def test_hosted_adapter_keeps_game_and_training_choices_independent(self):
        if shutil.which('node') is None:
            self.skipTest('Node is required for the real hosted adapter contract')
        result = subprocess.run(['node', 'verify_game_choices.mjs'], cwd=ROOT,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(json.loads(result.stdout)['passed'])

    def test_sparse_default_preserves_all_previous_certified_game_choices(self):
        work = ROOT.parent
        with tempfile.TemporaryDirectory(dir=work, prefix='game-choice-build-') as temp:
            output = Path(temp)
            selected = json.loads((ROOT/'runs/sparse-perfect-selection.json').read_text())
            run = ROOT / 'runs' / selected['run_id']
            model = json.loads((run/'model.json').read_text())
            self.assertLess(model['parameter_count'], 74)
            if selected.get('previous_run_ids'):
                self.assertLess(model['parameter_count'], 53)
            if model['parameter_count'] < 53:
                self.assertIn('perfect-sparse-53', selected.get('previous_run_ids', []))
            with contextlib.redirect_stdout(io.StringIO()):
                build_site.build(run, output, 'https://example.invalid/repo', 'https://example.invalid')
            index = json.loads((output / 'data/networks.json').read_text())
            self.assertEqual(index['default_id'], 'perfect-sparse')
            self.assertEqual(index['default_game_id'], 'perfect-sparse')
            expected = {
                'compact': 'compression-20261113-selected',
                'dense': 'compression-20261113-a',
                'perfect-compact': 'perfect-optimal-74',
                'perfect-sparse': selected['run_id'],
                'original': '20261113T233235Z-7e5be6',
            }
            expected.update({'history-'+name: name for name in selected.get('previous_run_ids', [])})
            expected_counts={key: json.loads((ROOT/'runs'/name/'model.json').read_text())['parameter_count']
                             for key, name in expected.items()}
            self.assertEqual({row['id']: row['parameters'] for row in index['networks']}, expected_counts)
            for row in index['networks']:
                with self.subTest(network=row['id']):
                    source = ROOT / 'runs' / expected[row['id']]
                    self.assertEqual((output / row['model_url'].lstrip('/')).read_bytes(),
                                     (source / 'model.json').read_bytes())
                    proof = json.loads((output / row['certification_url'].lstrip('/')).read_text())
                    audit = json.loads((source / 'independent-verification.json').read_text())
                    audit.pop('fixtures', None)
                    self.assertEqual(proof, audit)
            self.assertEqual((output / 'data/model.json').read_bytes(), (run / 'model.json').read_bytes())
            config = json.loads((output / 'site-config.json').read_text())
            self.assertEqual(config['run_id'], selected['run_id'])
            # The previous default can still be explicitly built and selected.
            with contextlib.redirect_stdout(io.StringIO()):
                build_site.build(ROOT/'runs/perfect-optimal-74', output,
                                 'https://example.invalid/repo', 'https://example.invalid')
            self.assertEqual(json.loads((output/'data/networks.json').read_text())['default_game_id'],
                             'perfect-compact')

    def test_historical_sparse_choice_requires_complete_proof_before_deduplication(self):
        def read(path):
            return json.loads(path.read_text())
        baseline=ROOT/'runs/compression-20261113-selected'
        selected=ROOT/'runs/perfect-sparse-53'
        historical=ROOT/'runs/perfect-optimal-74'
        names=('model.json','independent-verification.json','portable-verification.json')
        base={name: read(baseline/name) for name in names+('status.json','evaluation.json')}
        chosen={name: read(selected/name) for name in names}
        previous={name: read(historical/name) for name in names}
        for documents in (base,chosen,previous):
            documents['independent-verification.json'].pop('fixtures',None)
        with tempfile.TemporaryDirectory(dir=ROOT.parent,prefix='history-choice-test-') as temp:
            project=Path(temp)
            (project/'web').mkdir()
            for folder,documents in (('published',base),('chosen',chosen),('previous',previous)):
                target=project/'runs'/folder
                target.mkdir(parents=True)
                for name,value in documents.items():
                    (target/name).write_text(json.dumps(value))
            registry=project/'runs/sparse-perfect-selection.json'
            registry.write_text(json.dumps({'run_id':'chosen','previous_run_ids':['previous']}))
            with patch.object(build_site,'ROOT',project), contextlib.redirect_stdout(io.StringIO()):
                build_site.build(project/'runs/published',project/'output','https://example.invalid/repo','https://example.invalid')
                index=read(project/'output/data/networks.json')
                history=next(row for row in index['networks'] if row['id']=='history-previous')
                self.assertEqual(history['parameters'],74)
                self.assertEqual((project/'output'/history['model_url'].lstrip('/')).read_bytes(),
                                 (project/'runs/previous/model.json').read_bytes())
                mutations=[('independent-verification.json','perfect_policy',False),
                           ('independent-verification.json','optimal_count',4519),
                           ('independent-verification.json','total',4519),
                           ('independent-verification.json','no_loss_from_empty',False),
                           ('portable-verification.json','passed',False),
                           ('portable-verification.json','action_mismatches',1),
                           ('portable-verification.json','optimal_count',4519)]
                for filename,field,value in mutations:
                    with self.subTest(receipt=filename,field=field):
                        changed=copy.deepcopy(previous)
                        changed[filename][field]=value
                        for name,document in changed.items():
                            (project/'runs/previous'/name).write_text(json.dumps(document))
                        with self.assertRaisesRegex(ValueError,'historical sparse perfect choice'):
                            build_site.build(project/'runs/published',project/'output','https://example.invalid/repo','https://example.invalid')
                # An imperfect already-listed model cannot evade the history gate by deduplication.
                registry.write_text(json.dumps({'run_id':'chosen','previous_run_ids':['published']}))
                with self.assertRaisesRegex(ValueError,'historical sparse perfect choice'):
                    build_site.build(project/'runs/published',project/'output','https://example.invalid/repo','https://example.invalid')

    def test_historical_sparse_registry_rejects_invalid_or_duplicate_names(self):
        baseline=ROOT/'runs/compression-20261113-selected'
        selected=ROOT/'runs/perfect-sparse-53'
        with tempfile.TemporaryDirectory(dir=ROOT.parent,prefix='history-registry-test-') as temp:
            project=Path(temp)
            (project/'web').mkdir()
            for folder,source,names in (
                ('published',baseline,('model.json','status.json','evaluation.json','independent-verification.json','portable-verification.json')),
                ('chosen',selected,('model.json','independent-verification.json','portable-verification.json'))):
                destination=project/'runs'/folder
                destination.mkdir(parents=True)
                for name in names:
                    shutil.copyfile(source/name,destination/name)
            registry=project/'runs/sparse-perfect-selection.json'
            invalid=[None,'chosen',{},['chosen'],['missing'],['..'],['../chosen'],['a\\b'],[''],[1],['previous','previous']]
            with patch.object(build_site,'ROOT',project), contextlib.redirect_stdout(io.StringIO()):
                for previous in invalid:
                    with self.subTest(previous=previous):
                        registry.write_text(json.dumps({'run_id':'chosen','previous_run_ids':previous}))
                        with self.assertRaises(ValueError):
                            build_site.build(project/'runs/published',project/'output','https://example.invalid/repo','https://example.invalid')


if __name__ == '__main__':
    unittest.main()
