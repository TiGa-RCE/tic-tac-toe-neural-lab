"""Game-choice contracts using frozen metadata and the actual hosted adapter.

No learned-network forward pass, model backend, browser, or training is run.
"""
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

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
            with contextlib.redirect_stdout(io.StringIO()):
                build_site.build(run, output, 'https://example.invalid/repo', 'https://example.invalid')
            index = json.loads((output / 'data/networks.json').read_text())
            self.assertEqual(index['default_id'], 'perfect-sparse')
            self.assertEqual(index['default_game_id'], 'perfect-sparse')
            self.assertEqual({row['id']: row['parameters'] for row in index['networks']},
                             {'compact': 38, 'dense': 123, 'perfect-compact': 74,
                              'perfect-sparse': model['parameter_count'], 'original': 6537})
            expected = {
                'compact': 'compression-20261113-selected',
                'dense': 'compression-20261113-a',
                'perfect-compact': 'perfect-optimal-74',
                'perfect-sparse': selected['run_id'],
                'original': '20261113T233235Z-7e5be6',
            }
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


if __name__ == '__main__':
    unittest.main()
