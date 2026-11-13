"""API/lifecycle checks that never load or launch a neural network."""
from functools import partial
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import Mock, patch

import server

WORK = Path(__file__).resolve().parents[3] / 'work'
WORK.mkdir(exist_ok=True)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(dir=WORK, prefix='server-test-')
        self.root = Path(self.temp.name)
        self.controller = server.Controller(self.root)

    def tearDown(self):
        self.controller.process = None
        self.controller.close()
        self.temp.cleanup()

    def test_invalid_seed_never_calls_admission_or_worker(self):
        with patch.object(server.subprocess, 'run') as admit, patch.object(server.subprocess, 'Popen') as popen:
            for seed in (True, -1, 2**32, '42', None, 1.5):
                with self.assertRaises(ValueError):
                    self.controller.start(seed)
            admit.assert_not_called()
            popen.assert_not_called()

    def test_storage_refusal_prevents_worker_and_new_run(self):
        admission = self.root / 'admission.py'
        admission.touch()
        refusal = Mock(returncode=1, stdout='reserve refused')
        with patch.object(server, 'ADMISSION', admission), patch.object(server.subprocess, 'run', return_value=refusal), patch.object(server.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(RuntimeError, 'Storage admission refused'):
                self.controller.start(42)
            popen.assert_not_called()
        self.assertEqual(list(self.controller.runs.iterdir()), [])

    def test_guarded_launch_and_control_messages(self):
        admission = self.root / 'admission.py'
        guard = self.root / 'memory-run'
        admission.touch(); guard.touch()
        process = Mock()
        process.poll.return_value = None
        result = Mock(returncode=0, stdout='{"admitted":true}')
        with patch.object(server, 'ADMISSION', admission), patch.object(server, 'GUARD', guard), patch.object(server.subprocess, 'run', return_value=result) as admit, patch.object(server.subprocess, 'Popen', return_value=process) as popen:
            started = self.controller.start(42)
            command = popen.call_args.args[0]
            self.assertEqual(command[:6], [str(guard), '--kind', 'model', '--peak-bytes', '536870912', '--'])
            self.assertIn('100000000', admit.call_args.args[0])
            self.assertEqual(popen.call_args.kwargs['env']['TMPDIR'], str(self.controller.current))
            self.assertEqual(json.loads((self.controller.runs/'active.json').read_text())['run_id'], started['run_id'])
            with self.assertRaisesRegex(RuntimeError, 'already active'):
                self.controller.start(43)
            self.assertEqual(popen.call_count, 1)
            for action in ('pause', 'resume', 'stop'):
                self.assertEqual(self.controller.control(action), {'requested': action})
                self.assertEqual(json.loads((self.controller.current/'control.json').read_text()), {'action': action})
            process.poll.return_value = 0
            with self.assertRaisesRegex(RuntimeError, 'No owned'):
                self.controller.control('resume')

    def test_unowned_running_status_is_interrupted_without_touching_checkpoint(self):
        run = self.controller.runs / 'old'
        run.mkdir()
        original = {'status': 'running', 'run_id': 'old', 'model_available': True}
        (run/'status.json').write_text(json.dumps(original))
        (run/'model.json').write_text('{"checkpoint":"saved"}')
        self.controller.current = run
        state = self.controller.status()
        self.assertEqual(state['status'], 'interrupted')
        self.assertTrue(state['model_available'])
        self.assertEqual(json.loads((run/'status.json').read_text()), original)
        self.assertEqual((run/'model.json').read_text(), '{"checkpoint":"saved"}')


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory(dir=WORK, prefix='api-test-')
        cls.controller = server.Controller(Path(cls.temp.name))
        cls.http = ThreadingHTTPServer(('127.0.0.1', 0), partial(server.Handler, controller=cls.controller))
        cls.port = cls.http.server_port
        cls.thread = threading.Thread(target=cls.http.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown(); cls.http.server_close(); cls.thread.join()
        cls.temp.cleanup()

    def request(self, method, path, body=None, *, origin=True, host=None, content_type='application/json'):
        headers = {'Host': host or f'127.0.0.1:{self.port}', 'Content-Type': content_type}
        if origin:
            headers['Origin'] = origin if isinstance(origin, str) else f'http://127.0.0.1:{self.port}'
        conn = HTTPConnection('127.0.0.1', self.port, timeout=3)
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        data = json.loads(response.read())
        status = response.status
        conn.close()
        return status, data

    def test_cross_site_mutations_are_rejected_before_controller(self):
        with patch.object(self.controller, 'start') as start:
            for origin in (False, 'https://untrusted.example'):
                status, _ = self.request('POST', '/api/train', '{}', origin=origin)
                self.assertEqual(status, 403)
            status, _ = self.request('POST', '/api/train', '{}', host='untrusted.example')
            self.assertEqual(status, 403)
            start.assert_not_called()

    def test_request_validation_and_missing_model(self):
        self.assertEqual(self.request('GET', '/api/model')[0], 404)
        self.assertEqual(self.request('POST', '/api/train', '{}', content_type='text/plain')[0], 415)
        for body in ('[]', 'null', '{'):
            self.assertEqual(self.request('POST', '/api/train', body)[0], 400)
        self.assertEqual(self.request('POST', '/api/train', 'x'*4097)[0], 400)
        self.assertEqual(self.request('GET', '/api/oracle?board=1,1,1,1,1,1,1,1,1')[0], 400)
        self.assertEqual(self.request('GET', '/api/oracle?board=a')[0], 400)

    def test_oracle_draw_and_same_origin_start_contract(self):
        status, result = self.request('GET', '/api/oracle?board=0,0,0,0,0,0,0,0,0')
        self.assertEqual(status, 200)
        self.assertEqual(result['value'], 0)
        self.assertEqual(result['optimal_moves'], list(range(9)))
        with patch.object(self.controller, 'start', return_value={'run_id': 'fixture', 'status': 'starting'}) as start:
            status, result = self.request('POST', '/api/train', '{"seed":123}')
            self.assertEqual(status, 202)
            self.assertEqual(result['run_id'], 'fixture')
            start.assert_called_once_with(123)


if __name__ == '__main__':
    unittest.main()
