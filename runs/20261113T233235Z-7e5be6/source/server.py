"""Loopback-only dashboard and bounded, guarded training controller."""
import argparse
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid
from urllib.parse import parse_qs, urlparse

from solver import solve, action_values

ROOT=Path(__file__).resolve().parent
GUARD=Path(os.environ.get('TTT_MEMORY_GUARD',str(Path.home()/'.local/bin/memory-run')))
ADMISSION=Path(os.environ.get('TTT_STORAGE_ADMISSION',str(Path.home()/'.codex/bin/storage_admit.py')))

def atomic_json(path,data):
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(data)); temp.replace(path)

class Controller:
    def __init__(self,root=ROOT):
        self.root=Path(root); self.runs=self.root/'runs'; self.runs.mkdir(exist_ok=True)
        self.lock=threading.Lock(); self.process=None; self.log=None
        pointer=self.runs/'active.json'
        if pointer.exists():
            name=json.loads(pointer.read_text())['run_id']; self.current=self.runs/name
        else:
            dirs=[p for p in self.runs.iterdir() if p.is_dir() and (p/'status.json').exists()]
            self.current=max(dirs,key=lambda p:p.stat().st_mtime) if dirs else None

    def status(self):
        if self.current and (self.current/'status.json').exists():
            state=json.loads((self.current/'status.json').read_text())
        else:
            state={"status":"starting" if self.process and self.process.poll() is None else "idle", "run_id":self.current.name if self.current else None,
                   "phases":[],"history":[],"events":[],"architecture":[27,64,64,9],"parameter_count":6537,"model_available":False}
        if self.process is not None and self.process.poll() is not None and state['status'] in ('running','paused','starting'):
            state.update(status='failed',error='Training process ended before completion. See trainer.log for admission or runtime details.')
            atomic_json(self.current/'status.json',state)
        elif self.process is None and state['status'] in ('running','paused','starting'):
            state.update(status='interrupted',error='This run has no worker owned by the current server. Saved checkpoints remain available; start a new run.')
        if self.current and (self.current/'independent-verification.json').exists():
            evidence=json.loads((self.current/'independent-verification.json').read_text())
            state.setdefault('evaluation',{})['independent']={k:evidence[k] for k in ('optimal_count','total','no_loss_from_empty','perfect_policy')}
            state['evaluation']['worst_case']={"as_x":evidence['as_x']['result'],"as_o":evidence['as_o']['result']}
        if self.current and (self.current/'portable-verification.json').exists():
            state.setdefault('evaluation',{})['browser_export']=json.loads((self.current/'portable-verification.json').read_text())
        return state

    def start(self,seed,mode='foundation'):
        if type(seed) is not int or not 0<=seed<2**32:
            raise ValueError('Seed must be an integer between 0 and 4294967295.')
        if mode not in ('foundation','continue'):
            raise ValueError('Training mode must be foundation or continue.')
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError('A training run is already active.')
            # Each run is below 20 MB; admit 100 MB including receipts and logs.
            if ADMISSION.exists():
                result=subprocess.run([sys.executable,str(ADMISSION),'--path',str(self.root),'--expected-bytes','100000000'],capture_output=True,text=True)
                if result.returncode:
                    raise RuntimeError('Storage admission refused this run: '+result.stdout.strip())
            else:
                raise RuntimeError('Required local storage admission tool is unavailable.')
            if not GUARD.exists():
                raise RuntimeError('Required local training memory guard is unavailable.')
            worker=[sys.executable,str(self.root/'train.py')]
            if mode=='continue':
                parents=[]
                for prior in self.runs.iterdir():
                    if not prior.is_dir() or not (prior/'status.json').exists(): continue
                    prior_state=json.loads((prior/'status.json').read_text())
                    if prior_state.get('status')!='completed' or not (prior/'model.json').exists(): continue
                    model=json.loads((prior/'model.json').read_text())
                    checkpoint=prior/(model['checkpoint']+'.npz')
                    if checkpoint.is_file(): parents.append(checkpoint)
                if not parents: raise RuntimeError('Complete a foundation run before continuing training.')
                parent=max(parents,key=lambda p:p.stat().st_mtime)
                worker=[sys.executable,str(self.root/'continue_train.py'),'--parent',str(parent)]
            name=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]
            run=self.runs/name; run.mkdir()
            atomic_json(run/'storage-admission.json',json.loads(result.stdout))
            env=os.environ.copy(); env['TMPDIR']=str(run); env['PYTHONDONTWRITEBYTECODE']='1'
            self.log=(run/'trainer.log').open('w')
            try:
                self.process=subprocess.Popen([str(GUARD),'--kind','model','--peak-bytes','536870912','--',*worker,'--run-dir',str(run),'--seed',str(seed)],
                                              cwd=self.root,env=env,stdout=self.log,stderr=subprocess.STDOUT)
            except Exception:
                self.log.close(); self.log=None; raise
            self.current=run; atomic_json(self.runs/'active.json',{'run_id':name})
            return {'run_id':name,'status':'starting'}

    def control(self,action):
        if action not in ('pause','resume','stop'): raise ValueError('Unknown training action.')
        with self.lock:
            if not self.process or self.process.poll() is not None: raise RuntimeError('No owned training run is active.')
            atomic_json(self.current/'control.json',{'action':action})
            return {'requested':action}

    def close(self):
        if self.process and self.process.poll() is None:
            atomic_json(self.current/'control.json',{'action':'stop'})
            try: self.process.wait(timeout=10)
            except subprocess.TimeoutExpired: print('Worker is still stopping; its memory guard retains ownership.',file=sys.stderr)
        if self.log: self.log.close()

class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,controller=None,**kwargs):
        self.controller=controller
        super().__init__(*args,directory=str(controller.root/'web'),**kwargs)

    def json_response(self,data,status=200):
        body=json.dumps(data,allow_nan=False).encode()
        self.send_response(status); self.send_header('Content-Type','application/json'); self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)

    def valid_host(self):
        return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')

    def do_GET(self):
        if not self.valid_host(): return self.json_response({'error':'Local host required.'},403)
        parsed=urlparse(self.path)
        try:
            if parsed.path=='/api/status': return self.json_response(self.controller.status())
            if parsed.path=='/api/model':
                p=self.controller.current/'model.json' if self.controller.current else None
                if not p or not p.exists(): return self.json_response({'error':'No trained checkpoint is available yet.'},404)
                return self.json_response(json.loads(p.read_text()))
            if parsed.path=='/api/oracle':
                board=tuple(int(v) for v in parse_qs(parsed.query).get('board',[''])[0].split(','))
                value,moves=solve(board)
                return self.json_response({'value':value,'optimal_moves':moves,'action_values':action_values(board)})
            if parsed.path=='/api/runs':
                return self.json_response({'runs':[p.name for p in sorted(self.controller.runs.iterdir(),reverse=True) if p.is_dir()]})
            if parsed.path.startswith('/api/'): return self.json_response({'error':'Unknown API route.'},404)
            if parsed.path not in ('/','/index.html','/app.js','/style.css','/network.js','/favicon.ico'):
                return self.json_response({'error':'Not found.'},404)
            super().do_GET()
        except (ValueError,KeyError,TypeError) as exc: self.json_response({'error':str(exc)},400)

    def do_POST(self):
        if not self.valid_host(): return self.json_response({'error':'Local host required.'},403)
        origin=self.headers.get('Origin')
        allowed=(f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}')
        if origin not in allowed: return self.json_response({'error':'Same-origin browser request required.'},403)
        if self.headers.get_content_type()!='application/json': return self.json_response({'error':'JSON required.'},415)
        try:
            size=int(self.headers.get('Content-Length','0'))
            if size<1 or size>4096: raise ValueError('Request body must be between 1 and 4096 bytes.')
            body=json.loads(self.rfile.read(size))
            if not isinstance(body,dict): raise ValueError('JSON object required.')
            if self.path=='/api/train':
                started=self.controller.start(body.get('seed',42),body['mode']) if 'mode' in body else self.controller.start(body.get('seed',42))
                return self.json_response(started,202)
            if self.path=='/api/control': return self.json_response(self.controller.control(body.get('action')))
            return self.json_response({'error':'Unknown API route.'},404)
        except (ValueError,TypeError) as exc: return self.json_response({'error':str(exc)},400)
        except RuntimeError as exc: return self.json_response({'error':str(exc)},409)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--port',type=int,default=8765); args=p.parse_args()
    controller=Controller(); server=ThreadingHTTPServer(('127.0.0.1',args.port),partial(Handler,controller=controller))
    print(f'Tic-Tac-Toe Lab: http://127.0.0.1:{args.port}',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close(); controller.close()

if __name__=='__main__': main()
