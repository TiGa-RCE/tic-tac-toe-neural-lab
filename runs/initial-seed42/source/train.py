"""Solver distillation with phase receipts; run through the local memory guard."""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from solver import enumerate_states, winner, current_player, legal_moves, play, solve, canonical_key

ARCH = [27, 64, 64, 9]
PARAMETERS = sum(a*b+b for a,b in zip(ARCH, ARCH[1:]))
PHASES = [
    ("teacher", "Qualify the teacher", "Enumerate reachable positions; minimax labels; symmetry-disjoint partitions."),
    ("founder", "Supervised foundation", "1,024 Adam updates on TRAIN, with every equally optimal action accepted."),
    ("phase1", "01 · Collect & review", "Native rollouts, targeted errors and retention replay; 512 updates per branch."),
    ("phase2", "02 · Repair & retain", "Refresh native trajectories; protect founder-correct decisions with temperature-2 KL."),
    ("phase3", "03 · Consolidate", "Final bounded refinement; preserve all phase checkpoints and repairs / breaks."),
    ("selection", "Freeze the selection", "Validation ranking, META tie break, then seal the selected identity before TEST."),
    ("qualification", "Sealed evaluation", "One TEST evaluation and separate exhaustive finite-state / adversarial audit."),
    ("export", "Portable student", "Export frozen tensors for neural-only browser play; retain the full experiment."),
]

def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, allow_nan=False, separators=(",", ":")))
    temp.replace(path)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def encode(boards):
    b = np.array(boards, dtype=np.int8)
    turn = np.where((b == 1).sum(1) == (b == -1).sum(1), 1, -1)
    rel = b * turn[:, None]
    return np.stack((rel == 1, rel == -1, rel == 0), axis=2).reshape(-1,27).astype(np.float64)

def softmax(logits, legal, temperature=1.):
    z = np.where(legal, logits / temperature, -1e9)
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z) * legal
    return p / p.sum(axis=1, keepdims=True)

class Network:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.params = []
        for a,b in zip(ARCH, ARCH[1:]):
            self.params += [rng.normal(0, np.sqrt(2/(a+b)), (a,b)), np.zeros(b)]

    def clone(self):
        n = Network(0)
        n.params = [p.copy() for p in self.params]
        return n

    def forward(self, x):
        activations = [x]
        for i in range(0,len(self.params),2):
            x = x @ self.params[i] + self.params[i+1]
            if i != len(self.params)-2:
                x = np.tanh(x)
            activations.append(x)
        return x, activations

    def objective(self, x, legal, optimal, parent=None, protect=None):
        logits, acts = self.forward(x)
        p = softmax(logits, legal)
        mass = (p * optimal).sum(1,keepdims=True).clip(1e-30)
        loss = -np.log(mass).mean()
        delta = (p - p * optimal / mass) / len(x)
        if parent is not None:
            q = softmax(parent, legal, 2.)
            pt = softmax(logits, legal, 2.)
            mask = protect[:,None]
            loss += float(.4 * (mask * q * (np.log(q.clip(1e-30))-np.log(pt.clip(1e-30)))).sum() / len(x))
            delta += .2 * mask * (pt-q) / len(x)
        grads = [None] * len(self.params)
        for layer in range(2,-1,-1):
            grads[2*layer] = acts[layer].T @ delta
            grads[2*layer+1] = delta.sum(0)
            if layer:
                delta = (delta @ self.params[2*layer].T) * (1-acts[layer]**2)
        return float(loss), grads

    def export(self, checkpoint, run_id):
        return {"architecture": ARCH, "parameter_count": PARAMETERS, "encoding": "relative-one-hot-interleaved",
                "output": "policy_logits", "trained": True, "checkpoint": checkpoint, "run_id": run_id,
                "layers": [{"weights":self.params[i].tolist(), "bias":self.params[i+1].tolist(),
                            "activation":"linear" if i==4 else "tanh"} for i in (0,2,4)]}

class Adam:
    def __init__(self, net):
        self.m = [np.zeros_like(p) for p in net.params]
        self.v = [np.zeros_like(p) for p in net.params]
        self.t = 0

    def step(self, net, grads, lr):
        norm = np.sqrt(sum(float((g*g).sum()) for g in grads))
        scale = min(1., 1./max(norm,1e-30))
        self.t += 1
        for i,(p,g) in enumerate(zip(net.params,grads)):
            g = g * scale
            self.m[i] = .9*self.m[i] + .1*g
            self.v[i] = .999*self.v[i] + .001*g*g
            p -= lr * (self.m[i]/(1-.9**self.t)) / (np.sqrt(self.v[i]/(1-.999**self.t))+1e-8)

def gradient_check():
    n = Network(19)
    x = encode([(0,)*9, (1,0,0,0,-1,0,0,0,0)])
    legal = np.array([[True]*9,[False,True,True,True,False,True,True,True,True]])
    optimal = np.array([[False,True,False,True,False,False,False,False,False], [False,True,True,False,False,False,False,False,False]])
    parent = n.forward(x)[0] + np.arange(9)[None,:]*.1
    _,grads=n.objective(x,legal,optimal,parent,np.array([True,False]))
    error=0.
    for k,index in [(0,(4,2)),(1,(2,)),(2,(2,3)),(3,(1,)),(4,(1,2)),(5,(1,))]:
        old=n.params[k][index]; eps=1e-5
        n.params[k][index]=old+eps
        hi=n.objective(x,legal,optimal,parent,np.array([True,False]))[0]
        n.params[k][index]=old-eps
        lo=n.objective(x,legal,optimal,parent,np.array([True,False]))[0]
        n.params[k][index]=old
        error=max(error,abs((hi-lo)/(2*eps)-grads[k][index]))
    if error>1e-7:
        raise ValueError(f"Gradient qualification failed: {error}")
    return error

class Experiment:
    def __init__(self, run_dir, seed):
        self.root=Path(run_dir); self.root.mkdir(parents=True,exist_ok=True)
        self.seed=seed; self.started=time.monotonic(); self.step=0
        self.state={"status":"running","run_id":self.root.name,"phase_id":"teacher","seed":seed,
                    "phases":[{"id":i,"title":t,"detail":d,"status":"pending","progress":0.,"metrics":{}} for i,t,d in PHASES],
                    "architecture":ARCH,"parameter_count":PARAMETERS,"metrics":{},"history":[],"events":[],
                    "dataset":{},"evaluation":{},"model_available":False,"model_version":None,
                    "recipe_note":"Adapted from supplied Boundaries 0.11: native collection → targeted review + retention → checkpoint selection → sealed test. Sudoku reference not available."}
        self.flush()

    def flush(self):
        self.state['elapsed_seconds']=round(time.monotonic()-self.started,2)
        atomic_json(self.root/'status.json',self.state)

    def event(self,message):
        row={"time":datetime.now(timezone.utc).isoformat(),"message":message}
        self.state['events'].append(row); self.state['events']=self.state['events'][-100:]
        with (self.root/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        self.flush()

    def phase(self,id,progress=0.,metrics=None):
        self.state['phase_id']=id
        for p in self.state['phases']:
            if p['id']==id:
                p.update(status='running' if progress<1 else 'completed',progress=progress)
                if metrics is not None: p['metrics']=metrics
        self.flush()

    def control(self):
        while True:
            p=self.root/'control.json'
            action=json.loads(p.read_text()).get('action') if p.exists() else 'resume'
            if action=='stop': raise InterruptedError('Stopped by dashboard; saved checkpoints are retained.')
            if action!='pause':
                if self.state['status']=='paused':
                    self.state['status']='running'; self.event('Training resumed.')
                return
            if self.state['status']!='paused':
                self.state['status']='paused'; self.event('Training paused at an update boundary.')
            time.sleep(.2)

    def stats(self,net,indices):
        if len(indices)==0: return {"accuracy":None,"count":0}
        p=softmax(net.forward(self.x[indices])[0],self.legal[indices])
        chosen=p.argmax(1)
        good=self.optimal[indices,chosen]
        return {"accuracy":float(good.mean()),"correct":int(good.sum()),"count":len(indices),
                "loss":float(-np.log((p*self.optimal[indices]).sum(1).clip(1e-30)).mean())}

    def checkpoint(self,net,adam,name):
        arrays={f'p{i}':a for i,a in enumerate(net.params)}
        arrays.update({f'm{i}':a for i,a in enumerate(adam.m)})
        arrays.update({f'v{i}':a for i,a in enumerate(adam.v)})
        arrays['adam_step']=np.array(adam.t)
        np.savez(self.root/f'{name}.npz',**arrays)
        atomic_json(self.root/f'{name}.json',net.export(name,self.root.name))
        atomic_json(self.root/'model.json',net.export(name,self.root.name))
        self.state.update(model_available=True,model_version=name)

    def log(self,net,loss,phase,epoch,branch=None):
        tr=self.stats(net,self.parts['train']); va=self.stats(net,self.parts['validation'])
        row={"step":self.step,"phase":phase,"epoch":epoch,"loss":float(loss),"train_accuracy":tr['accuracy'],"val_accuracy":va['accuracy'],"branch":branch}
        self.state['metrics']=row; self.state['history'].append(row)
        with (self.root/'metrics.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        self.flush()

    def collect(self,net,rng):
        """Execute honest network moves; retain only TRAIN-orbit pre-action states."""
        counts=Counter(); trace=[]
        starts=rng.choice(self.parts['train'],128,replace=True)
        train_set=set(self.parts['train'])
        for idx in starts:
            board=self.boards[int(idx)]
            while winner(board) is None:
                state_idx=self.lookup[board]
                logits=net.forward(encode([board]))[0][0]
                chosen=max(legal_moves(board),key=lambda m: logits[m])
                retained=state_idx in train_set
                if retained: counts[state_idx]+=1
                trace.append({"state":state_idx,"move":chosen,"retained_for_training":retained})
                board=play(board,chosen)
        return counts,trace

    def run(self):
        err=gradient_check()
        self.boards=[b for b in enumerate_states() if winner(b) is None]
        self.lookup={b:i for i,b in enumerate(self.boards)}
        self.x=encode(self.boards); self.legal=np.array([[c==0 for c in b] for b in self.boards])
        groups=[canonical_key(b) for b in self.boards]
        keys=sorted(set(groups)); rng=np.random.default_rng(self.seed); rng.shuffle(keys)
        n=len(keys); cut=[int(n*.7),int(n*.8),int(n*.9)]
        membership={k:name for name,ks in zip(('train','meta','validation','test'),np.split(np.array(keys),cut)) for k in ks}
        self.parts={name:np.array([i for i,g in enumerate(groups) if membership[g]==name]) for name in ('train','meta','validation','test')}
        # Optimization labels exist only for TRAIN. Other labels are consulted by grading after inference.
        self.optimal=np.zeros((len(self.boards),9),bool)
        for i in np.concatenate([self.parts['train'],self.parts['validation'],self.parts['meta']]):
            self.optimal[i,list(solve(self.boards[int(i)])[1])]=True
        corpus={"boards":self.boards,"orbit_keys":groups,"partitions":{k:v.tolist() for k,v in self.parts.items()},"orbit_membership":membership}
        atomic_json(self.root/'corpus.json',corpus)
        src={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
        protocol={"architecture":ARCH,"parameters":PARAMETERS,"master_seed":self.seed,"branch_seeds":[11,23,37],
                  "founder_updates":1024,"branch_phases":3,"updates_per_phase":512,"batch_size":128,
                  "founder_lr":.003,"branch_lr":.0003,"gradient_clip":1.,"retention_coefficient":.1,"temperature":2.,
                  "sampling":"64 native + 64 retention; first 32 updates replace 32 native samples with current TRAIN errors when available",
                  "selection":"maximum validation repairs, then minimum breaks, then validation accuracy; META breaks exact ties; earliest checkpoint final tie",
                  "corpus_sha256":digest(self.root/'corpus.json'),"source_hashes":src,"gradient_check_max_error":err,
                  "test_policy":"TEST grading after selection-before-test.json only; no retuning on final outcomes"}
        atomic_json(self.root/'protocol-before-training.json',protocol)
        self.state['dataset']={"reachable":5478,"decision_states":len(self.boards),"symmetry_orbits":n,
                               **{k:len(v) for k,v in self.parts.items()},"split":"70 / 10 / 10 / 10% of D4 orbits"}
        self.event(f'Teacher qualified: 5,478 reachable states; {len(self.boards)} decision states in {n} symmetry orbits. Gradient check {err:.2g}.')
        self.phase('teacher',1.,self.state['dataset'])
        self.phase('founder')
        founder=Network(self.seed); opt=Adam(founder)
        schedule=rng.choice(self.parts['train'],(1024,128),replace=True)
        atomic_json(self.root/'founder-schedule.json',schedule.tolist())
        for u,row in enumerate(json.loads((self.root/'founder-schedule.json').read_text())):
            ids=np.array(row); loss,g=founder.objective(self.x[ids],self.legal[ids],self.optimal[ids]); opt.step(founder,g,.003); self.step+=1
            if u%16==0: self.control()
            if (u+1)%32==0:
                self.log(founder,loss,'founder',u+1); self.phase('founder',(u+1)/1024)
        self.checkpoint(founder,opt,'founder'); self.phase('founder',1.)
        self.event('Shared founder saved. Seeds 11 / 23 / 37 start from the same tensors with fresh Adam state.')
        parent=founder.forward(self.x)[0]
        parent_choice=softmax(parent,self.legal).argmax(1)
        protect=self.optimal[np.arange(len(self.boards)),parent_choice]
        baseline=self.optimal[self.parts['validation'],parent_choice[self.parts['validation']]]
        candidates=[]
        def candidate(net,name,order):
            va_ids=self.parts['validation']; chosen=softmax(net.forward(self.x[va_ids])[0],self.legal[va_ids]).argmax(1)
            good=self.optimal[va_ids,chosen]
            row={"checkpoint":name,"order":order,"repairs":int((good&~baseline).sum()),"breaks":int((~good&baseline).sum()),
                 "validation_correct":int(good.sum()),"validation_count":len(good)}
            candidates.append(row); return row
        candidate(founder,'founder',0)
        branches={s:(founder.clone(),None,np.random.default_rng(s)) for s in (11,23,37)}
        branches={s:(net,Adam(net),rg) for s,(net,_,rg) in branches.items()}
        native_counts={s:Counter({int(i):1 for i in self.parts['train']}) for s in branches}
        for phase in range(1,4):
            phase_id=f'phase{phase}'; self.phase(phase_id)
            for branch,(net,adam,rg) in branches.items():
                self.control()
                counts,trace=self.collect(net,rg); native_counts[branch].update(counts)
                stem=f's{branch}-p{phase}'
                atomic_json(self.root/f'{stem}-native.json',trace)
                pool=np.array(sorted(native_counts[branch])); freq=np.array([native_counts[branch][int(i)] for i in pool],float); freq/=freq.sum()
                chosen=softmax(net.forward(self.x[self.parts['train']])[0],self.legal[self.parts['train']]).argmax(1)
                errors=self.parts['train'][~self.optimal[self.parts['train'],chosen]]
                schedule=np.concatenate([rg.choice(pool,(512,64),p=freq),rg.choice(self.parts['train'],(512,64))],axis=1)
                if len(errors): schedule[:32,:32]=rg.choice(errors,(32,32))
                atomic_json(self.root/f'{stem}-schedule.json',{"indices":schedule.tolist(),"native_pool":pool.tolist(),"multiplicity":[native_counts[branch][int(i)] for i in pool],"target_errors":errors.tolist()})
                self.event(f'Phase {phase}, seed {branch}: saved literal schedule; {len(trace)} native actions, {len(errors)} TRAIN errors targeted.')
                schedule=json.loads((self.root/f'{stem}-schedule.json').read_text())['indices']
                for u,row in enumerate(schedule):
                    ids=np.array(row)
                    loss,g=net.objective(self.x[ids],self.legal[ids],self.optimal[ids],parent[ids],protect[ids])
                    adam.step(net,g,.0003); self.step+=1
                    if u%16==0: self.control()
                    if (u+1)%32==0:
                        self.log(net,loss,phase_id,u+1,branch)
                        done=((11,23,37).index(branch)*512+u+1)/1536
                        self.phase(phase_id,min(done,.999),{"seed":branch,"updates":u+1,"target_errors":len(errors)})
                self.checkpoint(net,adam,stem)
                row=candidate(net,stem,len(candidates))
                self.state['candidates']=candidates
                self.event(f'{stem}: validation {row["validation_correct"]}/{row["validation_count"]}; repairs {row["repairs"]}, breaks {row["breaks"]} versus founder.')
            self.phase(phase_id,1.)
        self.phase('selection'); self.control()
        rank=lambda c:(c['repairs'],-c['breaks'],c['validation_correct'])
        best=max(map(rank,candidates)); finalists=[c for c in candidates if rank(c)==best]
        for c in finalists:
            net=Network(0)
            with np.load(self.root/f'{c["checkpoint"]}.npz') as d: net.params=[d[f'p{i}'].copy() for i in range(6)]
            c['meta']=self.stats(net,self.parts['meta'])
        selected=max(finalists,key=lambda c:(c['meta']['correct'],-c['order']))
        model_path=self.root/f'{selected["checkpoint"]}.json'
        seal={"selected":selected,"candidates":candidates,"model_sha256":digest(model_path),"protocol_sha256":digest(self.root/'protocol-before-training.json'),
              "sealed_at":datetime.now(timezone.utc).isoformat(),"test_consumed":False}
        atomic_json(self.root/'selection-before-test.json',seal)
        self.state['selection']=selected
        self.event(f'Selected {selected["checkpoint"]} and sealed its tensor identity before TEST.')
        self.phase('selection',1.,selected)
        self.phase('qualification'); self.control()
        net=Network(0)
        with np.load(self.root/f'{selected["checkpoint"]}.npz') as d: net.params=[d[f'p{i}'].copy() for i in range(6)]
        for i in self.parts['test']: self.optimal[i,list(solve(self.boards[int(i)])[1])]=True
        scores=net.forward(self.x)[0]; choices=np.where(self.legal,scores,-1e9).argmax(1)
        good=self.optimal[np.arange(len(self.boards)),choices]
        errors=[{"board":b,"move":int(choices[i]),"optimal_moves":np.flatnonzero(self.optimal[i]).tolist()} for i,b in enumerate(self.boards) if not good[i]]
        atomic_json(self.root/'counterexamples.json',errors)
        self.state['evaluation']={"test":self.stats(net,self.parts['test']),"exhaustive":{"correct":int(good.sum()),"count":len(good),"accuracy":float(good.mean()),"mistakes":len(errors)},
                                  "scope":"TEST is symmetry-disjoint. Exhaustive coverage includes training positions; it is not another held-out test.","selected_checkpoint":selected['checkpoint']}
        atomic_json(self.root/'evaluation.json',self.state['evaluation'])
        self.event(f'Sealed TEST: {self.state["evaluation"]["test"]["correct"]}/{len(self.parts["test"])} optimal. Exhaustive: {int(good.sum())}/{len(good)}. No post-test tuning.')
        self.phase('qualification',1.,self.state['evaluation']['exhaustive'])
        self.phase('export')
        atomic_json(self.root/'model.json',net.export(selected['checkpoint'],self.root.name))
        self.state.update(model_version=selected['checkpoint'],model_available=True,status='completed')
        self.phase('export',1.,{"checkpoint":selected['checkpoint'],"parameters":PARAMETERS})
        self.event('Training complete. Neural-only portable student ready; all raw-final and selected checkpoints retained.')

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run-dir',required=True); parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args(); exp=Experiment(args.run_dir,args.seed)
    try: exp.run()
    except InterruptedError as exc:
        exp.state['status']='cancelled'; exp.event(str(exc))
    except Exception as exc:
        exp.state['status']='failed'; exp.state['error']=str(exc)
        (exp.root/'traceback.txt').write_text(traceback.format_exc()); exp.event(f'Training failed: {exc}'); raise

if __name__=='__main__': main()
