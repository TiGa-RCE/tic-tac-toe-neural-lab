"""Bounded raw-board neural compression; exact adversarial diagnosis is training only."""
from train import np, Adam, softmax, atomic_json, digest, Experiment
from solver import enumerate_states, winner, solve, action_values, play, EMPTY
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse, json, shutil, subprocess, traceback

ENCODINGS={9:'relative-signed',18:'relative-two-channel',27:'relative-one-hot-interleaved'}
D4=[]
for reflected in (False,True):
    for turns in range(4):
        perm=[]
        for j in range(9):
            row,col=divmod(j,3)
            if reflected:col=2-col
            for _ in range(turns):row,col=col,2-row
            perm.append(3*row+col)
        D4.append(perm)

def features(boards,width):
    b=np.array(boards); turn=np.where(b.sum(1)==0,1,-1); rel=b*turn[:,None]
    if width==9:return rel.astype(float)
    if width==18:return np.stack((rel==1,rel==-1),2).reshape(-1,18).astype(float)
    return np.stack((rel==1,rel==-1,rel==0),2).reshape(-1,27).astype(float)

class Dense:
    def __init__(self,architecture,seed,symmetry=False):
        self.architecture=list(architecture); rng=np.random.default_rng(seed); self.params=[]
        self.symmetry=symmetry
        for a,b in zip(architecture,architecture[1:]):
            self.params.extend([rng.normal(0,np.sqrt(2/(a+b)),(a,b)),np.zeros(b)])

    def forward(self,x):
        count=len(x)
        if self.symmetry:
            view=x.reshape(count,9,-1);blocks=[]
            for perm in D4:
                transformed=np.empty_like(view);transformed[:,perm,:]=view;blocks.append(transformed.reshape(count,-1))
            x=np.concatenate(blocks)
        acts=[x]
        for i in range(0,len(self.params),2):
            x=x@self.params[i]+self.params[i+1]
            if i<len(self.params)-2:x=np.tanh(x)
            acts.append(x)
        if self.symmetry:
            averaged=sum(x[t*count:(t+1)*count,perm] for t,perm in enumerate(D4))/8
            return averaged,{'acts':acts,'count':count}
        return x,acts

    def objective(self,x,legal,target,parent,protect,kl=.01):
        logits,acts=self.forward(x); p=softmax(logits,legal)
        mass=(p*target).sum(1,keepdims=True).clip(1e-30)
        loss=float(-np.log(mass).mean()); delta=(p-p*target/mass)/len(x)
        if kl:
            q=softmax(parent,legal,2); pt=softmax(logits,legal,2); mask=protect[:,None]
            loss+=float(kl*4*(mask*q*(np.log(q.clip(1e-30))-np.log(pt.clip(1e-30)))).sum()/len(x))
            delta+=kl*2*mask*(pt-q)/len(x)
        if self.symmetry:
            blocks=[]
            for perm in D4:
                routed=np.empty_like(delta);routed[:,perm]=delta/8;blocks.append(routed)
            delta=np.concatenate(blocks);acts=acts['acts']
        gradients=[None]*len(self.params)
        for k in range(len(self.params)//2-1,-1,-1):
            gradients[2*k]=acts[k].T@delta;gradients[2*k+1]=delta.sum(0)
            if k:delta=(delta@self.params[2*k].T)*(1-acts[k]**2)
        return loss,gradients

    def export(self,name,run):
        count=sum(p.size for p in self.params)
        return dict(architecture=self.architecture,parameter_count=count,encoding=ENCODINGS[self.architecture[0]],symmetry_average='D4' if self.symmetry else None,forward_passes_per_move=8 if self.symmetry else 1,
                    trained=True,output='policy_logits',checkpoint=name,run_id=run,
                    layers=[dict(weights=self.params[i].tolist(),bias=self.params[i+1].tolist(),
                                 activation='linear' if i==len(self.params)-2 else 'tanh') for i in range(0,len(self.params),2)])

class Compression(Experiment):
    def __init__(self,run_dir,phases=8,resume=None,symmetry=False):
        root=Path(run_dir)
        if (root/'compression-protocol.json').exists():raise ValueError('Use a new immutable run directory.')
        super().__init__(root,42);self.budget=phases;self.resume=resume;self.symmetry=symmetry
        self.state.update(training_mode='finite-domain-compression',recipe_note='Raw-board dense networks; finite-domain solver teaching, native collection, adversarial critical-state review, balanced retention. Only no-loss as both sides is the acceptance gate; full optimality is reported separately. This bounded search does not establish a global minimum.',phases=[])

    def assess(self,net,x):
        z=net.forward(x)[0];choices=np.where(self.legal,z,-1e9).argmax(1)
        good=self.optimal[np.arange(len(x)),choices];safe=self.safe[np.arange(len(x)),choices]
        counts=Counter(); critical=set()
        def visit(board,mark,seen):
            result=winner(board)
            if result is not None:return result*mark
            if board in seen:return seen[board]
            idx=self.lookup[board]; turn=1 if sum(board)==0 else -1
            if turn==mark:
                counts[idx]+=1
                if not safe[idx]:critical.add(idx)
                value=visit(play(board,int(choices[idx])),mark,seen)
            else:value=min(visit(play(board,m),mark,seen) for m,c in enumerate(board) if c==0)
            seen[board]=value;return value
        vx=visit(EMPTY,1,{});vo=visit(EMPTY,-1,{})
        return dict(choices=choices,good=good,safe=safe,counts=counts,critical=sorted(critical),
                    correct=int(good.sum()),safe_correct=int(safe.sum()),as_x=vx,as_o=vo,no_loss_from_empty=vx>=0 and vo>=0)

    def run(self):
        probe=Dense([9,2,9],19,self.symmetry);px=features([EMPTY,(1,0,0,0,-1,0,0,0,0)],9)
        mask=np.ones((2,9),bool);target=np.zeros((2,9),bool);target[:,[0,3]]=True
        zero=np.zeros((2,9));protect=np.ones(2,bool)
        _,grads=probe.objective(px,mask,target,zero,protect)
        gradient_error=0.
        for layer,index in ((0,(2,1)),(1,(0,)),(2,(1,4)),(3,(2,))):
            old=probe.params[layer][index];eps=1e-5
            probe.params[layer][index]=old+eps;hi=probe.objective(px,mask,target,zero,protect)[0]
            probe.params[layer][index]=old-eps;lo=probe.objective(px,mask,target,zero,protect)[0]
            probe.params[layer][index]=old
            gradient_error=max(gradient_error,abs((hi-lo)/(2*eps)-grads[layer][index]))
        if gradient_error>1e-7:raise ValueError('Compression gradient qualification failed')
        self.boards=[b for b in enumerate_states() if winner(b) is None];self.lookup={b:i for i,b in enumerate(self.boards)}
        self.legal=np.array([[c==0 for c in b] for b in self.boards]);self.optimal=np.zeros_like(self.legal);self.safe=np.zeros_like(self.legal)
        for i,b in enumerate(self.boards):
            value,moves=solve(b);self.optimal[i,list(moves)]=True
            self.safe[i]=[(q is not None and q>=0) for q in action_values(b)] if value>=0 else self.legal[i]
        atomic_json(self.root/'corpus.json',{'boards':self.boards,'optimal_moves':[np.flatnonzero(r).tolist() for r in self.optimal],
                    'safe_moves':[np.flatnonzero(r).tolist() for r in self.safe],'scope':'All 4520 states are TRAIN; no held-out claim.'})
        teacher_path=Path(__file__).parent/'runs/20261113T233235Z-7e5be6/model.json'
        teacher=json.loads(teacher_path.read_text());parent=Dense(teacher['architecture'],0)
        parent.params=[np.array(v) for l in teacher['layers'] for v in (l['weights'],l['bias'])]
        parent_logits=parent.forward(features(self.boards,27))[0]
        architectures=[[9,h,9] for h in (16,8,4,2,1,3,5,6,7,10,12)]
        architectures += [[18,h,9] for h in (1,2,3,4,5,6,8)]+[[27,h,9] for h in (1,2,3,4)]
        architectures += [[9,9],[18,9],[9,2,2,9],[9,3,2,9],[9,3,3,9],[9,4,2,9],[9,4,3,9]]
        if self.symmetry:architectures=[[9,h,9] for h in (4,2,1)]+[[9,2,1,9],[9,1,1,9]]
        if self.resume:
            resume_model=json.loads(Path(self.resume).read_text());architectures=[resume_model['architecture']]
        sources={}
        for p in list(Path(__file__).parent.glob('*.py'))+list((Path(__file__).parent/'web').glob('*.js')):
            dest=self.root/'source'/p.relative_to(Path(__file__).parent);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
            sources[str(p.relative_to(Path(__file__).parent))]=digest(p)
        protocol=dict(architectures=architectures,seeds=[11,23,37],phases_per_trial=self.budget,updates_per_phase=512,batch_size=128,symmetry_average='D4' if self.symmetry else None,forward_passes_per_move=8 if self.symmetry else 1,gradient_check_max_error=gradient_error,
                      learning_rate='0.01 first two phases, 0.003 thereafter',teacher_kl=.01,temperature=2,gradient_clip=1,
                      objective='First phase optimal-set CE; later safe-move-set CE. Losing positions accept all legal actions, conserving capacity for no-loss.',
                      target='No losing opponent continuation from empty as either X or O; full optimality graded separately',
                      sampler='Half native/adversarial-state weighted, half uniform retention. First 32 updates use25%critical or unsafe states.',
                      baseline_parameters=6537,teacher_sha256=digest(teacher_path),corpus_sha256=digest(self.root/'corpus.json'),source_hashes=sources,
                      minimality_claim='Smallest verified successful model in this bounded search only; failed optimization is not an impossibility proof.')
        atomic_json(self.root/'compression-protocol.json',protocol)
        self.state.update(dataset={'reachable':5478,'decision_states':4520,'train':4520,'held_out':0},compression={'baseline_parameters':6537,'candidates':[],'scope':protocol['minimality_claim']})
        self.state['phases']=[dict(id='search',title='Search smaller dense policies',detail='Native collection → counterexample review → retention, then exact adversarial diagnosis.',status='running',progress=0,metrics={}),dict(id='qualification',title='Independently certify the smallest success',detail='Every opponent continuation as X/O; all-state coverage and browser parity reported separately.',status='pending',progress=0,metrics={})]
        best=None; trials=[];self.parts={'train':np.arange(4520),'validation':np.array([],dtype=int)}
        for ai,arch in enumerate(architectures):
            size=sum((a+1)*b for a,b in zip(arch,arch[1:]))
            if best and size>=best['parameters'] and not self.resume:continue
            x=features(self.boards,arch[0]);tag='-'.join(map(str,arch));arch_best=None
            for seed in (11,23,37):
                self.control();net=Dense(arch,seed,self.symmetry)
                if self.resume:net.params=[np.array(v) for l in resume_model['layers'] for v in (l['weights'],l['bias'])]
                adam=Adam(net);rng=np.random.default_rng(seed);history=[]
                for phase in range(1,self.budget+1):
                    before=self.assess(net,x);counts=np.ones(4520)
                    for idx,n in before['counts'].items():counts[idx]+=n*20
                    for idx in before['critical']:counts[idx]+=100
                    # Collect actual current-network actions from seeded TRAIN starts.
                    native=[]
                    for start in rng.choice(4520,128):
                        b=self.boards[int(start)]
                        while winner(b) is None:
                            idx=self.lookup[b];move=int(before['choices'][idx]);counts[idx]+=1;native.append([idx,move]);b=play(b,move)
                    targets=before['critical'] or np.flatnonzero(~before['safe']).tolist()
                    schedule=np.concatenate([rng.choice(4520,(512,64),p=counts/counts.sum()),rng.choice(4520,(512,64))],1)
                    if targets:schedule[:32,:32]=rng.choice(targets,(32,32))
                    stem=f'a{tag}-s{seed}-p{phase}';np.savez_compressed(self.root/(stem+'-schedule.npz'),indices=schedule,native=np.array(native),critical=np.array(before['critical']),multiplicity=counts)
                    with np.load(self.root/(stem+'-schedule.npz'),allow_pickle=False) as d:schedule=d['indices'].copy()
                    for u,ids in enumerate(schedule,1):
                        target=self.optimal[ids] if phase==1 else self.safe[ids]
                        loss,grads=net.objective(x[ids],self.legal[ids],target,parent_logits[ids],np.ones(128,dtype=bool),kl=.01)
                        adam.step(net,grads,.01 if phase<=2 else .003);self.step+=1
                        if u%64==0:self.control()
                    result=self.assess(net,x)
                    name=stem;model=net.export(name,self.root.name)
                    row=dict(architecture=arch,parameters=size,seed=seed,phase=phase,checkpoint=name,symmetry_average='D4' if self.symmetry else None,forward_passes_per_move=8 if self.symmetry else 1,no_loss_from_empty=result['no_loss_from_empty'],optimal_count=result['correct'],total=4520,safe_count=result['safe_correct'],as_x=result['as_x'],as_o=result['as_o'],loss=float(loss),critical_count=len(result['critical']))
                    history.append(row);self.state['metrics']={'step':self.step,'epoch':phase*512,'branch':seed,'loss':float(loss),'train_accuracy':result['correct']/4520,'val_accuracy':None}
                    self.state['history'].append(dict(self.state['metrics'],phase='search'))
                    self.phase('search',min((ai+(seed==23)/3+(seed==37)*2/3+phase/self.budget/3)/len(architectures),.99),{'architecture':tag,'parameters':size,'seed':seed,'phase':phase,'critical_states':len(result['critical'])})
                    if arch_best is None or (row['no_loss_from_empty'],row['optimal_count'])>(arch_best['no_loss_from_empty'],arch_best['optimal_count']):arch_best=row.copy()
                    if row['no_loss_from_empty']:
                        atomic_json(self.root/(name+'.json'),model)
                        np.savez(self.root/(name+'.npz'),**{f'p{i}':p for i,p in enumerate(net.params)})
                        if best is None or (size,-row['optimal_count'])<(best['parameters'],-best['optimal_count']):
                            best=row.copy();atomic_json(self.root/'best-so-far.json',best);atomic_json(self.root/'model.json',model)
                            self.state.update(model_available=True,model_version=name)
                        self.event(f'Certified by training diagnosis: {tag}, {size} parameters, seed{seed}, phase{phase}; {row["optimal_count"]}/4520 optimal, non-losing both sides.')
                        break
                trials.append(dict(architecture=arch,parameters=size,seed=seed,phases=history))
                atomic_json(self.root/'search-trials.json',trials)
                if best and best['parameters']==size:break
            self.state['compression']['candidates'].append(arch_best);self.state['compression']['trial_count']=len(trials);self.flush()
        if not best:raise RuntimeError('No smaller no-loss policy found in the registered budget.')
        self.phase('search',1);self.state['selection']=best
        model_path=self.root/(best['checkpoint']+'.json');model=json.loads(model_path.read_text())
        atomic_json(self.root/'selection-before-qualification.json',{'selected':best,'model_sha256':digest(model_path),'protocol_sha256':digest(self.root/'compression-protocol.json'),'frozen_at':datetime.now(timezone.utc).isoformat()})
        self.phase('qualification')
        from verify_export import qualify
        independent=qualify(model,include_fixtures=True);atomic_json(self.root/'independent-verification.json',independent)
        if not independent['no_loss_from_empty']:raise RuntimeError('Independent no-loss gate failed.')
        check=subprocess.run(['node',str(Path(__file__).parent/'verify_portable.mjs'),'--model',str(model_path),'--fixtures',str(self.root/'independent-verification.json'),'--output',str(self.root/'portable-verification.json')],capture_output=True,text=True)
        if check.returncode:raise RuntimeError(check.stderr[-1000:])
        self.state['compression'].update(selected_parameters=best['parameters'],selected_checkpoint=best['checkpoint'],reduction_fraction=1-best['parameters']/6537)
        self.state.update(architecture=best['architecture'],parameter_count=best['parameters'],mastery_achieved=True)
        self.state['evaluation']={'exhaustive':{'correct':independent['optimal_count'],'count':4520,'accuracy':independent['optimal_rate'],'mistakes':independent['mistake_count']},
            'independent':{k:independent[k] for k in ('optimal_count','total','no_loss_from_empty','perfect_policy')},'worst_case':{'as_x':independent['as_x']['result'],'as_o':independent['as_o']['result']},
            'browser_export':json.loads((self.root/'portable-verification.json').read_text()),'scope':protocol['minimality_claim']+' All states were training data; no held-out claim.','no_solver_fallback':True,'selected_checkpoint':best['checkpoint']}
        atomic_json(self.root/'evaluation.json',self.state['evaluation']);atomic_json(self.root/'model.json',model)
        atomic_json(self.root/'qualification-consumption.json',{'selection_sha256':digest(self.root/'selection-before-qualification.json'),'independent_sha256':digest(self.root/'independent-verification.json'),'portable_sha256':digest(self.root/'portable-verification.json'),'consumed_at':datetime.now(timezone.utc).isoformat()})
        self.phase('qualification',1);self.state['status']='completed';self.event(f'Selected {best["parameters"]}-parameter network; independent no-loss X/O and browser parity pass.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--phases',type=int,default=8);p.add_argument('--seed',type=int,default=42);p.add_argument('--resume');p.add_argument('--symmetry',action='store_true');a=p.parse_args()
    exp=Compression(a.run_dir,a.phases,a.resume,a.symmetry)
    try:exp.run()
    except InterruptedError as exc:exp.state['status']='cancelled';exp.event(str(exc))
    except Exception as exc:exp.state.update(status='failed',error=str(exc));exp.event(str(exc));(exp.root/'traceback.txt').write_text(traceback.format_exc());raise
