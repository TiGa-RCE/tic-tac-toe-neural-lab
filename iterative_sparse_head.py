"""Bounded sparse-head refitting with refreshed optimal neural witnesses.

Execute only under the supported model-work guard. This trainer writes masked
dense intermediates; deployment requires a separate sparse export qualification.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
import traceback

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy import sparse

from fit_sparse_head import build_basis, assess, digest, write_json
from solver import enumerate_states, solve, winner, play

ROOT=Path(__file__).resolve().parent


def finite_stat(value):
    return float(value) if value is not None and np.isfinite(value) else None


def read_masks(model):
    supplied=model.get('learned_parameter_mask')
    if supplied is None:
        arrays=[np.array(value,dtype=float) for layer in model['layers']
                for value in (layer['weights'],layer['bias'])]
        return [array!=0 for array in arrays]
    if isinstance(supplied[0],dict):
        supplied=[value for layer in supplied for value in (layer['weights'],layer['bias'])]
    masks=[np.array(value,dtype=bool) for value in supplied]
    params=[np.array(value,dtype=float) for layer in model['layers']
            for value in (layer['weights'],layer['bias'])]
    if len(masks)!=len(params) or any(mask.shape!=p.shape for mask,p in zip(masks,params)):
        raise ValueError('Source mask dimensions do not match its layers.')
    if any(np.any(p[~mask]!=0) for mask,p in zip(masks,params)):
        raise ValueError('Source structural zeros must be exact zeros.')
    return [mask & (p!=0) for mask,p in zip(masks,params)]


def candidate_model(parent,hidden_masks,head,active,checkpoint,run_id):
    model=json.loads(json.dumps(parent))
    head=head.copy();head[~active]=0.
    first_masks=[mask.copy() for mask in hidden_masks]
    dead=[]
    for unit in range(4):
        if not np.any(active[unit*8:(unit+1)*8]):
            first_masks[0][:,unit]=False;first_masks[1][unit]=False;dead.append(unit)
    first=model['layers'][0]
    w=np.array(first['weights'],dtype=float);b=np.array(first['bias'],dtype=float)
    w[~first_masks[0]]=0.;b[~first_masks[1]]=0.
    first['weights']=w.tolist();first['bias']=b.tolist()
    model['layers'][1]['weights']=head[:32].reshape(4,8).tolist()
    model['layers'][1]['bias']=head[32:].tolist()
    masks=first_masks+[active[:32].reshape(4,8),active[32:]]
    count=sum(int(mask.sum()) for mask in masks)
    model.update(checkpoint=checkpoint,run_id=run_id,parameter_count=count,
                 learned_parameter_count=count,effective_parameter_count=count,
                 coefficient_precision='float64',learned_parameter_mask=[mask.tolist() for mask in masks],
                 head_active_mask=active.tolist(),dropped_hidden_units=dead,
                 search_only_masked_dense=True,
                 deployment_requirement='Omit exact zeros and independently qualify the final exported browser model.')
    return model,count


def main(args):
    root=args.run_dir
    if root.exists():raise ValueError('Choose a new immutable run directory.')
    root.mkdir(parents=True)
    began=time.monotonic()
    parent=json.loads(args.model.read_text())
    masks=read_masks(parent)
    hidden_masks=masks[:2]
    base_count=sum(int(mask.sum()) for mask in masks)
    boards=[board for board in enumerate_states() if winner(board) is None]
    if len(boards)!=4520:raise ValueError('Expected exactly 4,520 decision boards.')
    legal=np.array([[cell==0 for cell in board] for board in boards])
    optimal=np.zeros_like(legal)
    for index,board in enumerate(boards):optimal[index,list(solve(board)[1])]=True
    basis,parent_head=build_basis(parent,boards)
    choices,correct,margin=assess(basis,parent_head,legal,optimal)
    if correct!=4520 or margin<=0:raise ValueError('The supplied source must already be strictly optimal everywhere.')
    write_json(root/'corpus.json',{'boards':boards,'optimal_moves':[np.flatnonzero(row).tolist() for row in optimal],
        'scope':'All 4,520 positions are deliberate teaching data; no held-out generalization claim.'})
    source_hashes={}
    for name in ('iterative_sparse_head.py','fit_sparse_head.py','compress_train.py','solver.py'):
        target=root/'source'/name;target.parent.mkdir(exist_ok=True)
        shutil.copyfile(ROOT/name,target);source_hashes[name]=digest(target)
    protocol={'registered_at':datetime.now(timezone.utc).isoformat(),
        'source_model':str(args.model),'source_model_sha256':digest(args.model),
        'source_hashes':source_hashes,'corpus_sha256':digest(root/'corpus.json'),
        'hidden_learned_coefficients_frozen':sum(int(mask.sum()) for mask in hidden_masks),
        'source_active_coefficients':base_count,'head_slots':34,'rounds_per_branch':args.rounds,
        'branch_count':args.orders,'seed':args.seed,'maximum_pruning_attempts_per_round':args.max_attempts,
        'lp_time_limit_seconds':args.time_limit,'margin_target':1.,'clean_zero_threshold':args.clean_zero,
        'milp':{'time_limit_seconds':args.milp_seconds,'calls':'At most once, first branch/round, before LP refitting.',
                'head_bound_M':args.milp_bound,'margin':args.milp_margin,
                'anchor':'Largest-magnitude source head coefficient fixed to its sign ±1 after positive common scaling.',
                'objective':'Minimize 34 binary support indicators, with all frozen optimal-versus-bad constraints.',
                'scope':'M and positive margin restrict the search; an unsuccessful bounded solve is not an impossibility proof.'},
        'witness_refresh':'Before every round, actual greedy choices from the last accepted head are checked against all exact optimal-action sets and frozen before LP calls.',
        'objective':'Weighted L1 head magnitude; round1 ordinary L1, later rounds capped inverse-magnitude reweighting. All frozen optimal-versus-suboptimal inequalities retained.',
        'native_collection':'128 current-policy trajectories per round; exact native actions/counts saved before fitting. Native counts order constraint rows only; full-domain retention remains mandatory.',
        'selection':'Fewest actually retained coefficients (including hidden mask and excluding only proven dead units), then minimum head L1, then earlier attempt.',
        'acceptance':'Independent export qualification is separate. Training candidates must satisfy all frozen margin inequalities and every one of 4,520 rounded-greedy decisions.',
        'scope':'Bounded witness/reweighting/pruning search; failed fits do not establish global minimality; no solver fallback during exported play.'}
    write_json(root/'iterative-head-protocol.json',protocol)
    lookup={board:index for index,board in enumerate(boards)}
    trials=[];rounds=[];best=None
    state={'status':'running','run_id':root.name,'training_mode':'finite-domain-iterative-sparse-head',
           'trial_count':0,'selection':None,'rounds':rounds}

    def flush():
        state.update(trial_count=len(trials),selection=best,elapsed_seconds=round(time.monotonic()-began,3))
        write_json(root/'status.json',state)
        write_json(root/'iterative-head-trials.json',trials)

    def save_candidate(head,active,record):
        nonlocal best
        checkpoint=f'head-b{record["branch"]}-r{record["round"]}-a{record["attempt"]}'
        model,count=candidate_model(parent,hidden_masks,head,active,checkpoint,root.name)
        record.update(checkpoint=checkpoint,parameters=count,active_head_count=int(active.sum()))
        write_json(root/(checkpoint+'.json'),model)
        if best is None or (count,record['head_l1'])<(best['parameters'],best['head_l1']):
            best=dict(record)
            write_json(root/'model.json',model)
            write_json(root/'best-before-qualification.json',best)

    identity=sparse.eye(34,format='csr')
    for branch in range(args.orders):
        current_head=parent_head.copy()
        order_rng=np.random.default_rng(args.seed+branch)
        for round_index in range(1,args.rounds+1):
            choices,correct,margin=assess(basis,current_head,legal,optimal)
            if correct!=4520 or margin<=0:raise AssertionError('An accepted incumbent lost full optimality.')
            stem=f'b{branch}-r{round_index}'
            starts=order_rng.choice(len(boards),128,replace=True)
            trace=[];counts=Counter()
            for start in starts:
                board=boards[int(start)]
                while winner(board) is None:
                    index=lookup[board];action=int(choices[index]);trace.append((index,action));counts[index]+=1
                    board=play(board,action)
            np.savez_compressed(root/(stem+'-native.npz'),starts=starts,
                                trace=np.array(trace,dtype=np.int64),choices=choices,
                                native_counts=np.array([counts[index] for index in range(len(boards))],dtype=np.int64))
            row_ids=[]
            # A native-prioritized ordering never drops any full-domain constraint.
            for index in sorted(range(len(boards)),key=lambda index:(-counts[index],index)):
                for bad in np.flatnonzero(legal[index]&~optimal[index]):
                    row_ids.append((index,int(choices[index]),int(bad)))
            constraints=np.array([basis[index,chosen]-basis[index,bad] for index,chosen,bad in row_ids])
            np.savez_compressed(root/(stem+'-constraints.npz'),matrix=constraints,
                                board_chosen_bad=np.array(row_ids,dtype=np.int64),rhs=np.ones(len(row_ids)))
            abs_head=np.abs(current_head)
            if round_index==1:
                objective_weights=np.ones(34)
            else:
                scale=max(float(np.median(abs_head[abs_head>0])),1e-6)
                objective_weights=np.clip(scale/(abs_head+scale*.05),.1,20.)
            if branch:objective_weights*=1+order_rng.uniform(0,1e-6,34)
            round_record={'branch':branch,'round':round_index,'registered_at':datetime.now(timezone.utc).isoformat(),
                'incumbent_choices_correct':correct,'incumbent_minimum_margin':margin,
                'native_file':stem+'-native.npz','native_sha256':digest(root/(stem+'-native.npz')),
                'constraints_file':stem+'-constraints.npz','constraints_sha256':digest(root/(stem+'-constraints.npz')),
                'constraint_rows':len(row_ids),'objective_weights':objective_weights.tolist(),
                'incumbent_head':current_head.tolist()}
            write_json(root/(stem+'-before-fit.json'),round_record)
            rounds.append(round_record);flush()
            a_ub=sparse.vstack((sparse.hstack((-sparse.csr_matrix(constraints),sparse.csr_matrix((len(row_ids),34)))),
                                sparse.hstack((identity,-identity)),sparse.hstack((-identity,-identity))),format='csr')
            b_ub=np.r_[-np.ones(len(row_ids)),np.zeros(68)]

            if branch==0 and round_index==1 and args.milp_seconds>0:
                anchor=int(np.argmax(np.abs(current_head)))
                normalized=current_head/abs(float(current_head[anchor]))
                anchor_sign=1. if normalized[anchor]>0 else -1.
                normalized_margin=float((constraints@normalized).min())
                bound=args.milp_bound
                milp_record={'branch':branch,'round':round_index,'attempt':'milp','method':'MILP',
                    'before_fit_sha256':digest(root/(stem+'-before-fit.json')),
                    'anchor_slot':anchor,'anchor_sign':anchor_sign,'head_bound':bound,'target_margin':args.milp_margin,
                    'source_normalized_frozen_margin':normalized_margin,
                    'source_within_registered_bounds':bool(np.abs(normalized).max()<=bound and normalized_margin>=args.milp_margin),
                    'accepted':False}
                write_json(root/'milp-before-fit.json',{**milp_record,'registered_at':datetime.now(timezone.utc).isoformat(),
                    'time_limit_seconds':args.milp_seconds})
                # Variables are continuous head34 and binary support34. An
                # existing zero may reactivate; all exact teacher margins remain.
                matrix=sparse.vstack((sparse.hstack((sparse.csr_matrix(constraints),sparse.csr_matrix((len(row_ids),34)))),
                    sparse.hstack((identity,-bound*identity)),sparse.hstack((-identity,-bound*identity))),format='csc')
                lower=np.r_[np.full(len(row_ids),args.milp_margin),np.full(68,-np.inf)]
                upper=np.r_[np.full(len(row_ids),np.inf),np.zeros(68)]
                lb=np.r_[np.full(34,-bound),np.zeros(34)]
                ub=np.r_[np.full(34,bound),np.ones(34)]
                lb[anchor]=ub[anchor]=anchor_sign
                lb[34+anchor]=ub[34+anchor]=1.
                began_milp=time.monotonic()
                result=milp(np.r_[np.zeros(34),np.ones(34)],integrality=np.r_[np.zeros(34),np.ones(34)],
                    bounds=Bounds(lb,ub),constraints=LinearConstraint(matrix,lower,upper),
                    options={'time_limit':args.milp_seconds,'mip_rel_gap':0.,'presolve':True})
                milp_record.update(solver_status=int(result.status),solver_message=str(result.message),
                    elapsed_seconds=round(time.monotonic()-began_milp,6),
                    mip_gap=finite_stat(getattr(result,'mip_gap',None)),
                    mip_dual_bound=finite_stat(getattr(result,'mip_dual_bound',None)))
                if result.x is not None and np.isfinite(result.x).all():
                    head=result.x[:34].copy();indicator=result.x[34:].copy()
                    head[np.abs(head)<=args.clean_zero]=0.
                    actual_active=head!=0
                    frozen_margin=float((constraints@head).min())
                    _,count,gap=assess(basis,head,legal,optimal)
                    integral=bool(np.abs(indicator-np.rint(indicator)).max()<=1e-5)
                    bounds_ok=bool(np.abs(head).max()<=bound+1e-7 and abs(head[anchor]-anchor_sign)<=1e-7)
                    links_ok=bool(np.all(np.abs(head)<=bound*np.rint(indicator)+1e-7))
                    milp_record.update(actual_active_mask=actual_active.tolist(),optimal_count=count,total=4520,
                        minimum_frozen_constraint_margin=frozen_margin,minimum_margin=gap,head_l1=float(np.abs(head).sum()),
                        integral_indicators=integral,bounds_satisfied=bounds_ok,links_satisfied=links_ok)
                    np.savez_compressed(root/'milp-result.npz',head=head,indicator=indicator)
                    milp_record['accepted']=bool(integral and bounds_ok and links_ok and frozen_margin>=args.milp_margin-1e-8 and
                                                  count==4520 and gap>0)
                    if milp_record['accepted']:save_candidate(head,actual_active,milp_record)
                trials.append(milp_record);flush()

            def fit(active,attempt,removed):
                started=time.monotonic()
                bounds=[(None,None) if yes else (0.,0.) for yes in active]+[(0.,None) if yes else (0.,0.) for yes in active]
                result=linprog(np.r_[np.zeros(34),objective_weights],A_ub=a_ub,b_ub=b_ub,bounds=bounds,
                    method='highs',options={'time_limit':args.time_limit,'primal_feasibility_tolerance':1e-8,
                                            'dual_feasibility_tolerance':1e-8})
                record={'branch':branch,'round':round_index,'attempt':attempt,'removed_slot':removed,
                    'proposed_active_mask':active.tolist(),'before_fit_sha256':digest(root/(stem+'-before-fit.json')),
                    'solver_status':int(result.status),'solver_message':str(result.message),
                    'elapsed_seconds':round(time.monotonic()-started,6),'accepted':False}
                head=None;actual_active=active.copy()
                if result.success and result.x is not None:
                    head=result.x[:34].copy();head[~active]=0.
                    head[np.abs(head)<=args.clean_zero]=0.
                    actual_active &= head!=0
                    frozen_margin=float((constraints@head).min())
                    _,correct,gap=assess(basis,head,legal,optimal)
                    record.update(actual_active_mask=actual_active.tolist(),optimal_count=correct,total=4520,
                                  minimum_frozen_constraint_margin=frozen_margin,minimum_margin=gap,
                                  head_l1=float(np.abs(head).sum()))
                    record['accepted']=bool(np.isfinite(head).all() and frozen_margin>=1.-1e-6 and correct==4520 and gap>0)
                    if record['accepted']:save_candidate(head,actual_active,record)
                trials.append(record);flush()
                return (head,actual_active) if record['accepted'] else None

            initial=fit(np.ones(34,dtype=bool),0,None)
            if initial is None:continue
            head,active=initial
            if branch==0:
                order=np.argsort(np.abs(head))
            else:
                order=order_rng.permutation(34)
            for attempt,slot in enumerate(order[:args.max_attempts],1):
                if not active[slot]:continue
                proposed=active.copy();proposed[slot]=False
                fitted=fit(proposed,attempt,int(slot))
                if fitted is not None:head,active=fitted
            current_head=head
    state['status']='awaiting-qualification' if best and best['parameters']<base_count else 'incomplete'
    if best:
        write_json(root/'selection-before-qualification.json',{'selected':best,'model_sha256':digest(root/'model.json'),
            'source_protocol_sha256':digest(root/'iterative-head-protocol.json'),
            'frozen_at':datetime.now(timezone.utc).isoformat(),'scope':'Training-only success; independent sparse export qualification remains required.'})
    flush()
    print(json.dumps({'run_id':root.name,'status':state['status'],'best':best,'trial_count':len(trials),
                      'elapsed_seconds':state['elapsed_seconds']}))
    return 0 if state['status']=='awaiting-qualification' else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--model',type=Path,default=ROOT/'runs/perfect-pruning-b/model.json')
    p.add_argument('--rounds',type=int,default=3);p.add_argument('--orders',type=int,default=4)
    p.add_argument('--max-attempts',type=int,default=34);p.add_argument('--time-limit',type=float,default=5.)
    p.add_argument('--seed',type=int,default=11);p.add_argument('--clean-zero',type=float,default=1e-10)
    p.add_argument('--milp-seconds',type=float,default=0.)
    p.add_argument('--milp-bound',type=float,default=1.)
    p.add_argument('--milp-margin',type=float,default=1e-5)
    args=p.parse_args()
    if not (1<=args.rounds<=20 and 1<=args.orders<=20 and 0<=args.max_attempts<=34 and
            0<args.time_limit<=60 and 0<=args.seed<2**32-20 and 0<=args.clean_zero<=1e-6 and
            0<=args.milp_seconds<=300 and 1<=args.milp_bound<=100 and 0<args.milp_margin<=.1):
        p.error('Expected bounded positive rounds, branches and LP limits.')
    if args.run_dir.exists():p.error('Run directory already exists; preserve previous evidence.')
    try:
        raise SystemExit(main(args))
    except Exception:
        if args.run_dir.exists():
            (args.run_dir/'traceback.txt').write_text(traceback.format_exc())
            write_json(args.run_dir/'failure.json',{'status':'failed','error':traceback.format_exc()})
        raise
