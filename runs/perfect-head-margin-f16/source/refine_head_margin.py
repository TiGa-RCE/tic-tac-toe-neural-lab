"""Maximize a sparse neural head's exact finite-domain margin at fixed support.

The optional hidden quantization happens before fitting. Original native neural
actions are frozen as witnesses; final independent export qualification remains
separate. Execute deliberate model work through the supported local guard.
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

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from fit_sparse_head import build_basis, assess, digest, write_json
from iterative_sparse_head import read_masks, candidate_model
from solver import enumerate_states, solve, winner, play

ROOT=Path(__file__).resolve().parent


def main(args):
    root=args.run_dir
    if root.exists():raise ValueError('Choose a new run directory; prior evidence is immutable.')
    root.mkdir(parents=True)
    began=time.monotonic()
    original=json.loads(args.model.read_text())
    masks=read_masks(original)
    if len(masks)!=4:raise ValueError('Expected the two-layer folded D4 policy.')
    head_support=np.r_[masks[2].ravel(),masks[3]]
    if not head_support.any():raise ValueError('Expected a nonzero head support.')
    boards=[board for board in enumerate_states() if winner(board) is None]
    if len(boards)!=4520:raise ValueError('Expected all 4,520 reachable decision boards.')
    legal=np.array([[cell==0 for cell in board] for board in boards])
    optimal=np.zeros_like(legal)
    for index,board in enumerate(boards):optimal[index,list(solve(board)[1])]=True
    original_basis,original_head=build_basis(original,boards)
    choices,correct,original_margin=assess(original_basis,original_head,legal,optimal)
    if correct!=4520 or original_margin<=0:
        raise ValueError('The source must already be strictly optimal on all 4,520 boards.')
    if not optimal[np.arange(len(boards)),choices].all():
        raise AssertionError('Every frozen witness must be teacher-certified optimal.')
    parent=json.loads(json.dumps(original))
    for field in ('weights','bias'):
        values=np.array(parent['layers'][0][field],dtype=args.hidden_precision).astype(float)
        parent['layers'][0][field]=values.tolist()
    # Every source structural zero remains zero after quantization.
    if any(np.any(np.array(parent['layers'][0][field])[~mask]!=0)
           for field,mask in zip(('weights','bias'),masks[:2])):
        raise AssertionError('Hidden quantization changed a fixed zero.')
    basis,_=build_basis(parent,boards)
    anchor=int(np.argmax(np.where(head_support,np.abs(original_head),-np.inf)))
    anchor_sign=1. if original_head[anchor]>0 else -1.
    if original_head[anchor]==0:raise ValueError('Expected a nonzero output anchor.')
    normalized_source_head=original_head/abs(float(original_head[anchor]))

    lookup={board:index for index,board in enumerate(boards)}
    rng=np.random.default_rng(args.seed)
    starts=rng.choice(len(boards),128,replace=True)
    trace=[];counts=Counter()
    for start in starts:
        board=boards[int(start)]
        while winner(board) is None:
            index=lookup[board];action=int(choices[index]);trace.append((index,action));counts[index]+=1
            board=play(board,action)
    np.savez_compressed(root/'native-before-fit.npz',starts=starts,
        trace=np.array(trace,dtype=np.int64),choices=choices,
        native_counts=np.array([counts[index] for index in range(len(boards))],dtype=np.int64))
    write_json(root/'corpus.json',{'boards':boards,
        'optimal_moves':[np.flatnonzero(row).tolist() for row in optimal],
        'scope':'Every reachable decision board is supervised; no held-out generalization claim.'})
    row_ids=[]
    for index in sorted(range(len(boards)),key=lambda index:(-counts[index],index)):
        for bad in np.flatnonzero(legal[index]&~optimal[index]):
            row_ids.append((index,int(choices[index]),int(bad)))
    constraints=np.array([basis[index,chosen]-basis[index,bad] for index,chosen,bad in row_ids])
    np.savez_compressed(root/'constraints-before-fit.npz',matrix=constraints,
        board_chosen_bad=np.array(row_ids,dtype=np.int64),head_support=head_support,
        quantized_hidden_weights=np.array(parent['layers'][0]['weights']),
        quantized_hidden_bias=np.array(parent['layers'][0]['bias']))
    source_hashes={}
    for name in ('refine_head_margin.py','iterative_sparse_head.py','fit_sparse_head.py','compress_train.py','solver.py'):
        target=root/'source'/name;target.parent.mkdir(exist_ok=True)
        shutil.copyfile(ROOT/name,target);source_hashes[name]=digest(target)
    original_seal=args.model.parent/'selection-before-qualification.json'
    protocol={'registered_at':datetime.now(timezone.utc).isoformat(),
        'source_model':str(args.model),'source_model_sha256':digest(args.model),
        'source_selection_sha256':digest(original_seal) if original_seal.exists() else None,
        'source_protocols':{path.name:digest(path) for path in args.model.parent.glob('*protocol.json')},
        'source_hashes':source_hashes,'corpus_sha256':digest(root/'corpus.json'),
        'native_trace_sha256':digest(root/'native-before-fit.npz'),
        'constraints_sha256':digest(root/'constraints-before-fit.npz'),
        'hidden_precision':args.hidden_precision,'hidden_learned_count_frozen':sum(int(mask.sum()) for mask in masks[:2]),
        'head_support_count':int(head_support.sum()),'source_masked_count':sum(int(mask.sum()) for mask in masks),
        'head_support':head_support.tolist(),'head_bound':1.,'anchor_slot':anchor,'anchor_sign':anchor_sign,
        'seed':args.seed,'lp_time_limit_seconds':args.time_limit,
        'constraint_rows':len(row_ids),'original_optimal_count':correct,'original_minimum_margin':original_margin,
        'normalized_source_frozen_margin_after_hidden_quantization':float((constraints@normalized_source_head).min()),
        'objective':'Maximize t with every frozen witness-versus-suboptimal logit gap at least t, fixed head support, absolute head coefficients at most1, and one fixed unit anchor.',
        'witnesses':'Actual rounded-greedy actions of the original fully optimal neural source, frozen before hidden quantization and optimization.',
        'acceptance':'Positive actual frozen margin, every one of 4,520 rounded-greedy decisions optimal, bounds/support/unit-anchor retained. Final sparse quantized export qualification is separate.',
        'scope':'Fixed-support, fixed-hidden bounded margin refinement. It establishes neither a global support minimum nor a generalization claim.'}
    write_json(root/'refine-head-margin-protocol.json',protocol)
    # Variables are [head34,t]. A full-domain constraint is -C.head+t<=0.
    a_ub=sparse.hstack((-sparse.csr_matrix(constraints),np.ones((len(row_ids),1))),format='csr')
    bounds=[(-1.,1.) if active else (0.,0.) for active in head_support]+[(0.,None)]
    bounds[anchor]=(anchor_sign,anchor_sign)
    result=linprog(np.r_[np.zeros(34),-1.],A_ub=a_ub,b_ub=np.zeros(len(row_ids)),bounds=bounds,
        method='highs',options={'time_limit':args.time_limit,'primal_feasibility_tolerance':1e-9,
                                'dual_feasibility_tolerance':1e-9})
    row={'solver_status':int(result.status),'solver_message':str(result.message),'accepted':False,
         'hidden_precision':args.hidden_precision,'anchor_slot':anchor,'anchor_sign':anchor_sign}
    if result.x is not None and np.isfinite(result.x).all():
        head=result.x[:34].copy();head[~head_support]=0.
        actual_support=head_support&(head!=0)
        frozen_margin=float((constraints@head).min())
        _,count,gap=assess(basis,head,legal,optimal)
        bounds_ok=bool(np.abs(head).max()<=1.+1e-8 and abs(head[anchor]-anchor_sign)<=1e-8)
        row.update(optimal_count=count,total=4520,minimum_margin=gap,
                   minimum_frozen_constraint_margin=frozen_margin,optimized_margin=float(result.x[34]),
                   actual_head_support=actual_support.tolist(),head_l1=float(np.abs(head).sum()),bounds_satisfied=bounds_ok)
        np.savez_compressed(root/'lp-result.npz',head=head,margin=result.x[34],support=actual_support)
        checkpoint='refined-head-'+args.hidden_precision
        model,actual_count=candidate_model(parent,masks[:2],head,actual_support,checkpoint,root.name)
        model['hidden_coefficient_precision']=args.hidden_precision
        row.update(checkpoint=checkpoint,parameters=actual_count)
        write_json(root/(checkpoint+'.json'),model)
        row['accepted']=bool(bounds_ok and frozen_margin>0 and count==4520 and gap>0)
        if row['accepted']:
            write_json(root/'model.json',model)
            write_json(root/'selection-before-qualification.json',{'selected':row,
                'model_sha256':digest(root/'model.json'),
                'source_protocol_sha256':digest(root/'refine-head-margin-protocol.json'),
                'frozen_at':datetime.now(timezone.utc).isoformat(),
                'scope':'Training-only margin success; independent sparse export qualification remains required.'})
    write_json(root/'fit-result.json',row)
    status={'status':'awaiting-qualification' if row['accepted'] else 'incomplete','run_id':root.name,
        'training_mode':'finite-domain-fixed-support-margin-refinement','selection':row if row['accepted'] else None,
        'fit_result':row,'elapsed_seconds':round(time.monotonic()-began,3)}
    write_json(root/'status.json',status)
    print(json.dumps(status))
    return 0 if row['accepted'] else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--model',type=Path,default=ROOT/'runs/perfect-iterative-head-c/model.json')
    p.add_argument('--hidden-precision',choices=('float16','float64'),default='float64')
    p.add_argument('--time-limit',type=float,default=20.)
    p.add_argument('--seed',type=int,default=11)
    args=p.parse_args()
    if not 0<args.time_limit<=300 or not 0<=args.seed<2**32:p.error('Expected a bounded positive time limit and unsigned32-bit seed.')
    if args.run_dir.exists():p.error('Run directory already exists; preserve previous evidence.')
    try:
        raise SystemExit(main(args))
    except Exception:
        if args.run_dir.exists():
            (args.run_dir/'traceback.txt').write_text(traceback.format_exc())
            write_json(args.run_dir/'failure.json',{'status':'failed','error':traceback.format_exc()})
        raise
