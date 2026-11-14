"""Freeze and independently qualify explicit sparse/unit-constant neural weights."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import json
import shutil
import subprocess
from train import np, atomic_json, digest
from verify_export import qualify

ROOT=Path(__file__).resolve().parent


def pack_candidate(original, normalize):
    model=copy.deepcopy(original)
    masks=model.pop('learned_parameter_mask',None)
    if masks:
        if isinstance(masks[0],dict):
            masks=[np.array(value,dtype=bool) for layer in masks for value in (layer['weights'],layer['bias'])]
        else:
            masks=[np.array(value,dtype=bool) for value in masks]
        if len(masks)!=2*len(model['layers']):raise ValueError('Expected one mask per weight/bias array')
    else:
        masks=[np.ones_like(np.array(value),dtype=bool) for layer in model['layers'] for value in (layer['weights'],layer['bias'])]
    for layer,wm,bm in zip(model['layers'],masks[::2],masks[1::2]):
        for field,mask in (('weights',wm),('bias',bm)):
            values=np.array(layer[field],dtype=float)
            if values.shape!=mask.shape or np.any(values[~mask]!=0):raise ValueError('Only exact structural zeros may be omitted')
            layer[field]=values.tolist()
    if model['architecture'][-1]==9:
        if any(not mask.all() for mask in masks):raise ValueError('Fold nine-readout models before masked pruning')
        layer=model['layers'][-1];w=np.array(layer['weights']);b=np.array(layer['bias'])
        layer['weights']=(w[:,:8]-w[:,8,None]).tolist();layer['bias']=(b[:8]-b[8]).tolist()
        model['architecture'][-1]=8;model['reference_logit_zero']=True
        if model.get('symmetry_average')=='D4':
            corner=float(b[[0,2,6,8]].mean());edge=float(b[[1,3,5,7]].mean())
            layer['bias']=[edge-corner,float(b[4])-corner];model['output_bias_orbits']='D4-corner-zero'
        masks=[np.ones_like(np.array(value),dtype=bool) for layer in model['layers'] for value in (layer['weights'],layer['bias'])]
    for layer,wm,bm in zip(model['layers'],masks[::2],masks[1::2]):
        layer['weights']=[[float(value) if wm[i,j] and value!=0 else None for j,value in enumerate(row)] for i,row in enumerate(layer['weights'])]
        layer['bias']=[float(value) if bm[i] and value!=0 else None for i,value in enumerate(layer['bias'])]
    if normalize:
        head=model['layers'][-1]
        slots=[('weights',i,j,value) for i,row in enumerate(head['weights']) for j,value in enumerate(row) if value is not None]
        slots += [('bias',i,None,value) for i,value in enumerate(head['bias']) if value is not None]
        field,i,j,value=max(slots,key=lambda slot:abs(slot[3]))
        scale=abs(value)
        if scale==0:raise ValueError('Output scale cannot be zero')
        head['weights']=[[None if value is None else value/scale for value in row] for row in head['weights']]
        head['bias']=[None if value is None else value/scale for value in head['bias']]
        marker='fixed:+1' if value>0 else 'fixed:-1'
        if j is None:head[field][i]=marker
        else:head[field][i][j]=marker
        # The discarded scale is intentionally not stored in the model.
        model['output_scale_normalized']=True
    for key in ('search_only_masked_dense','deployment_requirement','learned_parameter_count','effective_parameter_count','head_active_mask','intermediate_format'):
        model.pop(key,None)
    model['coefficient_storage']='sparse-fixed-v1'
    return model


def numeric_values(model):
    return [value for layer in model['layers'] for field in ('weights','bias')
            for value in (sum(layer[field],[]) if field=='weights' else layer[field])
            if type(value) in (int,float)]


def main(args):
    source=Path(args.source);run=Path(args.run_dir);run.mkdir(parents=True,exist_ok=False)
    original=json.loads(source.read_text())
    transformed=pack_candidate(original,args.normalize)
    count=len(numeric_values(transformed))
    if count>=74:raise ValueError('This export must store fewer than 74 learned scalars')
    files=('finalize_sparse.py','verify_export.py','verify_portable.mjs','web/network.js','web/training-worker.js')
    hashes={}
    for name in files:
        target=run/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target);hashes[name]=digest(target)
    source_seal=source.parent/'selection-before-qualification.json'
    if not source_seal.exists():source_seal=source.parent/'best-before-qualification.json'
    protocol={'registered_at':datetime.now(timezone.utc).isoformat(),'source_run':source.parent.name,
              'source_model':source.name,'source_model_sha256':digest(source),
              'source_selection_sha256':digest(source_seal) if source_seal.exists() else None,
              'source_selection_file':source_seal.name if source_seal.exists() else None,
              'source_protocols':{path.name:digest(path) for path in source.parent.glob('*protocol.json')},
              'source_hashes':hashes,'normalization':args.normalize,
              'format':'sparse-fixed-v1: numeric learned scalars, null structural zeros, explicit fixed unit markers',
              'precision_order':['float16','float32','float64'],
              'acceptance':'Every 4520 independent decision optimal; no loss as X/O; exact browser actions and absolute raw-logit tolerance1e-9.',
              'minimality_scope':'Smallest verified in retained bounded searches, not a proof of global minimality.'}
    atomic_json(run/'transformation-before-qualification.json',protocol)
    attempts=[];selected=None
    for precision in protocol['precision_order']:
        model=copy.deepcopy(transformed)
        for layer in model['layers']:
            layer['weights']=[[float(np.array(value,dtype=precision)) if type(value) in (int,float) else value for value in row] for row in layer['weights']]
            layer['bias']=[float(np.array(value,dtype=precision)) if type(value) in (int,float) else value for value in layer['bias']]
        checkpoint=f'sparse-perfect-{count}-{precision}'
        slots=[value for layer in model['layers'] for value in sum(layer['weights'],[])+layer['bias']]
        model.update(parameter_count=count,checkpoint=checkpoint,run_id=run.name,coefficient_precision=precision,
                     coefficient_payload_bytes=count*np.dtype(precision).itemsize,
                     structural_zero_slots=sum(value is None for value in slots),
                     fixed_unit_slots=sum(type(value) is str for value in slots),
                     dense_coefficient_slots=len(slots))
        atomic_json(run/(checkpoint+'.json'),model)
        audit=qualify(model,include_fixtures=True);atomic_json(run/(checkpoint+'-independent.json'),audit)
        result=subprocess.run(['node',str(ROOT/'verify_portable.mjs'),'--model',str(run/(checkpoint+'.json')),
                '--fixtures',str(run/(checkpoint+'-independent.json')),'--output',str(run/(checkpoint+'-portable.json'))],capture_output=True,text=True)
        parity=json.loads((run/(checkpoint+'-portable.json')).read_text())
        row={'precision':precision,'checkpoint':checkpoint,'parameters':count,'optimal_count':audit['optimal_count'],
             'perfect_policy':audit['perfect_policy'],'no_loss_from_empty':audit['no_loss_from_empty'],
             'portable':parity,'model_sha256':digest(run/(checkpoint+'.json'))}
        attempts.append(row);atomic_json(run/'transformation-attempts.json',attempts)
        if audit['perfect_policy'] and audit['optimal_count']==4520 and audit['no_loss_from_empty'] and result.returncode==0 and parity['action_mismatches']==0:
            selected=model,audit,parity,row;break
    if not selected:raise RuntimeError('No precision passed the unchanged acceptance gate')
    model,audit,parity,row=selected
    for name,value in (('model',model),('independent-verification',audit),('portable-verification',parity)):
        atomic_json(run/(name+'.json'),value)
    np.array(numeric_values(model),dtype=model['coefficient_precision']).tofile(run/'coefficients.bin')
    evaluation={'exhaustive':{'correct':4520,'count':4520,'accuracy':1.,'mistakes':0},
                'independent':{key:audit[key] for key in ('optimal_count','total','no_loss_from_empty','perfect_policy')},
                'worst_case':{'as_x':audit['as_x']['result'],'as_o':audit['as_o']['result']},'browser_export':parity,
                'coefficient_payload_bytes':(run/'coefficients.bin').stat().st_size,'selected_checkpoint':model['checkpoint'],
                'structural_zero_slots':model['structural_zero_slots'],'fixed_unit_slots':model['fixed_unit_slots'],
                'scope':'Sparse learned coefficients; structural positions and runtime/JSON overhead are additional. All finite decision boards supervised.',
                'no_solver_fallback':True}
    atomic_json(run/'evaluation.json',evaluation)
    now=datetime.now(timezone.utc).isoformat()
    status={'status':'completed','run_id':run.name,'phase_id':'qualification','training_mode':'finite-domain-sparse-perfect-export',
            'architecture':model['architecture'],'parameter_count':count,'model_available':True,'model_version':model['checkpoint'],
            'dataset':{'reachable':5478,'decision_states':4520,'train':4520,'held_out':0},
            'metrics':{'train_accuracy':1.,'val_accuracy':None,'loss':None},'evaluation':evaluation,'selection':row,'history':[],
            'recipe_note':'Exact optimal-action teaching, native action review, sparse fitting and retention; independent full-domain qualification. No solver during play.',
            'phases':[{'id':key,'title':title,'detail':detail,'status':'completed','progress':1.,'metrics':{}} for key,title,detail in (
                ('search','Fit a smaller neural policy','Teacher labels, native collection, targeted review and full-domain retention.'),
                ('serialize','Freeze sparse neural coefficients','Structural zeros and fixed unit constants stored explicitly; no per-board correction table.'),
                ('qualification','Certify every decision','All 4,520 optimal; no losing continuation as X/O; exact browser actions.'))],
            'events':[{'time':now,'message':f'Frozen {count}-coefficient sparse policy independently passed all 4520 decisions and strict browser parity.'}]}
    atomic_json(run/'status.json',status)
    atomic_json(run/'qualification-consumption.json',{'transformation_sha256':digest(run/'transformation-before-qualification.json'),
                'model_sha256':digest(run/'model.json'),'independent_sha256':digest(run/'independent-verification.json'),
                'portable_sha256':digest(run/'portable-verification.json'),'consumed_at':now})
    print(json.dumps({'run_id':run.name,'parameters':count,'precision':model['coefficient_precision'],'payload_bytes':evaluation['coefficient_payload_bytes'],
                      'zeros':model['structural_zero_slots'],'units':model['fixed_unit_slots'],'optimal_count':4520,'portable':parity}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--run-dir',required=True)
    p.add_argument('--normalize',action='store_true');main(p.parse_args())
