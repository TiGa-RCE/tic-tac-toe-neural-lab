"""Fold and qualify a compact network only if all 4,520 neural decisions are optimal."""
from train import np, atomic_json, digest
from verify_export import qualify
from pathlib import Path
from datetime import datetime, timezone
import argparse, copy, json, shutil, subprocess

def main(source,root):
    root.mkdir(parents=True,exist_ok=False)
    selected=json.loads((source/'selection-before-qualification.json').read_text())['selected']
    original_path=source/(selected['checkpoint']+'.json');original=json.loads(original_path.read_text())
    sources={}
    for p in (Path(__file__),Path(__file__).parent/'verify_export.py',Path(__file__).parent/'web/network.js',Path(__file__).parent/'web/training-worker.js'):
        target=root/'source'/p.relative_to(Path(__file__).parent);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        sources[str(p.relative_to(Path(__file__).parent))]=digest(p)
    protocol={'goal':'100% optimal neural decisions on all 4520 reachable decision boards',
              'source_run':source.name,'source_model':original_path.name,'source_model_sha256':digest(original_path),
              'source_protocol_sha256':digest(source/'perfect-protocol.json'),'source_hashes':sources,
              'transform':'Remove common ninth readout; for D4 also share output biases over geometric orbits and subtract common corner bias.',
              'precision_order':['float16','float32','float64'],'acceptance':'perfect_policy AND optimal_count==4520 AND no_loss_from_empty AND exact browser action parity at unchanged 1e-9 logit tolerance',
              'minimality_scope':'Smallest successful model found in the recorded bounded search, not a proof of global minimality.',
              'registered_at':datetime.now(timezone.utc).isoformat()}
    atomic_json(root/'transformation-before-qualification.json',protocol)
    folded=copy.deepcopy(original);last=folded['layers'][-1];w=np.array(last['weights']);b=np.array(last['bias'])
    last['weights']=(w[:,:8]-w[:,8,None]).tolist();last['bias']=(b[:8]-b[8]).tolist()
    folded['architecture'][-1]=8;folded['reference_logit_zero']=True
    if original.get('symmetry_average')=='D4':
        c=float(b[[0,2,6,8]].mean());e=float(b[[1,3,5,7]].mean())
        last['bias']=[e-c,float(b[4])-c];folded['output_bias_orbits']='D4-corner-zero'
    attempts=[];chosen=None
    for precision in ('float16','float32','float64'):
        model=copy.deepcopy(folded)
        for layer in model['layers']:
            for field in ('weights','bias'):layer[field]=np.array(layer[field],dtype=precision).astype(float).tolist()
        count=sum(sum(map(len,l['weights']))+len(l['bias']) for l in model['layers'])
        name=f'perfect-{count}-{precision}';model.update(parameter_count=count,checkpoint=name,run_id=root.name,coefficient_precision=precision)
        atomic_json(root/(name+'.json'),model)
        audit=qualify(model,include_fixtures=True);atomic_json(root/(name+'-independent.json'),audit)
        result=subprocess.run(['node',str(Path(__file__).parent/'verify_portable.mjs'),'--model',str(root/(name+'.json')),
                  '--fixtures',str(root/(name+'-independent.json')),'--output',str(root/(name+'-portable.json'))],capture_output=True,text=True)
        parity=json.loads((root/(name+'-portable.json')).read_text())
        row={'checkpoint':name,'parameters':count,'precision':precision,'optimal_count':audit['optimal_count'],
             'perfect_policy':audit['perfect_policy'],'no_loss_from_empty':audit['no_loss_from_empty'],
             'portable':parity,'model_sha256':digest(root/(name+'.json'))}
        attempts.append(row);atomic_json(root/'transformation-attempts.json',attempts)
        if audit['perfect_policy'] and audit['optimal_count']==4520 and audit['no_loss_from_empty'] and result.returncode==0:
            chosen=model,audit,parity,row;break
    if not chosen:raise RuntimeError('No folded precision passed full optimality and strict browser parity.')
    model,audit,parity,row=chosen
    atomic_json(root/'model.json',model);atomic_json(root/'independent-verification.json',audit);atomic_json(root/'portable-verification.json',parity)
    coefficients=[v for layer in model['layers'] for field in ('weights','bias') for v in np.array(layer[field]).flat]
    np.array(coefficients,dtype=model['coefficient_precision']).tofile(root/'coefficients.bin')
    evaluation={'exhaustive':{'correct':4520,'count':4520,'accuracy':1.,'mistakes':0},
                'independent':{k:audit[k] for k in ('optimal_count','total','no_loss_from_empty','perfect_policy')},
                'worst_case':{'as_x':audit['as_x']['result'],'as_o':audit['as_o']['result']},'browser_export':parity,
                'coefficient_payload_bytes':(root/'coefficients.bin').stat().st_size,'selected_checkpoint':model['checkpoint'],
                'scope':'All 4520 positions deliberately supervised. Full finite-domain optimality; no held-out generalization claim.',
                'no_solver_fallback':True}
    atomic_json(root/'evaluation.json',evaluation)
    search=json.loads((source/'status.json').read_text())
    status={**search,'status':'completed','run_id':root.name,'phase_id':'qualification','training_mode':'finite-domain-perfect-export',
            'architecture':model['architecture'],'parameter_count':model['parameter_count'],'model_version':model['checkpoint'],'model_available':True,
            'evaluation':evaluation,'selection':row,'metrics':{'train_accuracy':1.,'val_accuracy':None,'loss':None},
            'recipe_note':'Optimal-action teaching on every board, error review and retention. Frozen pure-neural export independently optimal on all 4520 boards; no solver fallback.',
            'phases':[{'id':'search','title':'Search compact fully optimal networks','detail':'Optimal-set supervision throughout; every smaller failed trial retained.','status':'completed','progress':1,'metrics':{}},
                      {'id':'fold','title':'Fold redundant coefficients','detail':'Remove common logits and share D4 output biases where applicable.','status':'completed','progress':1,'metrics':{'parameters':count,'precision':model['coefficient_precision']}},
                      {'id':'qualification','title':'Certify all 4,520 optimal decisions','detail':'Independent oracle and inference, exact browser action parity and both-side adversarial check.','status':'completed','progress':1,'metrics':{'optimal':4520,'total':4520}}]}
    status.pop('error',None);atomic_json(root/'status.json',status)
    atomic_json(root/'qualification-consumption.json',{'transformation_sha256':digest(root/'transformation-before-qualification.json'),
                'model_sha256':digest(root/'model.json'),'independent_sha256':digest(root/'independent-verification.json'),
                'portable_sha256':digest(root/'portable-verification.json'),'consumed_at':datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'run_id':root.name,'parameters':count,'precision':model['coefficient_precision'],'payload_bytes':evaluation['coefficient_payload_bytes'],
                      'optimal_count':audit['optimal_count'],'no_loss_both':audit['no_loss_from_empty'],'portable':parity}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-run',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();main(a.source_run,a.run_dir)
