"""Freeze and independently qualify exact algebraic compression of a learned D4 policy."""
from train import np, atomic_json, digest
from verify_export import qualify
from pathlib import Path
from datetime import datetime, timezone
import argparse, copy, json, subprocess, shutil

def main(source,root):
    root.mkdir(parents=True,exist_ok=False)
    selection=json.loads((source/'selection-before-qualification.json').read_text())['selected']
    original_path=source/(selection['checkpoint']+'.json');original=json.loads(original_path.read_text())
    protocol={'source_run':source.name,'source_model':original_path.name,'source_model_sha256':digest(original_path),
              'source_protocol_sha256':digest(source/'compression-protocol.json'),
              'transform':'Subtract ninth output weight column from the first eight and use a fixed ninth zero. Under D4, share output biases by corner/edge/center, then subtract common corner bias, leaving two learned coefficients.',
              'policy_score_rule':'floor(raw averaged logit * 10^12 + 0.5); lowest legal cell index breaks ties',
              'precision_order':['float16','float32','float64'],'selection':'Smallest independently no-loss and strictly portable derived format, then lowest coefficient bit width.',
              'scope':'Algebraic/model-format compression, not a new optimization or held-out test. Original failed strict D4 numerical qualification is preserved.',
              'registered_at':datetime.now(timezone.utc).isoformat()}
    atomic_json(root/'transformation-before-qualification.json',protocol)
    for p in (Path(__file__),Path(__file__).parent/'verify_export.py',Path(__file__).parent/'web/network.js',Path(__file__).parent/'web/training-worker.js'):
        dest=root/'source'/p.relative_to(Path(__file__).parent);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    attempts=[];chosen=None
    for shared in (True,False):
        folded=copy.deepcopy(original);last=folded['layers'][-1]
        w=np.array(last['weights']);b=np.array(last['bias'])
        last['weights']=(w[:,:8]-w[:,8,None]).tolist()
        last['bias']=(b[:8]-b[8]).tolist()
        folded['architecture'][-1]=8;folded.update(reference_logit_zero=True,logit_round_decimals=12,forward_passes_per_move=8)
        if shared:
            corner=float(b[[0,2,6,8]].mean());edge=float(b[[1,3,5,7]].mean())
            last['bias']=[edge-corner,float(b[4])-corner]
            folded['output_bias_orbits']='D4-corner-zero'
        for precision in ('float16','float32','float64'):
            model=copy.deepcopy(folded)
            for layer in model['layers']:
                for field in ('weights','bias'):layer[field]=np.array(layer[field],dtype=precision).astype(float).tolist()
            count=sum(sum(len(r) for r in l['weights'])+len(l['bias']) for l in model['layers'])
            name=f'compact-{count}-{precision}';model.update(parameter_count=count,checkpoint=name,run_id=root.name,coefficient_precision=precision)
            atomic_json(root/(name+'.json'),model)
            audit=qualify(model,include_fixtures=True);atomic_json(root/(name+'-independent.json'),audit)
            portable_path=root/(name+'-portable.json')
            result=subprocess.run(['node',str(Path(__file__).parent/'verify_portable.mjs'),'--model',str(root/(name+'.json')),'--fixtures',str(root/(name+'-independent.json')),'--output',str(portable_path)],capture_output=True,text=True)
            portable=json.loads(portable_path.read_text())
            row={'checkpoint':name,'parameters':count,'precision':precision,'no_loss_from_empty':audit['no_loss_from_empty'],
                 'optimal_count':audit['optimal_count'],'total':4520,'portable':portable,'model_sha256':digest(root/(name+'.json'))}
            attempts.append(row);atomic_json(root/'transformation-attempts.json',attempts)
            if audit['no_loss_from_empty'] and result.returncode==0:
                chosen=(model,audit,portable,row);break
        if chosen:break
    if not chosen:raise RuntimeError('No derived policy passed independent no-loss and strict browser parity.')
    model,audit,portable,row=chosen
    atomic_json(root/'model.json',model);atomic_json(root/'independent-verification.json',audit);atomic_json(root/'portable-verification.json',portable)
    values=[v for layer in model['layers'] for field in ('weights','bias') for v in np.array(layer[field]).flat]
    np.array(values,dtype=model['coefficient_precision']).tofile(root/'coefficients.bin')
    evaluation={'exhaustive':{'correct':audit['optimal_count'],'count':4520,'accuracy':audit['optimal_rate'],'mistakes':audit['mistake_count']},
                'independent':{k:audit[k] for k in ('optimal_count','total','no_loss_from_empty','perfect_policy')},
                'worst_case':{'as_x':audit['as_x']['result'],'as_o':audit['as_o']['result']},'browser_export':portable,
                'coefficient_payload_bytes':(root/'coefficients.bin').stat().st_size,'scope':protocol['scope'],
                'no_solver_fallback':True,'selected_checkpoint':model['checkpoint']}
    atomic_json(root/'evaluation.json',evaluation)
    raw_root=source.parent/'compression-20261113-a';raw_status=json.loads((raw_root/'status.json').read_text())
    source_status=json.loads((source/'status.json').read_text())
    candidates=raw_status['compression']['candidates']+source_status['compression']['candidates']
    for candidate in candidates:
        candidate['qualification']='Training diagnosis'
        if candidate['checkpoint']==raw_status['selection']['checkpoint']:candidate['qualification']='Independent + browser passed'
        if candidate['checkpoint']==selection['checkpoint']:candidate['qualification']='Old exact-tie browser parity failed; preserved'
    selected=dict(row,architecture=model['architecture'],symmetry_average='D4',reference_logit_zero=True,forward_passes_per_move=8,qualification='Independent + browser passed',selected=True)
    candidates.append(selected)
    status=dict(source_status)
    status.pop('error',None)
    status.update(status='completed',run_id=root.name,phase_id='qualification',model_available=True,model_version=model['checkpoint'],
                  architecture=model['architecture'],parameter_count=model['parameter_count'],selection=selected,evaluation=evaluation,
                  training_mode='finite-domain-compression',compression={'baseline_parameters':6537,'selected_parameters':model['parameter_count'],
                  'selected_checkpoint':model['checkpoint'],'coefficient_payload_bytes':evaluation['coefficient_payload_bytes'],'coefficient_precision':model['coefficient_precision'],
                  'forward_passes_per_move':8,'trial_count':len(json.loads((raw_root/'search-trials.json').read_text()))+len(json.loads((source/'search-trials.json').read_text())),
                  'candidates':candidates,'scope':'Smallest verified result found in the registered searches plus exact algebraic folding. Smaller training failures do not prove a global minimum. Eight shared neural passes per move.'},
                  recipe_note='Solver-trained tiny shared network; all states are TRAIN. No-loss as X/O is certified separately from all-position optimality. Eight geometric neural passes; two shared output biases; one fixed zero readout; declared numerical tie rule. No tactical or solver fallback.')
    status['metrics']=dict(status.get('metrics',{}),train_accuracy=audit['optimal_rate'],loss=None)
    status['phases']=[{'id':'search','title':'Search smaller networks','detail':'Raw and symmetry-shared networks, multiple seeds; failed smaller trials preserved.','status':'completed','progress':1,'metrics':{'trials':status['compression']['trial_count']}},
                      {'id':'fold','title':'Fold redundant learned coefficients','detail':'Eight learned readouts plus a fixed zero; two D4-shared output biases.','status':'completed','progress':1,'metrics':{'parameters':model['parameter_count'],'precision':model['coefficient_precision']}},
                      {'id':'qualification','title':'Certify the exact deployed policy','detail':'Every opponent continuation as X/O, all-state grading, and strict browser action parity.','status':'completed','progress':1,'metrics':{'non_losing_both':True,'positions':4520,'action_mismatches':portable['action_mismatches']}}]
    status['events'].append({'time':datetime.now(timezone.utc).isoformat(),'message':f'Frozen {model["parameter_count"]}-parameter {model["coefficient_precision"]} policy passed independent no-loss and strict browser parity.'})
    atomic_json(root/'status.json',status)
    atomic_json(root/'qualification-consumption.json',{'transformation_sha256':digest(root/'transformation-before-qualification.json'),'model_sha256':digest(root/'model.json'),
                'independent_sha256':digest(root/'independent-verification.json'),'portable_sha256':digest(root/'portable-verification.json'),'consumed_at':datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'run_id':root.name,'parameters':model['parameter_count'],'precision':model['coefficient_precision'],'payload_bytes':evaluation['coefficient_payload_bytes'],
                     'optimal_count':audit['optimal_count'],'no_loss_both':audit['no_loss_from_empty'],'portable':portable}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-run',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();main(a.source_run,a.run_dir)
