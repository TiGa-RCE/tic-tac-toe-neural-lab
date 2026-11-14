"""Build a self-contained public dashboard from an immutable completed run."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from solver import enumerate_states, winner, solve

ROOT=Path(__file__).resolve().parent

def build(run, output, repository_url, site_url):
    status=json.loads((run/'status.json').read_text())
    if status['status']!='completed':
        raise ValueError('Publish only a completed, qualified run.')
    evidence=json.loads((run/'independent-verification.json').read_text())
    portable=json.loads((run/'portable-verification.json').read_text())
    if not evidence['no_loss_from_empty'] or not portable['passed']:
        raise ValueError('Public baseline must pass independent no-loss and portable inference checks.')
    output.mkdir(parents=True,exist_ok=True)
    for source in (ROOT/'web').iterdir():
        if source.is_file(): shutil.copyfile(source,output/source.name)
    data=output/'data'; data.mkdir(exist_ok=True)
    for name in ('model.json','status.json','evaluation.json','portable-verification.json'):
        shutil.copyfile(run/name,data/name)
    # Full scalar fixtures belong in GitHub evidence; public certification is small.
    public_audit={k:v for k,v in evidence.items() if k!='fixtures'}
    (data/'certification.json').write_text(json.dumps(public_audit,separators=(',',':')))
    selected_sha=hashlib.sha256((run/'model.json').read_bytes()).hexdigest()
    catalog=[]
    default_id=None
    perfect_run=ROOT/'runs/perfect-optimal-selected'
    perfect_index=ROOT/'runs/perfect-optimal-selection.json'
    if perfect_index.exists():
        name=json.loads(perfect_index.read_text())['run_id']
        if Path(name).name!=name or name in ('','.','..'):raise ValueError('Perfect model registry must contain a run name, not a path.')
        perfect_run=ROOT/'runs'/name
    sparse_run=None
    sparse_index=ROOT/'runs/sparse-perfect-selection.json'
    if sparse_index.exists():
        name=json.loads(sparse_index.read_text())['run_id']
        if Path(name).name!=name or name in ('','.','..'):raise ValueError('Sparse model registry must contain a run name, not a path.')
        sparse_run=ROOT/'runs'/name
        if not (sparse_run/'model.json').exists():raise ValueError('The registered sparse perfect model is missing.')
    compact_run=ROOT/'runs/compression-20261113-selected'
    if not (compact_run/'model.json').exists():compact_run=run
    choices=[('compact','Smallest verified shared network',compact_run),
             ('dense','Compact dense network (one pass)',ROOT/'runs/compression-20261113-a'),
             ('perfect-compact','Compact fully optimal network',perfect_run),
             ('original','Original perfect network',ROOT/'runs/20261113T233235Z-7e5be6')]
    if sparse_run is not None:choices.insert(3,('perfect-sparse','Sparse fully optimal network',sparse_run))
    seen=set()
    model_dir=data/'networks';model_dir.mkdir(exist_ok=True)
    for identifier,label,folder in choices:
        if not (folder/'model.json').exists():continue
        candidate=json.loads((folder/'model.json').read_text())
        sha=hashlib.sha256((folder/'model.json').read_bytes()).hexdigest()
        qualification=json.loads((folder/'independent-verification.json').read_text())
        parity=json.loads((folder/'portable-verification.json').read_text())
        if identifier in ('perfect-compact','perfect-sparse') and not (qualification.get('perfect_policy') is True and
                qualification.get('optimal_count')==4520 and qualification.get('total')==4520 and
                qualification.get('no_loss_from_empty') is True and parity.get('passed') is True and
                parity.get('action_mismatches')==0 and parity.get('optimal_count')==4520):
            kind='compact perfect choice' if identifier=='perfect-compact' else 'sparse perfect choice'
            raise ValueError(f'The {kind} must independently pass all 4,520 boards and exact browser actions.')
        if sha in seen:continue
        seen.add(sha)
        if not qualification['no_loss_from_empty'] or not parity['passed']:continue
        if sha==selected_sha:default_id=identifier
        shutil.copyfile(folder/'model.json',model_dir/(identifier+'.json'))
        (model_dir/(identifier+'-certification.json')).write_text(json.dumps({k:v for k,v in qualification.items() if k!='fixtures'},separators=(',',':')))
        catalog.append({'id':identifier,'label':label,'model_url':'/data/networks/'+identifier+'.json',
                        'parameters':candidate['parameter_count'],'architecture':candidate['architecture'],'encoding':candidate['encoding'],
                        'model_sha256':sha,
                        'symmetry_average':candidate.get('symmetry_average'),'forward_passes_per_move':candidate.get('forward_passes_per_move',1),
                        'reference_logit_zero':candidate.get('reference_logit_zero',False),'output_bias_orbits':candidate.get('output_bias_orbits'),
                        'coefficient_precision':candidate.get('coefficient_precision','float64'),
                        'coefficient_storage':candidate.get('coefficient_storage','dense'),
                        'structural_zero_slots':candidate.get('structural_zero_slots',0),
                        'fixed_unit_slots':candidate.get('fixed_unit_slots',0),
                        'coefficient_payload_bytes':candidate.get('coefficient_payload_bytes',
                            (folder/'coefficients.bin').stat().st_size if (folder/'coefficients.bin').exists() else None),
                        'no_loss_from_empty':True,'perfect_policy':qualification.get('perfect_policy',False),
                        'optimal_count':qualification['optimal_count'],'total':qualification['total'],
                        'certification_url':'/data/networks/'+identifier+'-certification.json'})
    if default_id is None:raise ValueError('The published default must appear in the qualified network catalog.')
    (data/'networks.json').write_text(json.dumps({'default_id':default_id,'default_game_id':default_id,
                                                'default_training_id':default_id,'networks':catalog},indent=2)+'\n')
    if (run/'coefficients.bin').exists():shutil.copyfile(run/'coefficients.bin',data/'coefficients.bin')
    boards=[b for b in enumerate_states() if winner(b) is None]
    labels=[solve(b) for b in boards]
    teaching={'boards':boards,'optimal_moves':[row[1] for row in labels],
              'values':[row[0] for row in labels],
              'scope':'All 4,520 reachable decision positions are supervised; finite-domain coverage, not a held-out generalization experiment.'}
    (data/'teaching.json').write_text(json.dumps(teaching,separators=(',',':')))
    config={'mode':'hosted','repository_url':repository_url,'site_url':site_url,
            'run_id':run.name,'model_sha256':hashlib.sha256((run/'model.json').read_bytes()).hexdigest(),
            'certification_url':'./data/certification.json','training_scope':teaching['scope'],'network_catalog_url':'/data/networks.json'}
    (output/'site-config.json').write_text(json.dumps(config,indent=2)+'\n')
    (output/'robots.txt').write_text('User-agent: *\nAllow: /\n')
    (output/'_headers').write_text('/*\n  X-Content-Type-Options: nosniff\n  Referrer-Policy: strict-origin-when-cross-origin\n')
    print(json.dumps({'run_id':run.name,'output':str(output),'model_sha256':config['model_sha256'],
                      'no_loss_both':evidence['no_loss_from_empty'],'perfect_policy':evidence['perfect_policy']}))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'site'/'dist')
    p.add_argument('--repository-url',default='https://github.com/TiGa-RCE/tic-tac-toe-neural-lab')
    p.add_argument('--site-url',default='https://three-in-a-row-neural-lab.tiga-rce.chatgpt.site')
    args=p.parse_args(); build(args.run_dir,args.output,args.repository_url,args.site_url)
