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
    selected_model=json.loads((run/'model.json').read_text())
    catalog=[]
    choices=[('compact','Smallest verified shared network',run),
             ('dense','Compact dense network (one pass)',ROOT/'runs/compression-20261113-a'),
             ('original','Original perfect network',ROOT/'runs/20261113T233235Z-7e5be6')]
    seen=set()
    model_dir=data/'networks';model_dir.mkdir(exist_ok=True)
    for identifier,label,folder in choices:
        if not (folder/'model.json').exists():continue
        candidate=json.loads((folder/'model.json').read_text())
        sha=hashlib.sha256((folder/'model.json').read_bytes()).hexdigest()
        if sha in seen:continue
        seen.add(sha)
        qualification=json.loads((folder/'independent-verification.json').read_text())
        parity=json.loads((folder/'portable-verification.json').read_text())
        if not qualification['no_loss_from_empty'] or not parity['passed']:continue
        shutil.copyfile(folder/'model.json',model_dir/(identifier+'.json'))
        catalog.append({'id':identifier,'label':label,'model_url':'/data/networks/'+identifier+'.json',
                        'parameters':candidate['parameter_count'],'architecture':candidate['architecture'],'encoding':candidate['encoding'],
                        'symmetry_average':candidate.get('symmetry_average'),'forward_passes_per_move':candidate.get('forward_passes_per_move',1),
                        'reference_logit_zero':candidate.get('reference_logit_zero',False),'output_bias_orbits':candidate.get('output_bias_orbits'),
                        'coefficient_precision':candidate.get('coefficient_precision','float64'),
                        'no_loss_from_empty':True,'optimal_count':qualification['optimal_count'],'total':qualification['total']})
    (data/'networks.json').write_text(json.dumps({'default_id':catalog[0]['id'],'networks':catalog},indent=2)+'\n')
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
