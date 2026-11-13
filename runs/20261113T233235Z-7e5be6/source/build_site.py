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
    boards=[b for b in enumerate_states() if winner(b) is None]
    labels=[solve(b) for b in boards]
    teaching={'boards':boards,'optimal_moves':[row[1] for row in labels],
              'values':[row[0] for row in labels],
              'scope':'All 4,520 reachable decision positions are supervised; finite-domain coverage, not a held-out generalization experiment.'}
    (data/'teaching.json').write_text(json.dumps(teaching,separators=(',',':')))
    config={'mode':'hosted','repository_url':repository_url,'site_url':site_url,
            'run_id':run.name,'model_sha256':hashlib.sha256((run/'model.json').read_bytes()).hexdigest(),
            'certification_url':'./data/certification.json','training_scope':teaching['scope']}
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
