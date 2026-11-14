"""Create a separately sealed, folded numeric parent for masked fine-tuning."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,copy
from train import atomic_json,digest,np
from finalize_sparse import pack_candidate

def main(args):
    source=Path(args.source);root=Path(args.run_dir);root.mkdir(parents=True,exist_ok=False)
    model=pack_candidate(json.loads(source.read_text()),False)
    masks=[]
    for layer in model['layers']:
        for field in ('weights','bias'):
            array=np.array(layer[field],dtype=object)
            mask=np.array([v is not None for v in array.flat]).reshape(array.shape)
            array=np.array([0. if v is None else v for v in array.flat],dtype=float).reshape(array.shape)
            layer[field]=array.tolist();masks.append(mask.tolist())
    if args.normalize:
        head=model['layers'][-1];scale=max(abs(v) for row in head['weights'] for v in row)
        if not scale:raise ValueError('Expected a nonzero output scale')
        head['weights']=[[v/scale for v in row] for row in head['weights']]
        head['bias']=[v/scale for v in head['bias']]
    model.pop('coefficient_storage',None)
    model.update(learned_parameter_mask=masks,parameter_count=sum(np.array(m,dtype=bool).sum().item() for m in masks),
                 checkpoint='folded-pruning-parent',run_id=root.name,search_only_masked_dense=True,coefficient_precision='float64')
    atomic_json(root/'model.json',model)
    atomic_json(root/'preparation-protocol.json',{'registered_at':datetime.now(timezone.utc).isoformat(),
        'source':str(source),'source_sha256':digest(source),'script_sha256':digest(__file__),
        'operation':'Fold common logits, D4 output biases and exactzeros. Optional common positive output scaling; every remaining numeric scalar still counts as learned until final export.',
        'normalize':args.normalize,'model_sha256':digest(root/'model.json')})
    print(json.dumps({'run_id':root.name,'parameters':model['parameter_count']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--run-dir',required=True);p.add_argument('--normalize',action='store_true');main(p.parse_args())
