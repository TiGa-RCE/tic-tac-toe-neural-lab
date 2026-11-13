/** Exercise actual browser training and cooperative controls under Node. */
import fs from 'node:fs';
import {Worker} from 'node:worker_threads';
import {createHash} from 'node:crypto';
const args=process.argv.slice(2),get=name=>args[args.indexOf(name)+1];
const parent=JSON.parse(fs.readFileSync(get('--model'),'utf8'));
const original=JSON.stringify(parent),hash=value=>createHash('sha256').update(value).digest('hex');
const worker=new Worker(new URL('./worker_node_shim.mjs',import.meta.url),{workerData:{module:new URL('./web/training-worker.js',import.meta.url).href}});
let requestedPause=false,paused=false,resumed=false;
const began=Date.now();
const timer=setTimeout(()=>{worker.terminate();throw new Error('Browser worker acceptance timed out.');},120000);
worker.on('error',error=>{clearTimeout(timer);throw error;});
worker.on('message',message=>{
  const s=message.status;
  if(message.type==='status'&&!requestedPause&&s.phase_id==='phase1'&&s.metrics?.step>=32){requestedPause=true;worker.postMessage({action:'pause'});}
  if(message.type==='status'&&s.status==='paused'&&!paused){paused=true;resumed=true;worker.postMessage({action:'resume'});}
  if(message.type!=='complete') return;
  clearTimeout(timer);
  const shapes=message.archive.schedules.map(phase=>({phase:phase.phase,updates:phase.indices.length,batch_size:phase.indices[0]?.length}));
  const report={status:s.status,steps:s.metrics?.step,phases:shapes,pause_acknowledged:paused,resume_requested:resumed,
    certified:s.evaluation?.browser_audit?.no_loss_from_empty,optimal_count:s.evaluation?.exhaustive?.correct,
    selected:s.selection?.checkpoint,incumbent_preserved:message.model===null&&hash(original)===hash(JSON.stringify(parent)),
    elapsed_seconds:(Date.now()-began)/1000};
  report.passed=report.status==='completed'&&report.steps===1536&&report.pause_acknowledged&&report.certified&&report.optimal_count===4520&&report.incumbent_preserved&&shapes.every(p=>p.updates===512&&p.batch_size===128);
  fs.writeFileSync(get('--output'),JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify(report));
  worker.terminate();
  if(!report.passed) process.exitCode=1;
});
worker.postMessage({action:'start',seed:42,run_id:'worker-acceptance-seed42',parent,teaching_url:get('--base-url')+'/data/teaching.json'});
