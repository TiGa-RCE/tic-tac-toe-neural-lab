/** Real CPU training. This module is loaded only after explicit Start training.
 * Teacher labels are training/audit data. The gameplay module never imports it.
 */
import {outcome} from './network.js';

let command='resume';
let started=false;
self.onmessage=event=>{
  if(event.data.action==='start' && !started) {started=true;run(event.data);}
  else command=event.data.action;
};
const yieldTask=()=>new Promise(resolve=>setTimeout(resolve,0));
async function control(status) {
  await yieldTask();
  if(command==='stop') throw new Error('STOP_REQUESTED');
  while(command==='pause') {
    if(status.status!=='paused') {status.status='paused';send(status);}
    await new Promise(resolve=>setTimeout(resolve,100));
  }
  if(command==='stop') throw new Error('STOP_REQUESTED');
  status.status='running';
}
function send(status) {self.postMessage({type:'status',status});}
function random(seed) {
  let s=seed>>>0;
  return ()=>{s+=0x6D2B79F5;let t=Math.imul(s^s>>>15,1|s);t^=t+Math.imul(t^t>>>7,61|t);return ((t^t>>>14)>>>0)/4294967296;};
}
function flatten(model) {
  return model.layers.map(layer=>({input:layer.weights.length,output:layer.bias.length,w:Float64Array.from(layer.weights.flat()),b:Float64Array.from(layer.bias)}));
}
function clone(net) {return net.map(l=>({...l,w:l.w.slice(),b:l.b.slice()}));}
function exportModel(net,parent,checkpoint,run_id) {
  return {...parent,checkpoint,run_id,layers:net.map((l,index)=>({weights:Array.from({length:l.input},(_,i)=>Array.from(l.w.slice(i*l.output,(i+1)*l.output))),bias:Array.from(l.b),activation:index===net.length-1?'linear':'tanh'}))};
}
function forward(net,input,cache=false) {
  const activations=[input];
  for(let k=0;k<net.length;k++) {
    const layer=net[k],a=activations[k],z=layer.b.slice();
    for(let i=0;i<layer.input;i++) {
      const value=a[i];if(value===0) continue;
      for(let j=0;j<layer.output;j++) z[j]+=value*layer.w[i*layer.output+j];
    }
    if(k<net.length-1) for(let j=0;j<z.length;j++) z[j]=Math.tanh(z[j]);
    activations.push(z);
  }
  return cache?activations:activations[activations.length-1];
}
function features(board) {
  const player=board.reduce((s,c)=>s+c,0)===0?1:-1;
  return Float64Array.from(board.flatMap(c=>[Number(c===player),Number(c===-player),Number(c===0)]));
}
function probabilities(logits,board,temperature=1) {
  let max=-Infinity;
  for(let i=0;i<9;i++) if(board[i]===0) max=Math.max(max,logits[i]/temperature);
  const p=new Float64Array(9);let sum=0;
  for(let i=0;i<9;i++) if(board[i]===0) {p[i]=Math.exp(logits[i]/temperature-max);sum+=p[i];}
  for(let i=0;i<9;i++) p[i]/=sum;
  return p;
}
function greedy(logits,board) {
  let move=null;
  for(let i=0;i<9;i++) if(board[i]===0 && (move===null || logits[i]>logits[move])) move=i;
  return move;
}
function adam(net) {
  return {t:0,layers:net.map(l=>({mw:new Float64Array(l.w.length),vw:new Float64Array(l.w.length),mb:new Float64Array(l.b.length),vb:new Float64Array(l.b.length)}))};
}
function update(net,opt,rows,samples,parentLogits,protectedRows) {
  const grads=net.map(l=>({w:new Float64Array(l.w.length),b:new Float64Array(l.b.length)}));
  let loss=0;
  for(const row of rows) {
    const s=samples[row],acts=forward(net,s.x,true),logits=acts[acts.length-1],p=probabilities(logits,s.board);
    let mass=0;
    for(const i of s.optimal) mass+=p[i];
    mass=Math.max(1e-30,mass);loss-=Math.log(mass)/rows.length;
    let delta=p.slice();
    for(const i of s.optimal) delta[i]-=p[i]/mass;
    if(protectedRows[row]) {
      const q=probabilities(parentLogits[row],s.board,2),pt=probabilities(logits,s.board,2);
      for(let i=0;i<9;i++) {
        delta[i]+=.2*(pt[i]-q[i]);
        if(q[i]) loss+=.4*q[i]*Math.log(q[i]/Math.max(pt[i],1e-30))/rows.length;
      }
    }
    for(let i=0;i<9;i++) delta[i]/=rows.length;
    for(let k=net.length-1;k>=0;k--) {
      const l=net[k],g=grads[k],a=acts[k];
      for(let j=0;j<l.output;j++) g.b[j]+=delta[j];
      for(let i=0;i<l.input;i++) {
        const v=a[i];if(v===0) continue;
        for(let j=0;j<l.output;j++) g.w[i*l.output+j]+=v*delta[j];
      }
      if(k>0) {
        const previous=new Float64Array(l.input);
        for(let i=0;i<l.input;i++) {
          let sum=0;for(let j=0;j<l.output;j++) sum+=l.w[i*l.output+j]*delta[j];
          previous[i]=sum*(1-a[i]*a[i]);
        }
        delta=previous;
      }
    }
  }
  let norm=0;
  for(const g of grads) for(const array of [g.w,g.b]) for(const v of array) norm+=v*v;
  const scale=Math.min(1,1/Math.max(Math.sqrt(norm),1e-30));opt.t++;
  for(let k=0;k<net.length;k++) for(const type of ['w','b']) {
    const weights=net[k][type],m=opt.layers[k]['m'+type],v=opt.layers[k]['v'+type],g=grads[k][type];
    for(let i=0;i<weights.length;i++) {
      const d=g[i]*scale;m[i]=.9*m[i]+.1*d;v[i]=.999*v[i]+.001*d*d;
      weights[i]-=.0003*(m[i]/(1-.9**opt.t))/(Math.sqrt(v[i]/(1-.999**opt.t))+1e-8);
    }
  }
  return loss;
}
async function grade(net,samples,status,parentCorrect=null) {
  const choices=new Map(),correct=[],errors=[];let repairs=0,breaks=0;
  for(let i=0;i<samples.length;i++) {
    const s=samples[i],move=greedy(forward(net,s.x),s.board),good=s.optimal.includes(move);
    choices.set(s.board.join(','),move);correct.push(good);
    if(!good) errors.push(i);
    if(parentCorrect) {repairs+=Number(good&&!parentCorrect[i]);breaks+=Number(!good&&parentCorrect[i]);}
    if(i%256===0) await control(status);
  }
  function worst(neuralPlayer) {
    const memo=new Map();
    function visit(board) {
      const result=outcome(board);if(result!==null) return result*neuralPlayer;
      const key=board.join(',');if(memo.has(key)) return memo.get(key);
      const mark=board.reduce((a,b)=>a+b,0)===0?1:-1;
      const moves=mark===neuralPlayer?[choices.get(key)]:board.map((v,i)=>v===0?i:-1).filter(i=>i>=0);
      let best=Infinity;
      for(const m of moves) {const b=board.slice();b[m]=mark;best=Math.min(best,visit(b));}
      memo.set(key,best);return best;
    }
    return visit(Array(9).fill(0));
  }
  return {correct,errors,optimal_count:samples.length-errors.length,total:samples.length,accuracy:1-errors.length/samples.length,repairs,breaks,as_x:worst(1),as_o:worst(-1)};
}
function nativePool(net,samples,lookup,rng) {
  const counts=new Uint32Array(samples.length);counts.fill(1);let nativeActions=0;
  for(let n=0;n<128;n++) {
    let b=samples[Math.floor(rng()*samples.length)].board.slice();
    while(outcome(b)===null) {
      const id=lookup.get(b.join(',')),move=greedy(forward(net,samples[id].x),b);
      counts[id]++;nativeActions++;
      b[move]=b.reduce((a,c)=>a+c,0)===0?1:-1;
    }
  }
  const cumulative=[];let total=0;
  for(const c of counts) {total+=c;cumulative.push(total);}
  const sample=()=>{
    const target=rng()*total;let lo=0,hi=cumulative.length-1;
    while(lo<hi) {const mid=(lo+hi)>>>1;if(target<cumulative[mid]) hi=mid;else lo=mid+1;}
    return lo;
  };
  return {sample,nativeActions};
}

async function run(options) {
  const phases=[
    ['teacher','Load the finite teacher','All reachable decision states are TRAIN; old held-out results remain historical.'],
    ['phase1','01 · Collect & review','One visitor-local branch; 512 Adam updates, native trajectories and targeted TRAIN errors.'],
    ['phase2','02 · Repair & retain','Half native, half retention; founder-correct KL at temperature 2.'],
    ['phase3','03 · Consolidate','Final 512-update phase, retaining all checkpoint coverage and break counts.'],
    ['qualification','Audit the frozen candidate','Exhaustive decision coverage and every adversarial continuation from empty, as both X and O.'],
    ['export','Keep the certified student','Only a no-loss candidate can replace the playable published incumbent.'],
  ];
  const status={status:'running',run_id:options.run_id,training_mode:'browser-finite-domain',architecture:options.parent.architecture,parameter_count:options.parent.parameter_count,
    phase_id:'teacher',phases:phases.map(([id,title,detail])=>({id,title,detail,status:'pending',progress:0,metrics:{}})),metrics:{},history:[],events:[],candidates:[],evaluation:{},dataset:{},model_available:true,
    recipe_note:'Real visitor-local training, one branch in three phases of 512 Adam updates. All finite decision states are TRAIN. No held-out generalization is claimed. Labels remain in this training worker; gameplay uses only the frozen neural network. The published incumbent remains available until a browser candidate passes the no-loss audit.'};
  const event=message=>{status.events.push({time:new Date().toISOString(),message});send(status);};
  const phase=(id,progress,metrics={})=>{
    status.phase_id=id;const p=status.phases.find(p=>p.id===id);Object.assign(p,{progress,status:progress>=1?'completed':'running',metrics});send(status);
  };
  const archive={protocol:{seed:options.seed,phases:3,updates_per_phase:512,batch_size:128,learning_rate:.0003,clip:1,temperature:2,kl_coefficient:.1,branch_count:1,training_domain:'all reachable decision states',selection:'never replace certified incumbent with an uncertified or lower-coverage candidate'},schedules:[],checkpoints:[]};
  try {
    phase('teacher',0);
    const response=await fetch(options.teaching_url);if(!response.ok) throw new Error('Teaching corpus unavailable.');
    const teaching=await response.json();
    if(teaching.boards.length!==4520 || teaching.optimal_moves.length!==4520) throw new Error('Expected all 4,520 reachable decision states.');
    const samples=teaching.boards.map((board,i)=>({board,x:features(board),optimal:teaching.optimal_moves[i]}));
    const lookup=new Map(samples.map((s,i)=>[s.board.join(','),i]));
    let net=flatten(options.parent),best=clone(net);const opt=adam(net),rng=random(options.seed);
    const parentLogits=samples.map(s=>forward(net,s.x));
    const protectedRows=samples.map((s,i)=>s.optimal.includes(greedy(parentLogits[i],s.board)));
    let base=await grade(net,samples,status),bestGrade=base,bestCheckpoint='published-incumbent';
    status.dataset={decision_states:4520,train:4520,held_out:0,scope:'Exhaustive finite-domain training and certification.'};
    phase('teacher',1,{states:4520,parent_correct:base.optimal_count});event('Teacher corpus loaded only for this explicit browser experiment. Published game weights remain fixed.');
    status.candidates.push({checkpoint:'published-incumbent',validation_correct:base.optimal_count,validation_count:4520,repairs:0,breaks:0});
    for(let p=1;p<=3;p++) {
      const id=`phase${p}`;phase(id,0);await control(status);
      const diagnosis=await grade(net,samples,status,base.correct);
      const pool=nativePool(net,samples,lookup,rng),schedule=[];
      for(let u=0;u<512;u++) {
        const row=Array.from({length:128},(_,j)=>j<64?pool.sample():Math.floor(rng()*samples.length));
        if(u<32 && diagnosis.errors.length) for(let j=0;j<32;j++) row[j]=diagnosis.errors[Math.floor(rng()*diagnosis.errors.length)];
        schedule.push(row);
      }
      archive.schedules.push({phase:id,indices:schedule,native_actions:pool.nativeActions,target_errors:diagnosis.errors});
      event(`Phase ${p}: literal 512 × 128 schedule saved in browser memory; ${pool.nativeActions} honest native actions; ${diagnosis.errors.length} targeted errors.`);
      let loss=0;
      for(let u=0;u<512;u++) {
        loss=update(net,opt,schedule[u],samples,parentLogits,protectedRows);
        if((u+1)%8===0) await control(status);
        if((u+1)%32===0) {
          const probe=samples.filter((_,i)=>i%17===0);let good=0;
          for(const s of probe) good+=Number(s.optimal.includes(greedy(forward(net,s.x),s.board)));
          const row={step:(p-1)*512+u+1,phase:id,epoch:u+1,branch:options.seed,loss,train_accuracy:good/probe.length};
          status.metrics=row;status.history.push(row);phase(id,(u+1)/512,{updates:u+1,probe_states:probe.length});
        }
      }
      const g=await grade(net,samples,status,base.correct),checkpoint=`browser-p${p}`;
      const candidate={checkpoint,validation_correct:g.optimal_count,validation_count:g.total,repairs:g.repairs,breaks:g.breaks};
      status.candidates.push(candidate);archive.checkpoints.push({checkpoint,optimal_count:g.optimal_count,as_x:g.as_x,as_o:g.as_o});
      // This is checkpoint selection after external grading, never a per-move fallback.
      if(g.as_x>=0 && g.as_o>=0 && g.optimal_count===samples.length && (bestGrade.optimal_count<samples.length || bestGrade.as_x<0 || bestGrade.as_o<0)) {
        best=clone(net);bestGrade=g;bestCheckpoint=checkpoint;
      }
      phase(id,1,{correct:g.optimal_count,total:g.total,repairs:g.repairs,breaks:g.breaks});
      event(`${checkpoint}: ${g.optimal_count}/4520 optimal; repairs ${g.repairs}, breaks ${g.breaks}.`);
    }
    phase('qualification',0);
    const certified=bestGrade.as_x>=0 && bestGrade.as_o>=0 && bestGrade.optimal_count===samples.length;
    status.evaluation={exhaustive:{correct:bestGrade.optimal_count,count:4520,accuracy:bestGrade.accuracy,mistakes:bestGrade.errors.length},browser_audit:{no_loss_from_empty:certified,as_x:bestGrade.as_x>=0?'draw':'loss',as_o:bestGrade.as_o>=0?'draw':'loss'},scope:'All states were TRAIN. Browser audit is separate from the independently verified published baseline.'};
    phase('qualification',1,{no_loss_from_empty:certified});
    status.selection={checkpoint:bestCheckpoint};
    phase('export',1,{checkpoint:bestCheckpoint,incumbent_retained:bestCheckpoint==='published-incumbent'});
    status.status='completed';event('Browser experiment complete. Only the selected certified checkpoint is eligible for the next game.');
    archive.status=status;
    self.postMessage({type:'complete',status,archive,model:certified && bestCheckpoint!=='published-incumbent'?exportModel(best,options.parent,bestCheckpoint,options.run_id):null});
  } catch(error) {
    status.status=error.message==='STOP_REQUESTED'?'cancelled':'failed';
    if(status.status==='failed') status.error=error.message;
    event(status.status==='cancelled'?'Browser experiment stopped. Published checkpoint remains available.':`Browser experiment failed: ${error.message}`);
    archive.status=status;self.postMessage({type:'complete',status,archive,model:null});
  }
}
