/** Metadata-only checks of the real hosted adapter; no neural inference. */
import assert from 'node:assert/strict';
import {detectHosting} from './web/hosted.js';

const ids=['compact','dense','perfect-compact','original','perfect-sparse'];
const counts=[38,123,74,6537,53];
const models=Object.fromEntries(ids.map((id,i)=>[id,{
  checkpoint:id,run_id:`frozen-${id}`,parameter_count:counts[i],
  architecture:[9,1,9],encoding:'relative-signed',layers:[{weights:[[i]],bias:[i]}],
}]));
const catalog=ids.map((id,i)=>({id,label:id,parameters:counts[i],model_url:`/data/networks/${id}.json`,
  no_loss_from_empty:true,perfect_policy:i>1,optimal_count:i>1?4520:3800,total:4520}));
const fixtures={
  '/site-config.json':{mode:'hosted'},
  '/data/status.json':{status:'completed',model_available:true,parameter_count:53,evaluation:{independent:{no_loss_from_empty:true}}},
  '/data/model.json':models['perfect-sparse'],
  '/data/networks.json':{default_id:'perfect-sparse',default_game_id:'perfect-sparse',default_training_id:'perfect-sparse',networks:catalog},
  ...Object.fromEntries(ids.map(id=>[`/data/networks/${id}.json`,models[id]])),
};
const fetched=[];
const unavailable=new Set();
const oldFetch=globalThis.fetch,oldWorker=globalThis.Worker;
globalThis.fetch=async path=>{
  fetched.push(String(path));
  const ok=Object.hasOwn(fixtures,path)&&!unavailable.has(path);
  return {ok,status:ok?200:404,json:async()=>structuredClone(fixtures[path])};
};
class FakeWorker {
  postMessage(message){this.message=structuredClone(message);}
  terminate(){this.terminated=true;}
}
globalThis.Worker=FakeWorker;

try {
  const host=await detectHosting();
  assert.equal(host.selectedGameId,'perfect-sparse','new visits default to the fully optimal sparse model');
  assert.equal(host.selectedTrainingId,'perfect-sparse');
  assert.deepEqual(host.gameNetworks.map(n=>n.id),ids);
  assert.equal((await host.request('/api/model')).checkpoint,'perfect-sparse');

  for(const id of ids){
    const model=await host.request(`/api/model?network_id=${id}`);
    assert.equal(model.checkpoint,id,`the requested ${id} weights are returned`);
    assert.equal(host.selectedGameId,'perfect-sparse','a load does not commit a UI selection before its token check');
    model.layers[0].weights[0][0]=999999;
    const reloaded=await host.request(`/api/model?network_id=${id}`);
    assert.deepEqual(reloaded,models[id],'callers cannot mutate a frozen model through a returned object');
  }

  host.selectedGameId='compact';
  host.selectedTrainingId='dense';
  assert.equal((await host.request('/api/model')).checkpoint,'compact');
  assert.equal(host.selectedTrainingId,'dense','opponent loads do not select a training parent');
  const before=JSON.stringify(host.publishedModel);
  await host.request('/api/train',{body:JSON.stringify({network_id:'original',seed:42})});
  assert.equal(host.worker.message.parent.checkpoint,'original');
  assert.equal(host.selectedGameId,'compact','training starts independently of the game choice');
  const worker=host.worker;
  worker.onmessage({data:{type:'complete',status:{status:'completed'},archive:{},model:{checkpoint:'unqualified-experiment'}}});
  assert.equal(worker.terminated,true);
  assert.equal((await host.request('/api/model')).checkpoint,'compact','an experiment never replaces the opponent');
  host.restorePublished();
  assert.equal(host.selectedGameId,'compact','viewing published training evidence does not replace the opponent');
  assert.equal((await host.request('/api/model')).checkpoint,'compact');
  assert.equal(JSON.stringify(host.publishedModel),before);

  const count=fetched.length;
  await assert.rejects(()=>host.request('/api/model?network_id=missing'),/available|unknown|choose/i);
  assert.equal(fetched.length,count,'an invalid choice does not fetch an arbitrary URL');
  const fresh=await detectHosting();
  unavailable.add('/data/networks/dense.json');
  await assert.rejects(()=>fresh.request('/api/model?network_id=dense'),/load|404/i);
  assert.equal(fresh.selectedGameId,'perfect-sparse','a failed fetch never commits a different game choice');
  assert(!fetched.some(path=>/teaching|oracle/.test(path)),'game selection never loads the solver teaching corpus');
  console.log(JSON.stringify({passed:true,choices:ids,default_game:'perfect-sparse',inference_performed:false,
    independent_training:true,frozen_copies:true,failed_load_keeps_selection:true,teacher_requests:0}));
} finally {
  globalThis.fetch=oldFetch;
  globalThis.Worker=oldWorker;
}
