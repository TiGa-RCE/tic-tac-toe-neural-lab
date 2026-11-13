/** Static-host adapter. Training lives in an isolated visitor-local worker. */
const copy = value => structuredClone(value);
async function readJSON(path) {
  const response = await fetch(path, {cache:'no-store'});
  if (!response.ok) throw new Error(`Unable to load ${path} (${response.status}).`);
  return response.json();
}

export async function detectHosting() {
  try {
    const response = await fetch('/site-config.json', {cache:'no-store'});
    if (!response.ok) return null;
    const config = await response.json();
    if (config.mode !== 'hosted') return null;
    const [status,model] = await Promise.all([readJSON('/data/status.json'),readJSON('/data/model.json')]);
    return new HostedTrainer(config,status,model);
  } catch (error) {
    // A missing config is the ordinary local Python mode. A malformed hosted
    // configuration should remain an explicit failure rather than fake API data.
    if (error instanceof SyntaxError) throw error;
    return null;
  }
}

class HostedTrainer {
  constructor(config,status,model) {
    this.config=config;
    this.publishedStatus=copy(status);
    this.publishedModel=copy(model);
    this.state=copy(status);
    this.model=copy(model);
    this.worker=null;
    this.teaching=null;
    this.archive=null;
    this.browserRun=false;
  }

  async request(path,options={}) {
    if(path==='/api/status') return this.state;
    if(path==='/api/model') return this.model;
    if(path.startsWith('/api/oracle?')) {
      this.teaching ||= await readJSON('/data/teaching.json');
      const key=new URLSearchParams(path.split('?')[1]).get('board');
      const index=this.teaching.boards.findIndex(board=>board.join(',')===key);
      if(index<0) return {value:0,optimal_moves:[],action_values:[]};
      return {value:this.teaching.values?.[index] ?? null,optimal_moves:this.teaching.optimal_moves[index]};
    }
    const body=JSON.parse(options.body || '{}');
    if(path==='/api/train') {
      if(this.worker) throw new Error('A browser experiment is already active.');
      if(typeof Worker==='undefined') throw new Error('This browser does not support background training workers.');
      this.browserRun=true;
      this.archive=null;
      this.state={...copy(this.publishedStatus),status:'starting',run_id:`browser-${body.seed}-${Date.now()}`,history:[],events:[],candidates:[],evaluation:{},selection:null,
        recipe_note:'Visitor-local, single-branch continuation: three phases of 512 Adam updates, native collection, targeted review and protected replay. All reachable states are teaching data. This is finite-domain coverage, not a fresh held-out test. The published policy remains playable until a candidate passes its browser audit.'};
      this.worker=new Worker(new URL('./training-worker.js',import.meta.url),{type:'module'});
      this.worker.onmessage=event=>{
        const message=event.data;
        if(message.type==='status') this.state=message.status;
        if(message.type==='complete') {
          this.state=message.status;
          this.archive=message.archive;
          if(message.model) this.model=message.model;
          this.worker?.terminate();this.worker=null;
        }
      };
      this.worker.onerror=event=>{
        this.state={...this.state,status:'failed',error:event.message || 'Browser worker failed.'};
        this.worker?.terminate();this.worker=null;
      };
      this.worker.postMessage({action:'start',seed:body.seed,run_id:this.state.run_id,parent:copy(this.publishedModel),teaching_url:'/data/teaching.json'});
      return {run_id:this.state.run_id,status:'starting'};
    }
    if(path==='/api/control') {
      if(!this.worker) throw new Error('No browser experiment is active.');
      this.worker.postMessage({action:body.action});
      return {requested:body.action};
    }
    throw new Error('This action is unavailable on the published site.');
  }

  restorePublished() {
    if(this.worker) throw new Error('Stop the browser experiment before returning to the published run.');
    this.state=copy(this.publishedStatus);this.model=copy(this.publishedModel);this.browserRun=false;
  }
}
