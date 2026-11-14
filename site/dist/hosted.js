/** Static-host adapter. Training lives in an isolated visitor-local worker. */
const copy = value => structuredClone(value);
async function readJSON(path) {
  const response = await fetch(path, {cache:'no-store'});
  if (!response.ok) throw new Error(`Unable to load ${path} (${response.status}).`);
  return response.json();
}

export async function detectHosting() {
  let response;
  try {
    response = await fetch('/site-config.json', {cache:'no-store'});
  } catch (error) {
    // A network-unavailable config can still be an ordinary local Python mode.
    return null;
  }
  if (!response.ok) return null;
  const config = await response.json();
  if (config.mode !== 'hosted') return null;
  // Once hosted mode is explicit, missing/corrupt evidence is an error. Never
  // silently substitute a fictitious local API or an empty published archive.
  const [status,model] = await Promise.all([readJSON('/data/status.json'),readJSON('/data/model.json')]);
  const indexResponse=await fetch('/data/networks.json',{cache:'no-store'});
  const index=indexResponse.ok?await indexResponse.json():null;
  return new HostedTrainer(config,status,model,index);
}

class HostedTrainer {
  constructor(config,status,model,index) {
    this.config=config;
    this.publishedStatus=copy(status);
    this.publishedModel=copy(model);
    this.state=copy(status);
    this.model=copy(model);
    this.worker=null;
    this.teaching=null;
    this.archive=null;
    this.browserRun=false;
    this.trainingNetworks=index?.networks?.length?copy(index.networks):[{
      id:'published',label:'Published champion',model_url:'/data/model.json',parameters:model.parameter_count,
      architecture:model.architecture,encoding:model.encoding,
      no_loss_from_empty:status.evaluation?.independent?.no_loss_from_empty,
    }];
    const availableDefault=id=>this.trainingNetworks.some(network=>network.id===id)?id:this.trainingNetworks[0].id;
    this.selectedTrainingId=availableDefault(index?.default_training_id || index?.default_id);
    this.gameNetworks=copy(this.trainingNetworks);
    this.selectedGameId=availableDefault(index?.default_game_id || index?.default_id);
    this.starting=false;
  }

  async request(path,options={}) {
    if(path==='/api/status') return this.state;
    if(path==='/api/model' || path.startsWith('/api/model?')) {
      const id=new URLSearchParams(path.split('?')[1] || '').get('network_id') || this.selectedGameId;
      const network=this.gameNetworks.find(candidate=>candidate.id===id);
      if(!network) throw new Error('Choose an available opponent network.');
      // Only frozen catalog weights are playable. Teacher data and visitor
      // training outputs never enter this route, including after load failures.
      return copy(await readJSON(network.model_url));
    }
    if(path.startsWith('/api/oracle?')) {
      this.teaching ||= await readJSON('/data/teaching.json');
      const key=new URLSearchParams(path.split('?')[1]).get('board');
      const index=this.teaching.boards.findIndex(board=>board.join(',')===key);
      if(index<0) return {value:0,optimal_moves:[],action_values:[]};
      return {value:this.teaching.values?.[index] ?? null,optimal_moves:this.teaching.optimal_moves[index]};
    }
    const body=JSON.parse(options.body || '{}');
    if(path==='/api/train') {
      if(this.worker || this.starting) throw new Error('A browser experiment is already active.');
      if(typeof Worker==='undefined') throw new Error('This browser does not support background training workers.');
      const trainingNetwork=this.trainingNetworks.find(network=>network.id===(body.network_id || this.selectedTrainingId));
      if(!trainingNetwork) throw new Error('Choose an available training network.');
      this.starting=true;
      let parent;
      try {parent=await readJSON(trainingNetwork.model_url);}
      finally {this.starting=false;}
      this.selectedTrainingId=trainingNetwork.id;
      this.browserRun=true;
      this.archive=null;
      this.state={...copy(this.publishedStatus),status:'starting',run_id:`browser-${body.seed}-${Date.now()}`,history:[],events:[],candidates:[],evaluation:{},selection:null,
        architecture:parent.architecture,parameter_count:parent.parameter_count,training_network:copy(trainingNetwork),
        recipe_note:'Visitor-local, single-branch continuation: three phases of 512 Adam updates, native collection, targeted review and protected replay. All reachable states are teaching data. This is finite-domain coverage, not a fresh held-out test. Your selected opponent stays frozen; trained weights are saved only in downloadable experiment evidence.'};
      this.worker=new Worker(new URL('./training-worker.js',import.meta.url),{type:'module'});
      this.worker.onmessage=event=>{
        const message=event.data;
        if(message.type==='status') this.state=message.status;
        if(message.type==='complete') {
          this.state=message.status;
          this.archive=message.archive;
          // Experimental exports belong in the run download. Gameplay keeps
          // the independently selected frozen catalog network.
          this.worker?.terminate();this.worker=null;
        }
      };
      this.worker.onerror=event=>{
        this.state={...this.state,status:'failed',error:event.message || 'Browser worker failed.'};
        this.worker?.terminate();this.worker=null;
      };
      this.worker.postMessage({action:'start',seed:body.seed,run_id:this.state.run_id,parent,training_network:copy(trainingNetwork),teaching_url:'/data/teaching.json'});
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
