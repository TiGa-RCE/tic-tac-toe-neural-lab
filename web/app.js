import { predict, chooseMove, outcome } from './network.js';
import { detectHosting } from './hosted.js';

const $ = id => document.getElementById(id);
const escape = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
const pretty = key => String(key).replace(/_/g, ' ').replace(/\b\w/g, ch => ch.toUpperCase());
const finite = value => typeof value === 'number' && Number.isFinite(value);
const percent = value => finite(value) ? `${(value * 100).toFixed(1)}%` : '—';
const number = value => finite(value) ? (value !== 0 && Math.abs(value) < .00001 ? value.toExponential(2) : value.toLocaleString(undefined, {maximumFractionDigits: 5})) : String(value ?? '—');
const format = (key, value) => /accuracy|rate|agreement|coverage|fraction/.test(key) && finite(value) && value >= 0 && value <= 1 ? percent(value) : typeof value === 'boolean' ? (value ? 'Yes' : 'No') : number(value);
let latestStatus = null;
let requestBusy = false;
let polling = false;
let gameModel = null;
let board = Array(9).fill(0);
let humanSide = 1;
let thinking = false;
let loadingGame = false;
let gameToken = 0;
let oracle = null;
let oracleBoard = '';
let disconnected = false;
let hosted = null;

function showError(message) {
  $('error').textContent = message;
  $('error').hidden = !message;
}

async function api(path, options = {}) {
  if(hosted) return hosted.request(path,options);
  const response = await fetch(path, {
    cache: 'no-store',
    signal: AbortSignal.timeout(10000),
    ...options,
    headers: {'Content-Type': 'application/json', ...options.headers},
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error?.message || data.error || `Request failed (${response.status}).`);
  return data;
}

function phaseMarkup(phase, index) {
  const active = phase.status === 'running' || (latestStatus.phase_id === phase.id && ['running', 'paused'].includes(latestStatus.status));
  const completed = phase.status === 'completed';
  const skipped = phase.status === 'skipped';
  const progress = finite(phase.progress) ? Math.max(0, Math.min(1, phase.progress)) : completed ? 1 : 0;
  const metrics = Object.entries(phase.metrics || {}).filter(([, v]) => v !== null && typeof v !== 'object').slice(0, 5);
  return `<article class="phase ${active ? 'active' : ''} ${completed ? 'completed' : ''} ${skipped ? 'skipped' : ''} ${phase.status === 'failed' ? 'failed' : ''}" ${active ? 'aria-current="step"' : ''}>
    <div class="phase-index">${completed ? '✓' : skipped ? '·' : String(index + 1).padStart(2, '0')}</div>
    <div><h3>${escape(phase.title || pretty(phase.id))}</h3><p>${escape(phase.detail || '')}</p>
      ${metrics.length ? `<div class="phase-metrics">${metrics.map(([k,v]) => `<span>${escape(pretty(k))}: ${escape(format(k,v))}</span>`).join('')}</div>` : ''}
      ${active ? `<div class="progress-track" role="progressbar" aria-label="${escape(phase.title)}" aria-valuenow="${Math.round(progress * 100)}" aria-valuemin="0" aria-valuemax="100"><span style="width:${progress * 100}%"></span></div>` : ''}
    </div><div class="phase-percent">${completed ? 'DONE' : skipped ? 'NOT NEEDED' : active ? `${Math.round(progress * 100)}%` : phase.status === 'failed' ? 'FAIL' : '—'}</div>
  </article>`;
}

function flattenEvidence(object, prefix = '', depth = 0) {
  const entries = [];
  for (const [key, value] of Object.entries(object || {})) {
    const label = prefix ? `${prefix} / ${pretty(key)}` : pretty(key);
    if (value === null || value === undefined) continue;
    if (Array.isArray(value)) {
      if (value.every(v => typeof v !== 'object')) entries.push([key, label, value.join(', ')]);
      else if (value.length) entries.push([key, label, JSON.stringify(value)]);
    } else if (typeof value === 'object' && depth < 2) entries.push(...flattenEvidence(value, label, depth + 1));
    else entries.push([key, label, value]);
  }
  return entries;
}

function renderEvidence(status) {
  const results = flattenEvidence(status.evaluation).slice(0, 36);
  $('evaluation').innerHTML = results.length ? results.map(([key, label, value]) => `<div class="evidence-item ${String(value).length > 48 ? 'wide' : ''}"><span>${escape(label)}</span><strong>${escape(format(key,value))}</strong></div>`).join('') : '<p class="muted">Qualification results will appear after evaluation.</p>';
  const dataset = flattenEvidence(status.dataset).slice(0, 24);
  $('dataset').innerHTML = dataset.map(([key, label, value]) => `<div><span>${escape(label)}</span><strong>${escape(format(key,value))}</strong></div>`).join('');
  $('recipe-note').textContent = status.recipe_note || 'The exact teacher generates labels; held-out evaluation and full-domain qualification are reported separately.';
}

function renderCandidates(status) {
  const candidates = Array.isArray(status.candidates) ? status.candidates : [];
  $('candidates-panel').hidden = candidates.length === 0;
  $('candidate-count').textContent = `${candidates.length} SAVED`;
  const selected = status.selection?.checkpoint || status.evaluation?.selected_checkpoint;
  const finiteDomain = /finite-domain|compress/.test(status.training_mode || '') || Boolean(status.compression);
  $('candidate-metric-heading').textContent = finiteDomain ? 'Full domain' : 'Validation';
  $('candidate-note').textContent = finiteDomain ? 'Every reachable decision state is teaching data in this continuation. Repairs and breaks compare with its frozen parent; this is finite-domain coverage, not held-out generalization.' : 'Measured against the shared founder on validation: repairs turn wrong moves correct; breaks turn correct moves wrong. Selection is frozen before TEST.';
  $('candidates').innerHTML = candidates.map(candidate => {
    const chosen = candidate.checkpoint === selected;
    const correct = finiteDomain ? candidate.training_correct ?? candidate.validation_correct : candidate.validation_correct;
    const count = finiteDomain ? candidate.training_count ?? candidate.validation_count : candidate.validation_count;
    const fraction = finite(correct) && finite(count) ? `${correct} / ${count}` : '—';
    return `<tr class="${chosen ? 'selected' : ''}"><th scope="row">${escape(candidate.checkpoint)}${chosen ? '<span class="selected-label">SELECTED</span>' : ''}</th><td>${escape(fraction)}</td><td class="repair-count">${escape(number(candidate.repairs))}</td><td class="${candidate.breaks ? 'break-count' : ''}">${escape(number(candidate.breaks))}</td></tr>`;
  }).join('');
}

function renderCompression(status) {
  const compression=status.compression;
  $('compression-panel').hidden=!compression;
  if(!compression) return;
  const baseline=compression.baseline_parameters;
  const selected=compression.selected_parameters ?? status.parameter_count;
  $('compression-baseline').textContent=number(baseline);
  $('compression-selected').textContent=number(selected);
  $('compression-reduction').textContent=finite(baseline)&&baseline>0&&finite(selected)?`${(100*(1-selected/baseline)).toFixed(2)}%`:'—';
  $('compression-storage').hidden=!finite(compression.coefficient_payload_bytes);
  $('compression-storage').textContent=`${number(compression.coefficient_payload_bytes)}-byte coefficient payload (${compression.coefficient_precision || 'declared precision'}; JSON and runtime code excluded). ${number(compression.forward_passes_per_move ?? 1)} neural forward passes per move.`;
  const candidates=Array.isArray(compression.candidates)?compression.candidates:[];
  $('compression-trials').textContent=`${number(compression.trial_count ?? candidates.length)} RECORDED`;
  $('compression-candidates').innerHTML=candidates.map(candidate=>{
    const parameters=candidate.parameters ?? candidate.parameter_count;
    const passes=candidate.forward_passes_per_move ?? (candidate.symmetry_average==='D4'?8:1);
    const reference=candidate.reference_logit_zero===true || candidate.architecture?.at(-1)===8;
    const architecture=(Array.isArray(candidate.architecture)?candidate.architecture.join(' → '):candidate.architecture || '—')+(reference?' + fixed 0':'')+(candidate.output_bias_orbits==='D4-corner-zero'?' · shared bias':'');
    const noLoss=candidate.no_loss_from_empty ?? candidate.independent?.no_loss_from_empty ?? candidate.evaluation?.independent?.no_loss_from_empty;
    const optimal=candidate.optimal_count ?? candidate.independent?.optimal_count ?? candidate.evaluation?.exhaustive?.correct;
    const total=candidate.total ?? candidate.count ?? candidate.independent?.total ?? candidate.evaluation?.exhaustive?.count ?? 4520;
    const seed=Array.isArray(candidate.seeds)?candidate.seeds.join(', '):candidate.seed;
    const trials=candidate.trials ?? candidate.attempts;
    const sampling=seed!==undefined?String(seed):finite(trials)?`${number(trials)} trials`:'—';
    const chosen=candidate.selected===true || (candidate.checkpoint && candidate.checkpoint===compression.selected_checkpoint);
    const qualification=candidate.qualification || 'No independent qualification recorded';
    return `<tr class="${chosen?'selected':''}"><th scope="row">${escape(architecture)}${chosen?'<span class="selected-label">SELECTED</span>':''}</th><td>${escape(number(parameters))}</td><td>${escape(number(passes))}${candidate.symmetry_average==='D4'?' · D4':''}</td><td>${escape(sampling)}</td><td class="${noLoss===true?'repair-count':noLoss===false?'break-count':''}">${noLoss===true?'Yes':noLoss===false?'No':'—'}</td><td>${finite(optimal)?`${escape(number(optimal))} / ${escape(number(total))}`:'—'}</td><td class="qualification ${/failed/i.test(qualification)?'break-count':/passed/i.test(qualification)?'repair-count':''}">${escape(qualification)}</td></tr>`;
  }).join('');
  $('compression-scope').textContent=compression.scope || 'Smallest qualified network in this recorded search. Failed training attempts do not prove that a smaller architecture is impossible. No solver or tactical fallback is used during play.';
}

function renderControls() {
  const state = latestStatus?.status || 'idle';
  const active = ['running', 'paused', 'starting'].includes(state);
  $('start').disabled = active || requestBusy || disconnected;
  $('pause').hidden = state === 'paused';
  $('pause').disabled = state !== 'running' || requestBusy || disconnected;
  $('resume').hidden = state !== 'paused';
  $('resume').disabled = state !== 'paused' || requestBusy || disconnected;
  $('stop').disabled = !active || requestBusy || disconnected;
  $('seed').disabled = active || requestBusy;
  $('training-network').disabled = active || requestBusy;
  $('published-run').hidden = !hosted?.browserRun;
  $('published-run').disabled = active || requestBusy;
  $('download-run').hidden = !hosted?.archive;
}

function renderTrainingChoices() {
  $('training-network-label').hidden=!hosted;
  $('training-network-note').hidden=!hosted;
  if(!hosted) return;
  const networks=hosted.trainingNetworks;
  const signature=JSON.stringify(networks.map(network=>[network.id,network.label,network.parameters,network.architecture,network.symmetry_average,network.output_bias_orbits]));
  if($('training-network').dataset.catalog!==signature) {
    $('training-network').innerHTML=networks.map(network=>`<option value="${escape(network.id)}">${escape(network.label || network.id)} · ${escape(number(network.parameters))} parameters · ${escape((network.architecture || []).join(' → '))}${network.reference_logit_zero===true || network.architecture?.at(-1)===8?' + fixed 0':''}${network.output_bias_orbits==='D4-corner-zero'?' · shared bias':''}${network.symmetry_average==='D4'?' · 8 passes':''}</option>`).join('');
    $('training-network').dataset.catalog=signature;
  }
  $('training-network').value=hosted.selectedTrainingId;
  $('training-network-note').textContent=`This choice trains only the browser experiment. The playable game keeps the published ${number(hosted.publishedModel.parameter_count)}-parameter champion; experiment weights are available in the run download.`;
}

function renderStatus(status) {
  latestStatus = status;
  const state = status.status || 'idle';
  $('run-status').textContent = pretty(state);
  $('run-status').className = `status-pill ${state}`;
  $('run-id').textContent = status.run_id || 'No run yet';
  const arch = status.architecture;
  if (Array.isArray(arch)) $('architecture').innerHTML = arch.map(escape).join(' <span>→</span> ')+(status.reference_logit_zero===true || arch.at(-1)===8?' <span>+ fixed 0</span>':'');
  if (finite(status.parameter_count)) $('parameter-count').textContent = status.parameter_count.toLocaleString();
  if (Array.isArray(status.phases) && status.phases.length) $('phases').innerHTML = status.phases.map(phaseMarkup).join('');
  const metrics = status.metrics || {};
  const finiteDomain = /finite-domain|compress/.test(status.training_mode || '') || Boolean(status.compression);
  const browserTraining = status.training_mode === 'browser-finite-domain';
  $('loss').textContent = finite(metrics.loss) ? metrics.loss.toFixed(4) : '—';
  $('train-accuracy').textContent = percent(metrics.train_accuracy);
  const lastCandidate = (status.candidates || []).at(-1);
  const fullCoverage = status.evaluation?.exhaustive?.accuracy ?? (lastCandidate && (lastCandidate.training_correct ?? lastCandidate.validation_correct) / (lastCandidate.training_count ?? lastCandidate.validation_count));
  $('validation-accuracy').textContent = percent(finiteDomain ? fullCoverage : metrics.val_accuracy ?? metrics.validation_accuracy);
  $('validation-metric-label').textContent = finiteDomain ? 'LAST FULL AUDIT' : 'LATEST VALIDATION';
  $('train-metric-label').textContent = browserTraining ? 'TRAIN PROBE' : 'LATEST TRAIN';
  $('validation-legend').hidden = finiteDomain;
  if(hosted) {
    $('start').innerHTML = 'Train in this browser <span aria-hidden="true">↗</span>';
    $('training-note').textContent = hosted.browserRun ? 'Real training on this device. One branch, three 512-update phases; all decision states are TRAIN. The published policy remains separate.' : 'Published training evidence is a saved snapshot. Start an independent, visitor-local browser experiment to watch real updates; it does not modify the public baseline.';
    $('journal-label').textContent = hosted.browserRun ? 'BROWSER LIVE' : 'SAVED EVIDENCE';
    $('runtime-note').textContent = hosted.browserRun ? 'Browser experiment · only this device' : 'Published game and training archive';
  } else {
    $('start').innerHTML = status.model_available ? 'Continue training <span aria-hidden="true">↗</span>' : 'Start training <span aria-hidden="true">↗</span>';
    if(finiteDomain) $('training-note').textContent = 'Continue from the saved network: collect → review → retain, then exhaustive no-loss certification as X and O. All finite decision states are TRAIN.';
  }
  if(status.status==='failed' && status.error) showError(status.error);
  $('epoch-label').textContent = finite(metrics.epoch) ? `Update ${number(metrics.epoch)}${metrics.branch !== null && metrics.branch !== undefined ? ` · seed ${number(metrics.branch)}` : ' · founder'}` : finite(metrics.step) ? `Update ${number(metrics.step)}` : 'No updates recorded';
  const events = Array.isArray(status.events) ? status.events : [];
  if (events.length) $('events').innerHTML = events.slice(-18).reverse().map(event => {
    let time = event.time;
    if (finite(time)) time = new Date(time < 1e12 ? time * 1000 : time);
    else time = new Date(time);
    const label = Number.isNaN(time.getTime()) ? '·' : time.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
    return `<li><time>${escape(label)}</time><span>${escape(event.message || event.text || '')}</span></li>`;
  }).join('');
  renderEvidence(status);
  renderCandidates(status);
  renderCompression(status);
  renderTrainingChoices();
  drawChart(status.history || []);
  renderControls();
  if (!gameModel && status.model_available && !loadingGame) newGame();
}

function drawChart(history) {
  const canvas = $('chart');
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const pad = {left:35, right:34, top:10, bottom:25};
  const w = width - pad.left - pad.right;
  const h = height - pad.top - pad.bottom;
  ctx.font = '9px ui-monospace,monospace';
  ctx.lineWidth = 1;
  ctx.strokeStyle = '#2b3530';
  ctx.fillStyle = '#788c7e';
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + h * i / 4;
    ctx.beginPath();ctx.moveTo(pad.left,y);ctx.lineTo(width-pad.right,y);ctx.stroke();
    ctx.textAlign = 'left';ctx.fillText(`${100-i*25}%`, width-pad.right+7,y+3);
  }
  const points = history.filter(p => finite(p.loss) || finite(p.train_accuracy) || finite(p.val_accuracy));
  $('chart-empty').hidden = points.length > 0;
  if (!points.length) return;
  const steps = points.map((p,i) => finite(p.step) ? p.step : i);
  const minStep = Math.min(...steps), maxStep = Math.max(...steps);
  const maxLoss = Math.max(0.01, ...points.filter(p => finite(p.loss)).map(p => p.loss)) * 1.08;
  for (let i = 0; i <= 2; i++) {
    ctx.textAlign = 'right';ctx.fillStyle = '#b28d70';
    ctx.fillText((maxLoss * (1-i/2)).toFixed(2), pad.left-7, pad.top+h*i/2+3);
  }
  const phaseNames = {founder:'Founder', phase1:'Phase 1', phase2:'Phase 2', phase3:'Phase 3'};
  let previousPhase = null;
  points.forEach((point, index) => {
    const previous = points[index - 1];
    if (previous && (point.branch !== previous.branch || point.phase !== previous.phase)) {
      const x = pad.left + w * ((steps[index] + steps[index - 1]) / 2 - minStep) / Math.max(1, maxStep - minStep);
      ctx.save();ctx.setLineDash([2, 4]);ctx.strokeStyle = '#62746666';ctx.lineWidth = 1;
      ctx.beginPath();ctx.moveTo(x, pad.top);ctx.lineTo(x, pad.top + h);ctx.stroke();ctx.restore();
    }
    if (point.phase !== previousPhase) {
      const x = pad.left + w * (steps[index] - minStep) / Math.max(1, maxStep - minStep);
      ctx.fillStyle = '#84998b';ctx.textAlign = 'left';
      ctx.fillText(phaseNames[point.phase] || point.phase, x + 3, pad.top + 9);
      previousPhase = point.phase;
    }
  });
  const lines = [['loss','#eab388',maxLoss],['train_accuracy','#b8edcb',1],['val_accuracy','#c4b3eb',1]];
  for (const [key,color,max] of lines) {
    ctx.strokeStyle = color;ctx.lineWidth = 1.8;ctx.beginPath();
    let started = false;
    points.forEach((point,index) => {
      if (!finite(point[key])) {started=false;return;}
      const x = pad.left + w * (steps[index]-minStep) / Math.max(1,maxStep-minStep);
      const y = pad.top + h * (1-Math.max(0,Math.min(max,point[key]))/max);
      const previous = points[index-1];
      if (!started || (previous && (point.branch !== previous.branch || point.phase !== previous.phase))) ctx.moveTo(x,y);else ctx.lineTo(x,y);
      started=true;
    });
    ctx.stroke();
    if (points.length===1 && finite(points[0][key])) {
      ctx.fillStyle=color;ctx.beginPath();ctx.arc(pad.left,pad.top+h*(1-points[0][key]/max),2,0,Math.PI*2);ctx.fill();
    }
  }
  ctx.fillStyle='#788c7e';ctx.textAlign='left';ctx.fillText(String(minStep),pad.left,height-5);
  ctx.textAlign='right';ctx.fillText(`${maxStep} steps`,width-pad.right,height-5);
}

async function poll() {
  if (polling) return;
  polling = true;
  try {
    const data = await api('/api/status');
    disconnected = false;
    $('connection').textContent = hosted ? (hosted.browserRun ? 'Browser training workspace' : 'Published checkpoint ready') : 'Local trainer connected';
    $('connection-dot').className = 'dot online';
    renderStatus(data);
  } catch (error) {
    disconnected = true;
    $('connection').textContent = 'Trainer disconnected';
    $('connection-dot').className = 'dot offline';
    renderControls();
  } finally {polling = false;}
}

async function trainingAction(path, body) {
  if (requestBusy) return;
  requestBusy = true;showError('');renderControls();
  try {await api(path, {method:'POST', body:JSON.stringify(body)});await poll();}
  catch(error) {showError(error.message);}
  finally {requestBusy=false;renderControls();}
}

function currentPlayer() {return board.filter(v=>v===1).length === board.filter(v=>v===-1).length ? 1 : -1;}
function policyProbabilities(logits, cells) {
  const legal = cells.map((v,i)=>v===0?i:-1).filter(i=>i>=0);
  const max = Math.max(...legal.map(i=>logits[i]));
  const values=logits.map((v,i)=>cells[i]===0?Math.exp(v-max):0);
  const sum=values.reduce((a,b)=>a+b,0);
  return values.map(v=>sum?v/sum:0);
}

function renderGame() {
  const result=outcome(board);
  let probabilities=null,best=null;
  if(gameModel && result===null) {
    try {probabilities=policyProbabilities(predict(gameModel,board),board);best=chooseMove(gameModel,board);}
    catch(error) {showError(error.message);}
  }
  const showScores=$('show-scores').checked;
  const showTeacher=$('teacher-overlay').checked && oracle && oracleBoard===board.join(',');
  $('board').innerHTML=board.map((value,index)=>{
    const disabled=!gameModel || loadingGame || thinking || result!==null || value!==0 || currentPlayer()!==humanSide;
    const label=value===1?'X':value===-1?'O':'Empty';
    const probability=probabilities?.[index];
    return `<button class="cell ${value===1?'mark-x':value===-1?'mark-o':'empty'} ${index===best?'best':''} ${showTeacher && oracle.optimal_moves.includes(index)?'teacher-optimal':''}" data-cell="${index}" aria-label="Row ${Math.floor(index/3)+1}, column ${index%3+1}: ${label}${value===0&&probabilities?`, policy probability ${percent(probability)}`:''}" ${disabled?'disabled':''}>
      <span class="cell-number" aria-hidden="true">${index+1}</span>${value===1?'×':value===-1?'○':''}${!value&&showScores&&probabilities?`<span class="score">${percent(probability)}</span>`:''}${!value&&index===best&&showScores?'<span class="score-caption">TOP</span>':''}</button>`;
  }).join('');
  $('human-symbol').textContent=`Playing ${humanSide===1?'X':'O'}`;
  $('network-symbol').textContent=`Playing ${humanSide===1?'O':'X'}`;
  document.querySelector('.player-symbol.human').textContent=humanSide===1?'×':'○';
  document.querySelector('.player-symbol.neural').textContent=humanSide===1?'○':'×';
  $('game-status').textContent=loadingGame?'Loading latest checkpoint…':!gameModel?'Waiting for a trained checkpoint':result!==null?(result===0?'A draw. Well played.':result===humanSide?'You won this game.':'The student wins this round.'):thinking?'The student is choosing…':currentPlayer()===humanSide?'Your turn. Pick a cell.':"The student's turn.";
  $('checkpoint').textContent=gameModel?`${gameModel.checkpoint || 'Saved weights'}${gameModel.run_id?` · ${gameModel.run_id}`:''}`:'No model loaded';
  const symmetry=gameModel?.symmetry_average==='D4';
  const passes=symmetry?8:1;
  $('inference-work').hidden=!gameModel;
  $('inference-work').textContent=(symmetry?`${number(gameModel.parameter_count)} learned parameters; ${number(passes)} neural passes per move. The same weights score all rotations and reflections. `:`${number(passes)} neural forward pass per move. `)+(gameModel?.reference_logit_zero===true?'Eight learned readouts plus a fixed zero produce nine move scores. ':'')+(gameModel?.output_bias_orbits==='D4-corner-zero'?'Two learned readout biases are shared across edge and center positions; corner biases are fixed at zero. ':'')+(Number.isInteger(gameModel?.logit_round_decimals)?`Policy ties use ${gameModel.logit_round_decimals}-decimal score keys, then lowest cell index. `:'')+'No solver or tactical fallback.';
  $('inference-explanation').textContent=symmetry?'The same network scores eight rotations/reflections, maps scores back to the original cells and averages them. Occupied cells are masked; the highest legal neural score chooses the move.':'The student scores the nine cells in one forward pass. Occupied cells are masked; the highest legal policy score chooses the move.';
  const publishedProof=hosted?.publishedStatus?.evaluation?.independent?.no_loss_from_empty;
  $('play-proof').hidden=!publishedProof || !gameModel || gameModel.run_id!==hosted?.publishedModel?.run_id;
  $('play-proof').textContent='Published checkpoint: independently verified no-loss as both X and O. Moves use only this frozen neural network.';
  $('export').disabled=!gameModel;
  $('new-game').disabled=loadingGame;
  $('teacher-note').hidden=!$('teacher-overlay').checked;
  if($('teacher-overlay').checked) $('teacher-note').textContent=showTeacher?(result===null?`Teacher value: ${oracle.value===1?'forced win':oracle.value===-1?'forced loss':'draw with perfect play'}. Outlined cells are optimal. Displayed probabilities are the network’s policy, not win chances.`:'Game over. The teacher is only an observer.'):'Comparing this position with the teacher…';
}

async function updateOracle() {
  if(!$('teacher-overlay').checked || !gameModel) return;
  const key=board.join(',');
  try {
    const data=await api(`/api/oracle?board=${key}`);
    if(key===board.join(',') && $('teacher-overlay').checked) {oracle=data;oracleBoard=key;renderGame();}
  } catch(error) {$('teacher-note').textContent=`Teacher comparison unavailable: ${error.message}`;}
}

function advanceStudent(token) {
  if(!gameModel || outcome(board)!==null || currentPlayer()===humanSide) return;
  thinking=true;renderGame();
  setTimeout(()=>{
    if(token!==gameToken) return;
    try {
      const move=chooseMove(gameModel,board);
      if(move!==null) board[move]=currentPlayer();
    } catch(error) {showError(error.message);}
    thinking=false;oracle=null;renderGame();updateOracle();
  },260);
}

async function newGame() {
  const token=++gameToken;
  thinking=false;loadingGame=true;oracle=null;board=Array(9).fill(0);humanSide=Number($('side').value);renderGame();
  try {
    const model=await api('/api/model');
    if(token!==gameToken) return;
    predict(model,board);
    gameModel=model;
  } catch(error) {if(!gameModel) showError(`No playable checkpoint yet. ${error.message}`);else showError(`Keeping the last loaded checkpoint: ${error.message}`);}
  finally {
    if(token===gameToken) {loadingGame=false;renderGame();updateOracle();advanceStudent(token);}
  }
}

$('start').addEventListener('click',()=>{
  const seed=Number($('seed').value);
  if(!Number.isInteger(seed)||seed<0||seed>2147483647) {showError('Choose an integer seed from 0 to 2,147,483,647.');return;}
  trainingAction('/api/train',{seed,mode:latestStatus?.model_available?'continue':'foundation',...(hosted?{network_id:$('training-network').value}:{})});
});
$('training-network').addEventListener('change',()=>{if(hosted) hosted.selectedTrainingId=$('training-network').value;});
for(const action of ['pause','resume','stop']) $(action).addEventListener('click',()=>trainingAction('/api/control',{action}));
$('new-game').addEventListener('click',newGame);
$('side').addEventListener('change',newGame);
$('show-scores').addEventListener('change',renderGame);
$('teacher-overlay').addEventListener('change',()=>{renderGame();updateOracle();});
$('board').addEventListener('click',event=>{
  const cell=event.target.closest('[data-cell]');
  if(!cell || cell.disabled || thinking || !gameModel || outcome(board)!==null || currentPlayer()!==humanSide) return;
  const move=Number(cell.dataset.cell);
  if(board[move]) return;
  board[move]=humanSide;oracle=null;renderGame();updateOracle();advanceStudent(gameToken);
});
$('export').addEventListener('click',()=>{
  if(!gameModel) return;
  const url=URL.createObjectURL(new Blob([JSON.stringify(gameModel,null,2)],{type:'application/json'}));
  const link=document.createElement('a');link.href=url;link.download='tic-tac-toe-student.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
window.addEventListener('resize',()=>drawChart(latestStatus?.history || []));
$('published-run').addEventListener('click',()=>{
  try {hosted.restorePublished();renderStatus(hosted.state);showError('');}
  catch(error) {showError(error.message);}
});
$('download-run').addEventListener('click',()=>{
  if(!hosted?.archive) return;
  const url=URL.createObjectURL(new Blob([JSON.stringify(hosted.archive)],{type:'application/json'}));
  const link=document.createElement('a');link.href=url;link.download=`${hosted.state.run_id}-evidence.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
async function initialize() {
  renderGame();renderControls();
  try {
    hosted=await detectHosting();
    if(hosted) {
      $('runtime-banner').hidden=false;
      if(hosted.config.repository_url) {$('repository-link').href=hosted.config.repository_url;$('repository-link').hidden=false;}
    }
    await poll();setInterval(poll,1000);
  } catch(error) {showError(`Unable to initialize the dashboard: ${error.message}`);}
}
initialize();
