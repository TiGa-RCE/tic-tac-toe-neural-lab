/** Small dense network inference. Gameplay uses these weights, never the teacher. */
export const D4_PERMUTATIONS=Object.freeze([false,true].flatMap(reflect=>[0,1,2,3].map(rotations=>Object.freeze(Array.from({length:9},(_,index)=>{
  let row=Math.floor(index/3),col=index%3;
  if(reflect) col=2-col;
  for(let rotation=0;rotation<rotations;rotation++) [row,col]=[col,2-row];
  return row*3+col;
})))));

export function modelEncoding(model) {
  const encoding = model?.encoding || ({9:'relative-signed',18:'relative-two-channel',27:'relative-one-hot-interleaved'}[model?.architecture?.[0]]);
  if (!['relative-signed','relative-two-channel','relative-one-hot-interleaved'].includes(encoding)) {
    throw new Error('Checkpoint has an unsupported board encoding.');
  }
  return encoding;
}

export function encode(board, encoding = 'relative-one-hot-interleaved') {
  if (!Array.isArray(board) || board.length !== 9 || board.some(v => ![-1, 0, 1].includes(v))) {
    throw new Error('Expected nine cells containing −1, 0, or 1.');
  }
  const player = board.filter(v => v === 1).length === board.filter(v => v === -1).length ? 1 : -1;
  if (encoding === 'relative-signed') return board.map(v => v * player);
  if (encoding === 'relative-two-channel') return board.flatMap(v => [Number(v === player), Number(v === -player)]);
  if (encoding !== 'relative-one-hot-interleaved') throw new Error('Unsupported board encoding.');
  return board.flatMap(v => [Number(v === player), Number(v === -player), Number(v === 0)]);
}

// Structural zeros and unit constants are explicit, rather than stored learned
// scalar values. Their positions and format metadata remain part of the file.
export function coefficientValue(value, storage = null) {
  if(storage!==null && storage!==undefined && storage!=='sparse-fixed-v1') throw new Error('Unsupported coefficient storage.');
  if(typeof value==='number' && Number.isFinite(value)) return value;
  if(storage==='sparse-fixed-v1') {
    if(value===null) return 0;
    if(value==='fixed:+1') return 1;
    if(value==='fixed:-1') return -1;
  }
  throw new Error('Invalid neural coefficient.');
}

function densePredict(model, board) {
  let input = encode(board, modelEncoding(model));
  if (!Array.isArray(model?.layers) || !model.layers.length) throw new Error('No trained network loaded.');
  const biasOrbits=model.output_bias_orbits ?? null;
  if(biasOrbits!==null && biasOrbits!=='D4-corner-zero') throw new Error('Checkpoint has unsupported shared output biases.');
  if(biasOrbits==='D4-corner-zero' && (model.symmetry_average!=='D4' || model.reference_logit_zero!==true || model.architecture?.at(-1)!==8)) throw new Error('Shared output biases require the declared D4 fixed-zero readout.');
  for (const [index,layer] of model.layers.entries()) {
    if (layer.weights.length !== input.length) throw new Error('Checkpoint input dimensions do not match.');
    const shared=biasOrbits==='D4-corner-zero' && index===model.layers.length-1;
    if(shared && (layer.bias.length!==2 || layer.weights.some(row=>row.length!==8))) throw new Error('Shared output readout must store eight weight columns and two biases.');
    const storedBias=layer.bias.map(value=>coefficientValue(value,model.coefficient_storage));
    const bias=shared?[0,storedBias[0],0,storedBias[0],storedBias[1],storedBias[0],0,storedBias[0]]:storedBias;
    const output = bias.map((bias, j) => {
      let value = bias;
      for (let i = 0; i < input.length; i++) value += input[i] * coefficientValue(layer.weights[i][j],model.coefficient_storage);
      return layer.activation === 'tanh' ? Math.tanh(value) : value;
    });
    if (output.some(v => !Number.isFinite(v))) throw new Error('Checkpoint produced a non-finite score.');
    input = output;
  }
  if(model.reference_logit_zero===true) {
    if(input.length!==8) throw new Error('Reference-logit checkpoint must learn eight scores.');
    input.push(0);
  }
  if (input.length !== 9) throw new Error('Checkpoint must produce nine move scores.');
  return input;
}

export function predict(model, board) {
  if(model?.symmetry_average===undefined || model.symmetry_average===null || model.symmetry_average===false || model.symmetry_average==='none') return densePredict(model,board);
  if(model.symmetry_average!=='D4') throw new Error('Checkpoint has unsupported symmetry averaging.');
  const average=Array(9).fill(0);
  for(const permutation of D4_PERMUTATIONS) {
    const transformed=Array(9);
    for(let cell=0;cell<9;cell++) transformed[permutation[cell]]=board[cell];
    const logits=densePredict(model,transformed);
    for(let cell=0;cell<9;cell++) average[cell]+=logits[permutation[cell]];
  }
  return average.map(value=>value/8);
}

export function outcome(board) {
  for (const [a, b, c] of [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]) {
    if (board[a] && board[a] === board[b] && board[a] === board[c]) return board[a];
  }
  return board.includes(0) ? null : 0;
}

export function policyScore(score, decimals = null) {
  if(decimals===null || decimals===undefined) return score;
  if(!Number.isInteger(decimals) || decimals<0 || decimals>12) throw new Error('Unsupported policy score rounding.');
  return Math.floor(score * 10**decimals + .5);
}

export function chooseMove(model, board) {
  if (outcome(board) !== null) return null;
  const scores = predict(model, board).map(score=>policyScore(score,model.logit_round_decimals));
  let move = null;
  for (let i = 0; i < 9; i++) {
    if (board[i] === 0 && (move === null || scores[i] > scores[move])) move = i;
  }
  return move;
}
