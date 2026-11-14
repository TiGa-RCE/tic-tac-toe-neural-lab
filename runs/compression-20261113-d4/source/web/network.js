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

function densePredict(model, board) {
  let input = encode(board, modelEncoding(model));
  if (!Array.isArray(model?.layers) || !model.layers.length) throw new Error('No trained network loaded.');
  for (const layer of model.layers) {
    if (layer.weights.length !== input.length) throw new Error('Checkpoint input dimensions do not match.');
    const output = layer.bias.map((bias, j) => {
      let value = bias;
      for (let i = 0; i < input.length; i++) value += input[i] * layer.weights[i][j];
      return layer.activation === 'tanh' ? Math.tanh(value) : value;
    });
    if (output.some(v => !Number.isFinite(v))) throw new Error('Checkpoint produced a non-finite score.');
    input = output;
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

export function chooseMove(model, board) {
  if (outcome(board) !== null) return null;
  const scores = predict(model, board);
  let move = null;
  for (let i = 0; i < 9; i++) {
    if (board[i] === 0 && (move === null || scores[i] > scores[move])) move = i;
  }
  return move;
}
