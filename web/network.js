/** Small dense network inference. Gameplay uses these weights, never the teacher. */
export function encode(board) {
  if (!Array.isArray(board) || board.length !== 9 || board.some(v => ![-1, 0, 1].includes(v))) {
    throw new Error('Expected nine cells containing −1, 0, or 1.');
  }
  const player = board.filter(v => v === 1).length === board.filter(v => v === -1).length ? 1 : -1;
  return board.flatMap(v => [Number(v === player), Number(v === -player), Number(v === 0)]);
}

export function predict(model, board) {
  let input = encode(board);
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
