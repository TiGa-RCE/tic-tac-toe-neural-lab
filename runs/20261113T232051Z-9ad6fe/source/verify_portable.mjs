import fs from 'node:fs';
import {predict, chooseMove} from './web/network.js';
const args=process.argv.slice(2);
const get=name=>args[args.indexOf(name)+1];
const model=JSON.parse(fs.readFileSync(get('--model'),'utf8'));
const expected=JSON.parse(fs.readFileSync(get('--fixtures'),'utf8'));
let maxError=0, mismatches=0, optimal=0;
for(const row of expected.fixtures){
  const logits=predict(model,row.board);
  for(let i=0;i<9;i++) maxError=Math.max(maxError,Math.abs(logits[i]-row.logits[i]));
  const move=chooseMove(model,row.board);
  if(move!==row.move) mismatches++;
  if(row.optimal_moves.includes(move)) optimal++;
}
const report={positions:expected.fixtures.length,action_mismatches:mismatches,max_logit_error:maxError,tolerance:1e-9,optimal_count:optimal,passed:mismatches===0&&maxError<=1e-9&&optimal===expected.optimal_count};
fs.writeFileSync(get('--output'),JSON.stringify(report));
if(!report.passed) throw new Error(JSON.stringify(report));
console.log(JSON.stringify(report));
