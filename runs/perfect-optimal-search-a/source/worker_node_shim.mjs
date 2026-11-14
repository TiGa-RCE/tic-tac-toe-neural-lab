// Test adapter for the identical browser module; no training logic is duplicated.
import {parentPort, workerData} from 'node:worker_threads';
globalThis.self={postMessage:message=>parentPort.postMessage(message)};
parentPort.on('message',data=>self.onmessage({data}));
await import(workerData.module);
