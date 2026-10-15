import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.mjs";
import chromaJs from 'https://cdn.jsdelivr.net/npm/chroma-js@3.1.2/+esm'

let session;
let x;
const scale = chromaJs.scale('viridis');//.scale('viridis').domain([0, 1]);
ort.env.wasm.simd = true;
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

// --- 4. Load ONNX Model ---
export async function loadModel(modelFile) {
    session = await ort.InferenceSession.create(modelFile, {
        executionProviders: ['wasm'],  // or 'webgpu' if supported
        graphOptimizationLevel: 'all', // enables full graph optimizations
    }); // your model path
}

function cloneInputTensors(inputDict) {
  const newDict = {};
  for (const key in inputDict) {
    if (Object.hasOwnProperty.call(inputDict, key)) {
      const tensor = inputDict[key];

      // Create a fresh ArrayBuffer copy
      const newData = tensor.data.slice();  // slice() creates a new ArrayBuffer
      newDict[key] = new ort.Tensor(tensor.type, newData, tensor.dims);
    }
  }
  return newDict;
}

// --- 5. Run ONNX Model ---
export async function runModel(model_inputs, edgeData, T, lastOut, Tau, chlorineInjection, wdsName) {
    model_inputs["Tau"] = new ort.Tensor('int64', new BigInt64Array(1), []);
    model_inputs["Tau"].cpuData[0] = String(Tau);
    model_inputs["offset"] = new ort.Tensor('int64', new BigInt64Array(1), []);
    model_inputs["offset"].cpuData[0] = String(T);
    
    switch (wdsName) {
        case "Hanoi":
            var injectionInput = chlorineInjection;
            var nSources = 1;
            break;
        case "l_town_no_tanks":
            var injectionInput = [...chlorineInjection, ...chlorineInjection];
            var nSources = 2;
            break;
    }

    model_inputs["boundary_values"] = new ort.Tensor(
        'float32', Float32Array.from(injectionInput), [nSources, chlorineInjection.length, 1]
    );
    // model_inputs["boundary_values"] = new ort.Tensor(
    //     'float32', Float32Array.from(chlorineInjection), [1, chlorineInjection.length, 1]
    // );


    if (lastOut !== undefined) {
        const lo = await lastOut;
        model_inputs["x"] = lo.rawOut;
    }
    console.log('Tick!')

    if (session === undefined) {
        //await loadModel();
        console.log('Call loadModel before runModel!')
    }
    if (ort.env.wasm.proxy) {
        model_inputs = cloneInputTensors(model_inputs)
    }
    const output = await session.run(model_inputs);//{ input: input });
    edgeData.push(output.xT);
    return { edgeData, rawOut : output.rawOut};
}


export function getSegmentColor(edgeData, edgeIdx, segmentIdx, timeIdx) {
    const numSegments = edgeData.dims[1];
    const numSteps = edgeData.dims[2];
    const idx = edgeIdx * (numSegments * numSteps) + (segmentIdx * numSteps) + timeIdx;
    return scale(edgeData.data[idx]);
}