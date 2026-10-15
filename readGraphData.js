import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.mjs";
import numpyParser from 'https://cdn.jsdelivr.net/npm/numpy-parser@1.2.3/+esm'

function convert32to64(int32Arr) {
    const bigInt64Arr = new BigInt64Array(int32Arr.length);
    for (let i = 0; i < int32Arr.length; i++) {
        bigInt64Arr[i] = BigInt(int32Arr[i]);
    }
    return bigInt64Arr
}

export async function loadAllDefaultNPY(manifestUrl, wdsName) {
    // Load the manifest JSON
    const manifestResp = await fetch(manifestUrl);
    const manifest = await manifestResp.json();

    const defaultInputs = {};

    for (const [key, fileFormat] of Object.entries(manifest)) {
        const filePath = fileFormat.replace('{{WDS}}', wdsName);
        // Fetch the .npy file
        const arrayBuffer = await fetch(filePath).then(r => r.arrayBuffer());

        const npArray = numpyParser.fromArrayBuffer(arrayBuffer);
        // defaultInputs[key] = npArray.data;
        var dtype;

        switch (npArray.data.constructor.name) {
            case 'Float32Array': dtype = 'float32'; break;
            case 'Float64Array': dtype = 'float64'; break;
            case 'Int32Array':   dtype = 'int64'; npArray.data = convert32to64(npArray.data); break;
            case 'Int16Array':   dtype = 'int16'; break;
            case 'Int8Array':    dtype = 'int8'; break;
            case 'Uint8Array':   dtype = 'uint8'; break;
            case 'Uint16Array':  dtype = 'uint16'; break;
            case 'Uint32Array':  dtype = 'uint32'; break;
        }
        var shape = npArray.shape;

        if (key === "boundary_index") {
            shape = [shape[0]];
        }
        
        // console.log(key, npArray.shape, dtype);
        defaultInputs[key] = new ort.Tensor(dtype, npArray.data, shape);
    }

    return defaultInputs;
}

export async function buildGraph(graphData, wdsName) {
    const edgeIndex = graphData["edge_index"];
    const NEdges = edgeIndex.dims[1]
    const senders = new Int32Array(NEdges);
    senders.set(Array.from(edgeIndex.data.slice(0, NEdges), x => Number(x)));
    const receivers = new Int32Array(NEdges);
    receivers.set(Array.from(edgeIndex.data.slice(NEdges, 2 * NEdges), x => Number(x)));
    const nodesSet = new Set([...senders, ...receivers]);
    const nodesArr = Array.from(nodesSet);
    // const cyNodesArr = nodesArr.map(id => ({ data: { id: id.toString() } }));

    const posFile = "default_model_inputs/" + wdsName + "/pos.json";
    console.log(posFile);
    const nodePositions = await fetch(posFile).then(response => response.json());
    // 1. Get all 'x' values into a separate array
    const xValues = nodesArr.map(p => nodePositions[p].x);

    // 2. Get all 'y' values into a separate array
    const yValues = nodesArr.map(p => nodePositions[p].y);

    // 3. Find the min and max for 'x'
    const minX = Math.min(...xValues);
    const maxX = Math.max(...xValues);
    // 4. Find the min and max for 'y'
    const minY = Math.min(...yValues);
    const maxY = Math.max(...yValues);

    const nodes = Object.fromEntries(
        nodesArr.map(p => (
            [p, {
                data: { id: p },
                position: nodePositions[p] // { x: p.x, y: p.y }
            }]
    )));
    const edges = Array.from(senders).map((src, i) => ({
        s: (src),
        t: (receivers[i])
    }));

    return {
        minX, maxX, minY, maxY, nodes, edges
    };
}
