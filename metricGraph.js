// import * as PIXI from "https://cdn.jsdelivr.net/npm/pixi.js@8.1.5/dist/pixi.mjs";
// const { Geometry, Mesh, Graphics } = PIXI;
// 
/**
 * Generate coordinates of rectangles (split into triangles) along a path.
 *
 * @param {Object} start - Starting point {x, y}.
 * @param {Object} end - Ending point {x, y}.
 * @param {number} width - Width of the rectangle strip.
 * @param {number} numSegments - Number of rectangular segments.
 * @returns {number[]} Flat list of triangle vertex coordinates [x0, y0, x1, y1, ...].
 */
function generatePathTriangles(start, end, width, numSegments) {
    const coords = [];
  
    // Direction vector along the path
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const length = Math.sqrt(dx * dx + dy * dy);
    if (length === 0) return coords;
  
    // Unit direction vector along the path
    const ux = dx / length;
    const uy = dy / length;
  
    // Perpendicular vector (normalized)
    const px = -uy;
    const py = ux;
  
    // Half width offset
    const hwx = (width / 2) * px;
    const hwy = (width / 2) * py;
  
    // Step size along the path
    const step = length / numSegments;
  
    for (let i = 0; i < numSegments; i++) {
      // Segment endpoints on centerline
      const x0 = start.x + i * step * ux;
      const y0 = start.y + i * step * uy;
      const x1 = start.x + (i + 1) * step * ux;
      const y1 = start.y + (i + 1) * step * uy;
  
      // Four rectangle corners
      const p0x = x0 + hwx, p0y = y0 + hwy; // top-left
      const p1x = x1 + hwx, p1y = y1 + hwy; // top-right
      const p2x = x1 - hwx, p2y = y1 - hwy; // bottom-right
      const p3x = x0 - hwx, p3y = y0 - hwy; // bottom-left
  
      // Split rectangle into two triangles: (p0, p1, p2) and (p0, p2, p3)
      coords.push(
        p0x, p0y, p1x, p1y, p2x, p2y,
        p0x, p0y, p2x, p2y, p3x, p3y
      );
    }
    
    return coords;
  }

function drawNode(pos, nodeSize, color){
    const node = new PIXI.Graphics()
        .circle(0, 0, nodeSize)
        .fill(color); // Red node
    node.x = Number(pos.x);
    node.y = Number(pos.y);
    //
    return node;//graphContainer.addChild(node);
}

export function createMeshStrip(start, end, width, numSegments) {
    const positions = generatePathTriangles(start, end, width, numSegments);
    const colors = [];

    positions.slice(0, numSegments*2*3).map((p, i) => {
        colors.push(1,i/(numSegments*2*3),0) 
    });
    const geometry = new PIXI.Geometry({
        attributes: {
          aPosition: positions,
          aColor: colors,
        },
      });

      const mesh = new PIXI.Mesh({
        geometry,
        shader,
      });


      return {
        geometry, mesh
      };
}

  const gl = { vertex, fragment };
          
  const gpu = {
    vertex: {
      entryPoint: 'mainVert',
      source,
    },
    fragment: {
      entryPoint: 'mainFrag',
      source,
    },
  };

  const shader = PIXI.Shader.from({
    gl,
    gpu,
  });

export function drawMetricGraph(graphContainer, graph, width, height, numSegments, nodeSize, edgeWidth) {
    const xRange = graph.maxX - graph.minX;
    const yRange = graph.maxY - graph.minY;

    // Screen Extents (from PixiJS app)
    const screenWidth = width;
    const screenHeight = height;

    // Padding (e.g., 50 pixels on all sides)
    const padding = [50, 200];
    const usableScreenWidth = screenWidth - 2 * padding[0];
    const usableScreenHeight = screenHeight - 2 * padding[1];

    // 1. Calculate the two potential scale factors (Screen Range / Data Range)
    const scaleFactorX = usableScreenWidth / xRange;
    const scaleFactorY = usableScreenHeight / yRange;

    // 2. Use the smaller scale factor to ensure the entire graph fits
    const uniformScale = Math.min(scaleFactorX, scaleFactorY);
    graphContainer.x = padding[0] - (graph.minX * uniformScale);
    graphContainer.y = padding[1] - (graph.minY * uniformScale);
    graphContainer.scale.set(uniformScale);
    // console.log(uniformScale, minX, minY, maxX, maxY)
    
    const edgeColor = 0x00ff00;
    const nodeColor = 0x00ff00;

    // Object.entries(graph.nodes).map(([n, p]) => (
    //     graphContainer.addChild(
    //         drawNode(p.position, nodeSize, nodeColor)
    //     )
    // ));

    const edgeMeshes = [];

    for (const e in graph.edges) {
        const edge = graph.edges[e];
        const start = graph.nodes[edge.s].position;
        const end = graph.nodes[edge.t].position;
        const edgeMesh = createMeshStrip(start, end, edgeWidth, numSegments);
        graphContainer.addChild(edgeMesh.mesh);
        edgeMeshes.push(edgeMesh);
    }

    const nodeGraphics = [];

    for (const n in graph.nodes) {
        const node = drawNode(graph.nodes[n].position, nodeSize, nodeColor);
        nodeGraphics.push(node);
        graphContainer.addChild(node)
    }
        
    graphContainer.scale.set(uniformScale);
    // // console.log(nodePositions[nodes[0]]);
    // const edgeSegments = cyEdgesAll.map((p, i) => 
    //     drawEdge(graphContainer, nodePositions[p.s], nodePositions[p.t], edgeWidth, edgeColor, nz)
    // ).flat();
    return { graphContainer, edgeMeshes, nodeGraphics };
}

const sliderX = 60;

function getSliderWidth(app) {
  return Math.min(500, app.screen.width * 0.4);
}

export function drawInjectionControl(app, callback) {
    // Make sure stage covers the whole scene
    app.stage.hitArea = app.screen;

    // Make the slider
    const sliderWidth = getSliderWidth(app);
    const slider = new PIXI.Graphics().rect(0, 0, sliderWidth, 4).fill({ color: 0x272d37 });
    slider.eventMode = 'static';
    slider.x = sliderX;
    slider.y = 80;

    // Draw the handle
    const handle_core = new PIXI.Graphics().circle(0, 0, 8).fill({ color: 0xffffff });
    handle_core.y = 0;
    handle_core.x = 0;
    handle_core.eventMode = 'static';
    handle_core.cursor = 'pointer';

    // Draw the handle
    const handle = new PIXI.Graphics().circle(0, 0, 24).fill({ color: [0,0,0,0] });
    handle.y = slider.height / 2;
    handle.x = sliderWidth / 2;
    handle.eventMode = 'static';
    handle.addChild(handle_core);
    app.stage.addChild(slider);
    slider.addChild(handle);
    
    handle.on('pointerdown', onDragStart).on('pointerup', onDragEnd).on('pointerupoutside', onDragEnd);
    slider.on('pointerdown', onDragStart).on('pointerup', onDragEnd).on('pointerupoutside', onDragEnd);

    const animate = (delta) => {
        handle.x = (Math.sin(delta.lastTime / 1000) * 0.5 + 0.5) * sliderWidth;
        const t = (handle.x / sliderWidth);
        callback(t);
    };
    // app.ticker.add(animate);

    // Listen to pointermove on stage once handle is pressed.
    function onDragStart() {
        app.stage.eventMode = 'static';
        app.stage.addEventListener('pointermove', onDrag);
        // app.ticker.remove(animate);
    }

    // Stop dragging feedback once the handle is released.
    function onDragEnd() {
        app.stage.eventMode = 'auto';
        app.stage.removeEventListener('pointermove', onDrag);
        // app.ticker.add(animate);
    }

    // Update the handle's position & bunny's scale when the handle is moved.
    function onDrag(e) {
        const halfHandleWidth = handle_core.width / 2;
        // Set handle y-position to match pointer, clamped to (4, screen.height - 4).
        handle.x = Math.max(halfHandleWidth, Math.min(slider.toLocal(e.global).x, sliderWidth - halfHandleWidth));
        // Normalize handle position between -1 and 1.
        const t = (handle.x / sliderWidth);
        
        callback(t);
    }

    return { handle, slider };
}

export async function drawWDSSelectionControl(app, image, offset, callback) {
    // Make sure stage covers the whole scene
    app.stage.hitArea = app.screen;
    const texture = await PIXI.Assets.load(image);
    const button = new PIXI.Sprite(texture);

    // const buttonWidth = 160;
    // const buttonHeight = 100;
    // const rt = PIXI.RenderTexture.create({
    //     width: buttonWidth,
    //     height: buttonHeight,
    //     scaleMode: PIXI.SCALE_MODES.LINEAR,
    //     resolution: 1,
    //   });
      
    // const button = new PIXI.Sprite(rt);

    // Make the slider
    // const button = new PIXI.Graphics().rect(0, 0, buttonWidth, 4).fill({ color: 0x272d37 });
    button.eventMode = 'static';
    button.cursor = 'pointer';
    const scale = button.height / button.width;
    button.width = Math.min(160, app.screen.width * 0.2)
    button.height = button.width * scale
    button.x = getSliderWidth(app) + sliderX + 0.7 * button.width;// app.screen.width * 0.85;
    button.x = button.x + offset * button.width * 1.2;
    button.y = 80;
    button.anchor.set(0.5);
    // button.scale.set(0.08);
    const areaWidth = button.width + 10;
    const areaHeight = button.height + 10;
    const buttonArea = new PIXI.Graphics();
    buttonArea.beginFill(0x9898bd).drawRoundedRect(
      0, 0, areaWidth, areaHeight, 10
    ).endFill();
    buttonArea.x = button.x - areaWidth/2;
    buttonArea.y = button.y - areaHeight/2;

    button.on('pointerdown', callback)

    app.stage.addChild(buttonArea);
    app.stage.addChild(button);

    return;
}