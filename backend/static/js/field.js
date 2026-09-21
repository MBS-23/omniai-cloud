/* ==========================================================================
   OmniAI Cloud — 3D field
   --------------------------------------------------------------------------
   A rotating point cloud with depth, drawn in raw WebGL. Points are arranged
   in three concentric shells and drift; the ones that come close enough get
   joined by lines, which reads as a routing graph rather than a starfield.

   Written against the WebGL API directly rather than with Three.js: this is
   about 6 KB and needs no dependency, where Three would add roughly 170 KB
   gzipped for one decorative canvas.

   It disables itself when: WebGL is unavailable, the viewport is narrow,
   prefers-reduced-motion is set, or the tab is hidden.
   ========================================================================== */
(() => {
  'use strict';

  const canvas = document.getElementById('field');
  if (!canvas) return;

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  if (reduceMotion.matches || window.innerWidth < 640) return;

  const gl =
    canvas.getContext('webgl', { antialias: true, alpha: true, depth: false }) ||
    canvas.getContext('experimental-webgl', { antialias: true, alpha: true, depth: false });
  if (!gl) return;

  /* ------------------------------------------------------------ shaders */

  const VERTEX_SRC = `
    attribute vec3 aPos;
    attribute float aSeed;
    uniform mat4 uRot;
    uniform float uTime;
    uniform float uAspect;
    uniform vec2 uPointer;
    varying float vDepth;

    void main() {
      vec3 p = aPos;

      // Each point breathes on its own phase so the cloud never pulses as one.
      float wob = sin(uTime * 0.5 + aSeed * 6.2831) * 0.05;
      p *= 1.0 + wob;

      vec4 rotated = uRot * vec4(p, 1.0);

      // Pointer parallax, scaled by depth so near points move further.
      rotated.xy += uPointer * (0.12 + rotated.z * 0.06);

      // Perspective divide. The camera sits at z = 3.
      float z = rotated.z + 3.0;
      vDepth = clamp((rotated.z + 1.6) / 3.2, 0.0, 1.0);

      vec2 projected = rotated.xy / z;
      projected.x /= uAspect;

      gl_Position = vec4(projected * 1.5, 0.0, 1.0);
      gl_PointSize = (1.0 + vDepth * 3.4) * (1.0 + wob * 2.0);
    }
  `;

  const FRAGMENT_SRC = `
    precision mediump float;
    uniform vec3 uNear;
    uniform vec3 uFar;
    uniform float uAlpha;
    varying float vDepth;

    void main() {
      // Round out the square point sprite and soften its edge.
      vec2 d = gl_PointCoord - vec2(0.5);
      float r = length(d);
      if (r > 0.5) discard;
      float edge = smoothstep(0.5, 0.18, r);

      vec3 colour = mix(uFar, uNear, vDepth);
      gl_FragColor = vec4(colour, edge * uAlpha * (0.25 + vDepth * 0.75));
    }
  `;

  const LINE_VERTEX_SRC = `
    attribute vec3 aPos;
    uniform mat4 uRot;
    uniform float uAspect;
    uniform vec2 uPointer;
    varying float vDepth;

    void main() {
      vec4 rotated = uRot * vec4(aPos, 1.0);
      rotated.xy += uPointer * (0.12 + rotated.z * 0.06);
      float z = rotated.z + 3.0;
      vDepth = clamp((rotated.z + 1.6) / 3.2, 0.0, 1.0);
      vec2 projected = rotated.xy / z;
      projected.x /= uAspect;
      gl_Position = vec4(projected * 1.5, 0.0, 1.0);
    }
  `;

  const LINE_FRAGMENT_SRC = `
    precision mediump float;
    uniform vec3 uNear;
    uniform float uAlpha;
    varying float vDepth;
    void main() {
      gl_FragColor = vec4(uNear, vDepth * vDepth * uAlpha);
    }
  `;

  function compile(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.warn('field: shader failed', gl.getShaderInfoLog(shader));
      return null;
    }
    return shader;
  }

  function link(vertexSrc, fragmentSrc) {
    const vertex = compile(gl.VERTEX_SHADER, vertexSrc);
    const fragment = compile(gl.FRAGMENT_SHADER, fragmentSrc);
    if (!vertex || !fragment) return null;

    const program = gl.createProgram();
    gl.attachShader(program, vertex);
    gl.attachShader(program, fragment);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.warn('field: link failed', gl.getProgramInfoLog(program));
      return null;
    }
    return program;
  }

  const pointProgram = link(VERTEX_SRC, FRAGMENT_SRC);
  const lineProgram = link(LINE_VERTEX_SRC, LINE_FRAGMENT_SRC);
  if (!pointProgram || !lineProgram) return;

  /* ---------------------------------------------------------- geometry */

  const COUNT = 320;
  const positions = new Float32Array(COUNT * 3);
  const seeds = new Float32Array(COUNT);

  // Three shells, so the cloud has visible structure at different depths
  // instead of reading as uniform noise.
  for (let i = 0; i < COUNT; i += 1) {
    const shell = i % 3;
    const radius = 0.55 + shell * 0.34 + Math.random() * 0.12;

    // Even distribution over a sphere.
    const theta = Math.acos(2 * Math.random() - 1);
    const phi = Math.random() * Math.PI * 2;

    positions[i * 3] = radius * Math.sin(theta) * Math.cos(phi);
    positions[i * 3 + 1] = radius * Math.sin(theta) * Math.sin(phi) * 0.7;
    positions[i * 3 + 2] = radius * Math.cos(theta);
    seeds[i] = Math.random();
  }

  // Precompute the pairs close enough to connect. Doing this once keeps the
  // frame loop free of an O(n^2) scan.
  const LINK_DISTANCE = 0.42;
  const MAX_LINKS = 260;
  const linkPositions = [];

  outer:
  for (let i = 0; i < COUNT; i += 1) {
    for (let j = i + 1; j < COUNT; j += 1) {
      const dx = positions[i * 3] - positions[j * 3];
      const dy = positions[i * 3 + 1] - positions[j * 3 + 1];
      const dz = positions[i * 3 + 2] - positions[j * 3 + 2];
      if (dx * dx + dy * dy + dz * dz < LINK_DISTANCE * LINK_DISTANCE) {
        linkPositions.push(
          positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2],
          positions[j * 3], positions[j * 3 + 1], positions[j * 3 + 2],
        );
        if (linkPositions.length / 6 >= MAX_LINKS) break outer;
      }
    }
  }

  const linkArray = new Float32Array(linkPositions);
  const linkCount = linkArray.length / 3;

  function buffer(data) {
    const handle = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, handle);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    return handle;
  }

  const positionBuffer = buffer(positions);
  const seedBuffer = buffer(seeds);
  const linkBuffer = buffer(linkArray);

  /* ----------------------------------------------------------- uniforms */

  const uniforms = (program) => ({
    rot: gl.getUniformLocation(program, 'uRot'),
    time: gl.getUniformLocation(program, 'uTime'),
    aspect: gl.getUniformLocation(program, 'uAspect'),
    pointer: gl.getUniformLocation(program, 'uPointer'),
    near: gl.getUniformLocation(program, 'uNear'),
    far: gl.getUniformLocation(program, 'uFar'),
    alpha: gl.getUniformLocation(program, 'uAlpha'),
  });

  const pointUniforms = uniforms(pointProgram);
  const lineUniforms = uniforms(lineProgram);
  const aPosPoint = gl.getAttribLocation(pointProgram, 'aPos');
  const aSeed = gl.getAttribLocation(pointProgram, 'aSeed');
  const aPosLine = gl.getAttribLocation(lineProgram, 'aPos');

  /* Colour follows the theme, read from the CSS custom properties so there
     is one source of truth for the palette. */
  let nearColour = [1.0, 0.42, 0.24];
  let farColour = [0.35, 0.42, 0.55];
  let alpha = 0.85;

  function hexToRgb(value) {
    const match = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec((value || '').trim());
    if (!match) return null;
    return [
      parseInt(match[1], 16) / 255,
      parseInt(match[2], 16) / 255,
      parseInt(match[3], 16) / 255,
    ];
  }

  function readTheme() {
    const style = getComputedStyle(document.documentElement);
    nearColour = hexToRgb(style.getPropertyValue('--accent')) || nearColour;
    farColour = hexToRgb(style.getPropertyValue('--text-4')) || farColour;
    // The light theme needs a fainter field or it fights the text.
    alpha = document.documentElement.dataset.theme === 'light' ? 0.5 : 0.85;
  }
  readTheme();
  window.addEventListener('omniai:themechange', readTheme);

  /* --------------------------------------------------------- resize */

  let width = 0;
  let height = 0;

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    width = Math.max(1, Math.round(rect.width * dpr));
    height = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    gl.viewport(0, 0, width, height);
  }

  let resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(resize, 120);
  });

  /* -------------------------------------------------------- pointer */

  const pointer = { x: 0, y: 0, tx: 0, ty: 0 };

  window.addEventListener('pointermove', (event) => {
    pointer.tx = (event.clientX / window.innerWidth - 0.5) * 2;
    pointer.ty = -(event.clientY / window.innerHeight - 0.5) * 2;
  }, { passive: true });

  /* ------------------------------------------------------------ loop */

  // Column-major rotation about Y then X, built by hand to avoid a matrix
  // library for two rotations.
  function rotationMatrix(yaw, pitch, out) {
    const cy = Math.cos(yaw);
    const sy = Math.sin(yaw);
    const cx = Math.cos(pitch);
    const sx = Math.sin(pitch);

    out[0] = cy;       out[1] = sy * sx;  out[2] = -sy * cx; out[3] = 0;
    out[4] = 0;        out[5] = cx;       out[6] = sx;       out[7] = 0;
    out[8] = sy;       out[9] = -cy * sx; out[10] = cy * cx; out[11] = 0;
    out[12] = 0;       out[13] = 0;       out[14] = 0;       out[15] = 1;
    return out;
  }

  const matrix = new Float32Array(16);
  let start = performance.now();
  let running = true;
  let frame = null;

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

  function draw(now) {
    frame = requestAnimationFrame(draw);
    if (!running) return;

    const time = (now - start) / 1000;

    // Ease the pointer so motion is inertial rather than jittery.
    pointer.x += (pointer.tx - pointer.x) * 0.045;
    pointer.y += (pointer.ty - pointer.y) * 0.045;

    rotationMatrix(time * 0.13 + pointer.x * 0.28, Math.sin(time * 0.09) * 0.22 + pointer.y * 0.2, matrix);

    const aspect = width / Math.max(1, height);

    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Links first, so points sit on top of them.
    gl.useProgram(lineProgram);
    gl.uniformMatrix4fv(lineUniforms.rot, false, matrix);
    gl.uniform1f(lineUniforms.aspect, aspect);
    gl.uniform2f(lineUniforms.pointer, pointer.x, pointer.y);
    gl.uniform3fv(lineUniforms.near, nearColour);
    gl.uniform1f(lineUniforms.alpha, alpha * 0.16);

    gl.bindBuffer(gl.ARRAY_BUFFER, linkBuffer);
    gl.enableVertexAttribArray(aPosLine);
    gl.vertexAttribPointer(aPosLine, 3, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.LINES, 0, linkCount);

    gl.useProgram(pointProgram);
    gl.uniformMatrix4fv(pointUniforms.rot, false, matrix);
    gl.uniform1f(pointUniforms.time, time);
    gl.uniform1f(pointUniforms.aspect, aspect);
    gl.uniform2f(pointUniforms.pointer, pointer.x, pointer.y);
    gl.uniform3fv(pointUniforms.near, nearColour);
    gl.uniform3fv(pointUniforms.far, farColour);
    gl.uniform1f(pointUniforms.alpha, alpha);

    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.enableVertexAttribArray(aPosPoint);
    gl.vertexAttribPointer(aPosPoint, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, seedBuffer);
    gl.enableVertexAttribArray(aSeed);
    gl.vertexAttribPointer(aSeed, 1, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.POINTS, 0, COUNT);
  }

  // Stop rendering when the tab is hidden: there is no reason to hold a GPU
  // at 60 fps for a canvas nobody is looking at.
  document.addEventListener('visibilitychange', () => {
    running = !document.hidden;
    if (running) start = performance.now() - 1000;
  });

  // Stop when the header scrolls away, for the same reason.
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(
      ([entry]) => { running = entry.isIntersecting && !document.hidden; },
      { threshold: 0 }
    );
    observer.observe(canvas.parentElement || canvas);
  }

  reduceMotion.addEventListener?.('change', (event) => {
    if (event.matches && frame) {
      cancelAnimationFrame(frame);
      canvas.classList.remove('is-ready');
    }
  });

  resize();
  canvas.classList.add('is-ready');
  frame = requestAnimationFrame(draw);
})();
