/* ==========================================================================
   OmniAI Cloud — pointer interaction
   --------------------------------------------------------------------------
   An orange glow that trails the cursor, a dot that rides exactly on it, and
   buttons that lean slightly toward the pointer with a ripple on click.

   All of it runs from a single requestAnimationFrame loop that writes CSS
   custom properties. Nothing here reads layout during the loop, so the
   browser composites a transform instead of repainting a gradient every frame.

   Disabled entirely on touch devices and under prefers-reduced-motion.
   ========================================================================== */
(() => {
  'use strict';

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const coarse = window.matchMedia('(pointer: coarse)');
  if (reduceMotion.matches || coarse.matches || window.innerWidth < 640) return;

  const glow = document.getElementById('cursorGlow');
  const dot = document.getElementById('cursorDot');
  if (!glow || !dot) return;

  // Target is where the pointer is; eased is where the glow currently is.
  const pointer = { x: innerWidth / 2, y: innerHeight / 2 };
  const eased = { x: pointer.x, y: pointer.y };
  let live = false;
  let frame = null;

  const GLOW_EASE = 0.12;   // the glow lags, which reads as weight
  const DOT_EASE = 0.42;    // the dot is nearly immediate

  const dotPos = { x: pointer.x, y: pointer.y };

  function onMove(event) {
    pointer.x = event.clientX;
    pointer.y = event.clientY;

    if (!live) {
      live = true;
      // Start both at the pointer so nothing flies in from the corner.
      eased.x = dotPos.x = pointer.x;
      eased.y = dotPos.y = pointer.y;
      glow.classList.add('is-live');
      dot.classList.add('is-live');
    }
  }

  function tick() {
    frame = requestAnimationFrame(tick);
    if (!live) return;

    eased.x += (pointer.x - eased.x) * GLOW_EASE;
    eased.y += (pointer.y - eased.y) * GLOW_EASE;
    dotPos.x += (pointer.x - dotPos.x) * DOT_EASE;
    dotPos.y += (pointer.y - dotPos.y) * DOT_EASE;

    glow.style.setProperty('--cx', `${eased.x}px`);
    glow.style.setProperty('--cy', `${eased.y}px`);
    dot.style.setProperty('--dx', `${dotPos.x}px`);
    dot.style.setProperty('--dy', `${dotPos.y}px`);
  }

  /* --- Interactive elements make the glow swell ---------------------- */
  const INTERACTIVE = 'a, button, .dropzone, .panel__head, input, select, textarea, .chip';

  document.addEventListener('pointerover', (event) => {
    if (event.target.closest(INTERACTIVE)) {
      glow.classList.add('is-hot');
      dot.classList.add('is-hot');
    }
  });
  document.addEventListener('pointerout', (event) => {
    if (event.target.closest(INTERACTIVE)) {
      glow.classList.remove('is-hot');
      dot.classList.remove('is-hot');
    }
  });

  /* --- Buttons lean toward the pointer -------------------------------- */
  const MAX_PULL = 3.5;   // px; beyond this it stops looking deliberate

  function attachButton(button) {
    button.addEventListener('pointermove', (event) => {
      const rect = button.getBoundingClientRect();
      const relX = (event.clientX - rect.left) / rect.width;
      const relY = (event.clientY - rect.top) / rect.height;

      // Local highlight position, and a small pull toward the cursor.
      button.style.setProperty('--px', `${relX * 100}%`);
      button.style.setProperty('--py', `${relY * 100}%`);
      button.style.setProperty('--bx', `${(relX - 0.5) * 2 * MAX_PULL}px`);
      button.style.setProperty('--by', `${(relY - 0.5) * 2 * MAX_PULL}px`);
    });

    button.addEventListener('pointerleave', () => {
      button.style.setProperty('--bx', '0px');
      button.style.setProperty('--by', '0px');
    });

    button.addEventListener('click', (event) => {
      const rect = button.getBoundingClientRect();
      const size = Math.max(rect.width, rect.height);
      const ripple = document.createElement('span');
      ripple.className = 'ripple';
      ripple.style.width = ripple.style.height = `${size}px`;
      ripple.style.left = `${event.clientX - rect.left - size / 2}px`;
      ripple.style.top = `${event.clientY - rect.top - size / 2}px`;
      button.appendChild(ripple);
      // Matches the animation duration in the stylesheet.
      setTimeout(() => ripple.remove(), 600);
    });
  }

  function attachAll(scope = document) {
    scope.querySelectorAll('.btn').forEach((button) => {
      if (button.dataset.pointerBound) return;
      button.dataset.pointerBound = '1';
      attachButton(button);
    });
  }

  /* --- Lifecycle ------------------------------------------------------ */
  window.addEventListener('pointermove', onMove, { passive: true });

  // Buttons are rendered into the results area after an analysis, so watch
  // for them rather than binding once at load.
  if ('MutationObserver' in window) {
    new MutationObserver(() => attachAll()).observe(document.body, {
      childList: true,
      subtree: true,
    });
  }

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      cancelAnimationFrame(frame);
      frame = null;
    } else if (frame === null) {
      frame = requestAnimationFrame(tick);
    }
  });

  reduceMotion.addEventListener?.('change', (event) => {
    if (!event.matches) return;
    cancelAnimationFrame(frame);
    glow.classList.remove('is-live');
    dot.classList.remove('is-live');
  });

  attachAll();
  frame = requestAnimationFrame(tick);
})();
