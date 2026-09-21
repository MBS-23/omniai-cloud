/* ==========================================================================
   OmniAI Cloud — interface motion and theming
   --------------------------------------------------------------------------
   Theme switching, scroll progress, scroll-triggered reveals, counting
   figures, pointer tilt, the sliding tab indicator, and the masthead's
   stuck state.

   No animation library. IntersectionObserver covers reveals, requestAnimation-
   Frame covers counters, and CSS transitions cover the rest, which is a few
   hundred bytes against roughly 70 KB for GSAP plus ScrollTrigger. If the
   project later needs timeline sequencing or scrubbing, that is the point to
   reconsider.

   Everything here is decorative. The page is fully usable with this file
   blocked: content is visible, and only the entrance transitions are lost.
   ========================================================================== */
(() => {
  'use strict';

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const root = document.documentElement;

  /* ------------------------------------------------------------- theme */

  const THEME_KEY = 'omniai.theme';

  function applyTheme(theme, { persist = true } = {}) {
    root.dataset.theme = theme;
    if (persist) {
      try { localStorage.setItem(THEME_KEY, theme); } catch { /* private mode */ }
    }
    const toggle = document.getElementById('themeToggle');
    if (toggle) {
      const next = theme === 'dark' ? 'light' : 'dark';
      toggle.setAttribute('aria-label', `Switch to ${next} theme`);
      toggle.setAttribute('title', `Switch to ${next} theme`);
    }
    // The 3D field re-reads its palette from CSS on this event.
    window.dispatchEvent(new CustomEvent('omniai:themechange', { detail: { theme } }));
  }

  function initTheme() {
    let stored = null;
    try { stored = localStorage.getItem(THEME_KEY); } catch { /* private mode */ }

    // Dark is the product default, not a system-preference mirror: the
    // interface is designed dark first and the light theme is the variant.
    // Only an explicit choice from the toggle overrides it.
    applyTheme(stored === 'light' ? 'light' : 'dark', { persist: false });

    const toggle = document.getElementById('themeToggle');
    if (!toggle) return;
    toggle.addEventListener('click', () => {
      applyTheme(root.dataset.theme === 'dark' ? 'light' : 'dark');
    });
  }

  /* --------------------------------------------------- scroll progress */

  function initScrollProgress() {
    const bar = document.getElementById('scrollProgress');
    if (!bar) return;

    let ticking = false;

    const update = () => {
      const scrollable = document.documentElement.scrollHeight - window.innerHeight;
      const ratio = scrollable > 0 ? window.scrollY / scrollable : 0;
      bar.style.transform = `scaleX(${Math.min(1, Math.max(0, ratio))})`;
      ticking = false;
    };

    window.addEventListener('scroll', () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(update);
    }, { passive: true });

    update();
  }

  /* ------------------------------------------------------ masthead state */

  function initMasthead() {
    const masthead = document.querySelector('.masthead');
    if (!masthead) return;

    // A zero-height sentinel above the masthead: when it leaves the viewport
    // the header is stuck. Cheaper and steadier than reading scrollY.
    const sentinel = document.createElement('div');
    sentinel.setAttribute('aria-hidden', 'true');
    masthead.parentNode.insertBefore(sentinel, masthead);

    if (!('IntersectionObserver' in window)) return;
    new IntersectionObserver(
      ([entry]) => masthead.classList.toggle('is-stuck', !entry.isIntersecting),
      { threshold: 0 }
    ).observe(sentinel);
  }

  /* ------------------------------------------------------------ reveals */

  let revealObserver = null;

  function revealNow(element) {
    element.classList.add('is-visible');
  }

  function observeReveals(scope = document) {
    const targets = scope.querySelectorAll('.reveal:not(.is-visible)');
    if (!targets.length) return;

    if (reduceMotion.matches || !('IntersectionObserver' in window)) {
      targets.forEach(revealNow);
      return;
    }

    if (!revealObserver) {
      revealObserver = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          revealNow(entry.target);
          revealObserver.unobserve(entry.target);
        });
      }, { rootMargin: '0px 0px -8% 0px', threshold: 0.08 });
    }

    targets.forEach((target) => revealObserver.observe(target));
  }

  /* ----------------------------------------------------------- counters */

  const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);

  /**
   * Count an element's number up from zero.
   * Falls back to the final value immediately under reduced motion.
   */
  function countUp(element, target, { duration = 900, decimals = 0, suffix = '' } = {}) {
    const end = Number(target);
    if (!Number.isFinite(end)) {
      element.textContent = String(target);
      return;
    }

    const render = (value) => {
      element.textContent = value.toFixed(decimals) + suffix;
    };

    if (reduceMotion.matches) {
      render(end);
      return;
    }

    const started = performance.now();

    const step = (now) => {
      const progress = Math.min(1, (now - started) / duration);
      render(end * easeOutCubic(progress));
      if (progress < 1) requestAnimationFrame(step);
    };

    render(0);
    requestAnimationFrame(step);
  }

  /** Fill a `.bar__fill` to a percentage once it scrolls into view. */
  function animateBar(fill, percent) {
    const width = `${Math.max(0, Math.min(100, Number(percent) || 0))}%`;

    if (reduceMotion.matches || !('IntersectionObserver' in window)) {
      fill.style.width = width;
      return;
    }

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        // Next frame, so the transition has a 0% start to animate from.
        requestAnimationFrame(() => { fill.style.width = width; });
        observer.unobserve(entry.target);
      });
    }, { threshold: 0.2 });

    observer.observe(fill);
  }

  /* --------------------------------------------------------------- tilt */

  const MAX_TILT = 5; // degrees; beyond this it reads as a gimmick

  function attachTilt(element) {
    if (reduceMotion.matches) return;
    if (window.matchMedia('(pointer: coarse)').matches) return; // no hover on touch

    const onMove = (event) => {
      const rect = element.getBoundingClientRect();
      const x = (event.clientX - rect.left) / rect.width - 0.5;
      const y = (event.clientY - rect.top) / rect.height - 0.5;
      element.style.setProperty('--ry', `${x * MAX_TILT * 2}deg`);
      element.style.setProperty('--rx', `${-y * MAX_TILT * 2}deg`);
    };

    const reset = () => {
      element.style.setProperty('--rx', '0deg');
      element.style.setProperty('--ry', '0deg');
    };

    element.addEventListener('pointermove', onMove);
    element.addEventListener('pointerleave', reset);
  }

  /** Track the pointer as CSS variables, for the dropzone's follow light. */
  function attachPointerLight(element) {
    if (reduceMotion.matches) return;
    element.addEventListener('pointermove', (event) => {
      const rect = element.getBoundingClientRect();
      element.style.setProperty('--mx', `${event.clientX - rect.left}px`);
      element.style.setProperty('--my', `${event.clientY - rect.top}px`);
    });
  }

  /* -------------------------------------------------- tab indicator */

  function initTabIndicator() {
    const tabs = document.querySelector('.tabs');
    if (!tabs) return;

    const indicator = tabs.querySelector('.tabs__indicator');
    const buttons = Array.from(tabs.querySelectorAll('.tabs__btn'));
    if (!indicator || !buttons.length) return;

    const move = () => {
      const active = buttons.find((b) => b.getAttribute('aria-selected') === 'true') || buttons[0];
      indicator.style.width = `${active.offsetWidth}px`;
      indicator.style.transform = `translateX(${active.offsetLeft - tabs.clientLeft}px)`;
    };

    buttons.forEach((button) => button.addEventListener('click', () => requestAnimationFrame(move)));
    window.addEventListener('resize', move);

    // Fonts can land after first paint and change the measurement.
    if (document.fonts?.ready) document.fonts.ready.then(move);
    requestAnimationFrame(move);
  }

  /* ----------------------------------------------------- keyboard tabs */

  /** Left/Right arrow navigation between tabs, per the ARIA tabs pattern. */
  function initTabKeys() {
    const list = document.querySelector('[role="tablist"]');
    if (!list) return;
    const buttons = Array.from(list.querySelectorAll('[role="tab"]'));

    list.addEventListener('keydown', (event) => {
      const index = buttons.indexOf(document.activeElement);
      if (index === -1) return;

      let next = null;
      if (event.key === 'ArrowRight') next = (index + 1) % buttons.length;
      else if (event.key === 'ArrowLeft') next = (index - 1 + buttons.length) % buttons.length;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = buttons.length - 1;
      if (next === null) return;

      event.preventDefault();
      buttons[next].focus();
      buttons[next].click();
    });
  }

  /* ------------------------------------------------------------- expose */

  window.OmniAIMotion = { observeReveals, countUp, animateBar, attachTilt, reduceMotion };

  function init() {
    initTheme();
    initScrollProgress();
    initMasthead();
    initTabIndicator();
    initTabKeys();
    observeReveals();

    document.querySelectorAll('[data-tilt]').forEach(attachTilt);
    document.querySelectorAll('[data-pointer-light]').forEach(attachPointerLight);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
