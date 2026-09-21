# Design system

## Direction

**Dark by default, with a light theme on a toggle.** Both are designed rather
than derived: dark is five layered surfaces with real elevation, not black plus
white plus neon, and light is a warm paper palette, not the dark one inverted.

The register is a *technical instrument*: depth, motion, a live 3D field behind
the header, and a pipeline diagram that reflects real state — over a structure
that is still a grid.

Tokens live on `:root` in `backend/static/css/omniai.css`. No component uses a
raw colour value.

## Colour

| Token | Dark | Light | Use |
|---|---|---|---|
| `--bg` | `#07080c` | `#f7f6f3` | Page ground |
| `--surface` | `#0d1017` | `#ffffff` | Panels |
| `--surface-2` | `#131824` | `#fbfaf8` | Raised surfaces, inputs |
| `--surface-3` | `#1a2030` | `#f3f1ed` | Tracks, code blocks |
| `--border` | `#222a3a` | `#e2e0da` | Hairlines |
| `--border-2` | `#2f3a4f` | `#cdcac2` | Field borders |
| `--text` | `#eef1f6` | `#0d0e12` | Body |
| `--text-2` | `#b6c0d0` | `#3f4451` | Supporting |
| `--text-3` | `#7e8b9f` | `#6c7382` | Labels, metadata |
| `--text-4` | `#566174` | `#9aa0ac` | Placeholders only |
| `--accent` | `#ff6a3d` | `#d64a20` | The one accent |
| `--accent-2` | `#ffa02b` | `#b8690a` | Gradient partner |

Semantic: `--ok`, `--warn`, `--danger`, `--info`, each with a `-soft` wash.
Every status carries an icon and a text label as well as a colour, so nothing
depends on hue alone.

One warm accent, deliberately not the default tech blue or violet. It appears
on section numbers, the primary button, the progress line, active states and
one gradient word in the headline — and nowhere else.

`tests/test_assets.py` asserts both themes define the same colour tokens; a
token defined in only one theme would silently fall back and produce
unreadable text.

## Type

System stacks: no network request, no CSP exception for a font host, no layout
shift. `--font-display` is an old-style serif for headlines and figures,
`--font-body` a grotesque for interface copy, `--font-mono` for anything
numeric so columns of figures align.

Scale is fluid via `clamp()`: `--text-hero` runs 2.75rem → 5.5rem. Body is
16px at `--leading-body: 1.6`, prose capped at `--measure: 66ch`.

To swap in web fonts: add the link to `index.html`, add the font host to
`style-src` and `font-src` in `_register_security_headers` (`backend/app.py`),
and change the two font tokens. Nothing else changes.

## Motion

Durations `--dur-1` 140ms through `--dur-4` 900ms, on `--ease` for normal
transitions and `--ease-spring` for anything that should feel physical.

| Effect | Mechanism | Where |
|---|---|---|
| Scroll reveals | IntersectionObserver + CSS transition | Sections, cards |
| Scroll progress | rAF-throttled scroll handler | Top edge |
| Counting figures | requestAnimationFrame, ease-out cubic | Hero and result figures |
| Confidence bars | Width transition, triggered on view | Result panels |
| Pointer tilt | CSS custom properties from pointermove | Figures |
| Follow light | Radial gradient tracking the cursor | Dropzone |
| Tab indicator | Transform on a spring easing | Input tabs |
| Live flow | `stroke-dashoffset` driven by pipeline percent | Pipeline diagram |
| 3D field | Raw WebGL point cloud | Behind the header |

No animation library. IntersectionObserver, rAF and CSS transitions cover all
of the above in a few hundred bytes, against roughly 70 KB for GSAP plus
ScrollTrigger. If the interface later needs timeline sequencing or scrubbing,
that is the point to reconsider — not before.

`prefers-reduced-motion: reduce` collapses every duration to 0.01ms, reveals
content immediately, and removes the 3D field and the tilt entirely.

## The 3D field

`backend/static/js/field.js` — a rotating point cloud in raw WebGL, about 6 KB.
Three concentric shells of 320 points with perspective projection, depth-scaled
size and opacity, precomputed link lines, and inertial pointer parallax.

Three.js would have cost roughly 170 KB gzipped for one decorative canvas, so
the shaders are written directly.

It disables itself when WebGL is unavailable, the viewport is under 640px,
`prefers-reduced-motion` is set, the tab is hidden, or the header has scrolled
out of view. The last two stop it holding a GPU at 60fps for a canvas nobody is
looking at.

**Its class is `.backdrop__field`, never `.field`.** `.field` is the form-field
wrapper; an earlier version shared the name, and the two rules that hide the
canvas — under 640px and under reduced motion — therefore hid every text input
and select on the page. `tests/test_assets.py` now fails if any rule sets
`display: none` on `.field`.

## Layout

12-column grid, `--shell-max: 1380px`, `--space-5` gutters. Main content on
columns 1–8, supporting column 9–12; 7/5 at 1080px; full width at 860px.

Breakpoints 1080px, 860px, 640px. Mobile is designed, not inherited: tabs go
full width and split evenly, definition rows stack, stage details move below
their label, buttons go full width, and the 3D field and flow diagram are
dropped for ones that read at that size.

## Accessibility

- One focus treatment (`:focus-visible`, an accent ring on a background halo),
  global and never removed. A test asserts no rule drops `outline` without
  providing a replacement.
- Skip link as the first tab stop.
- Interactive targets at least 44px.
- Tabs implement the ARIA tabs pattern including arrow-key navigation.
- The progress meter is a `role="progressbar"` with live `aria-valuenow`.
- Errors are `role="alert"`, status messages `role="status"`.
- Decorative SVG is `aria-hidden`; the backdrop and flow diagram are
  `aria-hidden` in full, since the stage list carries the same information.
- Model output reaches the DOM only via `textContent`.

## Before changing the interface

1. Raw colour value? Use a token, and define it in both themes.
2. New class name? Check it does not collide with an existing one — that
   specific mistake made the whole form invisible once.
3. New state? It needs loading, empty, error and disabled treatments.
4. Still readable at 375px?
5. Still usable under `prefers-reduced-motion: reduce`?
6. Does any untrusted string reach the DOM as markup? It must not.
7. Run `python -m pytest tests/test_assets.py`.
