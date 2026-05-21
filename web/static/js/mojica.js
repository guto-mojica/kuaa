// web/static/js/mojica.js
// Mojica · vanilla JS polish layer for v0.3 chrome.
//
// Phase 6 (Task 25): SSE log auto-scroll on the Processing tab.
//   The Processing tab streams pipeline log lines into
//   `#proc-log .lines` via htmx-sse. We want the viewport pinned to the
//   bottom while events arrive, BUT the moment the user scrolls up to
//   inspect a previous line, auto-scroll should suspend until they
//   scroll back to the bottom (or until they re-tick the
//   [data-autoscroll] checkbox — which is checked by default).
//
// Future phases extend this file (Phase 7: toast bus, command palette,
// keyboard help). No external dependencies; ships as a single IIFE so
// nothing leaks into the global namespace.

(function () {
  'use strict';

  // Tolerance in pixels for "the user is at the bottom". Browsers can
  // produce sub-pixel scrollTop values after a swap; 24px is generous
  // enough to absorb that and the typical line-height of a log row.
  var BOTTOM_TOLERANCE = 24;

  // Per-element state: bound nodes are tagged with `__mojicaBound` so
  // re-running bindLogAutoscroll() after an HTMX swap is idempotent.
  var BOUND_FLAG = '__mojicaLogAutoscrollBound';

  /**
   * Bind log auto-scroll behaviour to `#proc-log` when present.
   *
   * - Scrolls `.lines` to the bottom on SSE-driven swap events.
   * - Suspends auto-scroll once the user scrolls up manually.
   * - Resumes when the user scrolls back to (near) the bottom.
   * - Respects the [data-autoscroll] checkbox (checked by default).
   *
   * Safe to call repeatedly: the per-node BOUND_FLAG short-circuits
   * duplicate listener registration when HTMX re-renders the tab.
   */
  function bindLogAutoscroll() {
    var log = document.getElementById('proc-log');
    if (!log) return;
    var lines = log.querySelector('.lines');
    if (!lines) return;
    if (lines[BOUND_FLAG]) return;
    lines[BOUND_FLAG] = true;

    var autoscrollBox = log.querySelector('[data-autoscroll]');

    // pinned: are we tracking the bottom? Starts true so newly-rendered
    // logs auto-scroll on first paint.
    var pinned = true;

    // Track manual scroll: as soon as the user moves more than
    // BOTTOM_TOLERANCE px above the bottom, un-pin. Re-pin when they
    // scroll back down.
    lines.addEventListener('scroll', function () {
      var distanceFromBottom =
        lines.scrollHeight - lines.scrollTop - lines.clientHeight;
      pinned = distanceFromBottom < BOTTOM_TOLERANCE;
    });

    function scrollToBottomIfPinned() {
      var enabled = autoscrollBox ? autoscrollBox.checked : true;
      if (enabled && pinned) {
        lines.scrollTop = lines.scrollHeight;
      }
    }

    // SSE events fire on the element with hx-ext="sse". htmx dispatches
    // `htmx:sseMessage` synchronously per event; `htmx:afterSettle`
    // fires after the swap has been applied to the DOM. We bind both
    // to be robust against future swap-target changes and to handle
    // non-SSE swaps that land inside `#proc-log` (e.g. a future
    // "clear log" button).
    document.body.addEventListener('htmx:sseMessage', function (evt) {
      if (log.contains(evt.target) || evt.target === log) {
        // Defer to next tick so the swap has actually mutated the DOM.
        setTimeout(scrollToBottomIfPinned, 0);
      }
    });
    document.body.addEventListener('htmx:afterSettle', function (evt) {
      if (log.contains(evt.target) || evt.target === log) {
        scrollToBottomIfPinned();
      }
    });

    // When the user re-ticks the checkbox after un-ticking it, snap
    // to the bottom immediately (mirrors the prototype UX).
    if (autoscrollBox) {
      autoscrollBox.addEventListener('change', function () {
        if (autoscrollBox.checked) {
          pinned = true;
          lines.scrollTop = lines.scrollHeight;
        }
      });
    }

    // Initial scroll on load: server may have rendered seed lines.
    scrollToBottomIfPinned();
  }

  // ── Bootstrap ──────────────────────────────────────────────────────
  // HTMX tab swaps can replace `#proc-log` (or insert it for the first
  // time when navigating to /processing). Rebind on every settle that
  // brings the log into existence; the BOUND_FLAG guard makes repeats
  // cheap.
  function init() {
    bindLogAutoscroll();
    document.body.addEventListener('htmx:afterSettle', function () {
      bindLogAutoscroll();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

// ─── ToastBus (Phase 7 · Task 26) ─────────────────────────────────────
// Global notification bus. Any HTMX response can trigger a toast by
// emitting an `HX-Trigger` header with a "toast" event payload. JS code
// (e.g. future client-side flows) can call `window.ToastBus.push(spec)`
// directly. The bus creates `.toast` elements inside the
// `#toast-root` div (rendered by base.html under .fx-app).
//
// Server contract (FastAPI):
//   response.headers["HX-Trigger"] = json.dumps({"toast": {
//       "title": "Saved",
//       "sub": "optional second line",
//       "kind": "info" | "success" | "warn" | "error",
//       "duration": 3500  // optional, ms; 0 disables auto-dismiss
//   }})
//
// HTMX dispatches a CustomEvent named "toast" with `evt.detail` = the
// payload above; the listener below pipes it straight into `push()`.
//
// The bus is exposed on `window.ToastBus` so devtools / manual flows can
// call it without a server round-trip (`window.ToastBus.push({title:'Hi'})`).
window.ToastBus = (function () {
  'use strict';

  // Default auto-dismiss delay (ms). Matches the prototype's 3500ms.
  var DEFAULT_DURATION = 3500;
  // How long the exit animation runs before the element is removed.
  // Keep in sync with `@keyframes p-toast-out` in polish.css (200ms).
  var EXIT_MS = 200;

  function rootEl() {
    return document.getElementById('toast-root');
  }

  function makeId() {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
      return crypto.randomUUID();
    }
    return 't-' + Date.now() + '-' + Math.random().toString(36).slice(2, 8);
  }

  /**
   * Push a new toast onto the host. Returns the toast id so callers can
   * remove it manually (e.g. when a follow-up request resolves before
   * the auto-dismiss fires).
   *
   * @param {Object} spec
   * @param {string} spec.title — required, top line
   * @param {string} [spec.sub] — optional second line
   * @param {string} [spec.kind] — 'info' | 'success' | 'warn' | 'error'
   * @param {number} [spec.duration] — auto-dismiss ms; 0 disables
   */
  function push(spec) {
    spec = spec || {};
    var root = rootEl();
    if (!root) return null;

    var id = makeId();
    var kind = spec.kind || 'info';
    var duration = (typeof spec.duration === 'number')
      ? spec.duration
      : DEFAULT_DURATION;

    var el = document.createElement('div');
    el.dataset.toastId = id;
    el.className = 'toast ' + kind;
    // role='alert' for errors so screen readers interrupt; role='status'
    // (the default for #toast-root's aria-live="polite") for the rest.
    el.setAttribute('role', kind === 'error' ? 'alert' : 'status');

    // Build the inner DOM imperatively so the title/sub are set via
    // textContent (XSS-safe — server may pass user-supplied film titles
    // or scene tags through, and the HX-Trigger payload is JSON-decoded
    // by htmx with no markup stripping).
    var ic = document.createElement('div');
    ic.className = 'ic';

    var body = document.createElement('div');
    body.className = 'body';
    var ttl = document.createElement('div');
    ttl.className = 'ttl';
    ttl.textContent = spec.title || '';
    body.appendChild(ttl);
    if (spec.sub) {
      var sub = document.createElement('div');
      sub.className = 'sub';
      sub.textContent = spec.sub;
      body.appendChild(sub);
    }

    var close = document.createElement('button');
    close.className = 'close';
    close.type = 'button';
    close.setAttribute('aria-label', 'Close');
    close.textContent = '×'; // ×
    close.addEventListener('click', function () { remove(id); });

    el.appendChild(ic);
    el.appendChild(body);
    el.appendChild(close);
    root.appendChild(el);

    if (duration > 0) {
      setTimeout(function () { remove(id); }, duration);
    }
    return id;
  }

  /**
   * Remove a toast by id. Adds the .exiting class first so the
   * `p-toast-out` keyframes animation runs; the element is removed
   * from the DOM after EXIT_MS.
   */
  function remove(id) {
    if (!id) return;
    var el = document.querySelector('[data-toast-id="' + id + '"]');
    if (!el) return;
    if (el.classList.contains('exiting')) return; // idempotent
    el.classList.add('exiting');
    setTimeout(function () {
      if (el.parentNode) el.parentNode.removeChild(el);
    }, EXIT_MS);
  }

  // HX-Trigger integration: htmx fires a CustomEvent named exactly after
  // each key in the JSON payload. `HX-Trigger: {"toast": {...}}` ⇒ a
  // "toast" event with `evt.detail` = the inner object. We listen on
  // document.body so the bus picks up triggers from any HTMX-aware
  // response, including SSE-driven swaps.
  function onToastEvent(evt) {
    var detail = evt && evt.detail;
    if (!detail) return;
    // Some htmx versions wrap the detail in an array when the trigger
    // value is non-scalar; normalise both shapes.
    if (Array.isArray(detail)) detail = detail[0];
    push(detail);
  }
  document.body.addEventListener('toast', onToastEvent);

  return { push: push, remove: remove };
})();

// ─── Keyboard router (Phase 7 · Tasks 27 + 28) ────────────────────────
// Single global keydown listener that:
//   * Opens the command palette on ⌘K / Ctrl+K. Works from any focus,
//     including text inputs (the palette IS a text input — preventing it
//     from opening while the user is typing a search query would be the
//     opposite of useful).
//   * Toggles the keyboard help overlay on ``?`` (Shift + /). Guarded
//     against typing in form fields — typing "what?" into an input must
//     keep landing the character, not pop the legend.
//   * ``Esc`` closes the help overlay when it is the active surface.
//     (The palette has its own Esc handler in palette.js.)
//
// palette.js is loaded on demand the first time the palette is opened,
// not eagerly at page load — most sessions never press ⌘K, and the
// palette scaffold is already in the DOM (server-rendered partial), so
// JS-side wiring is the only deferrable cost. Once loaded, ``window.Palette``
// stays cached and subsequent ⌘K presses hit it directly.
//
// The help overlay is fully self-contained: its DOM is server-rendered
// into base.html and ``window.Help`` (defined below) flips ``[hidden]``
// on the outer ``#help`` div. No script-loading dance is required — the
// state machine is small enough to live in mojica.js next to the router.
(function () {
  'use strict';

  var paletteLoading = false;

  /**
   * Ensure ``window.Palette`` is available, then invoke ``callback``.
   *
   * Idempotent + race-safe: once a load is in flight, subsequent calls
   * queue their callback on the same <script> element's ``load`` event
   * instead of injecting a second tag.
   */
  function ensurePaletteLoaded(callback) {
    if (window.Palette) {
      callback();
      return;
    }
    // A previous call is already loading the script — re-attach the
    // callback to the same tag. The script's onload fires once and runs
    // every queued callback.
    if (paletteLoading) {
      var pending = document.getElementById('palette-script');
      if (pending) {
        pending.addEventListener('load', function () { callback(); });
      }
      return;
    }
    paletteLoading = true;
    var script = document.createElement('script');
    script.id = 'palette-script';
    script.src = '/static/js/palette.js';
    script.defer = true;
    script.addEventListener('load', function () { callback(); });
    document.head.appendChild(script);
  }

  // ── Help overlay state machine ──────────────────────────────────────
  // State now lives in an Alpine store (``Alpine.store('help').open``)
  // so the markup in partials/_help_overlay.html can react with
  // ``x-show`` / ``@click`` / ``@keydown.escape.window`` directives.
  // The functions below are thin wrappers that flip the store; we keep
  // their named-function form because the public JS contract test
  // (``test_mojica_js_contains_help_toggle``) and downstream call-sites
  // (the ⌘K block, palette callbacks, future TopBar "?" button) all
  // address them through ``window.Help.{open,close,toggle}`` rather than
  // touching Alpine directly. Keeping the wrappers also means the file
  // works on the rare error page that loads mojica.js without Alpine
  // — every helper no-ops cleanly when the store is unavailable.
  function helpStore() {
    return window.Alpine && window.Alpine.store
      ? window.Alpine.store('help')
      : null;
  }

  function openHelp() {
    // Mutual exclusion: only one polish surface is on screen at a time.
    // If the palette is open, dismiss it first so the help legend takes
    // its place rather than stacking on top.
    if (window.Palette && typeof window.Palette.close === 'function') {
      window.Palette.close();
    }
    var s = helpStore();
    if (s) s.open = true;
  }

  function closeHelp() {
    var s = helpStore();
    if (s) s.open = false;
  }

  function toggleHelp() {
    var s = helpStore();
    if (s) s.open = !s.open;
  }

  // Register the store as soon as Alpine boots. ``alpine:init`` fires
  // once, right before Alpine walks the DOM, so the store is ready by
  // the time _help_overlay's ``x-show="$store.help.open"`` is evaluated.
  // Deferred-script ordering guarantees mojica.js runs before
  // DOMContentLoaded (which is when the CDN build calls Alpine.start()),
  // so this listener is always attached in time.
  document.addEventListener('alpine:init', function () {
    if (window.Alpine && typeof window.Alpine.store === 'function') {
      window.Alpine.store('help', { open: false });
    }
  });

  // Backdrop click, close-button click, and Esc-to-dismiss are now
  // declared on the overlay element itself via Alpine directives
  // (@click.self / @click / @keydown.escape.window). The legacy
  // document.addEventListener('click', …) handler that walked the DOM
  // to detect backdrop vs panel clicks is gone — Alpine's $event.target
  // check is the same logic in a clearer location.

  // Expose a small public surface so future flows (e.g. a "?" button on
  // the TopBar, or Task 27's palette gaining a "Show shortcuts" action)
  // can drive the overlay without re-implementing the toggle.
  window.Help = { open: openHelp, close: closeHelp, toggle: toggleHelp };

  document.addEventListener('keydown', function (e) {
    // ⌘K / Ctrl+K — open palette. Bypasses the "in field" guard
    // intentionally: this shortcut should work from any focus context.
    // Mutual exclusion: close help first so the palette doesn't pop on
    // top of the legend.
    var isMod = e.metaKey || e.ctrlKey;
    if (isMod && (e.key === 'k' || e.key === 'K')) {
      e.preventDefault();
      closeHelp();
      ensurePaletteLoaded(function () {
        if (window.Palette && typeof window.Palette.open === 'function') {
          window.Palette.open();
        }
      });
      return;
    }

    // Esc-when-help-open is now handled declaratively in
    // _help_overlay.html (``@keydown.escape.window``) so the keyboard
    // router no longer needs a branch for it. The palette still owns
    // its own Esc handler in palette.js; mutual exclusion is preserved
    // because openHelp/openPalette dismiss each other before opening.

    // Below this line: single-key shortcuts. The "in field" guard
    // suppresses them when the user is typing into an input/textarea so
    // a query like "what?" doesn't pop the help overlay mid-word. The
    // palette is special-cased because its <input> is the surface the
    // shortcut targets — we don't want help to pop when the palette is
    // already on screen either.
    var ae = document.activeElement;
    var inField = ae && (ae.tagName === 'INPUT' || ae.tagName === 'TEXTAREA' || ae.isContentEditable);
    var paletteOpen = !!(document.getElementById('palette') && !document.getElementById('palette').hidden);
    if (inField || paletteOpen) return;

    // ? — toggle keyboard help overlay. Bare key only; modifier
    // combinations are reserved for browser/OS shortcuts (⌘? = "About
    // Browser" on macOS, etc.) and must not be hijacked.
    if (e.key === '?' && !(e.metaKey || e.ctrlKey || e.altKey)) {
      e.preventDefault();
      toggleHelp();
      return;
    }
  });
})();
