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

