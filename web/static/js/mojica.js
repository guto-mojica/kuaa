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
