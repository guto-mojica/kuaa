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
  // The DOM is rendered once at page load by partials/_help_overlay.html;
  // we just flip the [hidden] attribute on the outer #help node. Keeping
  // the markup in place means the open path renders the legend in the
  // same frame as the keypress, which is the whole point of a help
  // surface bound to a one-key shortcut.
  function openHelp() {
    // Mutual exclusion: only one polish surface is on screen at a time.
    // If the palette is open, dismiss it first so the help legend takes
    // its place rather than stacking on top.
    if (window.Palette && typeof window.Palette.close === 'function') {
      window.Palette.close();
    }
    var help = document.getElementById('help');
    if (!help) return;
    help.hidden = false;
  }

  function closeHelp() {
    var help = document.getElementById('help');
    if (!help) return;
    help.hidden = true;
  }

  function toggleHelp() {
    var help = document.getElementById('help');
    if (!help) return;
    if (help.hidden) openHelp();
    else closeHelp();
  }

  function helpIsOpen() {
    var help = document.getElementById('help');
    return !!(help && !help.hidden);
  }

  // Backdrop + close-button click handler. The .kh-back element IS the
  // backdrop, so a click whose target is the outer #help element (not
  // a child) means the user clicked outside the panel — dismiss. The
  // .kh-panel carries [data-prevent-close] (kept for parity with the
  // palette scaffold even though the target check below is the actual
  // guard) and stops events at its boundary because clicks on the panel
  // hit one of its descendants, not #help itself.
  document.addEventListener('click', function (e) {
    if (!helpIsOpen()) return;
    var help = document.getElementById('help');
    // Backdrop click — the click event's target is the #help element
    // itself only when nothing inside .kh-panel intercepted it.
    if (e.target === help) {
      closeHelp();
      return;
    }
    // Explicit close button (anywhere inside the panel).
    if (e.target.closest && e.target.closest('[data-action="close-help"]')) {
      closeHelp();
    }
  });

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

    // Esc — close help if it is the active surface. The palette installs
    // its own Esc handler inside palette.js; this branch only fires when
    // the palette is closed and help is open, so the two never fight.
    if (e.key === 'Escape' && helpIsOpen()) {
      e.preventDefault();
      closeHelp();
      return;
    }

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

// ── Buscar retrieval prefs (Hybrid Search Task E1) ─────────────────────
// Persisted Alpine store backing the Buscar tab's knob-row popovers
// (retriever / sem_w / bm25_w / top_k). Defaults mirror the canonical
// hybrid baseline so a first-paint UI never drifts from the server
// contract: ``retriever=hybrid``, ``sem_w=0.70``, ``bm25_w=0.30``,
// ``top_k=9`` (UI preference; the route's FastAPI default is 8, see the
// E1 task plan — the hidden HTMX mirror in E2 sends the UI value on
// every request, so the divergence is intentional).
//
// The IIFE pattern mirrors the Mojica-redesign prefs block (which
// already ships ``cenasGroup`` / ``cenasSort`` / ``buscarView`` on the
// design branch): same KEYS/DEFAULTS shape, same ``loadPrefs`` /
// ``savePrefs`` / ``persistOnChange`` helpers, same ``alpine:init``
// registration. When the two branches converge, the helpers collapse
// into one block and the KEYS/DEFAULTS objects merge — no schema rewrite
// is needed because each store is keyed independently in localStorage.
//
// A corrupt or absent payload silently falls back to defaults via the
// try/catch in ``loadPrefs`` — a malformed entry must not break Buscar.
(function () {
  'use strict';

  // localStorage key namespace. Prefixed so future prefs don't collide
  // with any other localStorage usage in the app (eval grader, etc.).
  var KEYS = {
    retrieval: 'mojica:buscar:retrieval',
  };

  // Defaults — also the source of truth for "what fields exist". The
  // ``mode`` / ``sem_w`` / ``bm25_w`` / ``top_k`` fields mirror the
  // ``retriever`` / ``sem_w`` / ``bm25_w`` / ``top_k`` query params on
  // ``/api/search`` and ``/api/search/aggregate`` (D1/D2). Keep this in
  // sync with ``api/routes/search.py`` if the route's defaults change.
  var DEFAULTS = {
    retrieval: { mode: 'hybrid', sem_w: 0.70, top_k: 9 },
  };

  /**
   * Load a JSON-encoded prefs payload from localStorage and merge it
   * onto a defaults object. Missing keys + parse failures fall back
   * to defaults rather than leaving the store half-populated.
   *
   * Per-key type-coercion against the defaults shape: a numeric
   * default rejects any incoming value that ``Number()`` can't make
   * finite (``null``, ``undefined``, ``"abc"`` all fall back to the
   * default), and a string default rejects empty / non-string values.
   * Without this, an old / hand-edited / partial localStorage entry
   * like ``{"mode":"hybrid","sem_w":null}`` would surface as
   * ``Number(null).toFixed(2) === "0.00"`` (OK by luck) or
   * ``Number(undefined).toFixed(2) === "NaN"`` (the actual hazard)
   * in the hidden HTMX form mirror, then the FastAPI route would
   * 422 on every search. Defensive coercion here is cheap and means
   * a corrupt prefs entry never wedges the search UI.
   */
  function loadPrefs(key, defaults) {
    try {
      var raw = window.localStorage && window.localStorage.getItem(key);
      if (!raw) return Object.assign({}, defaults);
      var parsed = JSON.parse(raw);
      if (!parsed || typeof parsed !== 'object') return Object.assign({}, defaults);
      var out = Object.assign({}, defaults);
      for (var k in defaults) {
        if (!Object.prototype.hasOwnProperty.call(defaults, k)) continue;
        if (!Object.prototype.hasOwnProperty.call(parsed, k)) continue;
        var v = parsed[k];
        var dt = typeof defaults[k];
        if (dt === 'number') {
          // Accept only actual JSON numbers (typeof === 'number') that
          // are finite. Reject null / undefined / strings / NaN /
          // Infinity outright — they fall back to defaults instead of
          // coercing to 0 via Number(null) and then surfacing as the
          // wrong slider value.
          if (typeof v === 'number' && Number.isFinite(v)) out[k] = v;
        } else if (dt === 'string') {
          if (typeof v === 'string' && v.length > 0) out[k] = v;
        } else if (dt === 'boolean') {
          if (typeof v === 'boolean') out[k] = v;
        } else {
          out[k] = v;
        }
      }
      return out;
    } catch (e) {
      return Object.assign({}, defaults);
    }
  }

  /** Persist a plain-object snapshot to localStorage. Errors swallowed. */
  function savePrefs(key, snapshot) {
    try {
      if (window.localStorage) {
        window.localStorage.setItem(key, JSON.stringify(snapshot));
      }
    } catch (e) { /* quota / private-mode — best-effort */ }
  }

  /**
   * Bind an Alpine.effect that snapshots the named store's listed
   * keys to localStorage whenever any of them changes. Reading each
   * key inside the effect is what subscribes the effect to changes
   * (Alpine tracks proxy property access).
   */
  function persistOnChange(storeName, key, fields) {
    if (!(window.Alpine && typeof window.Alpine.effect === 'function')) return;
    window.Alpine.effect(function () {
      var s = window.Alpine.store(storeName);
      if (!s) return;
      var snap = {};
      for (var i = 0; i < fields.length; i++) snap[fields[i]] = s[fields[i]];
      savePrefs(key, snap);
    });
  }

  document.addEventListener('alpine:init', function () {
    if (!(window.Alpine && typeof window.Alpine.store === 'function')) return;

    window.Alpine.store('buscarRetrieval', loadPrefs(KEYS.retrieval, DEFAULTS.retrieval));

    // Persistence effects must register AFTER the stores exist; same
    // alpine:init handler keeps the relative ordering deterministic.
    persistOnChange('buscarRetrieval', KEYS.retrieval, ['mode', 'sem_w', 'top_k']);
  });
})();
