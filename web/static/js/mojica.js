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
// directly. The bus is backed by ``Alpine.store('toasts')`` and the
// ``#toast-root`` div in base.html renders the queue with x-for —
// pushing a spec into ``store.items`` reactively materialises a
// ``.toast`` card, and the per-toast auto-dismiss + click handlers
// flip ``exiting`` on the item so the ``p-toast-out`` keyframe runs
// before the entry is spliced out of the array.
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
// payload above; the listener below pipes it straight into push().
//
// The bus is exposed on `window.ToastBus` so devtools / manual flows can
// call it without a server round-trip (`window.ToastBus.push({title:'Hi'})`).
// Tests pin both the ``ToastBus`` and ``toast-root`` literals plus the
// ``'toast'`` event-name string — all three survive this refactor.
(function () {
  'use strict';

  // Default auto-dismiss delay (ms). Matches the prototype's 3500ms.
  var DEFAULT_DURATION = 3500;
  // How long the exit animation runs before the element is removed.
  // Keep in sync with ``@keyframes p-toast-out`` in polish.css (200ms).
  var EXIT_MS = 200;

  function toastsStore() {
    return window.Alpine && window.Alpine.store
      ? window.Alpine.store('toasts')
      : null;
  }

  function makeId() {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
      return crypto.randomUUID();
    }
    return 't-' + Date.now() + '-' + Math.random().toString(36).slice(2, 8);
  }

  // Register the store on alpine:init. Methods live on the store
  // so x-for-bound buttons can call ``$store.toasts.remove(t.id)``
  // directly without bouncing through ``window.ToastBus``.
  document.addEventListener('alpine:init', function () {
    if (!(window.Alpine && typeof window.Alpine.store === 'function')) return;
    window.Alpine.store('toasts', {
      items: [],
      /**
       * Push a new toast onto the host. Returns the toast id so callers
       * can remove it manually (e.g. when a follow-up request resolves
       * before the auto-dismiss fires).
       *
       * @param {Object} spec
       * @param {string} spec.title — required, top line
       * @param {string} [spec.sub] — optional second line
       * @param {string} [spec.kind] — 'info' | 'success' | 'warn' | 'error'
       * @param {number} [spec.duration] — auto-dismiss ms; 0 disables
       */
      push: function (spec) {
        // Some htmx versions wrap the HX-Trigger detail in an array
        // when the value is non-scalar; normalise both shapes here so
        // every push() call site stays simple.
        if (Array.isArray(spec)) spec = spec[0];
        if (!spec) return null;
        var id = makeId();
        var duration = (typeof spec.duration === 'number')
          ? spec.duration
          : DEFAULT_DURATION;
        this.items.push({
          id: id,
          kind: spec.kind || 'info',
          title: spec.title || '',
          sub: spec.sub || '',
          exiting: false,
        });
        var self = this;
        if (duration > 0) {
          setTimeout(function () { self.remove(id); }, duration);
        }
        return id;
      },
      /**
       * Remove a toast by id. Flips ``exiting`` first so the
       * p-toast-out keyframes animation runs; the entry is spliced
       * out of ``items`` after EXIT_MS (which removes the DOM via
       * the x-for binding).
       */
      remove: function (id) {
        if (!id) return;
        var t = this.items.find(function (x) { return x.id === id; });
        if (!t || t.exiting) return; // idempotent
        t.exiting = true;
        var self = this;
        setTimeout(function () {
          self.items = self.items.filter(function (x) { return x.id !== id; });
        }, EXIT_MS);
      },
    });
  });

  // ``window.ToastBus`` is the public surface — thin wrapper that
  // routes into the store. Kept as a stable name so devtools, future
  // client-side flows, and test_mojica_js_contains_toast_bus all
  // continue to see ``ToastBus`` in the bundle.
  window.ToastBus = {
    push: function (spec) {
      var s = toastsStore();
      return s ? s.push(spec) : null;
    },
    remove: function (id) {
      var s = toastsStore();
      if (s) s.remove(id);
    },
  };

  // HX-Trigger integration: htmx fires a CustomEvent named exactly
  // after each key in the JSON payload. ``HX-Trigger: {"toast": {...}}``
  // ⇒ a "toast" event with ``evt.detail`` = the inner object. We
  // listen on document.body so the bus picks up triggers from any
  // HTMX-aware response, including SSE-driven swaps. The store's
  // push() handles the array-wrapping normalisation, so this listener
  // is a one-liner forwarder.
  document.body.addEventListener('toast', function (evt) {
    window.ToastBus.push(evt && evt.detail);
  });
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

  // Register the stores as soon as Alpine boots. ``alpine:init`` fires
  // once, right before Alpine walks the DOM, so both stores are ready
  // by the time _help_overlay's ``x-show="$store.help.open"`` and
  // _palette.html's ``x-show="$store.palette.open"`` are first
  // evaluated. Deferred-script ordering guarantees mojica.js runs
  // before DOMContentLoaded (which is when the CDN build calls
  // Alpine.start()), so this listener is always attached in time.
  //
  // The palette store is registered eagerly even though palette.js
  // itself is loaded on demand — the markup needs the store before
  // the user's first ⌘K so the closed state is reactive from the
  // first paint instead of relying on an inline ``hidden`` attribute.
  document.addEventListener('alpine:init', function () {
    if (window.Alpine && typeof window.Alpine.store === 'function') {
      window.Alpine.store('help', { open: false });
      window.Alpine.store('palette', { open: false });
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
    // Palette-open state lives in Alpine.store('palette'); fall back to a
    // simple ``false`` if the store hasn't initialised yet (the first
    // single-key shortcut after page load would otherwise race the
    // alpine:init handler, which is harmless here — the palette can't
    // be open if the store isn't even registered).
    var paletteStore = window.Alpine && window.Alpine.store
      ? window.Alpine.store('palette')
      : null;
    var paletteOpen = !!(paletteStore && paletteStore.open);
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

// ─── UI preferences (Cenas Appearance / Fields + Buscar view) ────────
// localStorage-backed Alpine stores driving toolrow popovers in
// scenes.html (Appearance + Fields) and the view-toggle segments in
// search.html (Grade / Lista / Compacto). The server can't see
// localStorage, so these toggles are client-only — Alpine reactively
// binds class names on the appropriate container to flip layout and
// per-field visibility. Persistence survives reloads and tab swaps.
//
// Store shape (all reactive Alpine proxies):
//   $store.cenasAppearance.density   // 'comfortable' | 'compact'
//   $store.cenasFields.{timecode,pin_count,version,sub,tipo}  // bool
//   $store.buscarView.mode           // 'grid' | 'list' | 'compact'
//
// Defaults are "everything visible, comfortable density, grid view";
// users only ever pay storage cost when they deviate from defaults.
// A corrupt or absent payload silently falls back to defaults — the
// try/catch keeps a malformed localStorage entry from breaking the
// page entirely.
(function () {
  'use strict';

  // localStorage key namespace. Prefixed so future prefs don't collide
  // with any other localStorage usage in the app (eval grader, etc.).
  var KEYS = {
    appearance: 'mojica:cenas:appearance',
    fields:     'mojica:cenas:fields',
    group:      'mojica:cenas:group',
    sort:       'mojica:cenas:sort',
    view:       'mojica:buscar:view',
  };

  // Defaults — also the source of truth for "what fields exist". The
  // Fields popover renders one row per key here in declaration order.
  // ``group`` / ``sort`` mirror the server's ``_VALID_GROUPS`` /
  // ``_VALID_SORTS`` defaults — keep the two in sync if either is
  // edited.
  var DEFAULTS = {
    appearance: { density: 'comfortable' },
    fields:     { timecode: true, pin_count: true, version: true, sub: true, tipo: true },
    group:      { by: 'film' },
    sort:       { by: 'timecode' },
    view:       { mode: 'grid' },
  };

  /**
   * Load a JSON-encoded prefs payload from localStorage and merge it
   * onto a defaults object. Missing keys + parse failures both fall
   * back to defaults rather than leaving the store half-populated.
   */
  function loadPrefs(key, defaults) {
    try {
      var raw = window.localStorage && window.localStorage.getItem(key);
      if (!raw) return Object.assign({}, defaults);
      var parsed = JSON.parse(raw);
      if (!parsed || typeof parsed !== 'object') return Object.assign({}, defaults);
      return Object.assign({}, defaults, parsed);
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

    window.Alpine.store('cenasAppearance', loadPrefs(KEYS.appearance, DEFAULTS.appearance));
    window.Alpine.store('cenasFields',     loadPrefs(KEYS.fields,     DEFAULTS.fields));
    window.Alpine.store('cenasGroup',      loadPrefs(KEYS.group,      DEFAULTS.group));
    window.Alpine.store('cenasSort',       loadPrefs(KEYS.sort,       DEFAULTS.sort));
    window.Alpine.store('buscarView',      loadPrefs(KEYS.view,       DEFAULTS.view));

    // Persistence effects must register AFTER the stores exist; same
    // alpine:init handler keeps the relative ordering deterministic.
    persistOnChange('cenasAppearance', KEYS.appearance, ['density']);
    persistOnChange('cenasFields',     KEYS.fields,     ['timecode', 'pin_count', 'version', 'sub', 'tipo']);
    persistOnChange('cenasGroup',      KEYS.group,      ['by']);
    persistOnChange('cenasSort',       KEYS.sort,       ['by']);
    persistOnChange('buscarView',      KEYS.view,       ['mode']);
  });
})();
