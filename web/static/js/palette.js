// web/static/js/palette.js
// Command palette client logic (Phase 7 / Task 27).
//
// Loaded on demand by mojica.js the first time ⌘K (or Ctrl+K) fires.
// Exposes ``window.Palette = { open, close }`` so the keyboard router in
// mojica.js can drive it without re-importing. The DOM scaffold is
// rendered server-side in partials/_palette.html and lives inside
// base.html — this file only wires interaction.
//
// Group order is fixed (navigate / actions / films / scenes_recent) and
// must match api/services/palette_service.py. The server already returns
// rows in the visual order it wants displayed; this file does not re-rank.
//
// Class names follow web/static/css/polish.css:
//   * ``.cp-item.sel``  — selected row (NOT ``.on``)
//   * ``.cp-item .ic``  — icon slot
//   * ``.cp-item .lab .nm`` + ``.lab .sub``  — label + caption
//   * ``.cp-item .kbd`` / ``.cp-item .badge``  — right-side hint chips
//   * ``.cp-empty .big`` — no-results glyph

(function () {
  'use strict';

  // Resolve scaffold elements once at IIFE evaluation. If they're missing
  // we silently no-op: this script is loaded only when the keyboard
  // router decides to open the palette, but a missing scaffold (e.g. an
  // error page that didn't extend base.html) should not throw.
  var root = document.getElementById('palette');
  if (!root) return;
  var input = document.getElementById('cp-input');
  var list = document.getElementById('cp-list');
  var count = document.getElementById('cp-count');
  if (!input || !list || !count) return;

  // Flat array of all currently-visible items, in render order. Used by
  // arrow-key navigation so the selection index maps trivially to the
  // .cp-item nodes (rows[i] ↔ items[i]).
  var items = [];
  var selected = 0;

  // ``lastQuery`` short-circuits repeated identical fetches when the
  // input event fires without a real change (IME composition end, paste
  // of the same string, etc.).
  var lastQuery = null;

  // ── Open / close ────────────────────────────────────────────────────
  // Note: open() resets the input + selection but does NOT scroll-lock
  // the body — the backdrop is fixed full-viewport with its own scroll,
  // and locking the underlying page mid-search causes layout jumps when
  // the user closes the palette and lands back on a different tab.
  function open() {
    root.hidden = false;
    input.value = '';
    selected = 0;
    lastQuery = null;
    // setTimeout(0) so the focus call runs after [hidden] has been
    // removed and the browser has committed the visibility change.
    setTimeout(function () { input.focus(); }, 0);
    refresh('');
  }

  function close() {
    root.hidden = true;
    items = [];
    list.innerHTML = '';
    count.textContent = '';
  }

  // ── Fetch + render ──────────────────────────────────────────────────
  function refresh(q) {
    if (lastQuery === q) return;
    lastQuery = q;
    fetch('/api/palette/search?q=' + encodeURIComponent(q), {
      credentials: 'same-origin',
    })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        // The user may have closed the palette while the fetch was in
        // flight (lastQuery is reset to null on close → ignore stale).
        if (lastQuery === null) return;
        render(data);
      })
      .catch(function () { /* network failure: keep last results visible */ });
  }

  // Group titles. Order is fixed and matches palette_service.py's keys.
  var GROUPS = [
    ['navigate', 'Navigate'],
    ['actions', 'Actions'],
    ['films', 'Films'],
    ['scenes_recent', 'Recent scenes'],
  ];

  function render(data) {
    list.innerHTML = '';
    items = [];
    var total = 0;
    for (var gi = 0; gi < GROUPS.length; gi++) {
      var key = GROUPS[gi][0];
      var label = GROUPS[gi][1];
      var arr = (data && data[key]) || [];
      if (!arr.length) continue;

      var head = document.createElement('div');
      head.className = 'cp-group';
      head.textContent = label;
      list.appendChild(head);

      for (var i = 0; i < arr.length; i++) {
        var item = arr[i];
        items.push(item);
        var idx = items.length - 1;
        var row = renderRow(item, idx);
        list.appendChild(row);
        total++;
      }
    }
    if (total === 0) {
      var empty = document.createElement('div');
      empty.className = 'cp-empty';
      var big = document.createElement('div');
      big.className = 'big';
      big.textContent = '∅';
      empty.appendChild(big);
      var msg = document.createElement('div');
      msg.textContent = 'No matches';
      empty.appendChild(msg);
      list.appendChild(empty);
    }
    count.textContent = total
      ? (total + ' ' + (total === 1 ? 'result' : 'results'))
      : '';
    selected = 0;
    paintSelected();
  }

  function renderRow(item, idx) {
    // Imperative DOM construction (textContent for user strings) keeps
    // server-supplied film titles XSS-safe. The icon slot stays empty
    // for now — Task 27's scaffold uses CSS-only iconography on the
    // .cp-item .ic slot; wiring per-item SVGs is a Phase-8 polish job.
    var row = document.createElement('div');
    row.className = 'cp-item';
    row.setAttribute('role', 'option');
    row.dataset.itemIdx = String(idx);

    var ic = document.createElement('span');
    ic.className = 'ic';
    row.appendChild(ic);

    var lab = document.createElement('span');
    lab.className = 'lab';
    var nm = document.createElement('div');
    nm.className = 'nm';
    nm.textContent = item.label || '';
    lab.appendChild(nm);
    if (item.sub) {
      var sub = document.createElement('div');
      sub.className = 'sub';
      sub.textContent = item.sub;
      lab.appendChild(sub);
    }
    row.appendChild(lab);

    // Right-side hint chip. ``kbd`` (a hotkey like "1") preferred over
    // ``badge`` (a category tag) when both are present — neither group
    // sets both today, but render is defensive.
    if (item.kbd) {
      var kbd = document.createElement('span');
      kbd.className = 'kbd';
      kbd.textContent = item.kbd;
      row.appendChild(kbd);
    } else if (item.badge) {
      var badge = document.createElement('span');
      badge.className = 'badge';
      badge.textContent = item.badge;
      row.appendChild(badge);
    }

    row.addEventListener('click', function () { execute(item); });
    row.addEventListener('mouseenter', function () { setSelected(idx); });
    return row;
  }

  // ── Selection ───────────────────────────────────────────────────────
  function setSelected(idx) {
    if (!items.length) { selected = 0; return; }
    selected = Math.max(0, Math.min(items.length - 1, idx));
    paintSelected();
  }

  function paintSelected() {
    var rows = list.querySelectorAll('.cp-item');
    for (var i = 0; i < rows.length; i++) {
      // ``.sel`` matches polish.css (.cp-item.sel — line 191).
      rows[i].classList.toggle('sel', i === selected);
    }
    var current = rows[selected];
    if (current && typeof current.scrollIntoView === 'function') {
      current.scrollIntoView({ block: 'nearest' });
    }
  }

  // ── Execute ─────────────────────────────────────────────────────────
  function execute(item) {
    if (!item) return;
    close();
    if (item.url) {
      // Same-window navigation. ``/api/locale/{code}`` is a redirector
      // route, so a full GET (not fetch) is the right call — it sets the
      // locale cookie server-side then 303s back to the referring tab.
      window.location.href = item.url;
    }
  }

  // ── Wiring ──────────────────────────────────────────────────────────
  input.addEventListener('input', function (e) {
    refresh(e.target.value);
  });

  root.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') {
      e.preventDefault();
      close();
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelected(selected + 1);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelected(selected - 1);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      execute(items[selected]);
    }
  });

  // Backdrop click closes; panel click does not (target check is
  // strict-equal so any descendant click bubbles through without closing).
  root.addEventListener('click', function (e) {
    if (e.target === root) close();
  });

  // Public surface — mojica.js reads window.Palette to call open() on ⌘K.
  window.Palette = { open: open, close: close };
})();
