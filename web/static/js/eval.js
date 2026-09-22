// web/static/js/eval.js
// Eval set builder — Alpine component for the keyboard-first grading
// workspace (Phase 9 · Task 32; Alpine migration).
//
// Loaded only on /eval (referenced from eval/layout.html, after
// alpine.min.js + mojica.js). The surface is keyboard-first: 0/1/2/3
// grade keys, S = skip, j/k = row nav, ⌘⏎ = save & advance, B/C =
// toggle blind/compare. Mouse clicks on .gb / .ev-save / .ev-skip
// take the same code path through @click directives.
//
// State lives in an Alpine component (``Alpine.data('evalApp', …)``)
// mounted on ``<body class="ev-app">``:
//   * currentRow — index of the focused .ev-row. Drives the reactive
//     ``cur`` class via :class in eval/rows.html; a $watch scrolls the
//     focused row into view.
//   * blind / compare — toggle state. Drives ``body.ev-blind`` /
//     ``body.ev-compare`` (:class on <body>) and the per-row
//     .score/.scbar ``blind`` class; the queue.html checkboxes are
//     x-model-bound, so a keypress (B/C) and a click both flow
//     through the same signal.
//
// The network + DOM-patch helpers (gradeRow → /api/eval/grade,
// updateRowGrade, refreshMetrics → /api/eval/metrics, setMetric) stay
// imperative: they patch server-rendered markup after a fetch and have
// no declarative Alpine analogue. This mirrors how palette.js keeps
// its fetch/render path imperative while the open/close surface state
// moved to a store.

(function () {
  'use strict';

  // Admin token: ``?token=`` on the URL is the path real graders take;
  // ``body[data-token]`` is the offline fallback for a bookmarked page
  // with the cookie set. Resolved once at load, independent of Alpine.
  function readToken() {
    var m = window.location.search.match(/[?&]token=([^&]+)/);
    var fromUrl = m ? decodeURIComponent(m[1]) : '';
    var body = document.querySelector('.ev-app');
    return fromUrl || (body && body.dataset.token) || '';
  }

  // ── Grader identity persistence ───────────────────────────────────
  // The server reads the ``grader`` cookie on every grade and defaults
  // it to "anon". A cookie is host-scoped and clearable, so a session
  // reached at localhost:8501 instead of 127.0.0.1:8501, or a browser
  // set to drop cookies on quit, silently resumes as a different
  // annotator — which is how one person's grading pass ends up split
  // across two names that share no judgment. localStorage is the same
  // origin as the cookie but survives the cookie being cleared, so it
  // can restore the name; the server refusing "anon" covers the rest.
  var GRADER_KEY = 'kuaa.eval.grader';
  var HEAL_FLAG = 'kuaa.eval.grader.healed';

  function readGraderCookie() {
    var m = document.cookie.match(/(?:^|;\s*)grader=([^;]*)/);
    return m ? decodeURIComponent(m[1]) : '';
  }

  function writeGraderCookie(name) {
    document.cookie =
      'grader=' + encodeURIComponent(name) +
      '; path=/; max-age=31536000; samesite=lax';
  }

  function storedGrader() {
    try {
      return window.localStorage.getItem(GRADER_KEY) || '';
    } catch (err) {
      return '';  // private mode / storage disabled
    }
  }

  function storeGrader(name) {
    try {
      window.localStorage.setItem(GRADER_KEY, name);
    } catch (err) {
      /* nothing to do — the cookie is still the source of truth */
    }
  }

  // Restore the cookie from localStorage when it has gone missing, and
  // reload once so the server-rendered header, resume point and
  // agreement panel all reflect the restored name rather than "anon".
  // The sessionStorage flag makes the reload at most once per tab.
  function healGraderCookie() {
    if (readGraderCookie()) return;
    var name = storedGrader();
    if (!name) return;
    writeGraderCookie(name);
    try {
      if (window.sessionStorage.getItem(HEAL_FLAG)) return;
      window.sessionStorage.setItem(HEAL_FLAG, '1');
    } catch (err) {
      return;  // cannot guard the reload → settle for the cookie alone
    }
    window.location.reload();
  }

  // i18n-invariant metric keys → canonical English label text. The
  // right-pane cards are matched by label text (see setMetric); in a
  // translated locale the match silently no-ops and the cards just
  // don't live-update — a Task-33 follow-up (server-rendered partial
  // swap) will remove the label-matching entirely.
  var METRIC_LABELS = {
    p_at_5: 'PRECISION@5',
    p_at_3: 'PRECISION@3',
    ndcg_at_5: 'NDCG@5',
    inversions: 'INVERSIONS',
  };

  // Session-scoped cache for nDCG@5 deltas. Keyed by query_id so the
  // delta resets when the grader switches queries (a fresh query has
  // no prior baseline). The value is the LAST nDCG@5 the grader saw
  // for that query; refreshMetrics writes the new value + a delta
  // badge before updating the cache for the next round-trip.
  var PREV_NDCG = Object.create(null);

  // Format a delta number as "+0.07" / "-0.04" / "" (empty when the
  // change is below the smallest displayable step).
  function fmtDelta(delta) {
    if (!isFinite(delta)) return '';
    var rounded = Math.round(delta * 100) / 100;
    if (Math.abs(rounded) < 0.005) return '';
    return (rounded > 0 ? '+' : '') + rounded.toFixed(2);
  }

  // Live row list. Re-queried on demand rather than cached because an
  // HTMX swap of the row partial replaces the nodes; reading the DOM
  // each call keeps the cursor logic correct without a stale array.
  function rowEls() {
    return Array.prototype.slice.call(
      document.querySelectorAll('#ev-rows .ev-row')
    );
  }

  document.addEventListener('alpine:init', function () {
    if (!(window.Alpine && typeof window.Alpine.data === 'function')) return;

    window.Alpine.data('evalApp', function (opts) {
      opts = opts || {};
      return {
        currentRow: typeof opts.row === 'number' ? opts.row : 0,
        blind: !!opts.blind,
        compare: !!opts.compare,
        // '?' toggles the shortcut overlay in layout.html. Not persisted —
        // it is a reminder, not a preference.
        help: false,
        rowCount: 0,
        token: readToken(),

        init: function () {
          var self = this;
          healGraderCookie();
          var cookieName = readGraderCookie();
          if (cookieName && cookieName !== 'anon') storeGrader(cookieName);
          this.countRows();
          // Re-count after an HTMX swap brings in a new row list
          // (forward-looking — no row partial swaps today, but the
          // cursor must survive one when Task 33 adds it).
          document.body.addEventListener('htmx:afterSettle', function () {
            self.countRows();
          });
          // Keep the focused row in view as the cursor moves. The
          // ``cur`` class itself is reactive (:class in rows.html);
          // only the scroll needs an imperative nudge.
          this.$watch('currentRow', function () { self.scrollToCurrent(); });
        },

        // ── Grader identity ─────────────────────────────────────────
        // Writes the ``grader`` cookie the server reads on every grade
        // (api/routes/eval.py) and keys inter-annotator agreement on.
        // Nothing else in the UI writes it, and its "anon" default
        // silently merges separate grading passes into one annotator.
        //
        // Reloads because the name reaches the page through the server
        // context — header, initials, the per-grader resume point and
        // the agreement panel all come from build_eval_context, not
        // from this component.
        setGrader: function (raw) {
          var name = String(raw || '')
            .replace(/[\x00-\x1f;,=]/g, ' ')  // control chars + cookie delimiters
            .replace(/\s+/g, ' ')
            .trim()
            .slice(0, 40);
          if (!name) return;
          writeGraderCookie(name);
          storeGrader(name);
          window.location.reload();
        },

        // ── Row cursor ──────────────────────────────────────────────
        countRows: function () {
          this.rowCount = rowEls().length;
          if (this.currentRow >= this.rowCount) {
            this.currentRow = Math.max(0, this.rowCount - 1);
          }
        },

        focusRow: function (idx) {
          if (idx < 0 || idx >= this.rowCount) return;
          this.currentRow = idx;
        },

        focusRowEl: function (rowEl) {
          var idx = rowEls().indexOf(rowEl);
          if (idx >= 0) this.currentRow = idx;
        },

        scrollToCurrent: function () {
          var el = rowEls()[this.currentRow];
          if (el && typeof el.scrollIntoView === 'function') {
            el.scrollIntoView({ block: 'nearest' });
          }
        },

        // ── Grading ─────────────────────────────────────────────────
        // Mouse path: a .gb click moves the cursor onto its row (the
        // user is telling us where their attention is) then grades it.
        onGradeClick: function (btnEl, grade) {
          var row = btnEl.closest('.ev-row');
          if (!row) return;
          this.focusRowEl(row);
          this.gradeRow(row, grade);
        },

        // Keyboard path: grade whatever row the cursor is on.
        gradeCurrent: function (grade) {
          var row = rowEls()[this.currentRow];
          if (row) this.gradeRow(row, grade);
        },

        // POST /api/eval/grade then patch the row. We deliberately do
        // not roll back on failure — instead we toast and leave the
        // prior chips in place so the grader can retry.
        gradeRow: function (rowEl, grade) {
          var queryId = rowEl.dataset.queryId;
          var sceneId = rowEl.dataset.sceneId;
          if (!queryId || !sceneId) return;

          var fd = new FormData();
          fd.append('query_id', queryId);
          fd.append('scene_id', sceneId);
          fd.append('grade', String(grade));

          var qs = this.token
            ? '?token=' + encodeURIComponent(this.token)
            : '';
          var self = this;
          fetch('/api/eval/grade' + qs, { method: 'POST', body: fd })
            .then(function (resp) {
              if (!resp.ok) {
                resp.json().then(function (body) {
                  self.toast(
                    'Grade failed',
                    (body && body.detail) || ('Server ' + resp.status),
                    'error'
                  );
                }).catch(function () {
                  self.toast('Grade failed', 'Server ' + resp.status, 'error');
                });
                return;
              }
              self.updateRowGrade(rowEl, grade);
              self.refreshMetrics(queryId);
            })
            .catch(function (err) {
              // Network / CORS / offline — keep the row paint untouched.
              // eslint-disable-next-line no-console
              console.error('eval grade error:', err);
              self.toast(
                'Grade failed',
                err && err.message ? err.message : 'Network error',
                'error'
              );
            });
        },

        // Patch the row's chips after a successful POST. Imperative:
        // the .gb chips are server-rendered and there is no per-row
        // Alpine state to make this reactive without a deeper rewrite
        // (every row would need its own x-data + grade signal).
        updateRowGrade: function (rowEl, grade) {
          rowEl.classList.add('graded');
          var buttons = rowEl.querySelectorAll('.gb');
          buttons.forEach(function (b) {
            b.classList.remove('on0', 'on1', 'on2', 'on3', 'ons', 'dim');
          });
          buttons.forEach(function (b) {
            var v = parseInt(b.dataset.grade, 10);
            if (Number.isNaN(v)) return;
            if (v === grade) {
              if (grade === -1) b.classList.add('ons');
              else b.classList.add('on' + grade);
            } else if (grade !== -1) {
              // SKIP (-1) does not dim the 0/1/2/3 chips; any real
              // grade dims its siblings (mirrors the server render).
              b.classList.add('dim');
            }
          });
        },

        // ── Metrics refresh ─────────────────────────────────────────
        // Hit /api/eval/metrics and patch the four .ev-met cards in
        // place. We don't re-render the whole right pane — htmx isn't
        // on those nodes and a full swap would drop the histogram +
        // session stats the JSON response doesn't carry.
        refreshMetrics: function (queryId) {
          if (!queryId) return;
          var qs =
            '?query_id=' + encodeURIComponent(queryId) +
            (this.token ? '&token=' + encodeURIComponent(this.token) : '');
          var self = this;
          fetch('/api/eval/metrics' + qs)
            .then(function (resp) { return resp.ok ? resp.json() : null; })
            .then(function (data) {
              if (!data) return;
              // nDCG delta lands BEFORE setMetric updates the value so
              // fmtDelta can compare against the previous reading. The
              // cache is per-query so switching queries doesn't carry
              // a misleading delta across context boundaries.
              var prev = PREV_NDCG[queryId];
              if (typeof prev === 'number' && typeof data.ndcg_at_5 === 'number') {
                self.setDelta('ndcg_at_5', data.ndcg_at_5 - prev);
              }
              if (typeof data.ndcg_at_5 === 'number') {
                PREV_NDCG[queryId] = data.ndcg_at_5;
              }
              self.setMetric('p_at_5', data.p_at_5);
              self.setMetric('p_at_3', data.p_at_3);
              self.setMetric('ndcg_at_5', data.ndcg_at_5);
              self.setMetric('inversions', data.inversions);
            })
            .catch(function (err) {
              // eslint-disable-next-line no-console
              console.error('eval metrics refresh error:', err);
            });
        },

        // Write a delta badge into ``.ev-met .val .delta[data-key="…"]``.
        // Adds .up / .dn classes so eval.css renders green-for-better
        // (nDCG/precision gains) and red-for-worse. Inversions invert
        // the polarity (more inversions = worse), but we only render
        // the nDCG delta today — the metric.html scaffolding leaves a
        // .delta hook only there. Adding inversions later is one
        // template edit + one cache key.
        setDelta: function (key, delta) {
          var el = document.querySelector('.ev-met .val .delta[data-key="' + key + '"]');
          if (!el) return;
          var text = fmtDelta(delta);
          el.classList.remove('up', 'dn');
          el.textContent = text;
          if (!text) return;
          // nDCG / precision: positive delta = improvement.
          if (delta > 0) el.classList.add('up');
          else if (delta < 0) el.classList.add('dn');
        },

        setMetric: function (key, value) {
          if (value === undefined || value === null) return;
          var labelText = METRIC_LABELS[key];
          if (!labelText) return;
          var mets = document.querySelectorAll('.ev-rp .ev-met');
          for (var i = 0; i < mets.length; i++) {
            var m = mets[i];
            var lab = m.querySelector('.lab');
            if (!lab || lab.textContent.trim() !== labelText) continue;
            var n = m.querySelector('.val .n');
            if (n) {
              if (key === 'p_at_5' || key === 'p_at_3') {
                n.textContent = Math.round(value * 100) + '%';
              } else if (key === 'ndcg_at_5') {
                n.textContent = (Number(value) || 0).toFixed(2);
              } else {
                n.textContent = String(value);
              }
            }
            var bar = m.querySelector('.bar');
            if (bar) {
              var pct = 0;
              if (key === 'p_at_5' || key === 'p_at_3' || key === 'ndcg_at_5') {
                pct = (Number(value) || 0) * 100;
              } else if (key === 'inversions') {
                // Same scaling as metrics.html — 10 inversions ≈ full bar.
                pct = Math.min(100, (Number(value) || 0) * 10);
              }
              bar.style.setProperty('--p', Math.round(pct) + '%');
            }
            if (key === 'inversions') {
              m.classList.toggle('warn', (Number(value) || 0) > 0);
            }
            break;
          }
        },

        // ── Save & advance / skip ───────────────────────────────────
        // Grades save on the grade keypress; ⌘⏎ only moves. Within a
        // query it steps the row cursor. Past the last row it opens the
        // next query in the queue — but only once every row here has a
        // judgment. Leaving with rows unjudged is how a query gets
        // half-graded and then read as done by whoever resumes, so an
        // early ⌘⏎ jumps back to the first unjudged row instead.
        saveAndAdvance: function () {
          if (this.rowCount === 0) return;
          var nextRow = this.currentRow + 1;
          if (nextRow < this.rowCount) {
            this.currentRow = nextRow;
            return;
          }
          var unjudged = this.firstUnjudgedRow();
          if (unjudged >= 0) {
            this.currentRow = unjudged;
            this.toast('Rows unjudged', 'Grade or skip every row before moving on', 'error');
            return;
          }
          var href = this.nextQueryHref();
          if (href) {
            window.location.assign(href);
          } else {
            this.toast('Queue complete', 'No query after this one in the current filter', 'success');
          }
        },

        // Index of the first row without a judgment, or -1. Server-rendered
        // rows carry .graded when a grade exists; updateRowGrade adds it on
        // a successful POST, so an in-flight grade does not count yet.
        firstUnjudgedRow: function () {
          var rows = rowEls();
          for (var i = 0; i < rows.length; i++) {
            if (!rows[i].classList.contains('graded')) return i;
          }
          return -1;
        },

        // The queue link after the current query, skipping links the
        // active filter hides (x-show sets display:none, which empties
        // offsetParent). Reading the DOM rather than component state keeps
        // the filter local to the queue aside, where it lives.
        nextQueryHref: function () {
          var links = document.querySelectorAll('.ev-q-list .ev-q');
          var cur = document.querySelector('.ev-q-list .ev-q.cur');
          var seen = !cur;
          for (var i = 0; i < links.length; i++) {
            var link = links[i];
            if (link === cur) { seen = true; continue; }
            if (!seen || link.offsetParent === null) continue;
            return link.getAttribute('href') || '';
          }
          return '';
        },

        skipCurrent: function () {
          var row = rowEls()[this.currentRow];
          if (row) this.gradeRow(row, -1);
        },

        // ── Keyboard router ─────────────────────────────────────────
        // Bound via @keydown.window on <body> so the router works even
        // when focus has wandered onto the body after a fetch round-
        // trip. Inputs / textareas suppress every key so graders can
        // type into the filter box; ⌘K etc. pass through to mojica.js.
        onKey: function (e) {
          var ae = document.activeElement;
          if (ae && (ae.tagName === 'INPUT' || ae.tagName === 'TEXTAREA')) {
            return;
          }
          if (e.metaKey && e.key === 'Enter') {
            e.preventDefault();
            this.saveAndAdvance();
            return;
          }
          // Let Cmd-K / Ctrl-K etc. reach the global palette router.
          if (e.ctrlKey || e.metaKey || e.altKey) return;

          var k = e.key;
          if (k === 'j' || k === 'ArrowDown') {
            e.preventDefault();
            this.focusRow(this.currentRow + 1);
          } else if (k === 'k' || k === 'ArrowUp') {
            e.preventDefault();
            this.focusRow(this.currentRow - 1);
          } else if (k === '0' || k === '1' || k === '2' || k === '3') {
            e.preventDefault();
            this.gradeCurrent(parseInt(k, 10));
          } else if (k === 's' || k === 'S') {
            e.preventDefault();
            this.gradeCurrent(-1);
          } else if (k === 'b' || k === 'B') {
            // Flip the signal directly — x-model on the queue.html
            // checkbox + :class on <body> handle the rest reactively.
            e.preventDefault();
            this.blind = !this.blind;
          } else if (k === 'c' || k === 'C') {
            e.preventDefault();
            this.compare = !this.compare;
          } else if (k === '?') {
            // The footer has advertised '?' since the pane shipped and no
            // branch ever handled it. Escape closes, matching every other
            // dismissible surface in the app.
            e.preventDefault();
            this.help = !this.help;
          } else if (k === 'Escape' && this.help) {
            e.preventDefault();
            this.help = false;
          }
        },

        // ── Toast helper ────────────────────────────────────────────
        // Routes through the shared ToastBus (mojica.js). eval/layout
        // now ships its own #toast-root (via partials/_toast_host.html)
        // so these surface visibly instead of no-op'ing.
        toast: function (title, sub, kind) {
          if (window.ToastBus) {
            window.ToastBus.push({ title: title, sub: sub, kind: kind });
          }
        },
      };
    });
  });
})();
