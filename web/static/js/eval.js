// web/static/js/eval.js
// Eval set builder keyboard router + grading interactions (Phase 9 · Task 32).
//
// Loaded only on /eval (referenced from eval/layout.html). The /eval surface
// is keyboard-first: 0/1/2/3 grade keys, S = skip, j/k = row nav, ⌘⏎ =
// save & advance, B/C = toggle blind/compare. Mouse clicks on .gb buttons
// take the same code path as keypresses. After a grade lands, this script
// updates the row's chips inline and re-fetches /api/eval/metrics to repaint
// the right-pane Precision@K / nDCG / Inversions cards without a full page
// reload — see the API contract in api/routes/eval.py.

(function () {
  'use strict';

  const root = document.querySelector('.ev-app');
  if (!root) return;

  // The token lives on body.data-token (rendered by eval/layout.html) and is
  // also passed as ?token= on the page URL. We read the URL first because
  // that's the path real graders take; body[data-token] is the offline
  // fallback used when the page was bookmarked with the cookie set.
  const tokenMatch = window.location.search.match(/[?&]token=([^&]+)/);
  const adminToken =
    (tokenMatch ? decodeURIComponent(tokenMatch[1]) : '') ||
    (root.dataset.token || '');

  // State
  let currentRow = 0;
  let rows = [];

  function refreshRows() {
    rows = Array.from(document.querySelectorAll('#ev-rows .ev-row'));
    if (rows.length === 0) {
      currentRow = 0;
      return;
    }
    // Preserve current focus when re-binding (HTMX swaps); clamp to range.
    const existingCur = rows.findIndex((r) => r.classList.contains('cur'));
    if (existingCur >= 0) {
      currentRow = existingCur;
    } else if (currentRow >= rows.length) {
      currentRow = Math.max(0, rows.length - 1);
    }
    paintCurrent();
  }

  function paintCurrent() {
    rows.forEach((r, i) => r.classList.toggle('cur', i === currentRow));
    const c = rows[currentRow];
    if (c && typeof c.scrollIntoView === 'function') {
      c.scrollIntoView({ block: 'nearest' });
    }
  }

  // ─── Grading (network + DOM) ─────────────────────────────────────────────
  // gradeRow POSTs /api/eval/grade then mutates the DOM. We deliberately do
  // not roll back on failure: instead we toast (when ToastBus is mounted)
  // and leave the prior chips in place so the grader can retry. The eval
  // page does NOT extend base.html, so #toast-root is absent and ToastBus
  // calls no-op; that's acceptable for M1 — the inline button paint is
  // the primary feedback signal anyway.

  async function gradeRow(rowEl, grade) {
    const queryId = rowEl.dataset.queryId;
    const sceneId = rowEl.dataset.sceneId;
    if (!queryId || !sceneId) return;

    const fd = new FormData();
    fd.append('query_id', queryId);
    fd.append('scene_id', sceneId);
    fd.append('grade', String(grade));

    const qs = adminToken ? '?token=' + encodeURIComponent(adminToken) : '';
    try {
      const resp = await fetch('/api/eval/grade' + qs, {
        method: 'POST',
        body: fd,
      });
      if (!resp.ok) {
        if (window.ToastBus) {
          window.ToastBus.push({
            title: 'Grade failed',
            sub: 'Server ' + resp.status,
            kind: 'error',
          });
        }
        return;
      }
      updateRowGrade(rowEl, grade);
      refreshMetrics(queryId);
    } catch (err) {
      // Network / CORS / offline. Keep the row paint untouched.
      // eslint-disable-next-line no-console
      console.error('eval grade error:', err);
      if (window.ToastBus) {
        window.ToastBus.push({
          title: 'Grade failed',
          sub: err && err.message ? err.message : 'Network error',
          kind: 'error',
        });
      }
    }
  }

  function updateRowGrade(rowEl, grade) {
    rowEl.classList.add('graded');
    const buttons = rowEl.querySelectorAll('.gb');
    buttons.forEach((b) => {
      b.classList.remove('on0', 'on1', 'on2', 'on3', 'ons', 'dim');
    });
    buttons.forEach((b) => {
      const v = parseInt(b.dataset.grade, 10);
      if (Number.isNaN(v)) return;
      if (v === grade) {
        if (grade === -1) b.classList.add('ons');
        else b.classList.add('on' + grade);
      } else {
        // SKIP doesn't dim the 0/1/2/3 chips per the prototype CSS — only
        // the non-skip path dims siblings. (For consistency with how the
        // server renders graded rows, we dim siblings for any grade != -1.)
        if (grade !== -1) b.classList.add('dim');
      }
    });
  }

  // ─── Metrics refresh ─────────────────────────────────────────────────────
  // We hit /api/eval/metrics?query_id=... and patch the four .ev-met cards
  // in place. We don't re-render the whole right pane because (a) htmx
  // isn't on those nodes and (b) full-pane swap would lose the histogram +
  // session stats which the JSON response doesn't fully carry. The label
  // strings come from the i18n-rendered template, so we match against the
  // canonical English source keys (PRECISION@5, NDCG@5, PRECISION@3,
  // INVERSIONS). In a translated locale this match silently no-ops; the
  // page still works, the right pane just doesn't live-update. That's a
  // Task-33 follow-up (server-rendered partial swap).

  const METRIC_LABELS = {
    p_at_5: 'PRECISION@5',
    p_at_3: 'PRECISION@3',
    ndcg_at_5: 'NDCG@5',
    inversions: 'INVERSIONS',
  };

  async function refreshMetrics(queryId) {
    if (!queryId) return;
    const qs =
      '?query_id=' + encodeURIComponent(queryId) +
      (adminToken ? '&token=' + encodeURIComponent(adminToken) : '');
    try {
      const resp = await fetch('/api/eval/metrics' + qs);
      if (!resp.ok) return;
      const data = await resp.json();
      setMetric('p_at_5', data.p_at_5);
      setMetric('p_at_3', data.p_at_3);
      setMetric('ndcg_at_5', data.ndcg_at_5);
      setMetric('inversions', data.inversions);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error('eval metrics refresh error:', err);
    }
  }

  function setMetric(key, value) {
    if (value === undefined || value === null) return;
    const labelText = METRIC_LABELS[key];
    if (!labelText) return;
    const mets = document.querySelectorAll('.ev-rp .ev-met');
    for (const m of mets) {
      const lab = m.querySelector('.lab');
      if (!lab || lab.textContent.trim() !== labelText) continue;
      const n = m.querySelector('.val .n');
      if (n) {
        if (key === 'p_at_5' || key === 'p_at_3') {
          n.textContent = Math.round(value * 100) + '%';
        } else if (key === 'ndcg_at_5') {
          n.textContent = (Number(value) || 0).toFixed(2);
        } else {
          n.textContent = String(value);
        }
      }
      const bar = m.querySelector('.bar');
      if (bar) {
        let pct = 0;
        if (key === 'p_at_5' || key === 'p_at_3' || key === 'ndcg_at_5') {
          pct = (Number(value) || 0) * 100;
        } else if (key === 'inversions') {
          // Same scaling as metrics.html — 10 inversions ≈ full bar.
          pct = Math.min(100, (Number(value) || 0) * 10);
        }
        bar.style.setProperty('--p', Math.round(pct) + '%');
      }
      // Toggle .warn on the inversions card when count > 0 (mirrors the
      // template's server-side branch).
      if (key === 'inversions') {
        m.classList.toggle('warn', (Number(value) || 0) > 0);
      }
      break;
    }
  }

  // ─── Save & advance ─────────────────────────────────────────────────────
  // M1 behaviour: advance the row cursor when there's a next row; otherwise
  // surface "query complete" via the toast bus. The cross-query advance
  // (navigate to ?query=<next_id>) lands with Task 33 once the queue panel
  // exposes a stable ordering attribute.

  function saveAndAdvance() {
    if (rows.length === 0) return;
    const nextRow = currentRow + 1;
    if (nextRow < rows.length) {
      currentRow = nextRow;
      paintCurrent();
    } else if (window.ToastBus) {
      window.ToastBus.push({
        title: 'Query complete',
        sub: 'Advance to next query',
        kind: 'success',
      });
    }
  }

  // ─── Mouse path ──────────────────────────────────────────────────────────
  root.addEventListener('click', (e) => {
    const btn = e.target.closest('.gb');
    if (btn) {
      const grade = parseInt(btn.dataset.grade, 10);
      if (Number.isNaN(grade)) return;
      const row = btn.closest('.ev-row');
      if (!row) return;
      // Move the row cursor onto whatever the user clicked — they're
      // telling us where their attention is.
      const idx = rows.indexOf(row);
      if (idx >= 0) {
        currentRow = idx;
        paintCurrent();
      }
      gradeRow(row, grade);
      return;
    }
    // Right-pane action buttons: SAVE & ADVANCE / SKIP.
    const act = e.target.closest('[data-action]');
    if (act) {
      const which = act.dataset.action;
      if (which === 'save-advance') {
        saveAndAdvance();
      } else if (which === 'skip') {
        const row = rows[currentRow];
        if (row) gradeRow(row, -1);
      }
    }
  });

  // ─── Toggle handlers (blind / compare) ───────────────────────────────────
  root.addEventListener('change', (e) => {
    const t = e.target.closest('[data-toggle]');
    if (!t) return;
    const which = t.dataset.toggle;
    const enabled = t.checked;
    if (which === 'blind') {
      document.body.classList.toggle('ev-blind', enabled);
      document
        .querySelectorAll('.ev-row .score, .ev-row .scbar')
        .forEach((el) => el.classList.toggle('blind', enabled));
    } else if (which === 'compare') {
      document.body.classList.toggle('ev-compare', enabled);
    }
    // The visual switch (.toggle.on) mirrors the underlying checkbox.
    const label = t.closest('.toggle');
    if (label) label.classList.toggle('on', enabled);
  });

  // ─── Keyboard router ─────────────────────────────────────────────────────
  // We listen on document so the router works even when focus has wandered
  // onto the body — common after a fetch round-trip. Inputs / textareas
  // suppress every key (graders should be able to type into the filter
  // box without grading the current row). Modifiers other than meta+Enter
  // are passed through to the browser unchanged.

  document.addEventListener('keydown', (e) => {
    const ae = document.activeElement;
    if (ae && (ae.tagName === 'INPUT' || ae.tagName === 'TEXTAREA')) return;

    if (e.metaKey && e.key === 'Enter') {
      e.preventDefault();
      saveAndAdvance();
      return;
    }
    // Allow Cmd-K / Cmd-/ etc. to reach the global palette router.
    if (e.ctrlKey || e.metaKey || e.altKey) return;

    const k = e.key;
    if (k === 'j' || k === 'ArrowDown') {
      e.preventDefault();
      if (currentRow < rows.length - 1) {
        currentRow++;
        paintCurrent();
      }
    } else if (k === 'k' || k === 'ArrowUp') {
      e.preventDefault();
      if (currentRow > 0) {
        currentRow--;
        paintCurrent();
      }
    } else if (k === '0' || k === '1' || k === '2' || k === '3') {
      e.preventDefault();
      const grade = parseInt(k, 10);
      const row = rows[currentRow];
      if (row) gradeRow(row, grade);
    } else if (k === 's' || k === 'S') {
      e.preventDefault();
      const row = rows[currentRow];
      if (row) gradeRow(row, -1);
    } else if (k === 'b' || k === 'B') {
      e.preventDefault();
      const t = document.querySelector('[data-toggle="blind"]');
      if (t) {
        t.checked = !t.checked;
        t.dispatchEvent(new Event('change', { bubbles: true }));
      }
    } else if (k === 'c' || k === 'C') {
      e.preventDefault();
      const t = document.querySelector('[data-toggle="compare"]');
      if (t) {
        t.checked = !t.checked;
        t.dispatchEvent(new Event('change', { bubbles: true }));
      }
    }
  });

  // ─── Init ────────────────────────────────────────────────────────────────
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', refreshRows);
  } else {
    refreshRows();
  }
  // Re-bind when HTMX swaps content (future row-list partial swap).
  document.body.addEventListener('htmx:afterSettle', refreshRows);
})();
