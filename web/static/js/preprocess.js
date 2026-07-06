/* Pre-processing review player — scrub the source video to validate/pick cut
 * points, then split at the playhead. Vanilla + event-delegated so it survives
 * the HTMX fragment swaps that every cut edit triggers. No Alpine dependency
 * on the critical editing path. Guards on the tab being present, so it is inert
 * on every other page.
 */
(function () {
  'use strict';

  function review() { return document.getElementById('pp-review'); }
  function video() { return document.getElementById('pp-video'); }

  function fps() {
    var el = review();
    var v = el ? parseFloat(el.getAttribute('data-fps')) : NaN;
    return v && v > 0 ? v : 24;
  }
  function slug() {
    var el = review();
    return el ? el.getAttribute('data-slug') || '' : '';
  }

  function pad(n) { return (n < 10 ? '0' : '') + n; }

  function readout() {
    var v = video();
    var out = document.getElementById('pp-readout');
    if (!v || !out) return;
    var t = v.currentTime || 0;
    var frame = Math.round(t * fps());
    var hh = Math.floor(t / 3600);
    var mm = Math.floor((t % 3600) / 60);
    var ss = Math.floor(t % 60);
    out.textContent = pad(hh) + ':' + pad(mm) + ':' + pad(ss) + ' · f' + frame;
  }

  function seekTo(seconds) {
    var v = video();
    if (!v) return;
    if (!isNaN(seconds)) {
      v.pause();
      v.currentTime = Math.max(0, seconds);
      readout();
    }
  }

  function stepFrames(delta) {
    var v = video();
    if (!v) return;
    v.pause();
    v.currentTime = Math.max(0, (v.currentTime || 0) + delta / fps());
    readout();
  }

  function splitAtPlayhead() {
    var v = video();
    if (!v || !window.htmx) return;
    var frame = Math.round((v.currentTime || 0) * fps());
    // Staging a split is a cheap JSON write (no rebuild), so this needs no
    // busy/disabled-elt wiring — only "Apply changes" (in the filmstrip
    // fragment) does.
    window.htmx.ajax('POST', '/api/preprocess/cut/split', {
      target: '#pp-filmstrip',
      swap: 'outerHTML',
      values: { slug: slug(), at_frame: frame }
    });
  }

  // timeupdate does not bubble; catch it in the capture phase on the document.
  document.addEventListener('timeupdate', function (e) {
    if (e.target && e.target.id === 'pp-video') readout();
  }, true);
  document.addEventListener('loadedmetadata', function (e) {
    if (e.target && e.target.id === 'pp-video') readout();
  }, true);

  document.addEventListener('click', function (e) {
    var seek = e.target.closest('[data-seek]');
    if (seek) { seekTo(parseFloat(seek.getAttribute('data-seek'))); return; }

    var step = e.target.closest('[data-frame-step]');
    if (step) { stepFrames(parseInt(step.getAttribute('data-frame-step'), 10) || 0); return; }

    if (e.target.closest('#pp-split-here')) { splitAtPlayhead(); return; }
  });

  // ── Keep the strip scroll + playhead across filmstrip swaps ───────────
  // Every edit outerHTML-swaps #pp-filmstrip, which replaces .pp-strip and
  // resets its horizontal scroll to 0 — visually "throwing the timeline back
  // to frame 1". Snapshot the scroll offset (and the playhead, defensively —
  // some engines drop media position on the sibling reflow) before the swap
  // and restore both once the new fragment settles.
  var swapState = { scrollLeft: 0, time: 0 };

  document.addEventListener('htmx:beforeSwap', function (e) {
    var target = e.detail && e.detail.target;
    if (!target || target.id !== 'pp-filmstrip') return;
    var strip = target.querySelector('.pp-strip');
    swapState.scrollLeft = strip ? strip.scrollLeft : 0;
    var v = video();
    swapState.time = v ? (v.currentTime || 0) : 0;
  });

  document.addEventListener('htmx:afterSettle', function (e) {
    var target = e.detail && e.detail.target;
    if (!target || target.id !== 'pp-filmstrip') return;
    var strip = document.querySelector('#pp-filmstrip .pp-strip');
    if (strip && swapState.scrollLeft) strip.scrollLeft = swapState.scrollLeft;
    var v = video();
    if (v && swapState.time > 0.05 && (v.currentTime || 0) < 0.05) {
      v.currentTime = swapState.time;
      readout();
    }
  });

  // ── Keyboard: ← / → jump the playhead between scene cuts ──────────────
  // The active scene is derived from the playhead (not a stored index), so
  // it stays correct across the filmstrip fragment swaps every edit triggers.
  function sceneButtons() {
    return Array.prototype.slice.call(
      document.querySelectorAll('#pp-filmstrip .pp-scene[data-seek]')
    );
  }

  function markActive(el) {
    sceneButtons().forEach(function (s) { s.removeAttribute('aria-current'); });
    el.setAttribute('aria-current', 'true');
    el.scrollIntoView({ block: 'nearest', inline: 'center' });
  }

  function gotoCut(dir) {
    var list = sceneButtons();
    if (!list.length) return;
    var v = video();
    var t = (v && v.currentTime) || 0;
    // Half a frame of tolerance so a playhead sitting exactly on a cut
    // still moves to the neighbouring one instead of re-selecting itself.
    var eps = 0.5 / fps();
    var target = null;
    if (dir > 0) {
      for (var i = 0; i < list.length; i++) {
        if (parseFloat(list[i].getAttribute('data-seek')) > t + eps) { target = list[i]; break; }
      }
    } else {
      for (var j = list.length - 1; j >= 0; j--) {
        if (parseFloat(list[j].getAttribute('data-seek')) < t - eps) { target = list[j]; break; }
      }
    }
    if (!target) return;
    markActive(target);
    seekTo(parseFloat(target.getAttribute('data-seek')));
  }

  document.addEventListener('keydown', function (e) {
    if (!review()) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    var t = e.target;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) return;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      e.preventDefault();  // also blocks the <video>'s native ±5s arrow seek
      gotoCut(e.key === 'ArrowRight' ? 1 : -1);
    }
  });
})();
