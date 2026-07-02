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
})();
