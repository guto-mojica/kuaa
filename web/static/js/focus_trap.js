// web/static/js/focus_trap.js
// Mojica · accessible focus management for modal overlays (U3).
//
// WCAG 2.1 keyboard contract for modal dialogs:
//   * 2.1.2 No Keyboard Trap (inverted, intentional): while a *modal*
//     overlay is open, Tab / Shift+Tab must cycle ONLY through the
//     overlay's own focusable controls and must not reach the page
//     behind it. The trap wraps at both ends.
//   * 2.4.3 Focus Order: opening the overlay moves focus into it; closing
//     it returns focus to the control that had focus when it opened (the
//     trigger), so the user lands back where they were.
//
// This is a single vendored IIFE (no build step, no deps). It is generic:
// any element tagged ``data-focus-trap`` is managed automatically — the
// utility never reaches into Alpine stores or HTMX internals, so it works
// identically for the three modal surfaces regardless of how each one
// toggles its own visibility:
//   * #palette  (command palette)  — Alpine ``x-show="$store.palette.open"``
//                                     toggles inline ``display:none``.
//   * #help     (keyboard help)    — Alpine ``x-show="$store.help.open"``.
//   * #about    (About modal)      — HTMX swaps the element in/out of
//                                     ``#modal-container``; visible the
//                                     whole time it is in the DOM.
//
// Anchored, non-modal popovers (the ``.knob-popover`` / ``.scenes-filter-
// popover`` toolbar controls — ``role="dialog"`` WITHOUT ``aria-modal``,
// dismissed by click-outside) are deliberately NOT trapped: trapping a
// transient toolbar popover would break expected tab-through behaviour and
// is not a WCAG requirement for non-modal disclosure widgets.
//
// Detection is visibility-driven (not event-driven) so it stays decoupled
// from each overlay's owner: a MutationObserver watches the document for
// the attribute flips (``style`` / ``class`` / ``hidden``) and node
// insert/remove that change a trap element's rendered state, then diffs
// "which traps are visible now" against "which were visible last tick".
// A trap that became visible is activated; one that became hidden (or was
// removed from the DOM) is deactivated and restores focus.

(function () {
  'use strict';

  // Composite selector for natively- or explicitly-focusable elements.
  // Order is irrelevant (we sort by DOM order via querySelectorAll); what
  // matters is coverage. ``[tabindex="-1"]`` is excluded because such
  // elements are programmatically focusable but NOT part of the Tab order,
  // which is exactly what the wrap logic walks. Hidden inputs are excluded
  // up front; ``:disabled`` filters out disabled form controls.
  var FOCUSABLE = [
    'a[href]',
    'area[href]',
    'button:not([disabled])',
    'input:not([disabled]):not([type="hidden"])',
    'select:not([disabled])',
    'textarea:not([disabled])',
    'iframe',
    'object',
    'embed',
    '[contenteditable="true"]',
    'audio[controls]',
    'video[controls]',
    'summary',
    '[tabindex]:not([tabindex="-1"])',
  ].join(',');

  /**
   * Is ``el`` currently rendered (occupies layout)? True iff it is not
   * ``display:none`` and is laid out somewhere in the page.
   *
   * ``getClientRects().length`` is the robust test here: it is empty for a
   * ``display:none`` element (or one inside a ``display:none`` ancestor) and
   * non-empty otherwise — crucially including ``position:fixed`` overlays,
   * for which ``offsetParent`` is unhelpfully ``null``. All three Mojica
   * modal backdrops are ``position:fixed/absolute; inset:0``, so this is the
   * predicate that works for every case (open ``x-show``, ``x-cloak``-hidden,
   * and the DOM-present-but-CSS-hidden interim).
   */
  function isRendered(el) {
    if (!el || !el.getClientRects) return false;
    return el.getClientRects().length > 0;
  }

  /**
   * Live, DOM-ordered list of the focusable elements inside ``container``
   * that are actually reachable right now. Recomputed on demand (every Tab
   * keypress + on activation) so dynamically-rendered content is handled:
   * the palette's result rows are built imperatively after open, and a
   * knob popover nested in a modal contributes its controls only while it
   * is itself open. ``isRendered`` drops anything currently hidden so Tab
   * never lands on a control the user cannot see.
   */
  function focusable(container) {
    var nodes = container.querySelectorAll(FOCUSABLE);
    var out = [];
    for (var i = 0; i < nodes.length; i++) {
      var n = nodes[i];
      // Skip hidden / collapsed / aria-hidden subtrees.
      if (n.closest('[aria-hidden="true"]')) continue;
      if (!isRendered(n)) continue;
      out.push(n);
    }
    return out;
  }

  /**
   * Choose the element to focus first when ``container`` opens:
   *   1. an explicit ``[data-autofocus]`` inside the container, else
   *   2. the first reachable focusable, else
   *   3. the container itself (made programmatically focusable via a
   *      ``tabindex="-1"`` we stamp on, so an empty dialog still pulls
   *      focus off the page behind it).
   *
   * If focus is ALREADY inside the container when we activate (e.g. the
   * palette's own ``input.focus()`` fired first on the same tick), we keep
   * it — re-focusing would be a redundant flicker and could fight the
   * owner's deliberate target.
   */
  function initialTarget(container) {
    var explicit = container.querySelector('[data-autofocus]');
    if (explicit && isRendered(explicit)) return explicit;
    var list = focusable(container);
    if (list.length) return list[0];
    if (!container.hasAttribute('tabindex')) {
      container.setAttribute('tabindex', '-1');
    }
    return container;
  }

  /**
   * Activate the trap for ``container``: remember the outgoing trigger so
   * we can restore to it, then move focus inside. Idempotent — a second
   * activate() on an already-active container is a no-op so a noisy
   * MutationObserver burst can't clobber the saved trigger or re-focus
   * mid-interaction.
   */
  function activate(container) {
    if (container.__focusTrapActive) return;
    container.__focusTrapActive = true;

    // The trigger is whatever had focus at open time. ``<body>`` is treated
    // as "nothing meaningful was focused" so we don't try to restore to it.
    var trigger = document.activeElement;
    container.__focusTrapTrigger =
      trigger && trigger !== document.body ? trigger : null;

    // Defer the focus move one frame so it runs AFTER the visibility flip
    // has committed (an element still mid-transition to visible can reject
    // .focus()), and after any same-tick owner focus (palette input).
    setTimeout(function () {
      if (!container.__focusTrapActive) return; // closed again before the tick
      if (container.contains(document.activeElement)) return; // already inside
      var target = initialTarget(container);
      if (target && typeof target.focus === 'function') {
        target.focus();
      }
    }, 0);
  }

  /**
   * Deactivate the trap for ``container`` and restore focus to the saved
   * trigger if it is still connected and focusable. Idempotent.
   */
  function deactivate(container) {
    if (!container.__focusTrapActive) return;
    container.__focusTrapActive = false;
    var trigger = container.__focusTrapTrigger;
    container.__focusTrapTrigger = null;
    if (
      trigger &&
      trigger.isConnected &&
      typeof trigger.focus === 'function' &&
      isRendered(trigger)
    ) {
      trigger.focus();
    }
  }

  /**
   * Keydown handler (registered once, at capture phase on the document).
   * On Tab / Shift+Tab while a trap is the active surface, keep focus
   * inside it by wrapping at both ends.
   *
   * Capture phase matters: it runs before the overlay's own bubble-phase
   * handlers (palette.js arrow/Enter router, Alpine ``@keydown``), so the
   * wrap decision is made on the current focus before anything else reacts,
   * and a wrapped Tab is ``preventDefault``-ed cleanly.
   */
  function onKeydown(e) {
    if (e.key !== 'Tab') return;
    // The topmost active trap owns the Tab. (Modal mutual-exclusion means at
    // most one is open in practice; if two were somehow active, the last in
    // DOM order is the visually-topmost, so it wins.)
    var container = activeContainer();
    if (!container) return;

    var list = focusable(container);
    if (!list.length) {
      // Empty dialog: pin focus to the container so Tab can't escape behind.
      e.preventDefault();
      if (typeof container.focus === 'function') container.focus();
      return;
    }

    var first = list[0];
    var last = list[list.length - 1];
    var active = document.activeElement;

    if (e.shiftKey) {
      // Backward: wrap from first (or from outside the list) to last.
      if (active === first || !container.contains(active)) {
        e.preventDefault();
        last.focus();
      }
    } else {
      // Forward: wrap from last (or from outside the list) to first.
      if (active === last || !container.contains(active)) {
        e.preventDefault();
        first.focus();
      }
    }
  }

  // ── Visibility tracking ─────────────────────────────────────────────
  // ``shown`` is the set of trap elements considered visible as of the last
  // observer tick. A WeakSet keyed on the element nodes themselves so that a
  // removed-from-DOM node (the About modal on close) is GC'd without a leak;
  // we explicitly delete on deactivation too.
  var shown = new WeakSet();

  function activeContainer() {
    // Return the last (DOM-order) visible, active trap. Cheap: there are at
    // most a handful of [data-focus-trap] elements on the page.
    var traps = document.querySelectorAll('[data-focus-trap]');
    var found = null;
    for (var i = 0; i < traps.length; i++) {
      if (traps[i].__focusTrapActive && isRendered(traps[i])) {
        found = traps[i];
      }
    }
    return found;
  }

  /**
   * Reconcile every ``[data-focus-trap]`` in the DOM against its previous
   * visibility: newly-visible ⇒ activate, newly-hidden / removed ⇒
   * deactivate. Runs on every relevant mutation; the per-container active
   * flag makes repeated calls idempotent so a burst of mutations costs only
   * one activate/deactivate per real transition.
   */
  function reconcile() {
    var traps = document.querySelectorAll('[data-focus-trap]');
    var seen = [];
    for (var i = 0; i < traps.length; i++) {
      var el = traps[i];
      seen.push(el);
      var visible = isRendered(el);
      var was = shown.has(el);
      if (visible && !was) {
        shown.add(el);
        activate(el);
      } else if (!visible && was) {
        shown.delete(el);
        deactivate(el);
      }
    }
    // A trap removed from the DOM entirely (About modal closed) won't appear
    // in ``traps`` above, so its deactivate() must be driven by the
    // childList branch of the observer (see below) which calls
    // deactivateDetached() with the removed nodes.
    return seen;
  }

  /**
   * Handle trap elements removed from the DOM (e.g. the About modal, which
   * HTMX deletes wholesale on close rather than hiding via CSS). The
   * MutationObserver hands us the removed nodes; we deactivate any that were
   * tracked traps (or contained one) so focus is restored to the trigger.
   */
  function deactivateDetached(removedNodes) {
    for (var i = 0; i < removedNodes.length; i++) {
      var node = removedNodes[i];
      if (node.nodeType !== 1) continue; // elements only
      var candidates = [];
      if (node.hasAttribute && node.hasAttribute('data-focus-trap')) {
        candidates.push(node);
      }
      if (node.querySelectorAll) {
        var nested = node.querySelectorAll('[data-focus-trap]');
        for (var j = 0; j < nested.length; j++) candidates.push(nested[j]);
      }
      for (var k = 0; k < candidates.length; k++) {
        var el = candidates[k];
        if (el.__focusTrapActive) {
          shown.delete(el);
          deactivate(el);
        }
      }
    }
  }

  function init() {
    // Capture-phase keydown so the wrap decision precedes overlay handlers.
    document.addEventListener('keydown', onKeydown, true);

    // Observe attribute flips that toggle CSS visibility (Alpine x-show sets
    // inline ``display``; x-cloak removal changes ``class``/attributes) and
    // node insert/remove (HTMX swaps for the About modal).
    var observer = new MutationObserver(function (mutations) {
      var sawChildList = false;
      for (var i = 0; i < mutations.length; i++) {
        var m = mutations[i];
        if (m.type === 'childList' && m.removedNodes && m.removedNodes.length) {
          deactivateDetached(m.removedNodes);
          sawChildList = true;
        } else if (m.type === 'childList' && m.addedNodes && m.addedNodes.length) {
          sawChildList = true;
        }
      }
      // One reconcile pass per mutation batch covers attribute-driven
      // show/hide and newly-inserted visible traps. (Removals are handled
      // above because a detached node is no longer queryable.)
      reconcile();
      // ``sawChildList`` is referenced to keep the branch meaningful to
      // readers / linters even though reconcile() is unconditional.
      void sawChildList;
    });

    observer.observe(document.documentElement, {
      subtree: true,
      childList: true,
      attributes: true,
      attributeFilter: ['style', 'class', 'hidden', 'x-cloak'],
    });

    // Initial pass: a trap that is somehow already visible at load (e.g. the
    // About standalone /about page renders the modal inline) is activated.
    reconcile();
  }

  // Public surface — exposed so tests / future flows can drive or inspect
  // the trap without poking the observer. Mirrors the ToastBus / Palette /
  // Help pattern in mojica.js.
  window.FocusTrap = {
    activate: activate,
    deactivate: deactivate,
    focusable: focusable,
    refresh: reconcile,
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
