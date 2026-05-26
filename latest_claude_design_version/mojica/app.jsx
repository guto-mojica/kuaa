// Cinemateca Mojica — Mojica hybrid shell (A's editorial typography
// applied to C's three-pane interconnected architecture, frames
// dialed up so they breathe). Single React component, scoped CSS.

const M_PALETTE = {
  ink:        '#0E1014',
  pane:       '#11141A',
  surface:    '#161A20',
  raised:     '#1B1F26',
  hairline:   '#262A33',
  hairline2:  '#363B45',
  paper:      '#ECE6D8',
  paperDim:   '#B8B0A0',
  muted:      '#7D7568',
  faint:      '#54504A',
  faintCool:  '#4F555E',
  accent:     '#DC462A',  // Brazilian vermilion
  accentDim:  '#8C2E1B',
  accentBg:   'rgba(220,70,42,0.10)',
  accentBg2:  'rgba(220,70,42,0.05)',
  gold:       '#C7A55C',
  good:       '#6FBE94',
  warn:       '#D9A85C',
};

const M_CSS = `
.m-app, .m-app * { box-sizing: border-box; }
.m-app *::-webkit-scrollbar { width: 8px; height: 8px; }
.m-app *::-webkit-scrollbar-thumb { background: ${M_PALETTE.hairline2}; }
.m-app *::-webkit-scrollbar-track { background: transparent; }
.m-app {
  --ink: ${M_PALETTE.ink};
  --pane: ${M_PALETTE.pane};
  --surface: ${M_PALETTE.surface};
  --raised: ${M_PALETTE.raised};
  --line: ${M_PALETTE.hairline};
  --line2: ${M_PALETTE.hairline2};
  --paper: ${M_PALETTE.paper};
  --paper-dim: ${M_PALETTE.paperDim};
  --muted: ${M_PALETTE.muted};
  --faint: ${M_PALETTE.faint};
  --accent: ${M_PALETTE.accent};
  --accent-dim: ${M_PALETTE.accentDim};
  --accent-bg: ${M_PALETTE.accentBg};
  --accent-bg2: ${M_PALETTE.accentBg2};
  --gold: ${M_PALETTE.gold};
  --good: ${M_PALETTE.good};
  --warn: ${M_PALETTE.warn};
  --serif: 'Newsreader', 'Source Serif 4', Georgia, serif;
  --sans: 'Geist', system-ui, sans-serif;
  --mono: 'JetBrains Mono', 'Courier New', monospace;
  display: grid;
  grid-template-rows: 30px 44px 1fr 26px;
  height: 100vh; width: 100vw;
  background: var(--ink); color: var(--paper);
  font-family: var(--sans); font-size: 13px; line-height: 1.55;
  letter-spacing: -0.005em;
  font-feature-settings: 'ss01' on, 'cv11' on;
  -webkit-font-smoothing: antialiased;
  overflow: hidden;
}

/* ─── TOP STATUS ────────────────────────────────────────────────────── */
.m-topbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 22px; border-bottom: 1px solid var(--line);
  background: var(--pane);
}
.m-topbar .left { display:flex; align-items:center; gap: 18px; }
.m-topbar .brand {
  display:flex; align-items:center; gap: 9px;
  font-family: var(--serif); font-size: 14.5px; font-weight: 400;
  color: var(--paper); letter-spacing: -0.005em;
}
.m-topbar .brand-sub {
  font-family: var(--mono); font-size: 9.5px;
  letter-spacing: 0.18em; text-transform: uppercase;
  color: var(--muted); padding-left: 12px;
  border-left: 1px solid var(--line2);
}
.m-topbar .crumb {
  font-family: var(--mono); font-size: 10.5px;
  letter-spacing: 0.04em; color: var(--muted);
}
.m-topbar .crumb .sep { color: var(--faint); margin: 0 8px; }
.m-topbar .crumb .cur { color: var(--paper); }
.m-topbar .right {
  display:flex; align-items:center; gap: 22px;
  font-family: var(--mono); font-size: 10px;
  letter-spacing: 0.06em; color: var(--muted);
  font-variant-numeric: tabular-nums;
}
.m-topbar .right .v { color: var(--paper); }
.m-topbar .right .dot {
  display:inline-block; width:6px; height:6px; background: var(--good);
  margin-right: 6px; vertical-align: 1px;
}
.m-topbar .right .dot.warn { background: var(--warn); }

/* ─── TAB BAR ───────────────────────────────────────────────────────── */
.m-tabbar {
  display: flex; align-items: stretch; gap: 0;
  padding: 0 22px; border-bottom: 1px solid var(--line);
}
.m-tab {
  display: flex; align-items: center; gap: 9px;
  padding: 0 18px; cursor: pointer;
  font-family: var(--sans); font-size: 12.5px; color: var(--muted);
  position: relative; letter-spacing: -0.005em;
}
.m-tab + .m-tab { border-left: 1px solid var(--line); }
.m-tab .dot { width:5px; height:5px; border-radius:50%; background: transparent; }
.m-tab.active { color: var(--paper); }
.m-tab.active .dot { background: var(--accent); }
.m-tab .k {
  font-family: var(--mono); font-size: 9.5px; color: var(--faint);
  margin-left: 4px;
}
.m-tab .pip {
  font-family: var(--mono); font-size: 9.5px;
  color: var(--gold); padding: 1px 5px;
  border: 1px solid var(--line2); border-radius: 9px;
  margin-left: 2px; font-variant-numeric: tabular-nums;
}
.m-tab:hover { color: var(--paper-dim); }
.m-tabbar .gap { flex: 1; }
.m-tabbar .ctx {
  display:flex; align-items:center; gap: 16px;
  font-family: var(--mono); font-size: 10px; color: var(--muted);
  letter-spacing: 0.04em;
}
.m-tabbar .ctx b { color: var(--paper); font-weight: 400; font-variant-numeric: tabular-nums; }

/* ─── BODY 3 PANES ──────────────────────────────────────────────────── */
.m-body {
  display: grid;
  grid-template-columns: 282px 1fr 380px;
  overflow: hidden;
  min-height: 0;
}

/* ─── LEFT PANE — Acervo / Films ────────────────────────────────────── */
.m-lp {
  border-right: 1px solid var(--line);
  display:flex; flex-direction: column; overflow: hidden;
  background: var(--ink);
}
.m-lp .head {
  padding: 16px 18px 12px;
  display:flex; align-items: baseline; justify-content: space-between;
  font-family: var(--mono); font-size: 9.5px;
  letter-spacing: 0.18em; text-transform: uppercase; color: var(--muted);
}
.m-lp .head .ct { color: var(--faint); }
.m-lp .filter {
  margin: 0 14px 10px; padding: 7px 10px;
  background: var(--surface); border: 1px solid var(--line);
  display: flex; align-items: center; gap: 8px;
  font-family: var(--mono); font-size: 11px; color: var(--muted);
}
.m-lp .filter input {
  flex: 1; background: transparent; border: none; outline: none;
  font: inherit; color: var(--paper);
}
.m-lp .filter .icon { color: var(--faint); }

.m-films { padding: 4px 8px 8px; overflow-y: auto; }
.m-film {
  display:grid; grid-template-columns: 36px 1fr auto;
  align-items: baseline; gap: 8px;
  padding: 9px 12px 9px 10px;
  cursor: pointer; position: relative;
  border-left: 2px solid transparent;
}
.m-film:hover { background: var(--surface); }
.m-film.active { background: var(--accent-bg); border-left-color: var(--accent); }
.m-film .y {
  font-family: var(--mono); font-size: 10px;
  color: var(--muted); letter-spacing: 0.04em;
  font-variant-numeric: tabular-nums;
}
.m-film .t {
  font-family: var(--serif); font-size: 14.5px; color: var(--paper-dim);
  letter-spacing: -0.005em; line-height: 1.2;
}
.m-film.active .t { color: var(--paper); }
.m-film .n {
  font-family: var(--mono); font-size: 10px; color: var(--faint);
  font-variant-numeric: tabular-nums;
}
.m-film .director {
  grid-column: 2 / 4;
  font-family: var(--sans); font-size: 11px; color: var(--muted);
  margin-top: 1px;
}
.m-film.proc .t { color: var(--gold); font-style: italic; }
.m-film.proc .n,
.m-film.proc .director { color: var(--gold); }

/* match bar — how many scenes from this film match current query */
.m-film .matchrow {
  grid-column: 2 / 4;
  display: flex; align-items: center; gap: 8px; margin-top: 6px;
  font-family: var(--mono); font-size: 9.5px;
  color: var(--muted); font-variant-numeric: tabular-nums;
}
.m-film .matchrow .bar {
  flex: 1; height: 3px; background: var(--line); position: relative;
}
.m-film .matchrow .bar::before {
  content: ''; position: absolute; left: 0; top: 0; bottom: 0;
  width: var(--p, 0%); background: var(--accent-dim);
}
.m-film .matchrow .ct { color: var(--paper-dim); }
.m-film.has-sel .matchrow .bar::before { background: var(--accent); }
.m-film .selptr {
  grid-column: 2 / 4;
  font-family: var(--mono); font-size: 10.5px; color: var(--accent);
  margin-top: 5px; display:none;
  letter-spacing: 0.04em;
}
.m-film.has-sel .selptr { display: block; }
.m-film.has-sel .selptr .gly { color: var(--accent); margin-right: 4px; }

.m-lp .divider {
  height: 1px; background: var(--line); margin: 8px 0 0;
}
.m-lp .filters-section {
  padding: 14px 18px 6px;
  font-family: var(--mono); font-size: 9.5px;
  letter-spacing: 0.18em; text-transform: uppercase; color: var(--muted);
  display:flex; align-items:baseline; justify-content:space-between;
}
.m-lp .filters-section a { color: var(--faint); cursor: pointer; text-decoration: none; }
.m-lp .filters-section a:hover { color: var(--accent); }
.m-lp .tagcloud {
  padding: 0 14px 14px;
  display: flex; flex-wrap: wrap; gap: 4px 6px;
}
.m-lp .tagcloud .tg {
  font-family: var(--mono); font-size: 10px; padding: 2px 7px;
  color: var(--paper-dim); border: 1px solid var(--line2);
  cursor: pointer;
}
.m-lp .tagcloud .tg.active {
  color: var(--accent); border-color: var(--accent-dim);
  background: var(--accent-bg2);
}
.m-lp .tagcloud .tg:hover { border-color: var(--muted); }

.m-lp .lp-foot {
  margin-top: auto;
  padding: 12px 18px 14px;
  border-top: 1px solid var(--line);
  font-family: var(--mono); font-size: 10px;
  color: var(--muted); letter-spacing: 0.04em;
  display: grid; grid-template-columns: 1fr auto; row-gap: 4px;
}
.m-lp .lp-foot .v { color: var(--paper); text-align: right; font-variant-numeric: tabular-nums; }
.m-lp .lp-foot .row-foot {
  grid-column: 1 / 3;
  display:flex; align-items: center; justify-content: space-between;
  margin-top: 6px; padding-top: 8px;
  border-top: 1px solid var(--line);
  font-size: 9.5px; letter-spacing: 0.14em; text-transform: uppercase;
}
.m-lp .lp-foot .row-foot .on { color: var(--paper); }
.m-lp .lp-foot .row-foot a { color: var(--paper-dim); text-decoration:none; cursor: pointer; }
.m-lp .lp-foot .row-foot a:hover { color: var(--accent); }

/* ─── CENTER PANE — Search + Scenes ─────────────────────────────────── */
.m-cp {
  display:flex; flex-direction:column; overflow:hidden; min-width: 0;
  background: var(--ink);
}
.m-search {
  padding: 22px 30px 14px; border-bottom: 1px solid var(--line);
}
.m-qrow {
  display:flex; align-items:center; gap: 14px;
  padding: 6px 0 14px;
  border-bottom: 1px solid var(--line2);
}
.m-qrow .pre {
  font-family: var(--mono); font-size: 11px;
  letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted);
}
.m-qrow .q {
  flex: 1; background: transparent; border: none; outline: none;
  font-family: var(--mono); font-size: 20px; color: var(--paper);
  letter-spacing: -0.005em; caret-color: var(--accent);
}
.m-qrow .q::placeholder { color: var(--faint); font-family: var(--serif); font-style: italic; }
.m-qrow .submit {
  font-family: var(--mono); font-size: 10.5px;
  letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--muted); background: transparent;
  border: 1px solid var(--line2); padding: 7px 12px;
  cursor: pointer; display:flex; align-items: center; gap: 8px;
}
.m-qrow .submit:hover { color: var(--accent); border-color: var(--accent-dim); }
.m-qrow .submit .k { color: var(--faint); }

.m-modes-row {
  display:flex; align-items:center; justify-content: space-between;
  padding-top: 12px; gap: 24px; flex-wrap: wrap;
}
.m-modes { display:flex; gap: 24px; }
.m-mode {
  font-family: var(--mono); font-size: 10.5px;
  letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted);
  cursor: pointer; display:flex; align-items:center; gap: 7px;
}
.m-mode .g { color: var(--faint); }
.m-mode.active { color: var(--paper); }
.m-mode.active .g { color: var(--accent); }
.m-knobs { display:flex; gap: 18px; flex-wrap: wrap; align-items: center; }
.m-knob {
  font-family: var(--mono); font-size: 10.5px;
  color: var(--muted); display:flex; align-items: center; gap: 7px;
  letter-spacing: 0.04em;
}
.m-knob .k {
  color: var(--faint); text-transform: uppercase; letter-spacing: 0.16em;
}
.m-knob .v { color: var(--paper-dim); font-variant-numeric: tabular-nums; }
.m-knob .v.acc { color: var(--accent); }

.m-caption {
  display: grid; grid-template-columns: auto 1fr auto auto;
  align-items: baseline; gap: 18px;
  padding: 14px 30px 12px;
}
.m-caption .head {
  font-family: var(--serif); font-style: italic; font-size: 16px;
  color: var(--paper); letter-spacing: -0.005em;
}
.m-caption .head em {
  color: var(--accent); font-style: italic; font-weight: 400;
}
.m-caption .lab {
  font-family: var(--mono); font-size: 10px;
  letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted);
}
.m-caption .lab b { color: var(--paper); font-weight: 400; }

/* GRID */
.m-grid {
  flex: 1; overflow-y: auto;
  padding: 6px 30px 30px;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(310px, 1fr));
  column-gap: 24px; row-gap: 30px;
  align-content: start;
}
.m-scene {
  display:flex; flex-direction: column; gap: 10px;
  cursor: pointer; position: relative;
  transition: transform .15s;
}
.m-scene .kf {
  width: 100%; aspect-ratio: 4/3;
  background: var(--surface) center/cover no-repeat;
  filter: contrast(1.04) brightness(0.96);
  position: relative; transition: filter .2s;
}
.m-scene:hover .kf { filter: contrast(1.06) brightness(1.0); }
.m-scene .kf::after {
  content: ''; position: absolute; inset: 0;
  outline: 2px solid transparent; outline-offset: -2px;
  transition: outline-color .15s;
  pointer-events: none;
}
.m-scene.selected .kf::after { outline-color: var(--accent); }
.m-scene .badge {
  position: absolute; top: 8px; left: 8px;
  font-family: var(--mono); font-size: 9.5px; color: var(--paper);
  padding: 2px 6px; background: rgba(14,16,20,0.78);
  letter-spacing: 0.06em;
  backdrop-filter: blur(3px);
}
.m-scene .rank {
  position: absolute; top: 8px; right: 8px;
  font-family: var(--mono); font-size: 9.5px; color: var(--paper);
  padding: 2px 6px; background: rgba(14,16,20,0.78);
  letter-spacing: 0.04em; font-variant-numeric: tabular-nums;
  backdrop-filter: blur(3px);
}
.m-scene.selected .rank { color: var(--accent); }

.m-scene .meta-top {
  display:flex; align-items: baseline; justify-content: space-between;
  padding: 4px 0 6px; border-bottom: 1px solid var(--line);
}
.m-scene .film-attr {
  font-family: var(--serif); font-style: italic; font-size: 16px;
  color: var(--paper); letter-spacing: -0.005em; line-height: 1.15;
}
.m-scene .film-attr .yr {
  font-family: var(--mono); font-style: normal; font-size: 10.5px;
  color: var(--muted); margin-left: 7px; letter-spacing: 0.04em;
}
.m-scene .score {
  font-family: var(--mono); font-size: 11px; color: var(--gold);
  font-variant-numeric: tabular-nums; letter-spacing: 0;
}
.m-scene.selected .score { color: var(--accent); }
.m-scene .ids {
  font-family: var(--mono); font-size: 10px;
  letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted);
  font-variant-numeric: tabular-nums;
}
.m-scene .desc {
  font-family: var(--sans); font-size: 13px; line-height: 1.5;
  color: var(--paper-dim); text-wrap: pretty;
}
.m-scene .tags {
  font-family: var(--mono); font-size: 10px; color: var(--faint);
  letter-spacing: 0.04em;
}
.m-scene .tags .t + .t::before { content: ' · '; color: var(--faint); }
.m-scene .tags .t.m { color: var(--accent); }

/* ─── RIGHT PANE — Inspector ────────────────────────────────────────── */
.m-rp {
  border-left: 1px solid var(--line);
  display:flex; flex-direction: column; overflow: hidden;
  background: var(--pane);
}
.m-rp .head {
  display:flex; align-items: baseline; justify-content: space-between;
  padding: 14px 18px 10px;
  font-family: var(--mono); font-size: 9.5px;
  letter-spacing: 0.18em; text-transform: uppercase; color: var(--muted);
  border-bottom: 1px solid var(--line);
}
.m-rp .head .ct {
  font-family: var(--mono); color: var(--paper);
}
.m-rp .head .ct .accent { color: var(--accent); }

.m-rp .inner { padding: 18px 22px 22px; overflow-y: auto; flex: 1; }
.m-insp-kf {
  width: 100%; aspect-ratio: 4/3;
  background: var(--ink) center/cover no-repeat;
  filter: contrast(1.05) brightness(0.97);
  outline: 1px solid var(--line2); outline-offset: -1px;
}
.m-insp-film {
  font-family: var(--serif); font-size: 28px; font-weight: 400;
  color: var(--paper); margin-top: 14px;
  letter-spacing: -0.012em; line-height: 1.1;
}
.m-insp-film .yr {
  font-family: var(--mono); font-size: 13px; color: var(--muted);
  margin-left: 10px; letter-spacing: 0.04em; font-weight: 400;
}
.m-insp-attr {
  font-family: var(--sans); font-size: 12px; color: var(--muted);
  margin-top: 4px;
}
.m-insp-attr .v { color: var(--paper-dim); }
.m-insp-ids {
  font-family: var(--mono); font-size: 10.5px;
  letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--accent); margin-top: 10px;
  font-variant-numeric: tabular-nums;
}

.m-insp-section {
  display:flex; align-items: baseline; justify-content: space-between;
  margin-top: 22px; padding-bottom: 6px;
  border-bottom: 1px solid var(--line);
  font-family: var(--mono); font-size: 9.5px;
  letter-spacing: 0.18em; text-transform: uppercase; color: var(--muted);
}
.m-insp-section .ct { color: var(--faint); }

.m-insp-desc {
  margin-top: 10px;
  font-family: var(--sans); font-size: 13px; line-height: 1.55;
  color: var(--paper-dim); text-wrap: pretty;
}

.m-signals { display:grid; gap: 8px; margin-top: 12px; }
.m-sig {
  display:grid; grid-template-columns: 78px 1fr 42px;
  align-items: center; gap: 12px;
  font-family: var(--mono); font-size: 11px; color: var(--paper-dim);
  font-variant-numeric: tabular-nums;
}
.m-sig .lab {
  color: var(--muted); letter-spacing: 0.06em;
  text-transform: uppercase; font-size: 10px;
}
.m-sig .track { height: 4px; background: var(--line2); position: relative; }
.m-sig .track::before {
  content: ''; position: absolute; left: 0; top: 0; bottom: 0;
  width: var(--p); background: var(--accent-dim);
}
.m-sig .v { text-align: right; color: var(--paper); }
.m-sig.fused .track::before { background: var(--accent); }
.m-sig.fused .lab { color: var(--accent); }

.m-insp-tags { display:flex; flex-wrap: wrap; gap: 5px; margin-top: 12px; }
.m-insp-tag {
  font-family: var(--mono); font-size: 10.5px;
  padding: 2px 8px; color: var(--paper-dim);
  border: 1px solid var(--line2); cursor: pointer;
  letter-spacing: 0.02em;
}
.m-insp-tag:hover { border-color: var(--muted); }
.m-insp-tag.m {
  color: var(--accent); border-color: var(--accent-dim);
  background: var(--accent-bg2);
}

/* Visual rhymes preview */
.m-rhymes {
  display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 8px; margin-top: 12px;
}
.m-rhyme { cursor: pointer; display:flex; flex-direction: column; gap: 4px; }
.m-rhyme .rkf {
  width: 100%; aspect-ratio: 4/3;
  background: var(--surface) center/cover no-repeat;
  filter: contrast(1.05) brightness(0.95);
  outline: 1px solid transparent; outline-offset: -1px;
  transition: outline-color .15s;
}
.m-rhyme:hover .rkf { outline-color: var(--accent); }
.m-rhyme .rfilm {
  font-family: var(--serif); font-style: italic; font-size: 11.5px;
  color: var(--paper-dim); line-height: 1.1;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.m-rhyme .rs {
  font-family: var(--mono); font-size: 9.5px; color: var(--muted);
  font-variant-numeric: tabular-nums; letter-spacing: 0.02em;
}
.m-rhyme .rs b { color: var(--gold); font-weight: 400; }

.m-actions {
  display:flex; flex-direction: column; gap: 6px;
  margin-top: 20px;
}
.m-action {
  display:flex; align-items: center; justify-content: space-between;
  padding: 9px 12px;
  font-family: var(--sans); font-size: 12.5px;
  border: 1px solid var(--line2); color: var(--paper-dim);
  cursor: pointer; transition: border-color .12s, color .12s, background .12s;
}
.m-action:hover { border-color: var(--accent-dim); color: var(--paper); }
.m-action.primary {
  background: var(--accent); color: #1A0F0C; border-color: var(--accent);
  font-weight: 500;
}
.m-action.primary:hover { background: var(--accent); }
.m-action .key {
  font-family: var(--mono); font-size: 10px; color: var(--muted);
  letter-spacing: 0.06em;
}
.m-action.primary .key { color: rgba(26,15,12,0.65); }

/* ─── BOTTOM STATUS ─────────────────────────────────────────────────── */
.m-botbar {
  display:flex; align-items: center; justify-content: space-between;
  padding: 0 22px;
  border-top: 1px solid var(--line);
  background: var(--pane);
  font-family: var(--mono); font-size: 10.5px; color: var(--muted);
  letter-spacing: 0.04em;
}
.m-botbar .mode {
  background: var(--accent); color: #1A0F0C;
  padding: 0 9px; height: 18px;
  display:inline-flex; align-items: center; margin-right: 14px;
  font-weight: 500; letter-spacing: 0.1em; font-size: 9.5px;
  text-transform: uppercase;
}
.m-botbar .keys { display:flex; align-items: center; gap: 16px; }
.m-botbar .keys .k b { color: var(--accent); font-weight: 400; margin-right: 4px; }
.m-botbar .keys .k { color: var(--paper-dim); }
.m-botbar .right { display:flex; align-items: center; gap: 18px; }
.m-botbar .right .v { color: var(--paper); }
.m-botbar .right .ok { color: var(--good); }

/* Helpers */
.m-x { color: var(--faint); }
`;

function Mark() {
  return (
    <svg width="20" height="20" viewBox="0 0 22 22" fill="none">
      <rect x="0.5" y="2.5" width="21" height="17" stroke={M_PALETTE.paper} strokeWidth="1"/>
      <rect x="2" y="4" width="3" height="3" fill={M_PALETTE.paper}/>
      <rect x="2" y="9.5" width="3" height="3" fill={M_PALETTE.paper}/>
      <rect x="2" y="15" width="3" height="3" fill={M_PALETTE.paper}/>
      <rect x="17" y="4" width="3" height="3" fill={M_PALETTE.paper}/>
      <rect x="17" y="9.5" width="3" height="3" fill={M_PALETTE.paper}/>
      <rect x="17" y="15" width="3" height="3" fill={M_PALETTE.paper}/>
      <rect x="7" y="5" width="8" height="12" fill={M_PALETTE.accent}/>
    </svg>
  );
}

function Mojica() {
  const films = window.FILMS;
  const results = window.RESULTS;
  const filmsById = Object.fromEntries(films.map(f => [f.id, f]));
  const [sel, setSel] = React.useState(0);  // selected result index
  const [query, setQuery] = React.useState('duas pessoas conversando ao ar livre');

  // Inject CSS once
  React.useEffect(() => {
    if (!document.getElementById('m-css')) {
      const s = document.createElement('style');
      s.id = 'm-css'; s.textContent = M_CSS;
      document.head.appendChild(s);
    }
  }, []);

  // Keyboard nav: j/k or ↑/↓ to move selection, Enter to "open"
  React.useEffect(() => {
    const onKey = (e) => {
      if (document.activeElement && document.activeElement.tagName === 'INPUT') return;
      if (e.key === 'j' || e.key === 'ArrowDown') {
        e.preventDefault();
        setSel(s => Math.min(results.length - 1, s + 1));
      } else if (e.key === 'k' || e.key === 'ArrowUp') {
        e.preventDefault();
        setSel(s => Math.max(0, s - 1));
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [results.length]);

  const r = results[sel];
  const f = filmsById[r.film];

  // Per-film match counts in current results
  const matchCounts = {};
  results.forEach(rr => { matchCounts[rr.film] = (matchCounts[rr.film] || 0) + 1; });
  const maxMatch = Math.max(...Object.values(matchCounts), 1);

  // Hybrid retrieval signals (derived for the mock)
  const sigSem  = Math.min(0.96, r.score + 0.06);
  const sigBm25 = Math.max(0.04, r.score - 0.52);
  const sigRk   = Math.min(0.96, r.score - 0.01);
  const sigFu   = r.score;

  // Visual rhymes preview: pick 3 scenes from OTHER films, sorted by score
  const rhymes = results
    .filter((x, i) => x.film !== r.film && i !== sel)
    .slice(0, 3)
    .map((x, i) => ({...x, sim: (0.94 - i*0.04).toFixed(2)}));

  // Common tags from current matched scenes that the selected scene shares
  const matchedTags = new Set();
  results.forEach(rr => rr.tags.forEach(t => {
    if (rr.tags.length && r.tags.includes(t)) matchedTags.add(t);
  }));

  return (
    <div className="m-app">
      {/* TOP STATUS */}
      <div className="m-topbar">
        <div className="left">
          <span className="brand"><Mark /> Cinemateca Mojica</span>
          <span className="brand-sub">Acervo Digital · v1.0</span>
          <span className="crumb">
            acervo <span className="sep">/</span>
            <span className="cur">buscar</span> <span className="sep">·</span>
            multimodal: <span style={{color: M_PALETTE.paper}}>texto</span> <span className="sep">·</span>
            escopo: <span style={{color: M_PALETTE.paper}}>acervo inteiro</span>
          </span>
        </div>
        <div className="right">
          <span><span className="dot"></span>índice ok</span>
          <span><span className="dot warn"></span>aruanda · 78%</span>
          <span>CLIP-L · md2</span>
          <span>PT · <span className="v">EN</span></span>
        </div>
      </div>

      {/* TAB BAR */}
      <div className="m-tabbar">
        <div className="m-tab active"><span className="dot"></span>Buscar <span className="k">⌘1</span></div>
        <div className="m-tab"><span className="dot"></span>Cenas <span className="k">⌘2</span></div>
        <div className="m-tab"><span className="dot"></span>Anotar <span className="k">⌘3</span></div>
        <div className="m-tab"><span className="dot"></span>Rimas visuais <span className="k">⌘4</span></div>
        <div className="m-tab"><span className="dot"></span>Processamento <span className="pip">1</span></div>
        <div className="gap"></div>
        <div className="ctx">
          <span>cenas selecionadas <b>1</b></span>
          <span>resultados <b>9</b></span>
          <span>filmes <b>6/6</b></span>
        </div>
      </div>

      {/* BODY */}
      <div className="m-body">

        {/* LEFT PANE — Acervo */}
        <aside className="m-lp">
          <div className="head"><span>Acervo · Programa 2026</span><span className="ct">06 filmes</span></div>
          <div className="filter">
            <span className="icon">⌕</span>
            <input placeholder="Filtrar acervo…" />
            <span className="m-x">⌘/</span>
          </div>
          <div className="m-films">
            {films.map(film => {
              const cnt = matchCounts[film.id] || 0;
              const isProc = film.id === 'aruanda';
              const hasSel = film.id === r.film;
              const pct = cnt > 0 ? (cnt / maxMatch * 100) : 0;
              return (
                <div key={film.id}
                     className={'m-film' + (hasSel ? ' active has-sel' : '') + (isProc ? ' proc' : '')}>
                  <span className="y">{film.year}</span>
                  <span className="t">{film.title}</span>
                  <span className="n">{film.scenes}</span>
                  <span className="director">{film.director}{isProc && ' · processando'}</span>
                  <div className="matchrow">
                    <span>{cnt > 0 ? cnt : '–'}</span>
                    <span className="bar" style={{'--p': pct + '%'}}></span>
                    <span className="ct">{cnt > 0 ? 'match' : ''}</span>
                  </div>
                  <span className="selptr">
                    <span className="gly">↳</span>cena {String(r.cena).padStart(3,'0')} · {r.tc}
                  </span>
                </div>
              );
            })}
          </div>
          <div className="filters-section">
            <span>Tags · ativas</span>
            <a>limpar</a>
          </div>
          <div className="tagcloud">
            <span className="tg active">+ duas-pessoas</span>
            <span className="tg active">+ exterior</span>
            <span className="tg">dia</span>
            <span className="tg">rural-field</span>
            <span className="tg">interior</span>
            <span className="tg">noite</span>
            <span className="tg">close-up</span>
            <span className="tg">crowd</span>
            <span className="tg">title-card</span>
            <span className="tg">wagon</span>
            <span className="tg">horse-rider</span>
            <span className="tg">sertão</span>
          </div>

          <div className="lp-foot">
            <span>cenas</span><span className="v">1.588</span>
            <span>runtime</span><span className="v">8h 54m</span>
            <span>embeddings</span><span className="v">CLIP-L/14</span>
            <span>descrição</span><span className="v">moondream-2</span>
            <div className="row-foot">
              <span><span className="on">PT</span> · EN</span>
              <a>Sobre</a>
            </div>
          </div>
        </aside>

        {/* CENTER PANE — Search + Scenes */}
        <section className="m-cp">
          <div className="m-search">
            <div className="m-qrow">
              <span className="pre">› buscar</span>
              <input className="q" value={query} onChange={e => setQuery(e.target.value)} />
              <button className="submit">Executar <span className="k">⏎</span></button>
            </div>
            <div className="m-modes-row">
              <div className="m-modes">
                <span className="m-mode active"><span className="g">●</span>Texto</span>
                <span className="m-mode"><span className="g">○</span>Imagem</span>
                <span className="m-mode"><span className="g">○</span>Trilha</span>
                <span className="m-mode"><span className="g">○</span>Multimodal</span>
              </div>
              <div className="m-knobs">
                <span className="m-knob"><span className="k">Escopo</span><span className="v">acervo · 06 filmes</span></span>
                <span className="m-knob"><span className="k">Híbrido</span><span className="v">sem 0.70 · bm25 0.30</span></span>
                <span className="m-knob"><span className="k">Rerank</span><span className="v acc">on</span></span>
                <span className="m-knob"><span className="k">MMR</span><span className="v">λ 0.50</span></span>
                <span className="m-knob"><span className="k">k</span><span className="v">09</span></span>
              </div>
            </div>
          </div>

          <div className="m-caption">
            <span className="head">Nove cenas. Seis filmes. <em>Sem ⊕ BM25 ⊕ rerank.</em></span>
            <span></span>
            <span className="lab">231 ms</span>
            <span className="lab"><b>↓</b> afinidade</span>
          </div>

          <div className="m-grid">
            {results.map((rr, i) => {
              const ff = filmsById[rr.film];
              return (
                <article key={rr.id}
                         className={'m-scene' + (i === sel ? ' selected' : '')}
                         onClick={() => setSel(i)}>
                  <div className="kf" style={{backgroundImage:`url(${rr.kf})`}}>
                    <span className="badge">{ff.title.toUpperCase()} · {ff.year}</span>
                    <span className="rank">#{String(i+1).padStart(2,'0')} · {rr.score.toFixed(3)}</span>
                  </div>
                  <div className="meta-top">
                    <span className="film-attr">{ff.title}<span className="yr">{ff.year}</span></span>
                    <span className="score">{rr.score.toFixed(3)}</span>
                  </div>
                  <span className="ids">cena {String(rr.cena).padStart(3,'0')} · {rr.tc}</span>
                  <p className="desc">{rr.desc}</p>
                  <div className="tags">
                    {rr.tags.slice(0,5).map((t,j) => (
                      <span key={j} className={'t' + (t==='duas-pessoas' || t==='exterior' ? ' m' : '')}>{t}</span>
                    ))}
                  </div>
                </article>
              );
            })}
          </div>
        </section>

        {/* RIGHT PANE — Inspector */}
        <aside className="m-rp">
          <div className="head">
            <span>Inspector</span>
            <span className="ct">cena <span className="accent">{String(r.cena).padStart(3,'0')}</span> · {String(sel+1).padStart(2,'0')}/09</span>
          </div>
          <div className="inner">
            <div className="m-insp-kf" style={{backgroundImage:`url(${r.kf})`}}></div>
            <div className="m-insp-film">
              {f.title}<span className="yr">{f.year}</span>
            </div>
            <div className="m-insp-attr">
              dir. <span className="v">{f.director}</span> · {f.runtime} min · {f.country}
            </div>
            <div className="m-insp-ids">cena {String(r.cena).padStart(3,'0')} · {r.tc}</div>

            <div className="m-insp-section">
              <span>descrição · moondream-2</span>
              <span className="ct">↶ editar</span>
            </div>
            <p className="m-insp-desc">{r.desc}</p>

            <div className="m-insp-section">
              <span>por que este resultado</span>
              <span className="ct">{r.score.toFixed(3)}</span>
            </div>
            <div className="m-signals">
              <div className="m-sig"><span className="lab">semântico</span><span className="track" style={{'--p': `${(sigSem*100).toFixed(0)}%`}}></span><span className="v">{sigSem.toFixed(3)}</span></div>
              <div className="m-sig"><span className="lab">bm25</span><span className="track" style={{'--p': `${(sigBm25*100).toFixed(0)}%`}}></span><span className="v">{sigBm25.toFixed(3)}</span></div>
              <div className="m-sig"><span className="lab">rerank</span><span className="track" style={{'--p': `${(sigRk*100).toFixed(0)}%`}}></span><span className="v">{sigRk.toFixed(3)}</span></div>
              <div className="m-sig fused"><span className="lab">fundido</span><span className="track" style={{'--p': `${(sigFu*100).toFixed(0)}%`}}></span><span className="v">{sigFu.toFixed(3)}</span></div>
            </div>

            <div className="m-insp-section">
              <span>tags · {r.tags.length}</span>
              <span className="ct">+ adicionar</span>
            </div>
            <div className="m-insp-tags">
              {r.tags.map((t,i) => (
                <span key={i} className={'m-insp-tag' + (matchedTags.has(t) && (t==='duas-pessoas'||t==='exterior') ? ' m' : '')}>{t}</span>
              ))}
            </div>

            <div className="m-insp-section">
              <span>rimas visuais · k-nn</span>
              <span className="ct">cross-film</span>
            </div>
            <div className="m-rhymes">
              {rhymes.map((x, i) => {
                const ff = filmsById[x.film];
                return (
                  <div key={i} className="m-rhyme">
                    <div className="rkf" style={{backgroundImage:`url(${x.kf})`}}></div>
                    <span className="rfilm">{ff.title}</span>
                    <span className="rs">{ff.year} · <b>{x.sim}</b></span>
                  </div>
                );
              })}
            </div>

            <div className="m-actions">
              <div className="m-action primary"><span>Abrir cena</span><span className="key">⏎</span></div>
              <div className="m-action"><span>Encontrar rimas visuais</span><span className="key">⌥R</span></div>
              <div className="m-action"><span>Anotar · adicionar tag</span><span className="key">A</span></div>
              <div className="m-action"><span>Copiar timecode</span><span className="key">⌘C</span></div>
            </div>
          </div>
        </aside>

      </div>

      {/* BOTTOM STATUS */}
      <div className="m-botbar">
        <div style={{display:'flex',alignItems:'center'}}>
          <span className="mode">Buscar</span>
          <div className="keys">
            <span className="k"><b>j/k</b> navegar</span>
            <span className="k"><b>⏎</b> abrir</span>
            <span className="k"><b>^F</b> foco busca</span>
            <span className="k"><b>⌘K</b> comandos</span>
            <span className="k"><b>⌥R</b> rimas</span>
          </div>
        </div>
        <div className="right">
          <span>cena <span className="v">{String(sel+1).padStart(2,'0')}/09</span></span>
          <span><span className="ok">●</span> 231 ms</span>
          <span>1.588 cenas</span>
          <span>v1.0.0</span>
        </div>
      </div>
    </div>
  );
}

window.Mojica = Mojica;
