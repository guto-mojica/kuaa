// Mojica · Frame.io branch · shared theme
// Palette, CSS, icon components, mark. Loaded first.

const FX = {
  // surfaces
  bg:        '#0E1014',
  panel:     '#171B22',
  raised:    '#1F242E',
  hover:     '#252B36',
  selected:  '#2A2F3D',
  border:    '#262C36',
  border2:   '#363D49',
  border3:   '#454B58',

  // text
  text:      '#F1F3F7',
  text2:     '#B8BCC7',
  muted:     '#787E8A',
  faint:     '#4E535E',

  // accent — Frame.io 2024 purple/violet
  ac:        '#8B7BD8',
  ac2:       '#A99CE5',
  acDim:     '#5C4FA8',
  acBg:      'rgba(139,123,216,0.14)',
  acBgLow:   'rgba(139,123,216,0.06)',

  // annotation accent — yellow (pins, timestamps, playhead)
  yellow:    '#F5C842',
  yellowDim: '#9C7E1E',
  yellowBg:  'rgba(245,200,66,0.16)',

  // viewer/social accent — pink
  pink:      '#E879C7',
  pinkBg:    'rgba(232,121,199,0.18)',

  // status
  green:     '#5CCB91',
  greenBg:   'rgba(92,203,145,0.12)',
  orange:    '#F59042',
  orangeBg:  'rgba(245,144,66,0.14)',
  red:       '#F56E5A',
  redBg:     'rgba(245,110,90,0.14)',

  // tag/category colors (used as inline pills)
  catCartela:  '#F5C842',
  catDialogo:  '#A99CE5',
  catExterior: '#5CCB91',
  catInterior: '#F59042',
  catTransicao:'#787E8A',
  catTitulo:   '#E879C7',
};

// Film color identity (used for project dots and badges)
const FX_FILM = {
  jeca:       FX.ac,
  limite:     FX.pink,
  rio40:      FX.orange,
  cangaceiro: FX.yellow,
  aruanda:    FX.green,
  pagador:    '#9D7AE8',
};

const FX_CSS = `
.fx-app, .fx-app * { box-sizing: border-box; }
.fx-app *::-webkit-scrollbar { width: 9px; height: 9px; }
.fx-app *::-webkit-scrollbar-thumb { background: ${FX.border2}; border-radius: 5px; }
.fx-app *::-webkit-scrollbar-thumb:hover { background: ${FX.border3}; }
.fx-app *::-webkit-scrollbar-track { background: transparent; }

.fx-app {
  --bg: ${FX.bg};
  --panel: ${FX.panel};
  --raised: ${FX.raised};
  --hover: ${FX.hover};
  --selected: ${FX.selected};
  --bd: ${FX.border};
  --bd2: ${FX.border2};
  --bd3: ${FX.border3};
  --t: ${FX.text};
  --t2: ${FX.text2};
  --muted: ${FX.muted};
  --faint: ${FX.faint};
  --ac: ${FX.ac};
  --ac2: ${FX.ac2};
  --ac-dim: ${FX.acDim};
  --ac-bg: ${FX.acBg};
  --ac-bg-low: ${FX.acBgLow};
  --yellow: ${FX.yellow};
  --yellow-dim: ${FX.yellowDim};
  --yellow-bg: ${FX.yellowBg};
  --pink: ${FX.pink};
  --pink-bg: ${FX.pinkBg};
  --green: ${FX.green};
  --green-bg: ${FX.greenBg};
  --orange: ${FX.orange};
  --orange-bg: ${FX.orangeBg};
  --red: ${FX.red};
  --red-bg: ${FX.redBg};
  --sans: 'Geist', system-ui, -apple-system, sans-serif;
  --mono: 'JetBrains Mono', 'Geist Mono', monospace;

  display: grid;
  grid-template-rows: 52px 1fr;
  height: 100vh; width: 100vw;
  background: var(--bg); color: var(--t);
  font-family: var(--sans); font-size: 13px; line-height: 1.5;
  letter-spacing: -0.003em;
  -webkit-font-smoothing: antialiased;
  font-feature-settings: 'ss01' on, 'ss02' on;
  overflow: hidden;
}

/* ─── ICON BUTTON (shared) ──────────────────────────────────────────── */
.fx-icbtn {
  width: 30px; height: 30px; border-radius: 6px;
  display:flex; align-items:center; justify-content:center;
  background: transparent; border: none; color: var(--t2);
  cursor: pointer; font-family: var(--mono); font-size: 14px;
  transition: background .12s, color .12s;
  position: relative;
}
.fx-icbtn:hover { background: var(--hover); color: var(--t); }
.fx-icbtn.on { background: var(--ac-bg); color: var(--ac); }
.fx-icbtn .nb {
  position: absolute; top: 6px; right: 6px;
  width: 6px; height: 6px; border-radius: 50%; background: var(--pink);
}
.fx-icbtn.sm { width: 26px; height: 26px; font-size: 13px; }

/* ─── PILLS / TAGS ──────────────────────────────────────────────────── */
.fx-pill {
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 11px; padding: 2px 8px; border-radius: 14px;
  background: var(--raised); color: var(--t2); font-weight: 500;
}
.fx-pill .dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.fx-pill.ac { background: var(--ac-bg); color: var(--ac); }
.fx-pill.yellow { background: var(--yellow-bg); color: var(--yellow); }
.fx-pill.pink { background: var(--pink-bg); color: var(--pink); }
.fx-pill.green { background: var(--green-bg); color: var(--green); }
.fx-pill.orange { background: var(--orange-bg); color: var(--orange); }
.fx-pill.red { background: var(--red-bg); color: var(--red); }
.fx-pill.outline {
  background: transparent; border: 1px solid var(--bd2);
}

/* ─── TIMESTAMP PILL ─────────────────────────────────────────────────── */
/* Frame.io's signature yellow inline timestamp */
.fx-tc {
  display: inline-flex; align-items: center; gap: 3px;
  font-family: var(--mono); font-size: 10.5px; font-weight: 600;
  padding: 1px 6px; border-radius: 3px;
  background: var(--yellow-bg); color: var(--yellow);
  font-variant-numeric: tabular-nums;
}
.fx-tc::before { content: ''; width: 5px; height: 5px; border-radius: 50%; background: var(--yellow); }
.fx-tc.bare { background: transparent; padding: 1px 3px; }

/* ─── INPUT ──────────────────────────────────────────────────────────── */
.fx-input {
  background: var(--panel); border: 1px solid var(--bd);
  padding: 7px 11px; border-radius: 6px;
  display:flex; align-items:center; gap: 9px;
  transition: border-color .12s, background .12s, box-shadow .12s;
}
.fx-input:focus-within {
  border-color: var(--ac); background: var(--raised);
  box-shadow: 0 0 0 3px var(--ac-bg-low);
}
.fx-input .ico { color: var(--muted); font-size: 14px; }
.fx-input input {
  flex: 1; background: transparent; border: none; outline: none;
  font: inherit; color: var(--t); font-size: 13.5px;
}
.fx-input input::placeholder { color: var(--muted); }
.fx-input .kbd {
  font-family: var(--mono); font-size: 10px; padding: 1px 6px;
  border: 1px solid var(--bd2); border-radius: 3px; color: var(--muted);
  background: var(--bg);
}

/* ─── BUTTONS ────────────────────────────────────────────────────────── */
.fx-btn {
  display: inline-flex; align-items: center; gap: 7px;
  padding: 6px 13px; border-radius: 6px;
  font-family: var(--sans); font-size: 12.5px; font-weight: 500;
  cursor: pointer; border: 1px solid transparent;
  transition: background .12s, color .12s, border-color .12s;
}
.fx-btn .kbd {
  font-family: var(--mono); font-size: 10px; padding: 0 5px;
  border-radius: 3px; opacity: 0.7;
}
.fx-btn.primary {
  background: var(--ac); color: #FFFFFF; font-weight: 600;
}
.fx-btn.primary:hover { background: var(--ac2); }
.fx-btn.primary .kbd { background: rgba(0,0,0,0.18); }
.fx-btn.secondary {
  background: var(--raised); border-color: var(--bd2); color: var(--t);
}
.fx-btn.secondary:hover { background: var(--hover); border-color: var(--bd3); }
.fx-btn.secondary .kbd { background: var(--bg); border: 1px solid var(--bd); color: var(--muted); }
.fx-btn.ghost {
  background: transparent; color: var(--t2);
}
.fx-btn.ghost:hover { background: var(--hover); color: var(--t); }

/* ─── PANEL HEADER (small caps) ─────────────────────────────────────── */
.fx-ph {
  display: flex; align-items: baseline; justify-content: space-between;
  padding: 14px 14px 6px;
  font-size: 11px; font-weight: 500; color: var(--muted);
  letter-spacing: 0.04em;
}
.fx-ph .ct {
  font-family: var(--mono); font-size: 10.5px; color: var(--faint);
}
.fx-ph .add {
  width: 20px; height: 20px; border-radius: 4px;
  display: flex; align-items: center; justify-content: center;
  background: transparent; border: none; color: var(--muted);
  cursor: pointer; font-size: 14px;
}
.fx-ph .add:hover { background: var(--hover); color: var(--t); }

/* ─── SECTION DIVIDER ───────────────────────────────────────────────── */
.fx-hr { border-top: 1px solid var(--bd); margin: 8px 0; }

/* generic utilities */
.fx-mono { font-family: var(--mono); font-variant-numeric: tabular-nums; }
.fx-truncate { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
`;

// ─── ICONS ──────────────────────────────────────────────────────────────
// 18px Lucide-ish line icons. All use stroke="currentColor", no fill.
const I = {
  home: () => <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M3 10.5L12 3l9 7.5V21H3z"/><path d="M9 21v-7h6v7"/></svg>,
  search: () => <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg>,
  grid: () => <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>,
  tag: () => <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M21 13l-9 9-9-9V3h10z"/><circle cx="7.5" cy="7.5" r="1.5"/></svg>,
  rhymes: () => <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><circle cx="6" cy="6" r="3"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="18" r="3"/><path d="M9 6h6M9 18h6M6 9v6M18 9v6"/></svg>,
  bell: () => <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M18 8a6 6 0 0 0-12 0c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.7 21a2 2 0 0 1-3.4 0"/></svg>,
  upload: () => <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>,
  share: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>,
  folder: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>,
  film: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><rect x="2" y="3" width="20" height="18" rx="2"/><line x1="7" y1="3" x2="7" y2="21"/><line x1="17" y1="3" x2="17" y2="21"/><line x1="2" y1="9" x2="7" y2="9"/><line x1="2" y1="15" x2="7" y2="15"/><line x1="17" y1="9" x2="22" y2="9"/><line x1="17" y1="15" x2="22" y2="15"/></svg>,
  chevR: () => <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="9 18 15 12 9 6"/></svg>,
  chevD: () => <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="6 9 12 15 18 9"/></svg>,
  chevL: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="15 18 9 12 15 6"/></svg>,
  panelL: () => <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="9" y1="3" x2="9" y2="21"/></svg>,
  panelR: () => <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="15" y1="3" x2="15" y2="21"/></svg>,
  panelB: () => <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="15" x2="21" y2="15"/></svg>,
  filter: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>,
  sort: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M3 6h18M6 12h12M10 18h4"/></svg>,
  group: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="6" rx="1"/><rect x="3" y="13" width="18" height="6" rx="1"/></svg>,
  fields: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2l3 7 7 .5-5.5 4.5 2 7L12 17l-6.5 4 2-7L2 9.5 9 9z"/></svg>,
  appearance: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>,
  comment: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M21 11.5a8.4 8.4 0 0 1-9 8.5 9 9 0 0 1-3.6-.7L3 21l1.7-5.4A8.4 8.4 0 0 1 12 3a8.4 8.4 0 0 1 9 8.5z"/></svg>,
  pin: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><path d="M12 17.5V22"/><path d="M15 9V4l-3-2-3 2v5L4 14h16z"/></svg>,
  more: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><circle cx="5" cy="12" r="1.6"/><circle cx="12" cy="12" r="1.6"/><circle cx="19" cy="12" r="1.6"/></svg>,
  download: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>,
  link: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M10 13a5 5 0 0 0 7.5.5l3-3a5 5 0 0 0-7-7L11.6 5"/><path d="M14 11a5 5 0 0 0-7.5-.5l-3 3a5 5 0 0 0 7 7l1.8-1.8"/></svg>,
  play: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="6 4 20 12 6 20"/></svg>,
  pause: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>,
  loop: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>,
  volume: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>,
  settings: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9c.36.16.7.4 1 .7.3.3.54.64.7 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>,
  expand: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>,
  send: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>,
  attach: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M21 11.5l-9 9a6 6 0 0 1-8.5-8.5l9-9a4 4 0 0 1 5.7 5.7L9.3 17.2a2 2 0 0 1-2.8-2.8L15 6"/></svg>,
  emoji: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="9"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>,
  check: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>,
  proc: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12a9 9 0 1 1-6.2-8.5"/></svg>,
  circle: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"><circle cx="12" cy="12" r="9"/></svg>,
  x: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>,
  plus: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>,
  image: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>,
  audio: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M2 12h2l3-9 4 18 3-12 2 6h6"/></svg>,
  doc: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>,
  globe: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="9"/><line x1="3" y1="12" x2="21" y2="12"/><path d="M12 3a14 14 0 0 1 4 9 14 14 0 0 1-4 9 14 14 0 0 1-4-9 14 14 0 0 1 4-9z"/></svg>,
  arrowR: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>,
};

// Mark — small film-strip with purple gel
function FXMark({size=22}) {
  return (
    <svg width={size} height={size} viewBox="0 0 22 22" fill="none">
      <rect x="0.5" y="2.5" width="21" height="17" stroke={FX.text} strokeWidth="1.2" rx="3"/>
      <rect x="2.5" y="4.5" width="2.5" height="2.5" rx="0.7" fill={FX.text}/>
      <rect x="2.5" y="9.5" width="2.5" height="2.5" rx="0.7" fill={FX.text}/>
      <rect x="2.5" y="14.5" width="2.5" height="2.5" rx="0.7" fill={FX.text}/>
      <rect x="17"  y="4.5" width="2.5" height="2.5" rx="0.7" fill={FX.text}/>
      <rect x="17"  y="9.5" width="2.5" height="2.5" rx="0.7" fill={FX.text}/>
      <rect x="17"  y="14.5" width="2.5" height="2.5" rx="0.7" fill={FX.text}/>
      <rect x="7"   y="6" width="8" height="10" rx="1.5" fill={FX.ac}/>
    </svg>
  );
}

window.FX = FX;
window.FX_FILM = FX_FILM;
window.FX_CSS = FX_CSS;
window.I = I;
window.FXMark = FXMark;
