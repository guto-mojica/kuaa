// Cinemateca Mojica — Linear branch
// 3-pane mechanic preserved. Visual language reskinned as a serious
// keyboard-first instrument: indigo accent, dense issue-style row list,
// minimal chrome, command palette as centerpiece, status icons everywhere.

const L_PALETTE = {
  bg:        '#08090A',
  panel:     '#101114',
  raised:    '#16171B',
  hover:     '#1B1D22',
  border:    '#21232B',
  border2:   '#2E3038',
  text:      '#E8E9EE',
  text2:     '#9CA0A8',
  muted:     '#6C7079',
  faint:     '#4A4D55',
  accent:    '#5E6AD2',
  accent2:   '#7E8AE6',
  accentBg:  'rgba(94,106,210,0.12)',
  accentBgLow:'rgba(94,106,210,0.05)',
  green:     '#4CB782',
  yellow:    '#F2C94C',
  orange:    '#F2994A',
  red:       '#EB5757',
  purple:    '#BB87FC',
};

// Per-film color identity (used for project badges / dots)
const FILM_COLORS = {
  jeca:       L_PALETTE.accent,
  limite:     L_PALETTE.red,
  rio40:      L_PALETTE.orange,
  cangaceiro: L_PALETTE.yellow,
  aruanda:    L_PALETTE.green,
  pagador:    L_PALETTE.purple,
};

const L_CSS = `
.l-app, .l-app * { box-sizing: border-box; }
.l-app *::-webkit-scrollbar { width: 8px; height: 8px; }
.l-app *::-webkit-scrollbar-thumb { background: ${L_PALETTE.border2}; border-radius: 4px; }
.l-app *::-webkit-scrollbar-track { background: transparent; }

.l-app {
  --bg: ${L_PALETTE.bg};
  --panel: ${L_PALETTE.panel};
  --raised: ${L_PALETTE.raised};
  --hover: ${L_PALETTE.hover};
  --bd: ${L_PALETTE.border};
  --bd2: ${L_PALETTE.border2};
  --t: ${L_PALETTE.text};
  --t2: ${L_PALETTE.text2};
  --muted: ${L_PALETTE.muted};
  --faint: ${L_PALETTE.faint};
  --ac: ${L_PALETTE.accent};
  --ac2: ${L_PALETTE.accent2};
  --ac-bg: ${L_PALETTE.accentBg};
  --ac-bg-low: ${L_PALETTE.accentBgLow};
  --green: ${L_PALETTE.green};
  --yellow: ${L_PALETTE.yellow};
  --orange: ${L_PALETTE.orange};
  --red: ${L_PALETTE.red};
  --purple: ${L_PALETTE.purple};
  --sans: 'Geist', system-ui, sans-serif;
  --mono: 'JetBrains Mono', 'Geist Mono', monospace;
  display: grid;
  grid-template-rows: 44px 1fr;
  height: 100vh; width: 100vw;
  background: var(--bg); color: var(--t);
  font-family: var(--sans); font-size: 13px; line-height: 1.45;
  letter-spacing: -0.005em;
  font-feature-settings: 'ss01' on, 'cv11' on;
  -webkit-font-smoothing: antialiased;
  overflow: hidden;
}

/* TOPBAR */
.l-top {
  display:flex; align-items:center; justify-content: space-between;
  padding: 0 14px; border-bottom: 1px solid var(--bd);
  background: var(--bg);
}
.l-top .left { display:flex; align-items:center; gap: 10px; }
.l-top .ws {
  display:flex; align-items:center; gap: 9px; padding: 5px 9px;
  border-radius: 5px; cursor: pointer; font-size: 13px; color: var(--t);
  font-weight: 500;
}
.l-top .ws:hover { background: var(--hover); }
.l-top .ws .av {
  width: 22px; height: 22px; border-radius: 5px;
  background: linear-gradient(135deg, var(--ac), var(--purple));
  display:flex; align-items:center; justify-content:center;
  font-family: var(--mono); font-size: 11px; font-weight: 600; color: #fff;
}
.l-top .ws .chev { color: var(--muted); font-size: 10px; }
.l-top .crumb {
  display:flex; align-items:center; gap: 6px;
  font-size: 12.5px; color: var(--t2);
}
.l-top .crumb .sep { color: var(--faint); margin: 0 1px; }
.l-top .crumb .cur { color: var(--t); }
.l-top .center {
  display:flex; align-items:center; gap: 6px;
  background: var(--panel); border: 1px solid var(--bd);
  padding: 5px 10px 5px 11px; border-radius: 6px;
  min-width: 320px; cursor: text; color: var(--muted);
  font-size: 12.5px;
}
.l-top .center .ico { font-size: 13px; }
.l-top .center .text { flex: 1; }
.l-top .center .kbd {
  font-family: var(--mono); font-size: 10px; padding: 1px 5px;
  border-radius: 3px; background: var(--raised); color: var(--t2);
  border: 1px solid var(--bd);
}
.l-top .right { display:flex; align-items: center; gap: 6px; }
.l-top .iconbtn {
  width: 28px; height: 28px; border-radius: 5px;
  display:flex; align-items:center; justify-content:center;
  color: var(--t2); cursor: pointer; font-size: 14px;
  background: transparent; border: none;
}
.l-top .iconbtn:hover { background: var(--hover); color: var(--t); }
.l-top .new {
  display:flex; align-items:center; gap: 6px;
  padding: 5px 11px; border-radius: 5px;
  background: var(--raised); border: 1px solid var(--bd2);
  color: var(--t); font-size: 12px; font-weight: 500; cursor: pointer;
}
.l-top .new:hover { background: var(--hover); }
.l-top .new .kbd {
  font-family: var(--mono); font-size: 10px; padding: 0 4px;
  background: var(--bg); border: 1px solid var(--bd); border-radius: 3px;
  color: var(--muted);
}

/* BODY */
.l-body {
  display: grid;
  grid-template-columns: 232px 1fr 380px;
  overflow: hidden; min-height: 0;
}

/* LEFT */
.l-lp {
  border-right: 1px solid var(--bd);
  display:flex; flex-direction:column; background: var(--bg); overflow: hidden;
}
.l-lp .sect {
  padding: 14px 14px 4px;
  font-size: 11px; font-weight: 500; letter-spacing: 0.04em;
  color: var(--muted); display:flex; align-items:center; justify-content:space-between;
}
.l-lp .sect .add {
  width: 18px; height: 18px; border-radius: 4px;
  display:flex; align-items:center; justify-content:center;
  color: var(--muted); cursor: pointer; font-size: 13px;
}
.l-lp .sect .add:hover { background: var(--hover); color: var(--t); }

.l-row {
  display:grid; grid-template-columns: 18px 1fr auto;
  align-items:center; gap: 8px;
  padding: 5px 14px; margin: 0; cursor: pointer;
  font-size: 13px; color: var(--t2); position: relative;
}
.l-row:hover { background: var(--hover); }
.l-row.on { background: var(--ac-bg); color: var(--t); }
.l-row.on::before {
  content:''; position:absolute; left: 0; top: 4px; bottom: 4px;
  width: 2px; background: var(--ac); border-radius: 2px;
}
.l-row .ico { color: var(--muted); font-size: 13px; }
.l-row.on .ico { color: var(--ac); }
.l-row .ct {
  font-family: var(--mono); font-size: 11px; color: var(--muted);
  font-variant-numeric: tabular-nums;
}
.l-row .ct.pill {
  background: var(--raised); padding: 1px 5px; border-radius: 3px; color: var(--t2);
}

.l-films { padding: 0 0 6px; }
.l-film {
  display:grid; grid-template-columns: 18px 1fr auto;
  align-items:center; gap: 8px;
  padding: 5px 14px; cursor: pointer; position: relative;
  font-size: 13px; color: var(--t2);
}
.l-film:hover { background: var(--hover); }
.l-film.on { background: var(--ac-bg); color: var(--t); }
.l-film.on::before {
  content:''; position:absolute; left: 0; top: 4px; bottom: 4px;
  width: 2px; background: var(--ac); border-radius: 2px;
}
.l-film .dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--c, var(--ac));
  display:inline-block; justify-self: center;
}
.l-film .name {
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.l-film .ct {
  font-family: var(--mono); font-size: 10.5px; color: var(--muted);
  font-variant-numeric: tabular-nums;
  display: flex; align-items: center; gap: 4px;
}
.l-film .ct .m {
  color: var(--ac); padding: 1px 5px; border-radius: 3px;
  background: var(--ac-bg);
}
.l-film.proc .name { color: var(--yellow); }
.l-film.proc .dot { background: var(--yellow) !important; }

/* foot */
.l-lp-foot {
  margin-top: auto; padding: 10px 14px;
  border-top: 1px solid var(--bd);
  display: grid; grid-template-columns: 1fr 1fr; gap: 4px;
  font-size: 11px; color: var(--muted);
}
.l-lp-foot .row { display:flex; align-items:center; justify-content:space-between; gap: 6px; }
.l-lp-foot .row .v { color: var(--t); font-family: var(--mono); font-size: 10.5px; }

/* CENTER */
.l-cp { display:flex; flex-direction:column; min-width: 0; overflow: hidden; background: var(--bg); }

.l-cp .header {
  display:flex; align-items:center; justify-content:space-between;
  padding: 14px 22px 6px; gap: 18px;
}
.l-cp .header .title {
  display:flex; align-items: center; gap: 10px;
  font-size: 15px; font-weight: 600; color: var(--t); letter-spacing: -0.01em;
}
.l-cp .header .title .cnt {
  font-family: var(--mono); font-size: 11.5px; color: var(--muted);
  font-weight: 400;
}
.l-cp .header .right { display:flex; align-items:center; gap: 4px; }
.l-cp .header .right .iconbtn {
  width: 26px; height: 26px; border-radius: 5px;
  display:flex; align-items:center; justify-content:center;
  background: transparent; color: var(--muted); border: none;
  cursor: pointer; font-size: 13px;
}
.l-cp .header .right .iconbtn:hover { background: var(--hover); color: var(--t); }

.l-search {
  display:flex; align-items:center; gap: 10px;
  margin: 10px 22px 0;
  background: var(--panel); border: 1px solid var(--bd);
  border-radius: 6px; padding: 7px 11px;
  transition: border-color .12s, background .12s;
}
.l-search:focus-within {
  border-color: var(--ac); background: var(--raised);
  box-shadow: 0 0 0 3px var(--ac-bg-low);
}
.l-search .ico { color: var(--muted); font-size: 13px; }
.l-search input {
  flex:1; background: transparent; border: none; outline: none;
  font: inherit; color: var(--t); font-size: 13.5px;
}
.l-search input::placeholder { color: var(--muted); }
.l-search .kbd {
  font-family: var(--mono); font-size: 10px; padding: 1px 5px;
  border: 1px solid var(--bd2); border-radius: 3px; color: var(--muted);
}
.l-search .submit {
  font-size: 11.5px; font-weight: 500; color: var(--t); cursor: pointer;
  padding: 4px 10px; border-radius: 4px;
  background: var(--ac); color: #fff; border: none;
  display:flex; align-items:center; gap: 5px;
}
.l-search .submit .k { font-family: var(--mono); font-size: 10px; opacity: 0.7; }

.l-filters {
  display:flex; align-items:center; gap: 6px; padding: 10px 22px;
  border-bottom: 1px solid var(--bd); flex-wrap: wrap;
}
.l-fchip {
  display:flex; align-items:center; gap: 6px;
  padding: 3px 10px 3px 8px; border-radius: 4px;
  font-size: 11.5px; color: var(--t2); cursor: pointer;
  background: transparent; border: 1px solid var(--bd);
  font-weight: 500;
}
.l-fchip:hover { background: var(--hover); border-color: var(--bd2); }
.l-fchip.on { background: var(--ac-bg); border-color: transparent; color: var(--ac); }
.l-fchip .lab { color: var(--muted); }
.l-fchip.on .lab { color: var(--ac2); }
.l-fchip .ico { font-size: 12px; }
.l-fchip .x { color: var(--muted); cursor: pointer; }
.l-fchip .x:hover { color: var(--t); }
.l-filters .grow { flex: 1; }
.l-filters .sort {
  display:flex; align-items:center; gap: 6px; font-size: 11.5px;
  color: var(--muted); cursor: pointer; padding: 3px 10px; border-radius: 4px;
}
.l-filters .sort:hover { background: var(--hover); }
.l-filters .sort b { color: var(--t); font-weight: 500; }

/* result column header */
.l-listhead {
  display:grid;
  grid-template-columns: 56px 18px 80px 1fr auto auto auto auto;
  gap: 12px; align-items:center;
  padding: 6px 22px; border-bottom: 1px solid var(--bd);
  font-size: 10.5px; font-weight: 500; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.06em;
  background: var(--bg);
}
.l-listhead .col {
  display:flex; align-items:center; gap: 5px; cursor: pointer; min-width: 0;
}
.l-listhead .col:hover { color: var(--t); }
.l-listhead .col.right { justify-content: flex-end; }

/* result list */
.l-list { flex: 1; overflow-y: auto; }
.l-issue {
  display:grid;
  grid-template-columns: 56px 18px 80px 1fr auto auto auto auto;
  gap: 12px; align-items:center;
  padding: 7px 22px; border-bottom: 1px solid var(--bd);
  cursor: pointer; position: relative;
  transition: background .08s;
}
.l-issue:hover { background: var(--hover); }
.l-issue.sel { background: var(--ac-bg); }
.l-issue.sel::before {
  content:''; position: absolute; left: 0; top: 0; bottom: 0; width: 2px;
  background: var(--ac);
}
.l-issue .thumb {
  width: 56px; height: 42px; border-radius: 4px;
  background-size: cover; background-position: center;
  filter: contrast(1.05) brightness(0.94);
  border: 1px solid var(--bd);
}
.l-issue .stat {
  width: 14px; height: 14px; border-radius: 50%;
  border: 1.5px solid var(--green); position: relative;
  display: inline-block;
}
.l-issue .stat::before {
  content: ''; position: absolute; inset: 2px; border-radius: 50%;
  background: var(--green);
}
.l-issue .stat.indexed { border-color: var(--green); }
.l-issue .stat.indexed::before { background: var(--green); }
.l-issue .stat.queued { border-color: var(--muted); }
.l-issue .stat.queued::before { background: transparent; }
.l-issue .stat.proc { border-color: var(--yellow); }
.l-issue .stat.proc::before { background: var(--yellow); width: 6px; height: 6px; top: 3px; left: 3px; }
.l-issue .id {
  font-family: var(--mono); font-size: 11.5px; color: var(--muted);
  letter-spacing: 0.01em; font-variant-numeric: tabular-nums;
}
.l-issue.sel .id { color: var(--t); }
.l-issue .desc {
  font-size: 13px; color: var(--t); min-width: 0;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  letter-spacing: -0.003em;
}
.l-issue .lab {
  display:flex; gap: 4px;
}
.l-issue .lab .l {
  display:flex; align-items:center; gap: 4px;
  font-size: 10.5px; padding: 1px 7px 1px 6px; border-radius: 9px;
  background: var(--raised); color: var(--t2); white-space: nowrap;
}
.l-issue .lab .l .d { width: 6px; height: 6px; border-radius: 50%; }
.l-issue .lab .l.m { background: var(--ac-bg); color: var(--ac); }
.l-issue .lab .l.m .d { background: var(--ac); }
.l-issue .prio {
  display:flex; align-items:center; gap: 5px;
  font-size: 11.5px; color: var(--t2); width: 80px;
}
.l-issue .prio .gly { font-family: var(--mono); font-size: 12px; }
.l-issue .prio .gly.u { color: var(--red); }
.l-issue .prio .gly.h { color: var(--orange); }
.l-issue .prio .gly.m { color: var(--yellow); }
.l-issue .prio .gly.l { color: var(--muted); }

.l-issue .proj {
  display:flex; align-items:center; gap: 6px;
  font-size: 11.5px; color: var(--t2); white-space: nowrap;
}
.l-issue .proj .dot {
  width: 8px; height: 8px; border-radius: 50%;
}
.l-issue .tc {
  font-family: var(--mono); font-size: 10.5px; color: var(--muted);
  font-variant-numeric: tabular-nums; min-width: 78px; text-align: right;
}
.l-issue .score {
  font-family: var(--mono); font-size: 11px; color: var(--t);
  font-variant-numeric: tabular-nums; min-width: 50px; text-align: right;
}
.l-issue.sel .score { color: var(--ac); }

/* RIGHT — Issue detail */
.l-rp {
  border-left: 1px solid var(--bd);
  background: var(--panel);
  display:flex; flex-direction:column; overflow: hidden;
}
.l-rp .htop {
  display:flex; align-items:center; justify-content:space-between;
  padding: 8px 14px; border-bottom: 1px solid var(--bd);
}
.l-rp .htop .id {
  display:flex; align-items:center; gap: 8px;
  font-family: var(--mono); font-size: 11.5px; color: var(--muted);
}
.l-rp .htop .id .stat {
  width: 14px; height: 14px; border-radius: 50%;
  border: 1.5px solid var(--green); position: relative;
}
.l-rp .htop .id .stat::before {
  content:''; position:absolute; inset: 2px; border-radius: 50%; background: var(--green);
}
.l-rp .htop .id .v { color: var(--t); }
.l-rp .htop .acts { display:flex; align-items:center; gap: 2px; }
.l-rp .htop .acts .iconbtn {
  width: 26px; height: 26px; display:flex; align-items:center; justify-content:center;
  border-radius: 4px; background: transparent; color: var(--muted);
  cursor: pointer; font-size: 13px; border: none;
}
.l-rp .htop .acts .iconbtn:hover { background: var(--hover); color: var(--t); }

.l-rp .inner { overflow-y: auto; flex:1; padding: 16px 18px 18px; }
.l-insp-kf {
  width: 100%; aspect-ratio: 4/3; border-radius: 6px;
  background-size: cover; background-position: center;
  background-color: var(--bg);
  filter: contrast(1.05) brightness(0.96);
  border: 1px solid var(--bd);
}
.l-rp h1 {
  margin: 14px 0 0; font-size: 19px; font-weight: 600; color: var(--t);
  letter-spacing: -0.014em; line-height: 1.25;
}
.l-rp .sub {
  font-size: 12px; color: var(--muted); margin-top: 4px;
}
.l-rp .sub b { color: var(--t2); font-weight: 500; }

.l-props {
  display: grid; grid-template-columns: 86px 1fr;
  row-gap: 6px; column-gap: 12px; margin-top: 18px;
  padding-top: 14px; border-top: 1px solid var(--bd);
  font-size: 12px;
}
.l-props .k {
  color: var(--muted); font-size: 11.5px;
  display: flex; align-items: center; gap: 6px;
}
.l-props .k .ico { color: var(--faint); font-size: 12px; }
.l-props .v {
  color: var(--t); display:flex; align-items:center; gap: 6px; flex-wrap: wrap;
}
.l-props .v .chip {
  display:inline-flex; align-items:center; gap: 5px;
  padding: 2px 8px; border-radius: 9px;
  font-size: 11.5px; background: var(--raised); color: var(--t);
}
.l-props .v .chip .dot { width: 7px; height: 7px; border-radius: 50%; }
.l-props .v .chip.prio { background: transparent; padding: 0; }
.l-props .v .chip.prio .gly { color: var(--orange); font-family: var(--mono); }
.l-props .v .chip.stat { background: var(--raised); }
.l-props .v .chip.stat .dot { background: var(--green); }
.l-props .v .chip.ac { background: var(--ac-bg); color: var(--ac); }
.l-props .v .mono {
  font-family: var(--mono); font-size: 11.5px; color: var(--t);
  font-variant-numeric: tabular-nums;
}

.l-sect {
  display:flex; align-items:center; justify-content:space-between;
  margin-top: 22px; padding-bottom: 7px;
  border-bottom: 1px solid var(--bd);
  font-size: 11px; color: var(--muted); font-weight: 500;
  letter-spacing: 0.02em;
}
.l-sect a {
  color: var(--ac); font-size: 11px; cursor: pointer; font-family: var(--mono);
}
.l-rp .desc {
  font-size: 13px; color: var(--t2); line-height: 1.55; margin-top: 10px;
  text-wrap: pretty;
}

.l-signals { margin-top: 10px; }
.l-sig {
  display:grid; grid-template-columns: 80px 1fr 48px;
  align-items: center; gap: 10px; margin-bottom: 6px;
  font-family: var(--mono); font-size: 11px; color: var(--t2);
  font-variant-numeric: tabular-nums;
}
.l-sig .lab { color: var(--muted); font-size: 10.5px; }
.l-sig .track { height: 4px; background: var(--bd); border-radius: 2px; position: relative; }
.l-sig .track::before {
  content:''; position:absolute; left:0; top:0; bottom:0;
  width: var(--p); background: var(--ac); border-radius: 2px;
}
.l-sig.fused .track::before { background: var(--green); }
.l-sig .v { text-align: right; color: var(--t); }

.l-rhymes { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 10px; }
.l-ry { cursor: pointer; }
.l-ry .kf {
  width: 100%; aspect-ratio: 4/3; border-radius: 4px;
  background-size: cover; background-position: center;
  background-color: var(--bg);
  filter: contrast(1.05) brightness(0.95);
  border: 1px solid var(--bd);
  transition: border-color .12s;
}
.l-ry:hover .kf { border-color: var(--ac); }
.l-ry .lab {
  font-size: 10.5px; color: var(--t2); margin-top: 4px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.l-ry .lab .dot {
  display:inline-block; width: 6px; height: 6px; border-radius: 50%;
  margin-right: 5px; vertical-align: middle;
}
.l-ry .sc { font-family: var(--mono); font-size: 10px; color: var(--muted); }

.l-comments { margin-top: 10px; display:flex; flex-direction:column; gap: 12px; }
.l-com {
  display:grid; grid-template-columns: 26px 1fr; gap: 9px;
}
.l-com .av {
  width: 24px; height: 24px; border-radius: 4px;
  background: var(--raised); display:flex; align-items:center; justify-content:center;
  font-family: var(--mono); font-size: 10px; font-weight: 600; color: var(--t);
}
.l-com.ai .av { background: linear-gradient(135deg, var(--ac), var(--purple)); color: #fff; }
.l-com .bod { display:flex; flex-direction: column; gap: 3px; }
.l-com .who {
  display:flex; align-items:baseline; gap: 6px;
  font-size: 12px;
}
.l-com .who .n { color: var(--t); font-weight: 500; }
.l-com .who .ago { color: var(--muted); font-family: var(--mono); font-size: 10.5px; }
.l-com .body { font-size: 12.5px; color: var(--t2); line-height: 1.5; text-wrap: pretty; }

.l-rp .composer {
  margin-top: 14px; padding: 10px 12px;
  background: var(--bg); border: 1px solid var(--bd); border-radius: 6px;
}
.l-rp .composer textarea {
  width: 100%; resize: none; background: transparent;
  border: none; outline: none; min-height: 36px;
  font: inherit; font-size: 12.5px; color: var(--t);
}
.l-rp .composer textarea::placeholder { color: var(--muted); }
.l-rp .composer .actions {
  display:flex; justify-content:space-between; align-items:center; margin-top: 6px;
}
.l-rp .composer .actions .l {
  font-family: var(--mono); font-size: 10.5px; color: var(--muted);
}
.l-rp .composer .actions .send {
  background: var(--ac); color: #fff; border: none; border-radius: 4px;
  padding: 5px 11px; font-size: 11.5px; font-weight: 500; cursor: pointer;
}
`;

function MojicaLinear() {
  const films = window.FILMS;
  const results = window.RESULTS;
  const byId = Object.fromEntries(films.map(f => [f.id, f]));
  const [sel, setSel] = React.useState(0);

  React.useEffect(() => {
    if (!document.getElementById('l-css')) {
      const s = document.createElement('style');
      s.id = 'l-css'; s.textContent = L_CSS;
      document.head.appendChild(s);
    }
  }, []);

  const r = results[sel];
  const f = byId[r.film];

  // Film ID used as project key prefix
  const filmKey = (id) => (
    id === 'jeca'       ? 'JECA' :
    id === 'limite'     ? 'LIMT' :
    id === 'rio40'      ? 'R40G' :
    id === 'cangaceiro' ? 'CANG' :
    id === 'aruanda'    ? 'ARUA' :
                          'PAGD'
  );

  // Per-film match counts
  const matches = {};
  results.forEach(rr => matches[rr.film] = (matches[rr.film] || 0) + 1);

  // Priority tier from score
  const prio = (s) => {
    if (s >= 0.85) return {gly: '⫷', cls: 'u', lab: 'Crítica'};
    if (s >= 0.80) return {gly: '⫸', cls: 'h', lab: 'Alta'};
    if (s >= 0.75) return {gly: '═', cls: 'm', lab: 'Média'};
    return {gly: '─', cls: 'l', lab: 'Baixa'};
  };

  const sigSem  = Math.min(0.96, r.score + 0.06);
  const sigBm25 = Math.max(0.04, r.score - 0.52);
  const sigRk   = Math.min(0.96, r.score - 0.01);
  const sigFu   = r.score;

  const rhymes = results.filter((x, i) => x.film !== r.film && i !== sel).slice(0, 3)
    .map((x, i) => ({...x, sim: (0.94 - i*0.04).toFixed(2)}));

  return (
    <div className="l-app">
      {/* TOP */}
      <div className="l-top">
        <div className="left">
          <div className="ws">
            <div className="av">M</div>
            <span>Mojica</span>
            <span className="chev">▾</span>
          </div>
          <div className="crumb">
            <span className="sep">/</span>
            <span>Acervo</span>
            <span className="sep">/</span>
            <span className="cur">Buscar</span>
          </div>
        </div>
        <div className="center">
          <span className="ico">⌕</span>
          <span className="text">Procurar cenas, filmes, tags…</span>
          <span className="kbd">⌘K</span>
        </div>
        <div className="right">
          <button className="iconbtn" title="Inbox">⌘</button>
          <button className="iconbtn" title="Help">?</button>
          <button className="new">+ Nova cena <span className="kbd">C</span></button>
        </div>
      </div>

      {/* BODY */}
      <div className="l-body">
        {/* LEFT */}
        <aside className="l-lp">
          <div style={{padding: '10px 14px 4px'}}>
            <div className="l-row on">
              <span className="ico">⌕</span>
              <span>Buscar</span>
              <span className="ct pill">9</span>
            </div>
            <div className="l-row">
              <span className="ico">⊞</span>
              <span>Cenas</span>
              <span className="ct">1.588</span>
            </div>
            <div className="l-row">
              <span className="ico">⊟</span>
              <span>Anotar</span>
              <span className="ct">231</span>
            </div>
            <div className="l-row">
              <span className="ico">∿</span>
              <span>Rimas visuais</span>
              <span className="ct">new</span>
            </div>
            <div className="l-row">
              <span className="ico">⟳</span>
              <span>Processamento</span>
              <span className="ct" style={{color: L_PALETTE.yellow}}>1 ativo</span>
            </div>
          </div>

          <div className="sect"><span>Filmes</span><span className="add">+</span></div>
          <div className="l-films">
            {films.map(film => {
              const cnt = matches[film.id] || 0;
              const isProc = film.id === 'aruanda';
              const hasSel = film.id === r.film;
              return (
                <div key={film.id}
                     className={'l-film' + (hasSel ? ' on' : '') + (isProc ? ' proc' : '')}
                     style={{'--c': FILM_COLORS[film.id]}}>
                  <span className="dot"></span>
                  <span className="name">{film.title}</span>
                  <span className="ct">
                    {cnt > 0 && <span className="m">{cnt}</span>}
                    <span>{film.scenes}</span>
                  </span>
                </div>
              );
            })}
          </div>

          <div className="sect"><span>Vistas</span><span className="add">+</span></div>
          <div style={{padding: '0 0 6px'}}>
            <div className="l-row"><span className="ico">◐</span><span>Exteriores rurais</span><span className="ct">142</span></div>
            <div className="l-row"><span className="ico">◐</span><span>Cartelas</span><span className="ct">28</span></div>
            <div className="l-row"><span className="ico">◐</span><span>Sem descrição</span><span className="ct">3</span></div>
            <div className="l-row"><span className="ico">◐</span><span>Anotações da semana</span><span className="ct">12</span></div>
          </div>

          <div className="l-lp-foot">
            <div className="row"><span>Cenas</span><span className="v">1.588</span></div>
            <div className="row"><span>Runtime</span><span className="v">8h54m</span></div>
            <div className="row"><span>Idx</span><span className="v" style={{color: L_PALETTE.green}}>ok</span></div>
            <div className="row"><span>Idioma</span><span className="v">PT</span></div>
          </div>
        </aside>

        {/* CENTER */}
        <section className="l-cp">
          <div className="header">
            <div className="title">
              Buscar <span className="cnt">· 9 cenas</span>
            </div>
            <div className="right">
              <button className="iconbtn" title="Group">⊟</button>
              <button className="iconbtn" title="Filter">⊕</button>
              <button className="iconbtn" title="Display">▤</button>
              <button className="iconbtn" title="Options">⋯</button>
            </div>
          </div>

          <div className="l-search">
            <span className="ico">⌕</span>
            <input defaultValue="duas pessoas conversando ao ar livre" />
            <span className="kbd">⌘F</span>
            <button className="submit">Buscar <span className="k">⏎</span></button>
          </div>

          <div className="l-filters">
            <span className="l-fchip on"><span className="ico">●</span><span className="lab">Modo:</span>texto<span className="x">×</span></span>
            <span className="l-fchip"><span className="ico">▣</span><span className="lab">Filme:</span>todos</span>
            <span className="l-fchip"><span className="ico">⌖</span><span className="lab">Híbrido:</span>0.70 / 0.30</span>
            <span className="l-fchip"><span className="ico">↻</span><span className="lab">Rerank:</span>on</span>
            <span className="l-fchip"><span className="ico">λ</span><span className="lab">MMR:</span>0.5</span>
            <span className="l-fchip on">+ duas-pessoas <span className="x">×</span></span>
            <span className="l-fchip on">+ exterior <span className="x">×</span></span>
            <span className="l-fchip" style={{borderStyle: 'dashed'}}>+ Filtro</span>
            <span className="grow"></span>
            <span className="sort">Ordem: <b>relevância</b> ▾</span>
          </div>

          <div className="l-listhead">
            <span></span>
            <span className="col">●</span>
            <span className="col">ID</span>
            <span className="col">Descrição da cena</span>
            <span className="col">Tags</span>
            <span className="col">Prio.</span>
            <span className="col">Filme</span>
            <span className="col right">Score</span>
          </div>

          <div className="l-list">
            {results.map((rr, i) => {
              const ff = byId[rr.film];
              const p = prio(rr.score);
              return (
                <div key={rr.id}
                     className={'l-issue' + (i === sel ? ' sel' : '')}
                     onClick={() => setSel(i)}>
                  <span className="thumb" style={{backgroundImage:`url(${rr.kf})`}}></span>
                  <span className="stat indexed"></span>
                  <span className="id">{filmKey(rr.film)}-{String(rr.cena).padStart(3,'0')}</span>
                  <span className="desc">{rr.desc}</span>
                  <span className="lab">
                    {rr.tags.slice(0,2).map((t,j) => (
                      <span key={j} className={'l' + (t==='duas-pessoas'||t==='exterior' ? ' m' : '')}>
                        <span className="d"></span>{t}
                      </span>
                    ))}
                  </span>
                  <span className="prio">
                    <span className={'gly ' + p.cls}>{p.gly}</span>
                    <span>{p.lab}</span>
                  </span>
                  <span className="proj">
                    <span className="dot" style={{background: FILM_COLORS[rr.film]}}></span>
                    {ff.title}
                  </span>
                  <span className="score">{rr.score.toFixed(3)}</span>
                </div>
              );
            })}
          </div>
        </section>

        {/* RIGHT */}
        <aside className="l-rp">
          <div className="htop">
            <div className="id">
              <span className="stat"></span>
              <span>Indexado</span>
              <span>·</span>
              <span className="v">{filmKey(r.film)}-{String(r.cena).padStart(3,'0')}</span>
            </div>
            <div className="acts">
              <button className="iconbtn" title="Copy">⌘</button>
              <button className="iconbtn" title="Open">↗</button>
              <button className="iconbtn" title="More">⋯</button>
            </div>
          </div>
          <div className="inner">
            <div className="l-insp-kf" style={{backgroundImage:`url(${r.kf})`}}></div>
            <h1>{r.desc.split(',')[0]}.</h1>
            <div className="sub">
              Cena <b>{String(r.cena).padStart(3,'0')}</b> em <b>{f.title}</b> ({f.year})
              · timecode <b style={{fontFamily:'var(--mono)'}}>{r.tc}</b>
            </div>

            <div className="l-props">
              <span className="k"><span className="ico">●</span>Status</span>
              <span className="v"><span className="chip stat"><span className="dot"></span>Indexado</span></span>

              <span className="k"><span className="ico">▤</span>Filme</span>
              <span className="v">
                <span className="chip">
                  <span className="dot" style={{background: FILM_COLORS[r.film]}}></span>
                  {f.title}
                </span>
                <span className="mono">{f.year}</span>
              </span>

              <span className="k"><span className="ico">◐</span>Diretor</span>
              <span className="v">{f.director}</span>

              <span className="k"><span className="ico">⏷</span>Prioridade</span>
              <span className="v">
                <span className="chip prio"><span className="gly">{prio(r.score).gly}</span></span>
                <span>{prio(r.score).lab}</span>
              </span>

              <span className="k"><span className="ico">∇</span>Score</span>
              <span className="v"><span className="mono" style={{color: L_PALETTE.accent}}>{r.score.toFixed(3)}</span></span>

              <span className="k"><span className="ico">⏱</span>Timecode</span>
              <span className="v"><span className="mono">{r.tc}</span></span>

              <span className="k"><span className="ico">#</span>Tags</span>
              <span className="v">
                {r.tags.slice(0,4).map((t,i) => (
                  <span key={i} className={'chip' + (t==='duas-pessoas'||t==='exterior' ? ' ac' : '')}>
                    <span className="dot" style={{background: (t==='duas-pessoas'||t==='exterior' ? L_PALETTE.accent : L_PALETTE.muted)}}></span>
                    {t}
                  </span>
                ))}
                <span className="chip" style={{background:'transparent', border: '1px dashed ' + L_PALETTE.border2, color: L_PALETTE.muted}}>+</span>
              </span>
            </div>

            <div className="l-sect"><span>Descrição</span><a>moondream-2</a></div>
            <p className="desc">{r.desc}</p>

            <div className="l-sect"><span>Por que este resultado</span><span style={{fontFamily:'var(--mono)', color: L_PALETTE.text}}>{r.score.toFixed(3)}</span></div>
            <div className="l-signals">
              <div className="l-sig"><span className="lab">Semântico</span><span className="track" style={{'--p': `${(sigSem*100).toFixed(0)}%`}}></span><span className="v">{sigSem.toFixed(3)}</span></div>
              <div className="l-sig"><span className="lab">BM25</span><span className="track" style={{'--p': `${(sigBm25*100).toFixed(0)}%`}}></span><span className="v">{sigBm25.toFixed(3)}</span></div>
              <div className="l-sig"><span className="lab">Rerank</span><span className="track" style={{'--p': `${(sigRk*100).toFixed(0)}%`}}></span><span className="v">{sigRk.toFixed(3)}</span></div>
              <div className="l-sig fused"><span className="lab">Fundido</span><span className="track" style={{'--p': `${(sigFu*100).toFixed(0)}%`}}></span><span className="v">{sigFu.toFixed(3)}</span></div>
            </div>

            <div className="l-sect"><span>Rimas visuais</span><a>ver todas</a></div>
            <div className="l-rhymes">
              {rhymes.map((x, i) => {
                const ff = byId[x.film];
                return (
                  <div key={i} className="l-ry">
                    <div className="kf" style={{backgroundImage:`url(${x.kf})`}}></div>
                    <span className="lab"><span className="dot" style={{background: FILM_COLORS[x.film]}}></span>{ff.title}</span>
                    <span className="sc">{x.sim} · cena {String(x.cena).padStart(3,'0')}</span>
                  </div>
                );
              })}
            </div>

            <div className="l-sect"><span>Atividade · 2</span><a>tudo</a></div>
            <div className="l-comments">
              <div className="l-com ai">
                <div className="av">md</div>
                <div className="bod">
                  <div className="who"><span className="n">moondream-2</span><span className="ago">há 4 dias</span></div>
                  <div className="body">Descrição gerada automaticamente. Tags propostas: <span style={{color: L_PALETTE.accent}}>duas-pessoas</span>, <span style={{color: L_PALETTE.accent}}>exterior</span>, <span style={{color: L_PALETTE.text2}}>rural-field</span>.</div>
                </div>
              </div>
              <div className="l-com">
                <div className="av">RG</div>
                <div className="bod">
                  <div className="who"><span className="n">Rafael · curador</span><span className="ago">há 2h</span></div>
                  <div className="body">Cena de referência para a vertente "diálogos no campo aberto". Boa candidata para a retrospectiva 2026.</div>
                </div>
              </div>
            </div>

            <div className="composer">
              <textarea placeholder="Escrever comentário…"></textarea>
              <div className="actions">
                <span className="l">⌘+⏎ enviar</span>
                <button className="send">Comentar</button>
              </div>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}

window.MojicaLinear = MojicaLinear;
